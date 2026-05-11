import socket
import sys
import time
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import struct
import numpy as np
from scipy.signal import butter, lfilter, iirnotch, sosfilt
from pylsl import StreamInfo, StreamOutlet


class Config:
    DEFAULT_PLOT_TIME = 1      # seconds
    UPDATE_RATE = 16           # milliseconds (~60 FPS)
    PLOT_HEIGHT = 600          # pixels
    WINDOW_SIZE = (1200, 800)  # width, height


class EMGProcessor:
    def __init__(self, fs=2000):
        self.fs = fs
        self.sos_band = butter(4, [20, 450], btype='band', fs=fs, output='sos')

        self.notches = []
        for freq in [50, 100, 150]:
            b, a = iirnotch(freq, 30, fs=fs)
            self.notches.append((b, a))

        self.zi_sos = None
        self.zi_notches = None

    def process(self, data_chunk):
        # data_chunk shape: (n_channels, n_samples)
        n_channels = data_chunk.shape[0]

        if self.zi_sos is None:
            n_sections = self.sos_band.shape[0]
            self.zi_sos = np.zeros((n_sections, n_channels, 2))

            self.zi_notches = []
            for b, a in self.notches:
                self.zi_notches.append(np.zeros((n_channels, max(len(a), len(b)) - 1)))

        filtered, self.zi_sos = sosfilt(self.sos_band, data_chunk, axis=1, zi=self.zi_sos)

        for i, (b, a) in enumerate(self.notches):
            filtered, self.zi_notches[i] = lfilter(b, a, filtered, axis=1, zi=self.zi_notches[i])

        return filtered


class Track:
    def __init__(self, title, frequency, num_channels, acq_channel, offset, conv_fact, plot_time=1):
        self.title = title
        self.frequency = frequency
        self.num_channels = num_channels
        self.acq_channel = acq_channel
        self.offset = offset
        self.conv_fact = conv_fact
        self.plot_time = plot_time
        self.buffer = np.zeros((num_channels, int(plot_time * frequency)))
        self.buffer_index = 0
        self.time_array = np.linspace(0, self.plot_time, self.buffer.shape[1])

        self.plot_widget = pg.PlotWidget(title=self.title)
        self.plot_widget.setXRange(0, self.plot_time)

        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        if 'HDsEMG' in title:
            self.plot_widget.setLabel('left', 'Amplitude', units='µV')
        else:
            self.plot_widget.setLabel('left', 'Amplitude', units='A.U.')

        self.plot_widget.setLabel('bottom', 'Time', units='s')

        self.plot_widget.getViewBox().setBackgroundColor((30, 30, 30))
        self.plot_widget.setAntialiasing(True)

        self.plot_widget.setYRange(-self.offset, self.num_channels * self.offset)

        self.curves = []
        for i in range(num_channels):
            if title in ('AUX 1', 'AUX 2', 'Buffer', 'Ramp'):
                pen = pg.mkPen(color=(255, 255, 255), width=1)
            else:
                pen = pg.mkPen(color=i, width=1)

            curve_name = f"Ch {i+1}" if i < 8 or num_channels <= 8 else None
            curve = self.plot_widget.plot(pen=pen, name=curve_name)
            self.curves.append(curve)

    def feed(self, packet):
        packet_size = packet.shape[1]
        if self.buffer_index + packet_size > self.buffer.shape[1]:
            end_space = self.buffer.shape[1] - self.buffer_index
            if end_space > 0:
                self.buffer[:, self.buffer_index:] = packet[:, :end_space]
            self.buffer[:, :packet_size-end_space] = packet[:, end_space:]
            self.buffer_index = packet_size - end_space
        else:
            self.buffer[:, self.buffer_index:self.buffer_index + packet_size] = packet
            self.buffer_index = (self.buffer_index + packet_size) % self.buffer.shape[1]

    def draw(self):
        for index, curve in enumerate(self.curves):
            curve.setData(self.time_array, self.buffer[index, :] * self.conv_fact + (self.offset * index))


class DataReceiverThread(QtCore.QThread):
    data_received = QtCore.pyqtSignal(np.ndarray)
    status_update = QtCore.pyqtSignal(str)

    def __init__(self, device, client_socket, tracks):
        super().__init__()
        self.device = device
        self.client_socket = client_socket
        self.tracks = tracks
        self.running = True
        self.packet_count = 0
        self.last_time = time.time()
        self.fps = 0
        self.processor = EMGProcessor(fs=self.device.frequency)
        self.leftover = b''

        self.emg_channels = self.tracks[0].num_channels
        self.lsl_info = StreamInfo('OTB_S64_EMG', 'EMG', self.emg_channels, self.device.frequency, 'float32', 'sessantaquattro_s01')
        self.lsl_outlet = StreamOutlet(self.lsl_info)

    def run(self):
        bytes_per_frame = self.device.nchannels * 2

        while self.running:
            try:
                data = self.client_socket.recv(bytes_per_frame * (self.device.frequency // 16))
                if not data:
                    print("No data received, connection may be closed")
                    break

                valid_bytes = (len(data) // bytes_per_frame) * bytes_per_frame
                if valid_bytes == 0:
                    continue

                valid_data = data[:valid_bytes]
                # Little-endian per il Sessantaquattro standard (manuale p.16)
                unpacked_data = struct.unpack(f'<{valid_bytes // 2}h', valid_data)
                reshaped_data = np.array(unpacked_data).reshape((-1, self.device.nchannels)).T

                emg_for_lsl = None

                for track in self.tracks:
                    if 'Bipolar' in track.title or 'Differential' in track.title:
                        raw_channels_needed = track.num_channels * 2
                        chunk_raw = reshaped_data[track.acq_channel : track.acq_channel + raw_channels_needed, :]

                        chunk_uv = chunk_raw * 0.286102

                        bipolar_chunk = np.zeros((track.num_channels, chunk_uv.shape[1]))
                        bipolar_chunk[0, :] = chunk_uv[0, :] - chunk_uv[1, :]
                        if track.num_channels > 1:
                            bipolar_chunk[1, :] = chunk_uv[2, :] - chunk_uv[3, :]

                        chunk_filtered = self.processor.process(bipolar_chunk)

                        emg_for_lsl = chunk_filtered
                        track.feed(chunk_filtered)

                    else:
                        chunk = reshaped_data[track.acq_channel:track.acq_channel + track.num_channels, :]
                        if 'HDsEMG' in track.title:
                            chunk_uv = chunk * 0.286102
                            chunk_filtered = self.processor.process(chunk_uv)
                            chunk = chunk_filtered
                            emg_for_lsl = chunk_filtered

                        track.feed(chunk)

                if emg_for_lsl is not None:
                    samples_to_push = emg_for_lsl.T.astype(np.float32)
                    self.lsl_outlet.push_chunk(samples_to_push.tolist())

                self.data_received.emit(reshaped_data)

                self.packet_count += 1
                if self.packet_count % 100 == 0:
                    current_time = time.time()
                    elapsed = current_time - self.last_time
                    self.fps = 100 / elapsed if elapsed > 0 else 0
                    self.last_time = current_time
                    self.status_update.emit(f"Data rate: {self.fps:.1f} packets/second")

            except Exception as e:
                print(f"Error receiving data: {e}")
                break

    def stop(self):
        print("Stopping data receiver thread")
        self.running = False


class Soundtrack(QtWidgets.QWidget):
    def __init__(self, device, client_socket):
        super().__init__()
        self.device = device
        self.client_socket = client_socket
        self.tracks = []
        self.plot_time = Config.DEFAULT_PLOT_TIME
        self.is_paused = False

        self.setWindowTitle("Sessantaquattro Data Visualization")
        self.setGeometry(100, 100, *Config.WINDOW_SIZE)

        self.main_layout = QtWidgets.QVBoxLayout(self)

        self.menu_widget = QtWidgets.QWidget()
        self.menu_layout = QtWidgets.QHBoxLayout(self.menu_widget)

        self.time_selector = QtWidgets.QComboBox()
        self.time_selector.addItems(['100ms', '250ms', '500ms', '1s', '5s', '10s'])
        self.time_selector.setCurrentText(f"{Config.DEFAULT_PLOT_TIME}s")
        self.time_selector.currentTextChanged.connect(self.change_plot_time)

        self.menu_layout.addWidget(QtWidgets.QLabel("Plot Time:"))
        self.menu_layout.addWidget(self.time_selector)

        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self.toggle_pause)
        self.menu_layout.addWidget(self.pause_button)

        self.status_label = QtWidgets.QLabel("Ready")
        self.menu_layout.addWidget(self.status_label)

        self.menu_layout.addStretch()

        self.main_layout.addWidget(self.menu_widget)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)

        self.main_layout.addWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_widget)

        self.init_tracks()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(Config.UPDATE_RATE)

        self.receiver_thread = DataReceiverThread(self.device, self.client_socket, self.tracks)
        self.receiver_thread.status_update.connect(self.update_status)
        self.receiver_thread.start()

    def init_tracks(self):
        # Sessantaquattro standard: 4 canali ausiliari fissi (AUX1, AUX2, Buffer, Ramp)
        # NCH=2 o NCH=3 con MODE=1: 32 canali EMG
        if self.device.nchannels == 36:
            track_info = [
                ('HDsEMG 32 channels', 32, 0, 500, 1),
                ('AUX 1', 1, 32, 1, 0.00014648),
                ('AUX 2', 1, 33, 1, 0.00014648),
                ('Buffer', 1, 34, 1, 1),
                ('Ramp',   1, 35, 1, 1),
            ]
        # NCH=3, MODE=0: 64 canali EMG
        elif self.device.nchannels == 68:
            track_info = [
                ('HDsEMG 64 channels', 64, 0, 500, 1),
                ('AUX 1', 1, 64, 1, 0.00014648),
                ('AUX 2', 1, 65, 1, 0.00014648),
                ('Buffer', 1, 66, 1, 1),
                ('Ramp',   1, 67, 1, 1),
            ]
        # NCH=0, MODE=0: 8 canali EMG, configurazione bipolare
        elif self.device.nchannels == 12:
            track_info = [
                ('Bipolar EMG (2 ch)', 2, 0, 500, 1),
                ('AUX 1', 1, 8,  1, 0.00014648),
                ('AUX 2', 1, 9,  1, 0.00014648),
                ('Buffer', 1, 10, 1, 1),
                ('Ramp',   1, 11, 1, 1),
            ]
        else:
            # Fallback generico: 4 canali ausiliari alla fine
            main_channels = self.device.nchannels - 4
            track_info = [
                (f'HDsEMG {main_channels} channels', main_channels, 0, 500, 1),
                ('AUX 1', 1, main_channels,     1, 0.00014648),
                ('AUX 2', 1, main_channels + 1, 1, 0.00014648),
            ]

        for title, n_channels, acq_channel, offset, conv_fact in track_info:
            track_container = QtWidgets.QWidget()
            track_layout = QtWidgets.QVBoxLayout(track_container)

            track = Track(title, self.device.frequency, n_channels, acq_channel, offset, conv_fact, self.plot_time)
            self.tracks.append(track)

            track.plot_widget.setMinimumHeight(300)

            track_layout.addWidget(track.plot_widget)

            self.scroll_layout.addWidget(track_container)

        self.scroll_layout.addStretch()

    def change_plot_time(self, time_str):
        if time_str.endswith('ms'):
            new_time = float(time_str[:-2]) / 1000
        else:
            new_time = float(time_str[:-1])

        print(f"Changing plot time to {new_time} seconds")

        for track in self.tracks:
            new_buffer = np.zeros((track.num_channels, int(new_time * track.frequency)))

            if track.buffer_index > 0:
                copy_size = min(new_buffer.shape[1], track.buffer.shape[1])
                new_buffer[:, -copy_size:] = track.buffer[:, -copy_size:]

            track.plot_time = new_time
            track.buffer = new_buffer
            track.buffer_index = min(track.buffer_index, new_buffer.shape[1])
            track.time_array = np.linspace(0, track.plot_time, track.buffer.shape[1])

            track.plot_widget.setXRange(0, new_time)

    def toggle_pause(self, checked):
        self.is_paused = checked
        self.pause_button.setText("Resume" if checked else "Pause")
        if checked:
            self.timer.stop()
            print("Visualization paused")
        else:
            self.timer.start(Config.UPDATE_RATE)
            print("Visualization resumed")

    def update_status(self, message):
        self.status_label.setText(message)

    def update_plot(self):
        if not self.is_paused:
            for track in self.tracks:
                track.draw()

    def closeEvent(self, event):
        print("Closing application")
        self.receiver_thread.stop()
        self.receiver_thread.wait()
        self.client_socket.close()
        event.accept()


class Sessantaquattro:
    def __init__(self, host="0.0.0.0", port=45454):
        self.host = host
        self.port = port
        self.nchannels = 68
        self.frequency = 2000
        self.server_socket = None
        self.client_socket = None

    def get_num_channels(self, NCH, MODE):
        """Sessantaquattro standard: 4 canali in meno rispetto al Plus (no IMU/quaternioni).

        NCH=3, MODE=0 → 68 (64 EMG + 2 AUX + Buffer + Ramp)
        """
        if NCH == 0:
            return 8  if MODE == 1 else 12
        elif NCH == 1:
            return 12 if MODE == 1 else 20
        elif NCH == 2:
            return 20 if MODE == 1 else 36
        elif NCH == 3:
            return 36 if MODE == 1 else 68
        return 68

    def get_sampling_frequency(self, FSAMP, MODE):
        frequencies = {
            0: 500,
            1: 1000,
            2: 2000,
            3: 4000
        }
        return frequencies.get(FSAMP, 2000)

    def create_command(self, FSAMP=2, NCH=3, MODE=0, HRES=0, HPF=0, EXTEN=0, TRIG=0, REC=0, GO=1):
        self.nchannels = self.get_num_channels(NCH, MODE)
        self.frequency = self.get_sampling_frequency(FSAMP, MODE)

        Command = 0
        Command = Command + GO
        Command = Command + (REC   << 1)
        Command = Command + (TRIG  << 2)
        Command = Command + (EXTEN << 4)
        Command = Command + (HPF   << 6)
        Command = Command + (HRES  << 7)
        Command = Command + (MODE  << 8)
        Command = Command + (NCH   << 11)
        Command = Command + (FSAMP << 13)

        print(f"Command in binary: {format(Command, '016b')}")
        return Command

    def start_server(self, command):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"Server listening on {self.host}:{self.port}...")

            self.client_socket, addr = self.server_socket.accept()
            print(f"Connection accepted from {addr}")
            # Little-endian per il Sessantaquattro standard (manuale p.16)
            self.client_socket.send(command.to_bytes(2, byteorder='little', signed=True))

        except socket.error as e:
            print(f"Error creating server: {e}")
            sys.exit(1)

    def stop_server(self):
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()


def main():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=True)

    device = Sessantaquattro()

    FSAMP = 2  # 2000 Hz (0=500Hz, 1=1000Hz, 2=2000Hz, 3=4000Hz)
    NCH = 3    # 64 canali (0=8, 1=16, 2=32, 3=64)
    MODE = 0   # Standard mode
    HRES = 0   # Normal resolution
    HPF = 1    # High-pass filter enabled
    EXTEN = 0  # External trigger disabled
    TRIG = 0   # Trigger mode disabled
    REC = 0    # Recording disabled
    GO = 1     # Start acquisition

    command = device.create_command(
        FSAMP=FSAMP, NCH=NCH, MODE=MODE,
        HRES=HRES, HPF=HPF, EXTEN=EXTEN,
        TRIG=TRIG, REC=REC, GO=GO
    )

    device.start_server(command)

    window = Soundtrack(device, device.client_socket)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
