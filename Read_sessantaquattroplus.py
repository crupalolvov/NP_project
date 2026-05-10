import socket
import sys
import time
import signal
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
        # 1. Passa-Banda in formato SOS (Stabilità matematica assoluta)
        self.sos_band = butter(4, [20, 450], btype='band', fs=fs, output='sos')
        
        # 2. Filtri Notch a 50Hz e armoniche (sono del 2° ordine, già stabili)
        self.notches = []
        for freq in [50, 150, 250, 350]:
            b, a = iirnotch(freq, 30, fs=fs)
            self.notches.append((b, a))
        
        # Inizializzazione delle variabili di stato
        self.zi_sos = None
        self.zi_notches = None

    def process(self, data_chunk):
        # data_chunk shape: (n_channels, n_samples)
        n_channels = data_chunk.shape[0]
        
        # Inizializziamo gli stati rigorosamente a ZERO per evitare derive
        if self.zi_sos is None:
            # Per SOS, lo stato ha dimensioni (canali, numero_sezioni, 2)
            n_sections = self.sos_band.shape[0]
            self.zi_sos = np.zeros((n_sections, n_channels, 2))
            
            self.zi_notches = []
            for b, a in self.notches:
                self.zi_notches.append(np.zeros((n_channels, max(len(a), len(b)) - 1)))
            
        # 1. Applicazione del Passa-Banda (Anti-Esplosione)
        filtered, self.zi_sos = sosfilt(self.sos_band, data_chunk, axis=1, zi=self.zi_sos)
        
        # 2. Applicazione a cascata dei filtri Notch
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

        # Create PlotWidget with enhanced interactive features
        self.plot_widget = pg.PlotWidget(title=self.title)
        self.plot_widget.setXRange(0, self.plot_time)

        # Enable mouse interactions
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Add labels and units
        if 'HDsEMG' in title:
            self.plot_widget.setLabel('left', 'Amplitude', units='µV')
        else:
            self.plot_widget.setLabel('left', 'Amplitude', units='A.U.')
        
        self.plot_widget.setLabel('bottom', 'Time', units='s')

        # Set background and enable antialiasing
        self.plot_widget.getViewBox().setBackgroundColor((30, 30, 30))
        self.plot_widget.setAntialiasing(True)
        
        # Disabilita l'autoscale impostando un range fisso sull'asse Y
        self.plot_widget.setYRange(-self.offset, self.num_channels * self.offset)
        
        # Get colors for this track type
        self.curves = []
        for i in range(num_channels):
            if title == 'AUX 1' or title == 'AUX 2':
                pen = pg.mkPen(color=(255, 255, 255), width=1)
            elif title == 'Quaternions':
                pen = pg.mkPen(color=(255, 255, 255), width=1)
            elif title == 'Buffer':
                pen = pg.mkPen(color=(255, 255, 255), width=1)
            elif title == 'Ramp':
                pen = pg.mkPen(color=(255, 255, 255), width=1)
            else:
                pen = pg.mkPen(color=i, width=1)

            curve_name = f"Ch {i+1}" if i < 8 or num_channels <= 8 else None
            curve = self.plot_widget.plot(pen=pen, name=curve_name)
            self.curves.append(curve)

    def feed(self, packet):
        packet_size = packet.shape[1]
        # Use buffer management
        if self.buffer_index + packet_size > self.buffer.shape[1]:
            # Calculate exactly how much data fits at the end
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

        # --- CONFIGURAZIONE LSL ---
        # Dinamicamente calcola i canali EMG (32)
        self.emg_channels = self.tracks[0].num_channels
        self.lsl_info = StreamInfo('OTB_S64_EMG', 'EMG', self.emg_channels, self.device.frequency, 'float32', 'sessantaquattro_sp01')
        self.lsl_outlet = StreamOutlet(self.lsl_info)

    def run(self):
        bytes_per_frame = self.device.nchannels * 2
        
        while self.running:
            try:
                data = self.client_socket.recv(bytes_per_frame * (self.device.frequency // 16))
                if not data:
                    print("No data received, connection may be closed")
                    break

                # TCP Framing Sicuro
                valid_bytes = (len(data) // bytes_per_frame) * bytes_per_frame
                if valid_bytes == 0:
                    continue
                    
                valid_data = data[:valid_bytes]
                unpacked_data = struct.unpack(f'>{valid_bytes // 2}h', valid_data)
                reshaped_data = np.array(unpacked_data).reshape((-1, self.device.nchannels)).T

                emg_for_lsl = None
                
                for track in self.tracks:
                    # --- GESTIONE DINAMICA EMG BIPOLARE ---
                    if 'Bipolar' in track.title or 'Differential' in track.title:
                        # Per 2 canali bipolari, dobbiamo leggere 4 canali fisici
                        raw_channels_needed = track.num_channels * 2
                        chunk_raw = reshaped_data[track.acq_channel : track.acq_channel + raw_channels_needed, :]
                        
                        # 1. Conversione in Microvolt
                        chunk_uv = chunk_raw * 0.286102
                        
                        # 2. Sottrazione Spaziale CORRETTA (Ripristino)
                        bipolar_chunk = np.zeros((track.num_channels, chunk_uv.shape[1]))
                        # Bipolare 1 (Rossa - Flessori): Ch0 - Ch1
                        bipolar_chunk[0, :] = chunk_uv[0, :] - chunk_uv[1, :]
                        # Bipolare 2 (Gialla - Estensori): Ch2 - Ch3
                        if track.num_channels > 1:
                            bipolar_chunk[1, :] = chunk_uv[2, :] - chunk_uv[3, :]
                        
                        # 3. Filtraggio (Passa-banda + Comb Notch)
                        chunk_filtered = self.processor.process(bipolar_chunk)
                        
                        emg_for_lsl = chunk_filtered
                        track.feed(chunk_filtered)
                        
                    # --- GESTIONE ALTRI CANALI (Es. AUX) ---
                    else:
                        chunk = reshaped_data[track.acq_channel:track.acq_channel + track.num_channels, :]
                        if 'HDsEMG' in track.title:
                            chunk_uv = chunk * 0.286102
                            chunk_filtered = self.processor.process(chunk_uv)
                            chunk = chunk_filtered
                            emg_for_lsl = chunk_filtered
                        
                        track.feed(chunk)

                # --- INVIO DATI LSL ---
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

        self.setWindowTitle("Sessantaquattro+ Data Visualization")
        self.setGeometry(100, 100, *Config.WINDOW_SIZE)

        # Create main layout
        self.main_layout = QtWidgets.QVBoxLayout(self)

        # Create menu bar widget
        self.menu_widget = QtWidgets.QWidget()
        self.menu_layout = QtWidgets.QHBoxLayout(self.menu_widget)

        # Create and setup the combo box for plot time
        self.time_selector = QtWidgets.QComboBox()
        self.time_selector.addItems(['100ms', '250ms', '500ms', '1s', '5s', '10s'])
        self.time_selector.setCurrentText(f"{Config.DEFAULT_PLOT_TIME}s")
        self.time_selector.currentTextChanged.connect(self.change_plot_time)

        # Add label and combo box to menu layout
        self.menu_layout.addWidget(QtWidgets.QLabel("Plot Time:"))
        self.menu_layout.addWidget(self.time_selector)
        
        # Add pause button
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self.toggle_pause)
        self.menu_layout.addWidget(self.pause_button)

        # Add status label
        self.status_label = QtWidgets.QLabel("Ready")
        self.menu_layout.addWidget(self.status_label)
        
        # Add stretch to push widgets to the left
        self.menu_layout.addStretch()  
        
        # Add menu widget to main layout
        self.main_layout.addWidget(self.menu_widget)
        
        # Create scroll area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Create widget to hold plots
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)
        
        # Add scroll area to main layout
        self.main_layout.addWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_widget)
        
        self.init_tracks()

        # Timer for plot updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(Config.UPDATE_RATE)

        self.receiver_thread = DataReceiverThread(self.device, self.client_socket, self.tracks)
        self.receiver_thread.status_update.connect(self.update_status)
        self.receiver_thread.start()

    def init_tracks(self):
        if self.device.nchannels == 40:  # Configurazione a 32 canali (32 + 8 accessori)
            track_info = [
                ('HDsEMG 32 channels', 32, 0, 500, 1),
                ('AUX 1', 1, 32, 1, 0.00014648),
                ('AUX 2', 1, 33, 1, 0.00014648),
                ('Quaternions', 4, 34, 1, 1),
                ('Buffer', 1, 38, 1, 1),
                ('Ramp', 1, 39, 1, 1),
            ]
        elif self.device.nchannels == 72:  # Full configuration (64 canali)
            track_info = [
                ('HDsEMG 64 channels', 64, 0, 500, 1),
                ('AUX 1', 1, 64, 1, 0.00014648),
                ('AUX 2', 1, 65, 1, 0.00014648),
                ('Quaternions', 4, 66, 1, 1),
                ('Buffer', 1, 70, 1, 1),
                ('Ramp', 1, 71, 1, 1),
            ]
        elif self.device.nchannels == 16:  # Configurazione hardware a 8 canali
            track_info = [
                ('Bipolar EMG (2 ch)', 2, 0, 500, 1),
                ('AUX 1', 1, 8, 1, 0.00014648),
                ('AUX 2', 1, 9, 1, 0.00014648),
            ]
        else:
            main_channels = self.device.nchannels - 8
            track_info = [
                (f'HDsEMG {main_channels} channels', main_channels, 0, 500, 1),
                ('AUX 1', 1, main_channels, 1, 0.00014648),
                ('AUX 2', 1, main_channels + 1, 1, 0.00014648),
            ]

        for title, n_channels, acq_channel, offset, conv_fact in track_info:
            # Create container widget for each track
            track_container = QtWidgets.QWidget()
            track_layout = QtWidgets.QVBoxLayout(track_container)
            
            track = Track(title, self.device.frequency, n_channels, acq_channel, offset, conv_fact, self.plot_time)
            self.tracks.append(track)

            # Set minimum height for the plot widget
            track.plot_widget.setMinimumHeight(300)

            # Add plot widget to track container
            track_layout.addWidget(track.plot_widget)

            # Add track container to scroll layout
            self.scroll_layout.addWidget(track_container)
        
        # Add stretch at the end to prevent unwanted spacing
        self.scroll_layout.addStretch()

    def change_plot_time(self, time_str):
        # Convert string time to seconds
        if time_str.endswith('ms'):
            new_time = float(time_str[:-2]) / 1000  # Convert milliseconds to seconds
        else:
            new_time = float(time_str[:-1])  # Remove 's' and convert to float
        
        print(f"Changing plot time to {new_time} seconds")
        
        # Update plot time for all tracks
        for track in self.tracks:
            # Create new buffer with new size
            new_buffer = np.zeros((track.num_channels, int(new_time * track.frequency)))
            
            # Copy existing data if possible
            if track.buffer_index > 0:
                # Calculate how much data to copy
                copy_size = min(new_buffer.shape[1], track.buffer.shape[1])
                new_buffer[:, -copy_size:] = track.buffer[:, -copy_size:]
            
            # Update track properties
            track.plot_time = new_time
            track.buffer = new_buffer
            track.buffer_index = min(track.buffer_index, new_buffer.shape[1])
            track.time_array = np.linspace(0, track.plot_time, track.buffer.shape[1])
            
            # Update plot x-axis range
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


class SessantaquattroPlus:
    def __init__(self, host="0.0.0.0", port=45454):
        self.host = host
        self.port = port
        self.nchannels = 72
        self.frequency = 2000
        self.server_socket = None
        self.client_socket = None

    def get_num_channels(self, NCH, MODE):
        """Calculate number of channels based on NCH and MODE settings"""
        if NCH == 0:  # 8 channels
            return 12 if MODE == 1 else 16
        elif NCH == 1:  # 16 channels
            return 16 if MODE == 1 else 24
        elif NCH == 2:  # 32 channels
            return 24 if MODE == 1 else 40
        elif NCH == 3:  # 64 channels
            return 40 if MODE == 1 else 72
        return 72  # default value

    def get_sampling_frequency(self, FSAMP, MODE):
        """Calculate sampling frequency based on FSAMP and MODE settings"""
        if MODE == 3:  # Accelerometer mode
            frequencies = {
                0: 2000,
                1: 4000,
                2: 8000,
                3: 16000
            }
        else:  # Other modes
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
        Command = Command + GO           # Bit 0
        Command = Command + (REC << 1)   # Bit 1
        Command = Command + (TRIG << 2)  # Bits 2-3
        Command = Command + (EXTEN << 4) # Bits 4-5
        Command = Command + (HPF << 6)   # Bit 6
        Command = Command + (HRES << 7)  # Bit 7
        Command = Command + (MODE << 8)  # Bits 8-10
        Command = Command + (NCH << 11)  # Bits 11-12
        Command = Command + (FSAMP << 13) # Bits 13-14+

        binary_command = format(Command, '016b')
        print(f"Command in binary: {binary_command}")
        return Command

    def start_server(self, command):
    # La riga "command = self.create_command()" è stata eliminata.
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"Server listening on {self.host}:{self.port}...")

            self.client_socket, addr = self.server_socket.accept()
            print(f"Connection accepted from {addr}")
            self.client_socket.send(command.to_bytes(2, byteorder='big', signed=True))

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
    
    # Create device instance with specific configuration
    device = SessantaquattroPlus()
    
    # Configure device with specific parameters
    FSAMP = 2  # 2000 Hz (0=500Hz, 1=1000Hz, 2=2000Hz, 3=4000Hz)
    NCH = 0   # 32 canali (0=8, 1=16, 2=32, 3=64)
    MODE = 0   # Standard mode
    HRES = 0   # Normal resolution
    HPF = 1    # High-pass filter enabled
    EXTEN = 0  # External trigger disabled
    TRIG = 0   # Trigger mode disabled
    REC = 0    # Recording disabled
    GO = 1     # Start acquisition

    # Create command and configure device
    command = device.create_command(
        FSAMP=FSAMP, NCH=NCH, MODE=MODE, 
        HRES=HRES, HPF=HPF, EXTEN=EXTEN, 
        TRIG=TRIG, REC=REC, GO=GO
    )
    
    # Start server with configured command
    device.start_server(command)

    # Create and show application window
    window = Soundtrack(device, device.client_socket)
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()