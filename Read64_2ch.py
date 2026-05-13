
import sys
import numpy as np
import socket
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from scipy.signal import butter, lfilter, iirnotch, sosfilt
from pylsl import StreamInfo, StreamOutlet
import communication_sessantaquattro as communication

class Config:
    DEFAULT_PLOT_TIME = 1      
    UPDATE_RATE = 16           
    WINDOW_SIZE = (1200, 800)

class EMGProcessor:
    def __init__(self, fs=2000):
        self.fs = fs
        nyq = self.fs / 2
        
        self.sos_band = butter(4, [20, min(450, nyq - 5)], btype='band', fs=fs, output='sos')
        self.notches = [iirnotch(freq, 30, fs=fs) for freq in [50,60, 100, 150] if freq < nyq]
        
        self.zi_sos = None
        self.zi_notches = None

    def process(self, data_chunk):
        n_channels = data_chunk.shape[0]
        if self.zi_sos is None:
            self.zi_sos = np.zeros((self.sos_band.shape[0], n_channels, 2))
            self.zi_notches = [np.zeros((n_channels, 2)) for _ in self.notches]

        filtered, self.zi_sos = sosfilt(self.sos_band, data_chunk, axis=1, zi=self.zi_sos)
        for i, (b, a) in enumerate(self.notches):
            filtered, self.zi_notches[i] = lfilter(b, a, filtered, axis=1, zi=self.zi_notches[i])
        return filtered

class Track:
    def __init__(self, title, frequency, num_channels, offset, conv_fact, plot_time=1):
        self.title = title
        self.num_channels = num_channels
        self.offset = offset
        self.conv_fact = conv_fact
        self.plot_time = plot_time
        
        self.buffer = np.zeros((num_channels, int(plot_time * frequency)))
        self.buffer_index = 0
        self.time_array = np.linspace(0, self.plot_time, self.buffer.shape[1])

        self.plot_widget = pg.PlotWidget(title=self.title)
        self.plot_widget.setXRange(0, self.plot_time)
        self.plot_widget.setYRange(-3500, 3500)  # Range fisso a +- 5 mV (+- 5000 µV)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude', units='µV' if 'EMG' in title else 'A.U.')
        self.plot_widget.getViewBox().setBackgroundColor((30, 30, 30))
        
        self.curves = [self.plot_widget.plot(pen=pg.mkPen(color=i, width=1)) for i in range(num_channels)]

    def feed(self, packet):
        p_size = packet.shape[1]
        if self.buffer_index + p_size > self.buffer.shape[1]:
            self.buffer = np.roll(self.buffer, -p_size, axis=1)
            self.buffer[:, -p_size:] = packet
            self.buffer_index = self.buffer.shape[1]
        else:
            self.buffer[:, self.buffer_index:self.buffer_index + p_size] = packet
            self.buffer_index += p_size

    def draw(self):
        for i, curve in enumerate(self.curves):
            curve.setData(self.time_array, (self.buffer[i, :] * self.conv_fact) + (self.offset * i))

class DataReceiverThread(QtCore.QThread):
    data_received = QtCore.pyqtSignal(np.ndarray)
    status_update = QtCore.pyqtSignal(str)

    def __init__(self, connection, num_channels, bytes_in_sample, sample_freq):
        super().__init__()
        self.connection = connection
        self.num_channels = num_channels 
        self.bytes_in_sample = bytes_in_sample
        self.sample_freq = sample_freq
        self.running = True
        
        self.active_bio_channels =8
        self.processor = EMGProcessor(fs=self.sample_freq)
        self.lsl_outlet = StreamOutlet(StreamInfo('Sessantaquattro_EMG', 'EMG', 
                                                  self.active_bio_channels, self.sample_freq, 
                                                  'float32', 's64_pisa'))

    def run(self):
        chunk_size = max(1, int(self.sample_freq / 60))
        
        while self.running:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    raw_bytes = communication.read_raw_bytes(self.connection, self.num_channels, self.bytes_in_sample)
                    if not raw_bytes: break
                    
                    integers = communication.bytes_to_integers(raw_bytes, self.num_channels, self.bytes_in_sample, False)
                    chunk_data.append(integers)

                if chunk_data:
                    reshaped = np.array(chunk_data).T
                    
                    # 1. Estrazione di tutti i canali
                    raw_channels = reshaped[:self.active_bio_channels, :]
                    
                    # 2. Conversione in microVolt e Filtraggio (Passa-banda + Notch)
                    channels_uv = raw_channels * 0.2861
                    channels_filtered = self.processor.process(channels_uv)

                    # 3. Stream su LSL dei canali puliti
                    self.lsl_outlet.push_chunk(channels_filtered.T.astype(np.float32).tolist())

                    # 4. Preparazione dati per i grafici
                    self.data_received.emit(channels_filtered)

            except Exception as e:
                self.status_update.emit(f"Errore: {e}")
                break

    def stop(self): self.running = False

class SoundtrackGUI(QtWidgets.QWidget):
    def __init__(self, connection, num_channels, sample_freq, bytes_in_sample):
        super().__init__()
        self.connection = connection
        self.num_channels, self.sample_freq, self.bytes_in_sample = num_channels, sample_freq, bytes_in_sample
        self.tracks, self.is_paused = [], False
        self.plot_time = Config.DEFAULT_PLOT_TIME  

        self.setWindowTitle("Sessantaquattro - Final Bipolar EMG & LSL")
        self.setGeometry(100, 100, *Config.WINDOW_SIZE)
        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel("Acquisizione in corso... LSL Attivo")
        layout.addWidget(self.status_label)
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.content)
        scroll.setWidget(self.content)
        layout.addWidget(scroll)
        
        self.init_tracks()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(Config.UPDATE_RATE)

        self.thread = DataReceiverThread(connection, num_channels, bytes_in_sample, sample_freq)
        self.thread.data_received.connect(self.on_data)
        self.thread.status_update.connect(lambda s: self.status_label.setText(s))
        self.thread.start()

    def init_tracks(self):
        # Mostriamo tutti i 8 canali
        t_info = [(f"Canale {i+1}", 1, 0, 1.0) for i in range(8)]
        
        for title, n, off, conv in t_info:
            t = Track(title, self.sample_freq, n, off, conv, self.plot_time)
            self.tracks.append(t)
            t.plot_widget.setMinimumHeight(200)
            self.scroll_layout.addWidget(t.plot_widget)

    def on_data(self, data):
        for i in range(8):
            self.tracks[i].feed(data[i:i+1, :])

    def update_plot(self):
        if not self.is_paused:
            for t in self.tracks: t.draw()

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        communication.disconnect_from_sq(self.connection)
        event.accept()
        # --- AGGIUNGI QUESTO BLOCCO ---
        if hasattr(self, 'server_socket'):
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
                self.server_socket.close()
            except Exception:
                pass

def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # --- AGGIUNGI QUESTA RIGA PER IL REUSE DELL'INDIRIZZO ---
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # --------------------------------------------------------
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    cmd, nch, fs, bis = communication.create_bin_command(start=1)
    print(f"Configurazione: {nch} canali (Hardware mode) @ {fs}Hz")
    
    try:
        conn = communication.connect_to_sq(sock, '0.0.0.0', 45454, cmd)
        win = SoundtrackGUI(conn, nch, fs, bis)
        # --- AGGIUNGI QUESTA RIGA PER PASSARE IL SERVER SOCKET ALLA GUI ---
        win.server_socket = sock
        # ------------------------------------------------------------------
        win.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Errore fatale: {e}")

if __name__ == "__main__": main()