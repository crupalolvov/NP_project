#!python3
import sys
import time
import numpy as np
import socket
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# Librerie per il processamento e lo streaming LSL
from scipy.signal import butter, lfilter, iirnotch, sosfilt
from pylsl import StreamInfo, StreamOutlet

# Importa il tuo modulo di comunicazione originale
import communication_sessantaquattro as communication

class Config:
    DEFAULT_PLOT_TIME = 1      # secondi di visualizzazione
    UPDATE_RATE = 16           # refresh GUI (~60 FPS)
    WINDOW_SIZE = (1200, 800)

class EMGProcessor:
    """
    Gestisce il filtraggio digitale in tempo reale.
    Applica un passa-banda 20-450Hz e filtri notch per il rumore di rete.
    """
    def __init__(self, fs=2000):
        self.fs = fs
        nyq = self.fs / 2
        
        # Filtro passa-banda 20-450Hz (adattivo alla freq. di campionamento)
        low_cut = 20
        high_cut = min(450, nyq - 5) 
        self.sos_band = butter(4, [low_cut, high_cut], btype='band', fs=fs, output='sos')

        # Filtri Notch a 50Hz e armoniche
        self.notches = []
        for freq in [50, 100, 150]:
            if freq < nyq:
                b, a = iirnotch(freq, 30, fs=fs)
                self.notches.append((b, a))

        self.zi_sos = None
        self.zi_notches = None

    def process(self, data_chunk):
        n_channels = data_chunk.shape[0]
        if self.zi_sos is None:
            self.zi_sos = np.zeros((self.sos_band.shape[0], n_channels, 2))
            self.zi_notches = [np.zeros((n_channels, 2)) for _ in self.notches]

        # Applicazione filtri con mantenimento dello stato (zi)
        filtered, self.zi_sos = sosfilt(self.sos_band, data_chunk, axis=1, zi=self.zi_sos)
        for i, (b, a) in enumerate(self.notches):
            filtered, self.zi_notches[i] = lfilter(b, a, filtered, axis=1, zi=self.zi_notches[i])
        return filtered

class Track:
    """ Gestisce la singola finestra di plot in pyqtgraph. """
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
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude', units='µV' if 'EMG' in title else 'A.U.')
        self.plot_widget.getViewBox().setBackgroundColor((30, 30, 30))
        
        self.curves = [self.plot_widget.plot(pen=pg.mkPen(color=i, width=1)) for i in range(num_channels)]

    def feed(self, packet):
        p_size = packet.shape[1]
        if self.buffer_index + p_size > self.buffer.shape[1]:
            roll = p_size
            self.buffer = np.roll(self.buffer, -roll, axis=1)
            self.buffer[:, -p_size:] = packet
            self.buffer_index = self.buffer.shape[1]
        else:
            self.buffer[:, self.buffer_index:self.buffer_index + p_size] = packet
            self.buffer_index += p_size

    def draw(self):
        for i, curve in enumerate(self.curves):
            curve.setData(self.time_array, (self.buffer[i, :] * self.conv_fact) + (self.offset * i))

class DataReceiverThread(QtCore.QThread):
    """ Thread dedicato alla ricezione TCP per evitare drop di campioni. """
    data_received = QtCore.pyqtSignal(np.ndarray)
    status_update = QtCore.pyqtSignal(str)

    def __init__(self, connection, num_channels, bytes_in_sample, sample_freq):
        super().__init__()
        self.connection = connection
        self.num_channels = num_channels # Saranno 8 (2 EMG + 2 Ghost + 2 AUX + 2 Acc)
        self.bytes_in_sample = bytes_in_sample
        self.sample_freq = sample_freq
        self.running = True
        
        # Setup specifico per 2 canali bipolari
        self.active_bio_channels = 2
        self.processor = EMGProcessor(fs=self.sample_freq)
        self.lsl_outlet = StreamOutlet(StreamInfo('Sessantaquattro_EMG', 'EMG', 
                                                  self.active_bio_channels, self.sample_freq, 
                                                  'float32', 's64_pisa'))

    def run(self):
        # Leggiamo pacchetti di circa 16ms per aggiornare la GUI fluidamente
        chunk_size = max(1, int(self.sample_freq / 60))
        
        while self.running:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    # Ricezione sicura tramite il tuo modulo communication
                    raw_bytes = communication.read_raw_bytes(self.connection, self.num_channels, self.bytes_in_sample)
                    if not raw_bytes: break
                    
                    integers = communication.bytes_to_integers(raw_bytes, self.num_channels, self.bytes_in_sample, False)
                    chunk_data.append(integers)

                if chunk_data:
                    reshaped = np.array(chunk_data).T
                    
                    # 1. Elaborazione segnali Bio (EMG Bipolari)
                    # LSB = 286.1 nV -> 0.286 µV
                    bio_uv = reshaped[:self.active_bio_channels, :] * 0.2861
                    bio_filtered = self.processor.process(bio_uv)

                    # 2. Invio a LSL
                    self.lsl_outlet.push_chunk(bio_filtered.T.astype(np.float32).tolist())

                    # 3. Preparazione dati per GUI (EMG + ultimi 4 canali: AUX e Accessory)
                    gui_data = np.vstack((bio_filtered, reshaped[-4:, :]))
                    self.data_received.emit(gui_data)

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

        self.setWindowTitle("Sessantaquattro Pisa - Real-Time EMG & LSL")
        self.setGeometry(100, 100, *Config.WINDOW_SIZE)
        layout = QtWidgets.QVBoxLayout(self)

        # Header
        self.status_label = QtWidgets.QLabel("Connessione stabilita...")
        layout.addWidget(self.status_label)
        
        # Area Plot
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.content)
        scroll.setWidget(self.content)
        layout.addWidget(scroll)
        
        self.init_tracks()

        # Timer Rendering
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(Config.UPDATE_RATE)

        # Avvio Thread Ricezione
        self.thread = DataReceiverThread(connection, num_channels, bytes_in_sample, sample_freq)
        self.thread.data_received.connect(self.on_data)
        self.thread.status_update.connect(lambda s: self.status_label.setText(s))
        self.thread.start()

    def init_tracks(self):
        # 2 EMG (offset 500µV) + 4 AUX/Sistema (offset 1000 unità)
        t_info = [("EMG Bipolare (µV)", 2, 500, 1.0), ("AUX & System Channels", 4, 1000, 1.0)]
        for title, n, off, conv in t_info:
            t = Track(title, self.sample_freq, n, off, conv)
            self.tracks.append(t)
            t.plot_widget.setMinimumHeight(350)
            self.scroll_layout.addWidget(t.plot_widget)

    def on_data(self, data):
        self.tracks[0].feed(data[:2, :])    # EMG
        self.tracks[1].feed(data[2:, :])    # AUX/Sistema

    def update_plot(self):
        if not self.is_paused:
            for t in self.tracks: t.draw()

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        communication.disconnect_from_sq(self.connection)
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    
    # Parametri per 2 canali bipolari con AD8x1SE
    # nch=0, mode=1 -> 8 canali totali trasmessi
    communication.nch, communication.mode = 0, 1 
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    cmd, nch, fs, bis = communication.create_bin_command(start=1)
    print(f"Configurazione: {nch} canali (8 totali) @ {fs}Hz")
    
    try:
        conn = communication.connect_to_sq(sock, '0.0.0.0', 45454, cmd)
        win = SoundtrackGUI(conn, nch, fs, bis)
        win.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Errore fatale: {e}")

if __name__ == "__main__": main()