#!python3
import sys
import numpy as np
import socket
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from scipy.signal import butter, lfilter, iirnotch, sosfilt
from pylsl import StreamInfo, StreamOutlet
import communication_sessantaquattro as communication
import time
import datetime
import pandas as pd
import threading

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
        
        if self.num_channels > 1:
            self.plot_widget.setYRange(-self.offset, self.num_channels * self.offset)
        else:
            self.plot_widget.setYRange(-3000, 3000)
            
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude', units='µV' if 'EMG' in title else 'A.U.')
        
        self.curves = [self.plot_widget.plot(pen=pg.mkPen(color=pg.intColor(i, hues=num_channels if num_channels > 1 else 10), width=1)) for i in range(num_channels)]

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
        
        self.active_bio_channels = 32
        self.processor = EMGProcessor(fs=self.sample_freq)
        self.lsl_outlet = StreamOutlet(StreamInfo('OTB_S64_EMG', 'EMG', 
                                                  self.active_bio_channels, self.sample_freq, 
                                                  'float32', 's64_pisa'))

        # Variabili per la registrazione
        self.is_recording = False
        self.recorded_data = []
        self.enable_preprocessing = False  # Flag per sospendere il preprocessing dei segnali

    def toggle_recording(self):
        if not self.is_recording:
            self.recorded_data = []
            self.is_recording = True
            return "Registrazione avviata..."
        else:
            self.is_recording = False
            data_to_save = self.recorded_data
            self.recorded_data = []
            
            if data_to_save:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"session_Read32_{timestamp_str}_EMG.csv"
                
                def save_data(data, fname):
                    all_times = np.concatenate([d[0] for d in data])
                    all_channels = np.concatenate([d[1] for d in data], axis=1)
                    cols = [f"EMG_{i+1}" for i in range(self.active_bio_channels)]
                    df = pd.DataFrame(all_channels.T, columns=cols)
                    df.insert(0, "Timestamp", all_times)
                    df.to_csv(fname, index=False)
                    self.status_update.emit(f"Registrazione completata e salvata in {fname}")
                
                threading.Thread(target=save_data, args=(data_to_save, filename)).start()
                return f"Salvataggio in corso in {filename}..."
            return "Registrazione interrotta (nessun dato)"

    def toggle_preprocessing(self):
        self.enable_preprocessing = not self.enable_preprocessing
        stato = "ATTIVATO" if self.enable_preprocessing else "DISATTIVATO"
        return f"Preprocessing {stato}", self.enable_preprocessing

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
                    
                    # 1. Estrazione di tutti i 32 canali EMG
                    raw_channels = reshaped[:self.active_bio_channels, :]
                    
                    # --- APPLICAZIONE DEL FILTRO SPAZIALE CAR ---
                    # Calcola il rumore di modo comune (media lungo l'asse dei canali)
                    # common_mode_noise = np.mean(raw_channels, axis=0)
                    
                    # # Sottrae il rumore globale da tutti i 32 canali contemporaneamente
                    # raw_channels_car = raw_channels - common_mode_noise
                    # --------------------------------------------
                    
                    # 2. Conversione in microVolt e Filtraggio (Passa-banda + Notch)
                    # Usiamo i dati appena "puliti" dal filtro CAR
                    channels_uv = raw_channels * 0.2861
                    
                    if self.enable_preprocessing:
                        final_data = self.processor.process(channels_uv)
                    else:
                        final_data = channels_uv

                    # Accumulo dei dati se la registrazione è attiva
                    if self.is_recording:
                        current_chunk_size = final_data.shape[1]
                        current_time = time.time()
                        times = np.linspace(current_time - current_chunk_size/self.sample_freq, current_time, current_chunk_size, endpoint=False)
                        self.recorded_data.append((times, final_data.copy()))

                    # 3. Stream su LSL dei canali puliti
                    self.lsl_outlet.push_chunk(final_data.T.astype(np.float32).tolist())

                    # 4. Preparazione dati per i grafici
                    self.data_received.emit(final_data)

            except Exception as e:
                self.status_update.emit(f"Errore: {e}")
                break

    def stop(self): self.running = False

class MultiplotWindow(QtWidgets.QWidget):
    def __init__(self, sample_freq, num_channels, plot_time):
        super().__init__()
        self.setWindowTitle("Sessantaquattro - Multiplot (Tutti i canali)")
        self.setGeometry(150, 150, *Config.WINDOW_SIZE)
        layout = QtWidgets.QVBoxLayout(self)
        
        # Sfasamento di 2000 µV per separare bene le linee
        self.track = Track("Multiplot 32 Canali", sample_freq, num_channels, offset=2000, conv_fact=1.0, plot_time=plot_time)
        layout.addWidget(self.track.plot_widget)

    def feed(self, data): self.track.feed(data)
    def draw(self): self.track.draw()

class SoundtrackGUI(QtWidgets.QWidget):
    def __init__(self, connection, num_channels, sample_freq, bytes_in_sample):
        super().__init__()
        self.connection = connection
        self.num_channels, self.sample_freq, self.bytes_in_sample = num_channels, sample_freq, bytes_in_sample
        self.tracks, self.is_paused = [], False
        self.plot_time = Config.DEFAULT_PLOT_TIME  

        self.setWindowTitle("Sessantaquattro - Final Bipolar EMG & LSL")
        self.setGeometry(100, 100, 500, 150)
        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel("Acquisizione in corso... LSL Attivo")
        layout.addWidget(self.status_label)
        
        self.instructions_label = QtWidgets.QLabel("Premi il tasto 'S' per aprire o chiudere la finestra dei 32 plot singoli")
        self.instructions_label.setStyleSheet("font-weight: bold; color: #555555;")
        layout.addWidget(self.instructions_label)
        
        self.record_instructions_label = QtWidgets.QLabel("Premi il tasto 'R' per avviare/fermare la registrazione su CSV")
        self.record_instructions_label.setStyleSheet("font-weight: bold; color: #555555;")
        layout.addWidget(self.record_instructions_label)
        
        self.prep_instructions_label = QtWidgets.QLabel("Premi il tasto 'P' per attivare/disattivare i filtri (Preprocessing: DISATTIVATO)")
        self.prep_instructions_label.setStyleSheet("font-weight: bold; color: #555555;")
        layout.addWidget(self.prep_instructions_label)
        
        # Finestra separata per i plot singoli
        self.single_plots_window = QtWidgets.QWidget()
        self.single_plots_window.setWindowTitle("Sessantaquattro - Plot Singoli (32 Canali)")
        self.single_plots_window.setGeometry(150, 150, *Config.WINDOW_SIZE)
        sp_layout = QtWidgets.QVBoxLayout(self.single_plots_window)
        
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.content)
        scroll_area.setWidget(self.content)
        sp_layout.addWidget(scroll_area)
        
        self.init_tracks()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(Config.UPDATE_RATE)

        self.thread = DataReceiverThread(connection, num_channels, bytes_in_sample, sample_freq)
        self.thread.data_received.connect(self.on_data)
        self.thread.status_update.connect(lambda s: self.status_label.setText(s))
        self.thread.start()

        self.multiplot_window = MultiplotWindow(self.sample_freq, 32, self.plot_time)
        self.multiplot_window.show()

        # Shortcut globale: Funziona a prescindere da quale finestra abbia il focus
        self.shortcut_s = QtWidgets.QShortcut("S", self)
        self.shortcut_s.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut_s.activated.connect(self.toggle_single_plots)

        self.shortcut_r = QtWidgets.QShortcut("R", self)
        self.shortcut_r.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut_r.activated.connect(self.toggle_recording)

        self.shortcut_p = QtWidgets.QShortcut("P", self)
        self.shortcut_p.setContext(QtCore.Qt.ApplicationShortcut)
        self.shortcut_p.activated.connect(self.toggle_preprocessing)

    def toggle_recording(self):
        msg = self.thread.toggle_recording()
        self.status_label.setText(msg)

    def toggle_preprocessing(self):
        msg, is_enabled = self.thread.toggle_preprocessing()
        self.status_label.setText(msg)
        stato = "ATTIVATO" if is_enabled else "DISATTIVATO"
        self.prep_instructions_label.setText(f"Premi il tasto 'P' per attivare/disattivare i filtri (Preprocessing: {stato})")

    def init_tracks(self):
        # Mostriamo tutti i 32 canali
        t_info = [(f"Canale {i+1}", 1, 0, 1.0) for i in range(32)]
        
        for title, n, off, conv in t_info:
            t = Track(title, self.sample_freq, n, off, conv, self.plot_time)
            self.tracks.append(t)
            t.plot_widget.setMinimumHeight(200)
            self.scroll_layout.addWidget(t.plot_widget)

    def on_data(self, data):
        for i in range(32):
            self.tracks[i].feed(data[i:i+1, :])
        if hasattr(self, 'multiplot_window'):
            self.multiplot_window.feed(data)

    def update_plot(self):
        if not self.is_paused:
            if self.single_plots_window.isVisible():
                for t in self.tracks: t.draw()
            if hasattr(self, 'multiplot_window'):
                self.multiplot_window.draw()

    def toggle_single_plots(self):
        if self.single_plots_window.isVisible():
            self.single_plots_window.hide()
        else:
            self.single_plots_window.show()

    def closeEvent(self, event):
        if hasattr(self, 'multiplot_window'):
            self.multiplot_window.close()
        if hasattr(self, 'single_plots_window'):
            self.single_plots_window.close()
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
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    
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