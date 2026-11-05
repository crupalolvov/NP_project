"""Real-time 32-channel MD-EMG acquisition, visualization and LSL streaming.

Connects to the OT Bioelettronica Sessantaquattro over TCP, converts the raw
samples to µV, optionally filters them (band-pass 10-450 Hz + 50 Hz notch
harmonics) and publishes them on Lab Streaming Layer as ``OTB_S64_EMG``.

Keyboard shortcuts (work from any window):
    S  show/hide the 32 single-channel plots
    M  show/hide the stacked multiplot
    P  enable/disable the digital filters (disabled at start-up)
    R  start/stop a quick CSV recording (debugging without the LSL recorder)

Usage:
    python -m acquisition.emg_stream_32ch
"""
import datetime
import sys
import threading

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pylsl import StreamInfo, StreamOutlet, local_clock
from PyQt5 import QtCore, QtWidgets

import config
from acquisition.emg_common import (ADC_TO_MICROVOLT, EMGProcessor, Track, close_device,
                                    open_device, read_chunk)

N_CHANNELS = 32
PLOT_TIME_S = 1         # visible time window
UPDATE_RATE_MS = 16     # ~60 FPS redraw
WINDOW_SIZE = (1200, 800)
DEBUG_RECORDINGS_DIR = config.DATA_DIR / "debug"


class DataReceiverThread(QtCore.QThread):
    """Reads the TCP stream, filters it, pushes it to LSL and to the GUI."""

    data_received = QtCore.pyqtSignal(np.ndarray)
    status_update = QtCore.pyqtSignal(str)

    def __init__(self, connection, num_channels, bytes_in_sample, sample_freq):
        super().__init__()
        self.connection = connection
        self.num_channels = num_channels
        self.bytes_in_sample = bytes_in_sample
        self.sample_freq = sample_freq
        self.running = True

        self.processor = EMGProcessor(fs=sample_freq, band=(10, 450),
                                      notch_freqs=(50, 100, 150, 200, 250), notch_q=80)
        self.lsl_outlet = StreamOutlet(StreamInfo(config.LSL_EMG_STREAM, "EMG", N_CHANNELS,
                                                  sample_freq, "float32", "s64_pisa"))

        self.is_recording = False
        self.recorded_data = []
        self.enable_preprocessing = False

    def toggle_recording(self):
        if not self.is_recording:
            self.recorded_data = []
            self.is_recording = True
            return "Recording started..."

        self.is_recording = False
        data_to_save, self.recorded_data = self.recorded_data, []
        if not data_to_save:
            return "Recording stopped (no data)"

        DEBUG_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = DEBUG_RECORDINGS_DIR / f"session_{timestamp_str}_EMG.csv"

        def save_data(data, fname):
            all_times = np.concatenate([d[0] for d in data])
            all_channels = np.concatenate([d[1] for d in data], axis=1)
            df = pd.DataFrame(all_channels.T, columns=[f"EMG_{i + 1}" for i in range(N_CHANNELS)])
            df.insert(0, "Timestamp_LSL", all_times)
            df.to_csv(fname, index=False)
            self.status_update.emit(f"Recording saved to {fname}")

        threading.Thread(target=save_data, args=(data_to_save, filename)).start()
        return f"Saving to {filename}..."

    def toggle_preprocessing(self):
        self.enable_preprocessing = not self.enable_preprocessing
        state = "ON" if self.enable_preprocessing else "OFF"
        return f"Preprocessing {state}", self.enable_preprocessing

    def run(self):
        chunk_size = max(1, int(self.sample_freq / 60))
        while self.running:
            try:
                chunk, alive = read_chunk(self.connection, self.num_channels,
                                          self.bytes_in_sample, chunk_size)
                if chunk is not None:
                    channels_uv = chunk[:N_CHANNELS, :] * ADC_TO_MICROVOLT
                    final_data = (self.processor.process(channels_uv)
                                  if self.enable_preprocessing else channels_uv)

                    if self.is_recording:
                        n = final_data.shape[1]
                        now = local_clock()
                        times = np.linspace(now - n / self.sample_freq, now, n, endpoint=False)
                        self.recorded_data.append((times, final_data.copy()))

                    self.lsl_outlet.push_chunk(final_data.T.astype(np.float32).tolist())
                    self.data_received.emit(final_data)

                if not alive:
                    self.status_update.emit("Connection closed by the device.")
                    break
            except Exception as e:
                self.status_update.emit(f"Error: {e}")
                break

    def stop(self):
        self.running = False


class MultiplotWindow(QtWidgets.QWidget):
    """All 32 channels stacked in a single plot."""

    def __init__(self, sample_freq, num_channels, plot_time):
        super().__init__()
        self.setWindowTitle("Sessantaquattro - Multiplot (all channels)")
        self.setGeometry(150, 150, *WINDOW_SIZE)
        layout = QtWidgets.QVBoxLayout(self)
        # 2 mV vertical offset between channels
        self.track = Track(f"Multiplot {num_channels} channels", sample_freq, num_channels,
                           offset=2000, conv_fact=1.0, plot_time=plot_time)
        layout.addWidget(self.track.plot_widget)

    def feed(self, data):
        self.track.feed(data)

    def draw(self):
        self.track.draw()


class EMGViewer(QtWidgets.QWidget):
    """Control window: status, shortcuts and the two plot windows."""

    def __init__(self, server_socket, connection, num_channels, sample_freq, bytes_in_sample):
        super().__init__()
        self.server_socket = server_socket
        self.connection = connection
        self.sample_freq = sample_freq
        self.tracks = []

        self.setWindowTitle("Sessantaquattro - MD-EMG & LSL")
        self.setGeometry(100, 100, 500, 150)
        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel("Acquiring... LSL stream active")
        layout.addWidget(self.status_label)
        for text in ("Press 'S' for single-channel plots, 'M' for the multiplot",
                     "Press 'R' to start/stop a CSV recording"):
            label = QtWidgets.QLabel(text)
            label.setStyleSheet("font-weight: bold; color: #555555;")
            layout.addWidget(label)
        self.prep_label = QtWidgets.QLabel()
        self.prep_label.setStyleSheet("font-weight: bold; color: #555555;")
        layout.addWidget(self.prep_label)
        self._set_prep_label(False)

        # Separate window with one plot per channel
        self.single_plots_window = QtWidgets.QWidget()
        self.single_plots_window.setWindowTitle(f"Sessantaquattro - Single plots ({N_CHANNELS} channels)")
        self.single_plots_window.setGeometry(150, 150, *WINDOW_SIZE)
        sp_layout = QtWidgets.QVBoxLayout(self.single_plots_window)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(content)
        scroll_area.setWidget(content)
        sp_layout.addWidget(scroll_area)
        for i in range(N_CHANNELS):
            t = Track(f"Channel {i + 1}", sample_freq, 1, 0, 1.0, PLOT_TIME_S)
            t.plot_widget.setMinimumHeight(200)
            self.tracks.append(t)
            self.scroll_layout.addWidget(t.plot_widget)

        self.multiplot_window = MultiplotWindow(sample_freq, N_CHANNELS, PLOT_TIME_S)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(UPDATE_RATE_MS)

        self.thread = DataReceiverThread(connection, num_channels, bytes_in_sample, sample_freq)
        self.thread.data_received.connect(self.on_data)
        self.thread.status_update.connect(self.status_label.setText)
        self.thread.start()

        # Application-wide shortcuts: they work whichever window has focus
        for key, slot in (("S", self.toggle_single_plots), ("M", self.toggle_multiplot),
                          ("R", self.toggle_recording), ("P", self.toggle_preprocessing)):
            shortcut = QtWidgets.QShortcut(key, self)
            shortcut.setContext(QtCore.Qt.ApplicationShortcut)
            shortcut.activated.connect(slot)

    def _set_prep_label(self, enabled):
        state = "ON" if enabled else "OFF"
        self.prep_label.setText(f"Press 'P' to toggle the digital filters (Preprocessing: {state})")

    def toggle_recording(self):
        self.status_label.setText(self.thread.toggle_recording())

    def toggle_preprocessing(self):
        msg, enabled = self.thread.toggle_preprocessing()
        self.status_label.setText(msg)
        self._set_prep_label(enabled)

    @staticmethod
    def _toggle(window):
        window.hide() if window.isVisible() else window.show()

    def toggle_single_plots(self):
        self._toggle(self.single_plots_window)

    def toggle_multiplot(self):
        self._toggle(self.multiplot_window)

    def on_data(self, data):
        for i in range(N_CHANNELS):
            self.tracks[i].feed(data[i:i + 1, :])
        self.multiplot_window.feed(data)

    def update_plot(self):
        if self.single_plots_window.isVisible():
            for t in self.tracks:
                t.draw()
        if self.multiplot_window.isVisible():
            self.multiplot_window.draw()

    def closeEvent(self, event):
        self.multiplot_window.close()
        self.single_plots_window.close()
        self.thread.stop()
        self.thread.wait()
        close_device(self.server_socket, self.connection)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    try:
        server, conn, nch, fs, bis = open_device()
    except Exception as e:
        print(f"Fatal error: {e}")
        return
    win = EMGViewer(server, conn, nch, fs, bis)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
