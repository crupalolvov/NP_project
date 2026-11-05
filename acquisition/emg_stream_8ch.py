"""Lightweight 8-channel EMG viewer for quick electrode checks.

Same device configuration as ``emg_stream_32ch`` but only the first 8
channels are filtered (band-pass 20-450 Hz + 50/60/100/150 Hz notches),
plotted one per row and streamed on LSL as ``Sessantaquattro_EMG``.
It is meant for setup/debugging (e.g. bipolar electrode placement), not for
dataset recording - the recorder expects the 32-channel ``OTB_S64_EMG`` stream.

Usage:
    python -m acquisition.emg_stream_8ch
"""
import sys

import numpy as np
import pyqtgraph as pg
from pylsl import StreamInfo, StreamOutlet
from PyQt5 import QtCore, QtWidgets

from acquisition.emg_common import (ADC_TO_MICROVOLT, EMGProcessor, Track, close_device,
                                    open_device, read_chunk)

N_CHANNELS = 8
PLOT_TIME_S = 1
UPDATE_RATE_MS = 16
WINDOW_SIZE = (1200, 800)
LSL_STREAM_NAME = "Sessantaquattro_EMG"


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

        self.processor = EMGProcessor(fs=sample_freq, band=(20, 450),
                                      notch_freqs=(50, 60, 100, 150), notch_q=30)
        self.lsl_outlet = StreamOutlet(StreamInfo(LSL_STREAM_NAME, "EMG", N_CHANNELS,
                                                  sample_freq, "float32", "s64_pisa"))

    def run(self):
        chunk_size = max(1, int(self.sample_freq / 60))
        while self.running:
            try:
                chunk, alive = read_chunk(self.connection, self.num_channels,
                                          self.bytes_in_sample, chunk_size)
                if chunk is not None:
                    channels_uv = chunk[:N_CHANNELS, :] * ADC_TO_MICROVOLT
                    filtered = self.processor.process(channels_uv)
                    self.lsl_outlet.push_chunk(filtered.T.astype(np.float32).tolist())
                    self.data_received.emit(filtered)
                if not alive:
                    self.status_update.emit("Connection closed by the device.")
                    break
            except Exception as e:
                self.status_update.emit(f"Error: {e}")
                break

    def stop(self):
        self.running = False


class EMGViewer(QtWidgets.QWidget):
    def __init__(self, server_socket, connection, num_channels, sample_freq, bytes_in_sample):
        super().__init__()
        self.server_socket = server_socket
        self.connection = connection
        self.tracks = []

        self.setWindowTitle(f"Sessantaquattro - {N_CHANNELS}-channel EMG & LSL")
        self.setGeometry(100, 100, *WINDOW_SIZE)
        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel("Acquiring... LSL stream active")
        layout.addWidget(self.status_label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        for i in range(N_CHANNELS):
            t = Track(f"Channel {i + 1}", sample_freq, 1, 0, 1.0, PLOT_TIME_S,
                      y_range=(-3500, 3500), dark_background=True)
            t.plot_widget.setMinimumHeight(200)
            self.tracks.append(t)
            scroll_layout.addWidget(t.plot_widget)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(UPDATE_RATE_MS)

        self.thread = DataReceiverThread(connection, num_channels, bytes_in_sample, sample_freq)
        self.thread.data_received.connect(self.on_data)
        self.thread.status_update.connect(self.status_label.setText)
        self.thread.start()

    def on_data(self, data):
        for i in range(N_CHANNELS):
            self.tracks[i].feed(data[i:i + 1, :])

    def update_plot(self):
        for t in self.tracks:
            t.draw()

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        close_device(self.server_socket, self.connection)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
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
