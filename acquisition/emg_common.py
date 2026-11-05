"""Building blocks shared by the real-time EMG viewers.

* ``EMGProcessor`` - causal IIR filter bank (band-pass + notch) that keeps its
  internal state between chunks, so it can be applied sample-by-sample.
* ``Track``        - scrolling pyqtgraph plot backed by a ring buffer.
* ``open_device``  - TCP server that waits for the Sessantaquattro.
"""
import socket

import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, iirnotch, lfilter, sosfilt

from acquisition import sessantaquattro as sq

ADC_TO_MICROVOLT = 0.2861       # 286.1 nV per LSB
DEVICE_PORT = 45454


class EMGProcessor:
    """Stateful band-pass + notch filter applied chunk by chunk (causal)."""

    def __init__(self, fs=2000, band=(10, 450), notch_freqs=(50, 100, 150, 200, 250), notch_q=80):
        self.fs = fs
        nyq = fs / 2
        self.sos_band = butter(4, [band[0], min(band[1], nyq - 5)], btype="band", fs=fs, output="sos")
        self.notches = [iirnotch(f, notch_q, fs=fs) for f in notch_freqs if f < nyq]
        self.zi_sos = None
        self.zi_notches = None

    def process(self, data_chunk):
        """Filter a ``(channels, samples)`` chunk, preserving state across calls."""
        n_channels = data_chunk.shape[0]
        if self.zi_sos is None:
            self.zi_sos = np.zeros((self.sos_band.shape[0], n_channels, 2))
            self.zi_notches = [np.zeros((n_channels, 2)) for _ in self.notches]

        filtered, self.zi_sos = sosfilt(self.sos_band, data_chunk, axis=1, zi=self.zi_sos)
        for i, (b, a) in enumerate(self.notches):
            filtered, self.zi_notches[i] = lfilter(b, a, filtered, axis=1, zi=self.zi_notches[i])
        return filtered


class Track:
    """A scrolling plot of one or more channels with a fixed time window."""

    def __init__(self, title, frequency, num_channels, offset, conv_fact, plot_time=1,
                 y_range=None, dark_background=False):
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
        if y_range is not None:
            self.plot_widget.setYRange(*y_range)
        elif num_channels > 1:
            self.plot_widget.setYRange(-offset, num_channels * offset)
        else:
            self.plot_widget.setYRange(-3000, 3000)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Amplitude", units="µV")
        if dark_background:
            self.plot_widget.getViewBox().setBackgroundColor((30, 30, 30))

        hues = num_channels if num_channels > 1 else 10
        self.curves = [
            self.plot_widget.plot(pen=pg.mkPen(color=pg.intColor(i, hues=hues), width=1))
            for i in range(num_channels)
        ]

    def feed(self, packet):
        """Append a ``(channels, samples)`` packet to the ring buffer."""
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
            curve.setData(self.time_array, self.buffer[i, :] * self.conv_fact + self.offset * i)


def read_chunk(connection, num_channels, bytes_in_sample, chunk_size):
    """Read up to ``chunk_size`` samples; returns ``(chunk, connection_alive)``.

    ``chunk`` has shape ``(num_channels, n_samples)`` in raw ADC units.
    """
    samples = []
    alive = True
    for _ in range(chunk_size):
        raw_bytes = sq.read_raw_bytes(connection, num_channels, bytes_in_sample)
        if not raw_bytes:
            alive = False
            break
        samples.append(sq.bytes_to_integers(raw_bytes, num_channels, bytes_in_sample, False))
    chunk = np.array(samples, dtype=np.float64).T if samples else None
    return chunk, alive


def open_device():
    """Start the TCP server and wait for the Sessantaquattro to connect.

    Returns ``(server_socket, connection, num_channels, fs, bytes_in_sample)``.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    cmd, nch, fs, bis = sq.create_bin_command(start=1)
    print(f"Configuration: {nch} channels (incl. 4 auxiliary) @ {fs} Hz")
    conn = sq.connect_to_sq(server, "0.0.0.0", DEVICE_PORT, cmd)
    return server, conn, nch, fs, bis


def close_device(server, conn):
    """Stop the acquisition and release both sockets."""
    try:
        sq.disconnect_from_sq(conn)
    except OSError:
        pass
    try:
        server.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    server.close()
