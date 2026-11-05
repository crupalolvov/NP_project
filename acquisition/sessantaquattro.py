"""TCP communication protocol for the OT Bioelettronica Sessantaquattro.

The PC acts as a TCP server: the Sessantaquattro connects to it over Wi-Fi,
receives a 16-bit configuration command and then streams samples as
big-endian two's-complement integers (one value per channel, 2 or 3 bytes).
"""

CONVERSION_FACTOR = 0.000286  # ADC LSB in mV (286.1 nV)


def integer_to_bytes(command):
    """Encode the 16-bit command word as two big-endian bytes."""
    return int(command).to_bytes(2, byteorder="big")


def convert_bytes_to_int(bytes_value, bytes_in_sample):
    """Decode one big-endian two's-complement sample (16 or 24 bit)."""
    if bytes_in_sample == 2:
        value = bytes_value[0] * 256 + bytes_value[1]
        if value >= 32768:
            value -= 65536
    elif bytes_in_sample == 3:
        value = bytes_value[0] * 65536 + bytes_value[1] * 256 + bytes_value[2]
        if value >= 8388608:
            value -= 16777216
    else:
        raise ValueError(f"Unknown bytes_in_sample value: {bytes_in_sample} (expected 2 or 3)")
    return value


def create_bin_command(start=1):
    """Build the command that starts (start=1) or stops (start=0) acquisition.

    Returns ``(command_bytes, number_of_channels, sample_frequency, bytes_in_sample)``.
    ``number_of_channels`` includes the 4 auxiliary/accessory channels.
    """
    rec = 0
    trig = 0
    ext = 0
    hpf = 1     # 1 = hardware high-pass filter enabled (recommended for EMG)
    hres = 0    # 0 = 16-bit resolution, 1 = 24-bit
    mode = 0    # 0 = monopolar, 1 = bipolar (differential)
    nch = 2     # 2 = 32 bio-signal channels
    fsamp = 2   # 2 = 2000 Hz
    getset = 0

    command = (start
               + rec * 2
               + trig * 4
               + ext * 16
               + hpf * 64
               + hres * 128
               + mode * 256
               + nch * 2048
               + fsamp * 8192
               + getset * 32768)

    # Total channels streamed (bio-signal + 4 auxiliary), from the device manual
    channels_table = {
        0: (12, 8),
        1: (20, 12),
        2: (36, 20),
        3: (68, 36),
    }
    if nch not in channels_table:
        raise ValueError(f"Wrong value for nch: {nch}")
    monopolar, bipolar = channels_table[nch]
    number_of_channels = bipolar if mode == 1 else monopolar

    # Sampling frequency (mode 3 = accelerometer-only mode, 4x faster)
    fs_table = {0: 500, 1: 1000, 2: 2000, 3: 4000}
    if fsamp not in fs_table:
        raise ValueError(f"Wrong value for fsamp: {fsamp}")
    sample_frequency = fs_table[fsamp] * (4 if mode == 3 else 1)

    bytes_in_sample = 3 if hres == 1 else 2

    return (integer_to_bytes(command),
            number_of_channels,
            sample_frequency,
            bytes_in_sample)


def bytes_to_integers(sample_from_channels_as_bytes, number_of_channels,
                      bytes_in_sample, output_milli_volts):
    """Split one multi-channel sample into per-channel integer values.

    If ``output_milli_volts`` is True, bio-signal channels are converted to mV
    (the last 4 auxiliary channels are left untouched).
    """
    channel_values = []
    for channel_index in range(number_of_channels):
        start = channel_index * bytes_in_sample
        end = start + bytes_in_sample
        value = convert_bytes_to_int(sample_from_channels_as_bytes[start:end], bytes_in_sample)

        if output_milli_volts and channel_index < (number_of_channels - 4):
            value *= CONVERSION_FACTOR
        channel_values.append(value)
    return channel_values


def read_raw_bytes(connection, number_of_all_channels, bytes_in_sample):
    """Read exactly one sample (all channels) from the socket.

    TCP may deliver fewer bytes than requested, so we keep reading until the
    whole sample is available; otherwise channels would become misaligned.
    Returns ``b""`` if the device closed the connection.
    """
    buffer_size = number_of_all_channels * bytes_in_sample
    data = bytearray()
    while len(data) < buffer_size:
        chunk = connection.recv(buffer_size - len(data))
        if not chunk:
            return b""
        data.extend(chunk)
    return bytes(data)


def connect_to_sq(sq_socket, ip_address, port, start_command):
    """Wait for the Sessantaquattro to connect, then send the start command."""
    sq_socket.bind((ip_address, port))
    sq_socket.listen(1)
    print("Waiting for the Sessantaquattro to connect...")
    conn, addr = sq_socket.accept()
    print(f"Connection from address: {addr}")
    conn.send(start_command)
    return conn


def disconnect_from_sq(conn):
    """Send the stop command and close the connection."""
    if conn is None:
        raise ConnectionError("Can't disconnect because the connection is not established")
    stop_command, _, _, _ = create_bin_command(start=0)
    conn.send(stop_command)
    conn.shutdown(2)
    conn.close()
