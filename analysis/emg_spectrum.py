"""Spectral analysis of EMG recordings: raw vs offline-filtered.

For every CSV it computes the Welch PSD (each channel + channel average) and
the channel-averaged spectrogram, before and after zero-phase filtering
(band-pass 10-450 Hz + notches at 50 Hz and harmonics). Useful to inspect
mains interference and to check that the notch filters work.

Usage:
    python -m analysis.emg_spectrum                       # trial 1
    python -m analysis.emg_spectrum --files a.csv b.csv   # any EMG CSV
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, spectrogram, tf2sos, welch

import config

NOTCH_FREQS = (50, 100, 150, 200, 250)


def offline_filter(raw, fs):
    nyq = fs / 2
    out = sosfiltfilt(butter(4, [10, min(450, nyq - 5)], btype="band", fs=fs, output="sos"), raw, axis=0)
    for freq in NOTCH_FREQS:
        if freq < nyq:
            out = sosfiltfilt(tf2sos(*iirnotch(freq, 80, fs=fs)), out, axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description="PSD and spectrogram of EMG recordings.")
    parser.add_argument("--files", nargs="+", type=Path, default=[config.trial_file(1, "EMG")])
    parser.add_argument("--fs", type=int, default=config.EMG_FS)
    parser.add_argument("--no-show", action="store_true", help="only save the figures")
    args = parser.parse_args()
    fs = args.fs

    output_dir = config.FIGURES_DIR / "emg_spectrum"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for path in args.files:
        print(f"Loading {path.name}...")
        df = pd.read_csv(path)
        raw = df[[c for c in df.columns if "EMG" in c or "CH_" in c]].to_numpy()
        datasets.append((f"Raw - {path.name}", raw))
        datasets.append((f"Filtered - {path.name}", offline_filter(raw, fs)))

    # ---- Power spectral density ----
    fig1 = plt.figure(figsize=(15, 5 * len(datasets)))
    for i, (label, data) in enumerate(datasets):
        print(f"PSD: {label}")
        psds = []
        for ch in range(data.shape[1]):
            freqs, psd = welch(data[:, ch], fs=fs, nperseg=1024)
            psds.append(psd)
        plt.subplot(len(datasets), 1, i + 1)
        for psd in psds:
            plt.semilogy(freqs, psd, color="gray", alpha=0.1)
        plt.semilogy(freqs, np.mean(psds, axis=0), color="red" if i % 2 == 0 else "blue",
                     linewidth=2, label=f"Average PSD ({label})")
        plt.title(f"Spectral analysis - {label}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (µV²/Hz)")
        plt.xlim(0, 500)
        plt.grid(True, which="both", linestyle="--", alpha=0.7)
        plt.legend()
    plt.tight_layout()
    fig1.savefig(output_dir / "psd_analysis.png", dpi=300, bbox_inches="tight")

    # ---- Spectrograms (shared colour scale) ----
    fig2 = plt.figure(figsize=(15, 5 * len(datasets)))
    spectra = []
    for label, data in datasets:
        print(f"Spectrogram: {label}")
        sxx = []
        for ch in range(data.shape[1]):
            f, t, s = spectrogram(data[:, ch], fs=fs, nperseg=512, noverlap=256)
            sxx.append(s)
        spectra.append((label, f, t, 10 * np.log10(np.mean(sxx, axis=0) + 1e-10)))
    vmin = min(s[3].min() for s in spectra)
    vmax = max(s[3].max() for s in spectra)
    for i, (label, f, t, sxx_db) in enumerate(spectra):
        plt.subplot(len(datasets), 1, i + 1)
        pm = plt.pcolormesh(t, f, sxx_db, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(pm, label="Power (dB)")
        plt.ylim(0, 500)
        plt.title(f"Average spectrogram ({datasets[i][1].shape[1]} channels) - {label}")
        plt.ylabel("Frequency (Hz)")
        if i == len(spectra) - 1:
            plt.xlabel("Time (s)")
    plt.tight_layout()
    fig2.savefig(output_dir / "spectrogram_analysis.png", dpi=300, bbox_inches="tight")
    print(f"Figures saved to {config.rel(output_dir)}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
