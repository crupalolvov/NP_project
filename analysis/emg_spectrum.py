import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, iirnotch, tf2sos, sosfiltfilt, butter
import os

def main():
    file_paths = [
        'session_EMG_IN_nofilt.csv',
        'session_EMG_OUT_nofilt.csv'
    ]

    fs = 2000
    print(f"Sampling frequency set to: {fs} Hz")

    # Definizione e creazione della cartella di output
    output_dir = "high_res_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"I plot ad alta definizione verranno salvati in: {output_dir}/")

    nyq = fs / 2
    sos_band = butter(4, [10, min(450, nyq - 5)], btype='band', fs=fs, output='sos')
    notches_freqs = [50, 100, 150, 200, 250]

    datasets = []

    for path in file_paths:
        print(f"Loading data from {path}...")
        df = pd.read_csv(path)
        emg_cols = [col for col in df.columns if 'EMG' in col]
        raw_data = df[emg_cols].to_numpy()

        print(f"Applying offline filtering (Bandpass + Notches) for {path}...")
        filtered_data = sosfiltfilt(sos_band, raw_data, axis=0)
        
        for freq in notches_freqs:
            if freq < nyq:
                b, a = iirnotch(freq, 80, fs=fs)
                sos = tf2sos(b, a)
                filtered_data = sosfiltfilt(sos, filtered_data, axis=0)

        # Estraiamo solo il nome del file (senza il percorso) per il titolo dei plot
        file_name = os.path.basename(path)
        datasets.append((f"Raw - {file_name}", raw_data))
        datasets.append((f"Filtered - {file_name}", filtered_data))

    # Altezza dinamica: calcola circa 5 pollici per ogni subplot generato
    fig1 = plt.figure(figsize=(15, 5 * len(datasets)))

    for i, (label, data) in enumerate(datasets):
        print(f"Computing spectrum for {label}...")
        psds = []

        # Calculate PSD for each channel
        for ch_idx in range(data.shape[1]):
            freqs, psd = welch(data[:, ch_idx], fs=fs, nperseg=1024)
            psds.append(psd)

        avg_psd = np.mean(psds, axis=0)
        
        plt.subplot(len(datasets), 1, i+1)
        
        # Display individual channels in the background
        for psd in psds:
            plt.semilogy(freqs, psd, color='gray', alpha=0.1)
            
        plt.semilogy(freqs, avg_psd, color='red' if i==0 else 'blue', linewidth=2, label=f'Average PSD ({label})')

        plt.title(f'Spectral Analysis - {label}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (uV^2/Hz)')
        plt.xlim(0, 500)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
    plt.tight_layout()

    # Salvataggio Figure 1
    psd_filename = os.path.join(output_dir, 'psd_analysis.png')
    fig1.savefig(psd_filename, dpi=300, bbox_inches='tight')
    print(f"Salvato PSD plot: {psd_filename}")

    # === SECOND FIGURE: SPECTROGRAMS OVER TIME ===
    fig2 = plt.figure(figsize=(15, 5 * len(datasets)))
    
    spectrogram_data = []
    global_vmin = np.inf
    global_vmax = -np.inf

    for i, (label, data) in enumerate(datasets):
        print(f"Computing spectrogram for {label}...")
        Sxx_list = []
        # Calculate spectrogram for each channel
        for ch_idx in range(data.shape[1]):
            f, t, Sxx = spectrogram(data[:, ch_idx], fs=fs, nperseg=512, noverlap=256)
            Sxx_list.append(Sxx)
            
        # Average the spectrograms to get an overview of activation
        avg_Sxx = np.mean(Sxx_list, axis=0)
        Sxx_db = 10 * np.log10(avg_Sxx + 1e-10)
        
        # Aggiorniamo il minimo e massimo globale per la scala dei colori
        global_vmin = min(global_vmin, np.min(Sxx_db))
        global_vmax = max(global_vmax, np.max(Sxx_db))
        
        spectrogram_data.append((label, f, t, Sxx_db))
        
    # Ora che conosciamo i limiti globali, generiamo i plot
    for i, (label, f, t, Sxx_db) in enumerate(spectrogram_data):
        plt.subplot(len(datasets), 1, i+1)
        # Usiamo vmin e vmax per fissare la scala colori
        pm = plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis', vmin=global_vmin, vmax=global_vmax)
        plt.colorbar(pm, label='Power (dB)')
        
        plt.ylim(0, 500) # Focus only on the 0-500 Hz range typical of EMG
        plt.title(f'Average Spectrogram (32 channels) - {label}')
        plt.ylabel('Frequency (Hz)')
        if i == len(datasets) - 1:
            plt.xlabel('Time (s)')

    plt.tight_layout()

    # Salvataggio Figure 2
    spectro_filename = os.path.join(output_dir, 'spectrogram_analysis.png')
    fig2.savefig(spectro_filename, dpi=300, bbox_inches='tight')
    print(f"Salvato Spectrogram plot: {spectro_filename}")

    plt.show()

if __name__ == "__main__":
    main()
