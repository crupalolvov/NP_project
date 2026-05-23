import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

def diagnose_emg_quality(df_emg, fs=2000, rest_duration=5.0):
    """
    Analizza la qualità del segnale EMG (PSD, SNR, Dinamica)
    """
    print("--- AVVIO DIAGNOSTICA EMG ---")
    emg_channels = [col for col in df_emg.columns if 'CH_' in col or 'EMG_' in col]
    raw_data = df_emg[emg_channels].values
    timestamps = df_emg['Timestamp_LSL'].values.copy()
    timestamps -= timestamps[0] # Normalizza a t=0
    
    # 1. Analisi SNR (Signal-to-Noise Ratio)
    # Assumiamo che i primi 'rest_duration' secondi siano di puro riposo
    rest_mask = timestamps < rest_duration
    active_mask = timestamps >= rest_duration
    
    rms_rest = np.sqrt(np.mean(raw_data[rest_mask]**2, axis=0))
    rms_active_max = np.sqrt(np.max(raw_data[active_mask]**2, axis=0)) # Stima del picco
    
    # Prevenzione divisione per zero
    rms_rest[rms_rest == 0] = 1e-6 
    
    snr_db = 20 * np.log10(rms_active_max / rms_rest)
    mean_snr = np.mean(snr_db)
    
    print(f"1. Rapporto Segnale-Rumore (SNR):")
    print(f"   SNR Medio tra i canali: {mean_snr:.2f} dB")
    if mean_snr < 10:
        print("   -> ATTENZIONE: SNR molto basso! Il segnale è dominato dal rumore.")
    elif mean_snr > 20:
        print("   -> OTTIMO: Il segnale muscolare emerge chiaramente dalla baseline.")
        
    # 2. Analisi Spettrale (PSD) sul Canale 0
    f, Pxx = welch(raw_data[:, 0], fs=fs, nperseg=4096)
    
    plt.figure(figsize=(12, 8))
    
    # Plot PSD
    plt.subplot(2, 1, 1)
    plt.semilogy(f, Pxx, color='blue')
    plt.axvspan(20, 450, color='green', alpha=0.2, label='Banda Fisiologica Ideale')
    plt.axvline(50, color='red', linestyle='--', label='Rumore di Rete (50 Hz)')
    plt.title("Densità Spettrale di Potenza (PSD) - Canale 0")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("PSD (V^2/Hz)")
    plt.xlim(0, 600)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Visualizzazione Temporale (Confronto 3 Canali Random)
    plt.subplot(2, 1, 2)
    ch_to_plot = [0, len(emg_channels)//2, len(emg_channels)-1] # Prende 3 canali distanti
    for idx, ch in enumerate(ch_to_plot):
        # Calcolo un RMS rapido per la visualizzazione
        window = int(fs * 0.1) # 100ms
        ch_data = raw_data[:, ch]
        ch_data = ch_data - np.mean(ch_data) # Rimuovi DC bias locale
        rectified = np.abs(ch_data)
        from scipy.ndimage import uniform_filter1d
        rms_vis = uniform_filter1d(rectified**2, window) ** 0.5
        
        plt.plot(timestamps, rms_vis + (idx * 1.5 * np.max(rms_vis)), label=f'Canale {ch}')
        
    plt.axvspan(0, rest_duration, color='gray', alpha=0.2, label='Fase di Riposo')
    plt.title("Inviluppo RMS (3 Canali rappresentativi)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza RMS (Sfalsata per visibilità)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Per chiamarla:
df_emg = pd.read_csv("recordings/trial_3_EMG.csv")
diagnose_emg_quality(df_emg)