import pandas as pd
import numpy as np
import torch
import os
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d
from IKA import process_full_kinematics

class BionicFeatureExtractor:
    def __init__(self, fs_emg=2000):
        self.fs_emg = fs_emg
        nyq = self.fs_emg / 2.0
        
        # 1. Configurazione Filtri
        self.b_band, self.a_band = butter(4, [20/nyq, 450/nyq], btype='band')
        # Filtri Notch europei (50 Hz e armoniche)
        self.notches = [iirnotch(f, 30, fs=fs_emg) for f in [50, 100, 150]]
        
        # 2. Parametri Inviluppo e Finestra
        self.rms_window = 200   # ~100 ms
        self.rms_step = 25      # step di ~12.5 ms -> genera un segnale a ~80 Hz
        self.norm_emg = 5000.0  # Normalizzazione a 5 mV (5000 µV)
        
        # 3. Parametri Tensori Deep Learning
        self.history_len = 64   # 0.8 secondi di memoria a 80 Hz
        self.emg_subsample = 4  # Prende 1 campione ogni 4 (16 totali)
        self.ang_subsample = 8  # Prende 1 campione ogni 8 (8 totali)

    def process_emg(self, emg_df):
        print("Elaborazione EMG: Filtri Spaziali, Temporali e calcolo RMS...")
        timestamps = emg_df['Timestamp_LSL'].values
        # Supporto per file salvati sia con 'CH_' (es. vecchi trial) che con 'EMG_' (record_LSL.py)
        emg_channels = [col for col in emg_df.columns if 'CH_' in col or 'EMG_' in col]
        raw_data = emg_df[emg_channels].values # Shape: (Samples, 32)
        
        # A. Common Average Reference (CAR)
        car_data = raw_data - np.mean(raw_data, axis=1, keepdims=True)
        
        # B. Filtraggio Temporale (Zero-phase filtfilt)
        filtered = filtfilt(self.b_band, self.a_band, car_data, axis=0)
        for b, a in self.notches:
            filtered = filtfilt(b, a, filtered, axis=0)
            
        # C. Rettificazione e Normalizzazione [0, 1]
        rectified = np.abs(filtered) / self.norm_emg
        np.clip(rectified, 0, 1, out=rectified)
        
        # D. Estrazione RMS scorrevole
        num_windows = (len(rectified) - self.rms_window) // self.rms_step + 1
        
        # Calcolo vettorializzato delle finestre (molto più veloce del ciclo for)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(rectified, window_shape=self.rms_window, axis=0)[::self.rms_step]
        rms_data = np.sqrt(np.mean(windows**2, axis=-1))
        
        end_indices = np.arange(num_windows) * self.rms_step + self.rms_window - 1
        rms_time = timestamps[end_indices]
            
        return rms_data, rms_time

    def process_kinematics(self, kin_df, target_timestamps):
        print("Elaborazione Cinematica: Normalizzazione DoF e Sincronizzazione Temporale...")
        time_kin = kin_df['Timestamp_LSL'].values
        dof_cols = [col for col in kin_df.columns if 'DoF_' in col]
        raw_angles = kin_df[dof_cols].values # Shape: (Samples, 24)
        
        # A. Normalizzazione Angoli: (q + 150) / 240
        norm_angles = (raw_angles + 150.0) / 240.0
        np.clip(norm_angles, 0, 1, out=norm_angles)
        
        # B. Interpolazione Lineare per allineare gli angoli ai timestamp dell'EMG RMS
        interpolator = interp1d(time_kin, norm_angles, axis=0, bounds_error=False, fill_value="extrapolate")
        aligned_angles = interpolator(target_timestamps)
        
        # Previene che l'estrapolazione lineare produca target anomali fuori da [0, 1]
        np.clip(aligned_angles, 0, 1, out=aligned_angles)
        
        return aligned_angles

    def create_tensors(self, rms_data, aligned_angles):
        print("Creazione dei tensori PyTorch (Generazione loop ricorsivo)...")
        num_valid_samples = len(rms_data) - self.history_len
        
        # Pre-allocazione tensori finali
        X_emg = np.zeros((num_valid_samples, 512), dtype=np.float32)
        X_ang = np.zeros((num_valid_samples, 192), dtype=np.float32)
        Y_target = np.zeros((num_valid_samples, 24), dtype=np.float32)
        
        for i in range(num_valid_samples):
            # L'istante t da predire è la fine della finestra storica
            t_idx = i + self.history_len
            
            # Estrazione finestre storiche
            emg_window = rms_data[i:t_idx, :] # Shape: (64, 32)
            ang_window = aligned_angles[i:t_idx, :] # Shape: (64, 24)
            
            # Sottocampionamento temporale
            emg_sampled = emg_window[::self.emg_subsample, :] # Diventa (16, 32)
            ang_sampled = ang_window[::self.ang_subsample, :] # Diventa (8, 24)
            
            # Flatten per l'input fully connected
            X_emg[i] = emg_sampled.flatten()
            X_ang[i] = ang_sampled.flatten()
            
            # Il target è l'angolo all'istante t
            Y_target[i] = aligned_angles[t_idx, :]
            
        return torch.tensor(X_emg), torch.tensor(X_ang), torch.tensor(Y_target)

def main():
    # --- 1. DEFINIZIONE PERCORSI FILE ---
    # Ricava il percorso assoluto della cartella corrente dello script (NP_project)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EMG_FILE = "recordings/trial_1_EMG.csv"       # Usa trial_1 per TRAIN, trial_2 per VAL
    KIN_FILE = "recordings/trial_1_Kinematics.csv" 
    OUTPUT_FILE = "train_tensors.pt" # Cambia in val_tensors.pt quando processi il trial_2
    
    KIN_IKA_FILE = KIN_FILE.replace('.csv', '_IKA_24DoF.csv')
    
    try:
        df_emg = pd.read_csv(EMG_FILE)
        df_kin_raw = pd.read_csv(KIN_FILE)
        
        # --- 1.5 CONTROLLO E CALCOLO IKA AUTOMATICO ---
        if 'LM_0_X' in df_kin_raw.columns:
            if not os.path.exists(KIN_IKA_FILE):
                print(f"File IKA non trovato. Avvio calcolo IKA automatico su {KIN_FILE}...")
                process_full_kinematics(KIN_FILE)
            else:
                print(f"File IKA già presente: {KIN_IKA_FILE}. Salto il calcolo per risparmiare tempo.")
            df_kin = pd.read_csv(KIN_IKA_FILE)
        elif 'DoF_0' in df_kin_raw.columns:
            print(f"Il file {KIN_FILE} contiene già gli angoli (DoF). IKA non necessaria.")
            df_kin = df_kin_raw
        else:
            print("Errore: il file di cinematica non contiene né i landmark (LM) né gli angoli (DoF).")
            return
    except FileNotFoundError as e:
        print(f"Errore: File non trovato. Assicurati che i CSV siano nella stessa cartella. Dettagli: {e}")
        return

    # --- 2. ESECUZIONE PIPELINE ---
    extractor = BionicFeatureExtractor(fs_emg=2000)
    
    # Processa EMG
    rms_data, rms_timestamps = extractor.process_emg(df_emg)
    
    # Processa Cinematica e allinea
    aligned_angles = extractor.process_kinematics(df_kin, rms_timestamps)
    
    # Crea tensori
    X_e, X_a, Y = extractor.create_tensors(rms_data, aligned_angles)
    
    # --- EXTRA: SALVATAGGIO EMG RMS PER INFERENCE ---
    # Esportiamo l'RMS a ~80Hz per darlo in pasto a inference.py
    rms_df = pd.DataFrame(rms_data, columns=[f'CH_{i}' for i in range(rms_data.shape[1])])
    rms_df.insert(0, 'Timestamp_LSL', rms_timestamps)
    rms_out = EMG_FILE.replace('.csv', '_RMS.csv')
    rms_df.to_csv(rms_out, index=False)
    print(f"File RMS esportato per l'inferenza: {rms_out}")

    # --- 3. SALVATAGGIO DEI DATI ---
    # Salviamo i tensori in un dizionario PyTorch
    torch.save({
        'X_emg': X_e,
        'X_ang': X_a,
        'Y_target': Y
    }, OUTPUT_FILE)
    
    print(f"\nOperazione completata con successo!")
    print(f"Dimensione Tensore EMG: {X_e.shape} (Dovrebbe essere N x 512)")
    print(f"Dimensione Tensore Angoli: {X_a.shape} (Dovrebbe essere N x 192)")
    print(f"Dimensione Tensore Target: {Y.shape} (Dovrebbe essere N x 24)")
    print(f"Dataset salvato in '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()