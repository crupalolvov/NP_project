import pandas as pd
import numpy as np
import torch
import os
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d
from core_kin import process_full_kinematics_core, extract_anatomy_from_csv

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
        
        # A. Common Average Reference (CAR) - NO! facciamo rimozione DC individuale per canale
        #car_data = raw_data - np.mean(raw_data, axis=1, keepdims=True)
        dc_removed_data = raw_data - np.mean(raw_data, axis=0, keepdims=True)

        # B. Filtraggio Temporale (Zero-phase filtfilt)
        filtered = filtfilt(self.b_band, self.a_band, dc_removed_data, axis=0)
        for b, a in self.notches:
            filtered = filtfilt(b, a, filtered, axis=0)
            
        # C. Rettificazione e Normalizzazione [0, 1]
        rectified = np.abs(filtered) / self.norm_emg
        #np.clip(rectified, 0, 1, out=rectified)
        
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
        print("Elaborazione Cinematica: Sottrazione Rest Angles e Normalizzazione DoF...")
        time_kin = kin_df['Timestamp_LSL'].values
        dof_cols = [col for col in kin_df.columns if 'DoF_' in col]
        raw_angles = kin_df[dof_cols].values # Shape: (Samples, 24)
        
        # 1. Calcolo degli angoli di riposo (Rest Angles)
        # Sfruttiamo il tuo protocollo: 8 secondi iniziali.
        # Scartiamo il primo secondo (assestamento sensori) e prendiamo i successivi 4 secondi.
        start_time = time_kin[0]
        mask_rest = (time_kin >= start_time + 1.0) & (time_kin <= start_time + 5.0)
        
        if not np.any(mask_rest):
            print("ATTENZIONE: Finestra di riposo non trovata, uso i primi 100 sample.")
            q_rest = np.mean(raw_angles[:100, :], axis=0)
        else:
            q_rest = np.mean(raw_angles[mask_rest, :], axis=0)
            
        # 2. "Subtraction of rest angles"
        centered_angles = raw_angles - q_rest
        # --- FIX CINEMATICO: ANGLE WRAPPING ---
        # Riporta tutti gli angoli nel range [-180, 180] gradi per eliminare i salti di 360°
        centered_angles = (centered_angles + 180.0) % 360.0 - 180.0

        # 3. Normalizzazione come da paper RPC-Net
        norm_angles = (centered_angles + 150.0) / 240.0
        
        # Ora la varianza è centrata. Niente np.clip() prima dell'interpolazione!
        
        # 4. Interpolazione Lineare
        interpolator = interp1d(time_kin, norm_angles, axis=0, bounds_error=False, fill_value=(norm_angles[0], norm_angles[-1]))
        aligned_angles = interpolator(target_timestamps)
        
        return aligned_angles, q_rest # Restituisci q_rest se ti serve salvarlo

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

        Y_target = np.clip(Y_target, 0.0, 1.0)
            
        return X_emg, X_ang, Y_target
        

def main():
    # --- 1. DEFINIZIONE PERCORSI FILE ---
    # Ricava il percorso assoluto della cartella corrente dello script (NP_project)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    CALIB_FILE = os.path.join(BASE_DIR, "hand_calibration.pt")
    
    # Parametri per la calibrazione anatomica (presi da core_kin.py)
    START_CALIB = 1530
    END_CALIB = 1700
    
    # --- 1.2 CONTROLLO E CALCOLO CALIBRAZIONE ANATOMICA ---
    if not os.path.exists(CALIB_FILE):
        ref_kin = os.path.join(BASE_DIR, "recordings/trial_1_Kinematics.csv")
        print(f"File di calibrazione anatomica '{CALIB_FILE}' non trovato.")
        if os.path.exists(ref_kin):
            print(f"Avvio calcolo calibrazione automatica da {ref_kin} (frame {START_CALIB}-{END_CALIB})...")
            try:
                anatomy = extract_anatomy_from_csv(ref_kin, START_CALIB, END_CALIB)
                torch.save(anatomy, CALIB_FILE)
                print(f"Calibrazione anatomica salvata in: {os.path.basename(CALIB_FILE)}")
            except Exception as e:
                print(f"Errore durante la creazione del file di calibrazione: {e}")
                return
        else:
            print(f"Errore: trial_1_Kinematics.csv non trovato per la calibrazione.")
            return

    splits = {
        'train': [1, 2, 3, 4, 7, 8, 10, 11, 13],
        'val': [5, 14],
        'test': [6, 9, 15]
    }

    extractor = BionicFeatureExtractor(fs_emg=2000)
    train_stats = None

    for split_name, trial_list in splits.items():
        print(f"\n==================================================")
        print(f" CREAZIONE SPLIT: {split_name.upper()}")
        print(f" Trials: {trial_list}")
        print(f"==================================================")
        
        all_X_e, all_X_a, all_Y = [], [], []
        last_q_rest = None
        
        for trial_num in trial_list:
            print(f"\n--- Processando Trial {trial_num} ---")
            emg_file = os.path.join(BASE_DIR, f"recordings/trial_{trial_num}_EMG.csv")
            kin_file = os.path.join(BASE_DIR, f"recordings/trial_{trial_num}_Kinematics.csv")
            kin_ika_file = kin_file.replace('.csv', '_core_IKA_24DoF.csv')
            
            if not os.path.exists(emg_file) or not os.path.exists(kin_file):
                print(f"TRIAL {trial_num} MANCANTE. Salto...")
                continue
                
            df_emg = pd.read_csv(emg_file)
            df_kin_raw = pd.read_csv(kin_file)
            
            if 'LM_0_X' in df_kin_raw.columns:
                if not os.path.exists(kin_ika_file):
                    print(f"File IKA non trovato. Avvio calcolo IKA automatico su {kin_file}...")
                    process_full_kinematics_core(kin_file)
                else:
                    print(f"File IKA già presente per trial {trial_num}.")
                df_kin = pd.read_csv(kin_ika_file)
            elif 'DoF_0' in df_kin_raw.columns:
                df_kin = df_kin_raw
            else:
                print(f"Errore: formato cinematica non valido in {kin_file}.")
                continue

            # --- CONTROLLO DURATA TEMPORALE ---
            start_emg = df_emg['Timestamp_LSL'].iloc[0]
            start_kin = df_kin['Timestamp_LSL'].iloc[0]
            durata_emg = df_emg['Timestamp_LSL'].iloc[-1] - start_emg
            durata_kin = df_kin['Timestamp_LSL'].iloc[-1] - start_kin
            print(f"Durata netta EMG: {durata_emg:.2f} sec | Durata netta KIN: {durata_kin:.2f} sec")

            # Processa e allinea
            rms_data, rms_timestamps = extractor.process_emg(df_emg)
            aligned_angles, q_rest = extractor.process_kinematics(df_kin, rms_timestamps)    
            
            # Estrazione array
            X_e, X_a, Y = extractor.create_tensors(rms_data, aligned_angles)
            
            all_X_e.append(X_e)
            all_X_a.append(X_a)
            all_Y.append(Y)
            last_q_rest = q_rest
            
            # --- SALVATAGGIO EMG RMS PER INFERENCE ---
            rms_df = pd.DataFrame(rms_data, columns=[f'CH_{i}' for i in range(rms_data.shape[1])])
            rms_df.insert(0, 'Timestamp_LSL', rms_timestamps)
            rms_out = emg_file.replace('.csv', '_RMS.csv')
            rms_df.to_csv(rms_out, index=False)
            print(f"File RMS esportato per l'inferenza: {rms_out}")

        if not all_X_e:
            print(f"Nessun dato valido processato per lo split {split_name}.")
            continue
            
        # --- CONCATENAZIONE E STANDARDIZZAZIONE ---
        X_e_tensor = torch.tensor(np.concatenate(all_X_e, axis=0))
        X_a_tensor = torch.tensor(np.concatenate(all_X_a, axis=0))
        Y_tensor = torch.tensor(np.concatenate(all_Y, axis=0))
        
        if split_name == 'train':
            emg_mean = X_e_tensor.mean(dim=0)
            emg_std = X_e_tensor.std(dim=0) + 1e-6
            ang_mean = X_a_tensor.mean(dim=0)
            ang_std = X_a_tensor.std(dim=0) + 1e-6
            train_stats = (emg_mean, emg_std, ang_mean, ang_std)
        else:
            if train_stats is None:
                print("ATTENZIONE: Le statistiche di Train non sono state calcolate! Uso statistiche dello split corrente.")
                emg_mean = X_e_tensor.mean(dim=0)
                emg_std = X_e_tensor.std(dim=0) + 1e-6
                ang_mean = X_a_tensor.mean(dim=0)
                ang_std = X_a_tensor.std(dim=0) + 1e-6
            else:
                emg_mean, emg_std, ang_mean, ang_std = train_stats
                
        # Applica standardizzazione globale
        X_e_tensor = (X_e_tensor - emg_mean) / emg_std
        X_a_tensor = (X_a_tensor - ang_mean) / ang_std
        
        # --- SALVATAGGIO DEL DATASET ---
        output_file = os.path.join(BASE_DIR, f"{split_name}_tensors.pt")
        torch.save({
            'X_emg': X_e_tensor, 'X_ang': X_a_tensor, 'Y_target': Y_tensor,
            'emg_mean': emg_mean, 'emg_std': emg_std,
            'ang_mean': ang_mean, 'ang_std': ang_std,
            'q_rest': last_q_rest
        }, output_file)
        
        print(f"\n✅ SPLIT '{split_name.upper()}' COMPLETATO E SALVATO IN: {os.path.basename(output_file)}")
        print(f"Dimensione Tensore EMG: {X_e_tensor.shape}")
        print(f"Dimensione Tensore Angoli: {X_a_tensor.shape}")
        print(f"Dimensione Tensore Target: {Y_tensor.shape}")

if __name__ == "__main__":
    main()