import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, lfilter, lfilter_zi
import time
import os
from core_kin import CleanHandFK

from RPC_Net import RPCNet_Exact

class SingleJointNet_Exact(nn.Module):
    def __init__(self, in_emg=512, in_ang=192):
        super().__init__()
        self.emg_branch = nn.Sequential(nn.Linear(in_emg, 512), nn.ReLU())
        self.ang_branch = nn.Sequential(
            nn.Linear(in_ang, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU()
        )
        self.merged = nn.Sequential(
            nn.Linear(512 + 24, 134), nn.ReLU(),
            nn.Linear(134, 134), nn.ReLU(),
            nn.Linear(134, 134), nn.ReLU(),
            nn.Linear(134, 1),
            nn.Sigmoid()
        )

    def forward(self, emg, ang):
        e_out = self.emg_branch(emg)
        a_out = self.ang_branch(ang)
        combined = torch.cat((e_out, a_out), dim=1)
        return self.merged(combined)

class RPCNet_Exact(nn.Module):
    def __init__(self, in_emg=512, in_ang=192):
        super().__init__()
        self.sub_nets = nn.ModuleList([SingleJointNet_Exact(in_emg, in_ang) for _ in range(24)])

    def forward(self, emg, ang):
        outputs = [net(emg, ang) for net in self.sub_nets]
        return torch.cat(outputs, dim=1)

# --- 2. DECODER OFFLINE ---
def run_offline_inference(emg_csv_path, model_path, output_csv_path, stats_file):
    print(f"Avvio decodifica offline sul file: {os.path.basename(emg_csv_path)}")
    
    # 1. Caricamento Dati EMG
    df_emg = pd.read_csv(emg_csv_path)
    timestamps = df_emg['Timestamp_LSL'].values if 'Timestamp_LSL' in df_emg.columns else np.arange(len(df_emg))
    
    emg_columns = [col for col in df_emg.columns if 'CH_' in col or 'EMG_' in col]
    emg_data = df_emg[emg_columns].values
    num_samples = len(emg_data)
    
    # 2. Inizializzazione Modello PyTorch
    model = RPCNet_Exact(in_emg=512, in_ang=192)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 3. Caricamento statistiche globali (solo per angoli)
    print(f"Caricamento statistiche cinematiche globali da: {os.path.basename(stats_file)}")
    train_stats = torch.load(stats_file, weights_only=False)
    ang_mean = train_stats['ang_mean'].numpy()
    ang_std = train_stats['ang_std'].numpy()
    
    # --- CALIBRAZIONE LOCALE EMG (Domain Adaptation) ---
    # Invece di usare l'EMG globale, calcoliamo le statistiche su questo specifico trial.
    # Questo annulla i cambiamenti di impedenza elettrodo-pelle!
    emg_mean_32 = np.mean(emg_data, axis=0)
    emg_std_32 = np.std(emg_data, axis=0) + 1e-6
    
    # La rete si aspetta 512 valori (16 time steps x 32 canali). 
    # Usiamo np.tile per ripetere le medie dei 32 canali per 16 volte.
    emg_mean_local = np.tile(emg_mean_32, 16)
    emg_std_local = np.tile(emg_std_32, 16)
    print("Calibrazione EMG locale (per-trial) applicata con successo.")
    
    # 4. Inizializzazione Buffer (0.78s di memoria)
    emg_buffer = np.zeros((64, 32), dtype=np.float32)
    # Posa di riposo normalizzata: 150/240 = 0.625
    ang_buffer = np.full((64, 24), 150.0 / 240.0, dtype=np.float32)
    
    # 5. Inizializzazione Filtro Passa-Basso
    fs_output = 80.0
    b, a = butter(4, 1.0 / (fs_output / 2.0), btype='low')
    zi_base = lfilter_zi(b, a)
    zi = np.array([zi_base * (150.0 / 240.0) for _ in range(24)])
    
    predicted_kinematics = []
    start_time = time.time()
    WINDOW_SIZE = 64
    
    # 6. Loop di Simulazione Temporale
    for i in range(num_samples):
        current_emg_rms = emg_data[i, :]
        emg_buffer = np.roll(emg_buffer, shift=-1, axis=0)
        emg_buffer[-1, :] = current_emg_rms
        
        if i < WINDOW_SIZE:
            predicted_kinematics.append(np.full(24, 150.0 / 240.0))
            continue
        
        emg_input = emg_buffer[::4, :].flatten()
        ang_input = ang_buffer[::8, :].flatten()
        
        # --- Z-SCORE Ibrido ---
        # EMG usa la statistica locale, Angoli usano la statistica globale
        emg_input_norm = (emg_input - emg_mean_local) / emg_std_local
        ang_input_norm = (ang_input - ang_mean) / ang_std
        
        with torch.no_grad():
            emg_tensor = torch.tensor(emg_input_norm, dtype=torch.float32).unsqueeze(0)
            ang_tensor = torch.tensor(ang_input_norm, dtype=torch.float32).unsqueeze(0)
            pred_angles = model(emg_tensor, ang_tensor).numpy()[0]
            
        smoothed_angles = np.zeros(24)
        for j in range(24):
            filtered_val, zi[j] = lfilter(b, a, [pred_angles[j]], zi=zi[j])
            smoothed_angles[j] = filtered_val[0]
            
        ang_buffer = np.roll(ang_buffer, shift=-1, axis=0)
        ang_buffer[-1, :] = smoothed_angles
        
        predicted_kinematics.append(smoothed_angles)
        
        if i % 1000 == 0 and i > 0:
            print(f"Processati {i} frame su {num_samples}...")

    print(f"Inferenza completata in {time.time() - start_time:.2f} secondi.")

    output_data = []
    for i in range(num_samples):
        row_dict = {'Timestamp_LSL': timestamps[i]}
        for j in range(24):
            row_dict[f'Pred_DoF_{j}'] = predicted_kinematics[i][j]
        output_data.append(row_dict)
        
    df_out = pd.DataFrame(output_data)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Predizioni salvate in: {output_csv_path}")

# --- 3. DECODIFICA IN COORDINATE 3D ---
def decode_predictions_to_csv(pred_csv_path, calibration_file, output_lms_path, stats_file):
    print(f"\n--- AVVIO DECODIFICA FK (Angoli -> Coordinate 3D) ---")
    df_pred = pd.read_csv(pred_csv_path)
    
    if not os.path.exists(calibration_file):
        print(f"Errore: File di calibrazione '{calibration_file}' non trovato.")
        return
        
    anatomy = torch.load(calibration_file, weights_only=False)
    fk = CleanHandFK(anatomy)
    
    train_stats = torch.load(stats_file, weights_only=False)
    q_rest = train_stats['q_rest'] 

    num_frames = len(df_pred)
    lms_data = []
    pred_cols = [f'Pred_DoF_{i}' for i in range(24)]
    
    # LA TUA LOGICA: fk.forward restituisce millimetri, il CSV originale è in metri.
    SCALE_MM_TO_METERS = 1000.0 
    
    for i in range(num_frames):
        row = df_pred.iloc[i]
        q_norm = row[pred_cols].values
        
        q_deg_centered = (q_norm * 240.0) - 150.0
        q_deg_absolute = q_deg_centered + q_rest
        q_rad = np.radians(q_deg_absolute)
        
        # 1. Output in millimetri
        lms_3d_mm = fk.forward(q_rad)
        
        # 2. Riconversione in METRI (come i World Landmarks)
        lms_3d_meters = lms_3d_mm / SCALE_MM_TO_METERS
        
        row_out = {'Timestamp_LSL': row['Timestamp_LSL']}
        for lm_idx in range(21):
            # Nessun offset: il polso resta (0,0,0)
            row_out[f'LM_{lm_idx}_X'] = lms_3d_meters[lm_idx, 0]
            row_out[f'LM_{lm_idx}_Y'] = lms_3d_meters[lm_idx, 1]
            row_out[f'LM_{lm_idx}_Z'] = lms_3d_meters[lm_idx, 2]
            
        lms_data.append(row_out)
            
    pd.DataFrame(lms_data).to_csv(output_lms_path, index=False)
    print(f"Coordinate decodificate salvate in: {output_lms_path}\n")

# --- ESECUZIONE ---
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # ---------------------------------------------------------
    # MODIFICA QUI IL NUMERO DEL TRIAL CHE VUOI TESTARE
    # Es. 6, 9, 15 (trial di Test non visti dalla rete)
    TRIAL_TEST = 15
    # ---------------------------------------------------------
    
    FILE_EMG_RMS = os.path.join(BASE_DIR, "recordings", f"trial_{TRIAL_TEST}_EMG_RMS.csv") 
    FILE_MODELLO = os.path.join(BASE_DIR, "rpc_net_weights.pth")
    FILE_OUTPUT_ANGLES = os.path.join(BASE_DIR, f"trial_{TRIAL_TEST}_predicted_kinematics_angles.csv")
    FILE_CALIB = os.path.join(BASE_DIR, "hand_calibration.pt")
    FILE_OUTPUT_LMS = os.path.join(BASE_DIR, f"trial_{TRIAL_TEST}_predicted_kinematics_lms.csv")
    FILE_STATS = os.path.join(BASE_DIR, "train_tensors.pt")
    
    if os.path.exists(FILE_EMG_RMS):
        # 1. Inferenza (EMG -> Angoli)
        run_offline_inference(FILE_EMG_RMS, FILE_MODELLO, FILE_OUTPUT_ANGLES, FILE_STATS)
        
        # 2. Decodifica (Angoli -> Coordinate 3D MediaPipe)
        decode_predictions_to_csv(FILE_OUTPUT_ANGLES, FILE_CALIB, FILE_OUTPUT_LMS, FILE_STATS)
    else:
        print(f"Errore: File {FILE_EMG_RMS} non trovato. Assicurati di aver generato i file RMS!")