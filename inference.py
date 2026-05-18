import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, lfilter
import time


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
            nn.Linear(134, 1)
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
def run_offline_inference(emg_csv_path, model_path, output_csv_path):
    print(f"Avvio decodifica offline sul file: {emg_csv_path}")
    
    # 1. Caricamento Dati EMG
    df_emg = pd.read_csv(emg_csv_path)
    
    # Estrazione dei Timestamp (se presenti) per sincronizzare l'output
    timestamps = df_emg['Timestamp_LSL'].values if 'Timestamp_LSL' in df_emg.columns else np.arange(len(df_emg))
    
    # Estrazione dei 32 canali (assumendo che le colonne si chiamino CH_0, CH_1, ... CH_31)
    # Adatta questo filtro in base all'intestazione reale del tuo CSV
    emg_columns = [col for col in df_emg.columns if 'CH_' in col or 'EMG_' in col]
    emg_data = df_emg[emg_columns].values
    
    num_samples = len(emg_data)
    
    # 2. Inizializzazione Modello PyTorch
    model = RPCNet_Exact(in_emg=512, in_ang=192)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 3. Inizializzazione Buffer (0.78s di memoria)
    emg_buffer = np.zeros((64, 32), dtype=np.float32)
    ang_buffer = np.zeros((64, 24), dtype=np.float32)
    
    # 4. Inizializzazione Filtro Passa-Basso (1 Hz su campionamento RMS a 80.0 Hz effettivi)
    fs_output = 80.0
    b, a = butter(4, 1.0 / (fs_output / 2.0), btype='low')
    zi = np.zeros((24, max(len(a), len(b)) - 1))
    
    predicted_kinematics = []
    
    start_time = time.time()
    
    # Dimensione della finestra in sample (0.78s * 81.92Hz = ~64 sample)
    WINDOW_SIZE = 64
    
    # 5. Loop di Simulazione Temporale
    for i in range(num_samples):
        # Acquisizione del dato "corrente" dal dataset
        current_emg_rms = emg_data[i, :]
        
        # Aggiornamento buffer EMG
        emg_buffer = np.roll(emg_buffer, shift=-1, axis=0)
        emg_buffer[-1, :] = current_emg_rms
        
        # "The initial 0.78 s of a session may not be used [...] due to the absence of sufficient earlier data."
        if i < WINDOW_SIZE:
            # Riempiamo l'output con zeri per mantenere allineati i timestamp di LSL
            predicted_kinematics.append(np.zeros(24))
            continue
        
        # Sottocampionamento per la rete (Feature Extraction Spaziale e Inerziale)
        emg_input = emg_buffer[::4, :].flatten()
        ang_input = ang_buffer[::8, :].flatten()
        
        # Inferenza
        with torch.no_grad():
            emg_tensor = torch.tensor(emg_input, dtype=torch.float32).unsqueeze(0)
            ang_tensor = torch.tensor(ang_input, dtype=torch.float32).unsqueeze(0)
            pred_angles = model(emg_tensor, ang_tensor).numpy()[0]
            
        # Filtraggio
        smoothed_angles = np.zeros(24)
        for j in range(24):
            filtered_val, zi[j] = lfilter(b, a, [pred_angles[j]], zi=zi[j])
            smoothed_angles[j] = filtered_val[0]
            
        # Aggiornamento buffer Angoli (Loop Ricorsivo)
        ang_buffer = np.roll(ang_buffer, shift=-1, axis=0)
        ang_buffer[-1, :] = smoothed_angles
        
        predicted_kinematics.append(smoothed_angles)
        
        if i % 1000 == 0 and i > 0:
            print(f"Processati {i} frame su {num_samples}...")

    print(f"Inferenza completata in {time.time() - start_time:.2f} secondi.")

    # 6. Salvataggio dei Risultati
    output_data = []
    for i in range(num_samples):
        row_dict = {'Timestamp_LSL': timestamps[i]}
        for j in range(24):
            # Denormalizzazione opzionale: se la rete predice nel range [0, 1], 
            # decommenta questa riga per riportare gli angoli in gradi reali
            # val_gradi = (predicted_kinematics[i][j] * 240.0) - 150.0
            
            row_dict[f'Pred_DoF_{j}'] = predicted_kinematics[i][j]
        output_data.append(row_dict)
        
    df_out = pd.DataFrame(output_data)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Predizioni salvate con successo in: {output_csv_path}")

# --- ESECUZIONE ---
if __name__ == "__main__":
    # Assicurati di avere il modello addestrato e il dataset pronto
    FILE_EMG_RMS = "NP_project/recordings/trial_3_EMG_RMS.csv" # Trial 3 ora è usato per il testing!
    FILE_MODELLO = "NP_project/rpc_net_weights.pth"
    FILE_OUTPUT = "NP_project/predicted_kinematics_offline.csv"
    
    run_offline_inference(FILE_EMG_RMS, FILE_MODELLO, FILE_OUTPUT)