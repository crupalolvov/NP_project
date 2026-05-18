import pandas as pd
import numpy as np
from scipy.optimize import minimize
import time

# --- 1. FUNZIONI MATEMATICHE (ROTOTRASLAZIONI) ---
def Rx(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

# --- 2. MODELLO CINEMATICO E OTTIMIZZAZIONE ---
class HandIKA:
    def __init__(self, calibration_data):
        self.num_dofs = 24
        self.calib = calibration_data
        
        # Limiti anatomici [Min, Max] in radianti per i 24 DoF
        self.bounds = [
            # Polso (3 DoF: Flex/Ext, Abd/Add, Pro/Sup)
            (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14),
            # Pollice (5 DoF: CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP_FE)
            (-1.0, 1.0), (-1.0, 1.5), (-1.0, 1.0), (-1.0, 1.5), (-1.0, 1.5),
            # Indice, Medio, Anulare, Mignolo (4 DoF x 4: MCP_FE, MCP_AA, PIP_FE, DIP_FE)
            (-0.2, 1.5), (-0.4, 0.4), (-0.1, 1.7), (-0.1, 1.5), # Indice
            (-0.2, 1.5), (-0.3, 0.3), (-0.1, 1.7), (-0.1, 1.5), # Medio
            (-0.2, 1.5), (-0.3, 0.3), (-0.1, 1.7), (-0.1, 1.5), # Anulare
            (-0.2, 1.5), (-0.4, 0.4), (-0.1, 1.7), (-0.1, 1.5)  # Mignolo
        ]
        
        # Stato corrente per il "warm start" (inizia da zero)
        self.q_current = np.zeros(self.num_dofs)

    def forward_kinematics(self, q):
        """ Calcola la posizione 3D teorica dei 21 landmark dati i 24 angoli q """
        LMs = np.zeros((21, 3))
        
        # 0. Polso (Origine + Rotazione Globale del polso)
        R_wrist = Rx(q[0]) @ Ry(q[1]) @ Rz(q[2])
        LMs[0] = [0, 0, 0] # Il polso è sempre all'origine locale
        
        # --- DITA LUNGHE (Indice=1, Medio=2, Anulare=3, Mignolo=4) ---
        finger_indices = {
            'Index': (5, 6, 7, 8, 8),   # ID dei landmark MediaPipe
            'Middle': (9, 10, 11, 12, 12),
            'Ring': (13, 14, 15, 16, 16),
            'Pinky': (17, 18, 19, 20, 20)
        }
        
        for idx, (finger_name, lm_idx) in enumerate(finger_indices.items()):
            q_idx = 8 + (idx * 4) # Indice di partenza nel vettore q (salta polso e pollice)
            
            # Vettore Metacarpo (Fisso rispetto al polso, calibrato)
            v_meta = self.calib['meta_vectors'][finger_name]
            # Direzione di estensione naturale del dito (invece di Y globale)
            finger_dir = v_meta / (np.linalg.norm(v_meta) + 1e-6)
            LMs[lm_idx[0]] = R_wrist @ v_meta
            
            # Matrice di splay per allineare l'asse Y locale al metacarpo
            theta_x = np.arcsin(np.clip(finger_dir[2], -1.0, 1.0))
            theta_z = np.arctan2(-finger_dir[0], finger_dir[1])
            R_splay = Rz(theta_z) @ Rx(theta_x)

            # Articolazione MCP (Rotazione su 2 assi: Flex/Ext e Abd/Add)
            R_mcp = R_wrist @ R_splay @ Rx(q[q_idx]) @ Rz(q[q_idx+1])
            v_prox = np.array([0, self.calib['lengths'][f'{finger_name}_Proximal'], 0])
            LMs[lm_idx[1]] = LMs[lm_idx[0]] + (R_mcp @ v_prox)
            
            # Articolazione PIP (Rotazione su 1 asse: Flex/Ext)
            R_pip = R_mcp @ Rx(q[q_idx+2])
            v_inter = np.array([0, self.calib['lengths'][f'{finger_name}_Intermediate'], 0])
            LMs[lm_idx[2]] = LMs[lm_idx[1]] + (R_pip @ v_inter)
            
            # Articolazione DIP (Rotazione su 1 asse: Flex/Ext)
            R_dip = R_pip @ Rx(q[q_idx+3])
            v_dist = np.array([0, self.calib['lengths'][f'{finger_name}_Distal'], 0])
            LMs[lm_idx[3]] = LMs[lm_idx[2]] + (R_dip @ v_dist)

        # --- POLLICE --- (Semplificato per adattarsi all'array)
        v_meta_t = self.calib['meta_vectors']['Thumb']
        thumb_dir = v_meta_t / (np.linalg.norm(v_meta_t) + 1e-6)
        LMs[1] = R_wrist @ v_meta_t
        
        theta_x_t = np.arcsin(np.clip(thumb_dir[2], -1.0, 1.0))
        theta_z_t = np.arctan2(-thumb_dir[0], thumb_dir[1])
        R_splay_t = Rz(theta_z_t) @ Rx(theta_x_t)

        R_cmc = R_wrist @ R_splay_t @ Rx(q[3]) @ Rz(q[4])
        v_prox_t = np.array([0, self.calib['lengths']['Thumb_Proximal'], 0])
        LMs[2] = LMs[1] + (R_cmc @ v_prox_t)
        
        R_mcp_t = R_cmc @ Rx(q[5]) @ Rz(q[6])
        v_inter_t = np.array([0, self.calib['lengths']['Thumb_Intermediate'], 0])
        LMs[3] = LMs[2] + (R_mcp_t @ v_inter_t)
        
        R_ip_t = R_mcp_t @ Rx(q[7])
        v_dist_t = np.array([0, self.calib['lengths']['Thumb_Distal'], 0])
        LMs[4] = LMs[3] + (R_ip_t @ v_dist_t)

        return LMs

    def objective_function(self, q, target_landmarks):
        estimated = self.forward_kinematics(q)
        # Errore Quadratico Medio pesato (diamo più peso alle punte delle dita)
        weights = np.ones(21)
        weights[[4, 8, 12, 16, 20]] = 2.0 
        
        # FIX MATEMATICO: Convertiamo i metri in millimetri (* 1000)
        # Questo impedisce al solutore SLSQP di fermarsi all'iterazione 0 per "Loss troppo piccola"
        diff_mm = (estimated - target_landmarks) * 1000.0
        error = np.sum(weights * np.linalg.norm(diff_mm, axis=1)**2)
        return error

    def solve_frame(self, target_landmarks):
        # Trasla i target in modo che il polso (LM 0) sia nell'origine
        wrist_pos = target_landmarks[0]
        centered_targets = target_landmarks - wrist_pos
        
        result = minimize(
            fun=self.objective_function,
            x0=self.q_current,             
            args=(centered_targets,),
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'ftol': 1e-6, 'maxiter': 60, 'disp': False}
        )
        
        # Assegniamo sempre il risultato calcolato
        self.q_current = result.x
        return self.q_current

# --- 3. PIPELINE AUTOMATIZZATA ---
def process_full_kinematics(csv_path):
    print("Avvio Pipeline IKA 24-DoF...")
    df = pd.read_csv(csv_path)
    
    # 3A. CARICAMENTO CALIBRAZIONE STATICA (Da Foto)
    import torch
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CALIB_FILE = os.path.join(BASE_DIR, "hand_calibration.pt")
    
    if not os.path.exists(CALIB_FILE):
        print(f"Errore: File di calibrazione '{CALIB_FILE}' non trovato!")
        print("Esegui prima lo script 'photo_calibration.py' con la foto della tua mano.")
        return
        
    calib_data = torch.load(CALIB_FILE, weights_only=False)
    print("Calibrazione statica (dalla foto) caricata con successo.")
    print("Avvio Ottimizzazione L-BFGS-B...")

    # 3B. FASE DI OTTIMIZZAZIONE IKA
    ika_solver = HandIKA(calib_data)
    angles_data = []
    
    start_time = time.time()
    for index, row in df.iterrows():
        target_landmarks = np.zeros((21, 3))
        for i in range(21):
            target_landmarks[i] = [row[f'LM_{i}_X'], row[f'LM_{i}_Y'], row[f'LM_{i}_Z']]
            
        calculated_angles = ika_solver.solve_frame(target_landmarks)
        
        row_dict = {'Timestamp_LSL': row['Timestamp_LSL']}
        for i in range(24):
            row_dict[f'DoF_{i}'] = np.degrees(calculated_angles[i]) # Salviamo in Gradi per leggibilità
            
        angles_data.append(row_dict)
        
        if index % 100 == 0 and index > 0:
            print(f"Processati {index} frame su {len(df)}...")

    print(f"Completato in {time.time() - start_time:.2f} secondi.")
    
    # 3C. SALVATAGGIO
    angles_df = pd.DataFrame(angles_data)
    output_name = csv_path.replace('.csv', '_IKA_24DoF_v7.csv')
    angles_df.to_csv(output_name, index=False)
    print(f"Dati salvati in: {output_name}")

if __name__ == "__main__":
    FILE_INPUT = "NP_project/recordings/trial_5_Kinematics.csv"
    process_full_kinematics(FILE_INPUT)