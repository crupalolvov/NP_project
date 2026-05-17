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
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), 2, 0], [0, 0, 1]]) # Corretto sotto

def Rz_corr(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

# --- 2. MODELLO CINEMATICO E OTTIMIZZAZIONE ---
class HandIKA:
    def __init__(self, calibration_data):
        self.num_dofs = 24
        self.calib = calibration_data
        
        # Limiti anatomici [Min, Max] in radianti per i 24 DoF
        self.bounds = [
            # Polso (3 DoF: Flex/Ext, Abd/Add, Pro/Sup)
            (-1.5, 1.5), (-0.5, 0.5), (-1.5, 1.5),
            # Pollice (5 DoF: CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP_FE)
            (-0.5, 0.5), (-0.5, 1.5), (-0.5, 0.5), (-0.5, 1.5), (-0.5, 1.5),
            # Indice, Medio, Anulare, Mignolo (4 DoF x 4: MCP_FE, MCP_AA, PIP_FE, DIP_FE)
            (-0.2, 1.5), (-0.3, 0.3), (0.0, 1.7), (0.0, 1.5), # Indice
            (-0.2, 1.5), (-0.2, 0.2), (0.0, 1.7), (0.0, 1.5), # Medio
            (-0.2, 1.5), (-0.2, 0.2), (0.0, 1.7), (0.0, 1.5), # Anulare
            (-0.2, 1.5), (-0.3, 0.3), (0.0, 1.7), (0.0, 1.5)  # Mignolo
        ]
        
        # Stato corrente per il "warm start" (inizia da zero)
        self.q_current = np.zeros(self.num_dofs)

    def forward_kinematics(self, q):
        """ Calcola la posizione 3D teorica dei 21 landmark dati i 24 angoli q """
        LMs = np.zeros((21, 3))
        
        # 0. Polso (Origine + Rotazione Globale del polso)
        R_wrist = Rx(q[0]) @ Ry(q[1]) @ Rz_corr(q[2])
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
            LMs[lm_idx[0]] = R_wrist @ v_meta
            
            # Articolazione MCP (Rotazione su 2 assi: Flex/Ext e Abd/Add)
            R_mcp = R_wrist @ Rx(q[q_idx]) @ Ry(q[q_idx+1])
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
        LMs[1] = R_wrist @ v_meta_t
        
        R_cmc = R_wrist @ Rx(q[3]) @ Ry(q[4])
        v_prox_t = np.array([0, self.calib['lengths']['Thumb_Proximal'], 0])
        LMs[2] = LMs[1] + (R_cmc @ v_prox_t)
        
        R_mcp_t = R_cmc @ Rx(q[5]) @ Ry(q[6])
        v_inter_t = np.array([0, self.calib['lengths']['Thumb_Distal'], 0]) # Uso Distal per coprire il segmento
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
        
        error = np.sum(weights * np.linalg.norm(estimated - target_landmarks, axis=1)**2)
        return error

    def solve_frame(self, target_landmarks):
        # Trasla i target in modo che il polso (LM 0) sia esattamente nell'origine (0,0,0)
        wrist_pos = np.copy(target_landmarks[0])
        centered_targets = target_landmarks - wrist_pos
        
        result = minimize(
            fun=self.objective_function,
            x0=self.q_current,             
            args=(centered_targets,),
            method='SLSQP',
            bounds=self.bounds,
            options={'ftol': 1e-5, 'maxiter': 100, 'disp': False}
        )
        
        if result.success:
            self.q_current = result.x
        return self.q_current

# --- 3. PIPELINE AUTOMATIZZATA ---
def process_full_kinematics(csv_path):
    print("Avvio Pipeline IKA 24-DoF...")
    df = pd.read_csv(csv_path)
    
    # 3A. FASE DI CALIBRAZIONE (Primi 5 secondi)
    # Assumiamo che Timestamp_LSL sia in secondi. Troviamo l'indice dei primi 5s.
    t0 = df['Timestamp_LSL'].iloc[0]
    calib_df = df[df['Timestamp_LSL'] <= t0 + 5.0]
    
    bones = {
        'Thumb_Proximal': (2, 3), 'Thumb_Distal': (3, 4),
        'Index_Proximal': (5, 6), 'Index_Intermediate': (6, 7), 'Index_Distal': (7, 8),
        'Middle_Proximal': (9, 10), 'Middle_Intermediate': (10, 11), 'Middle_Distal': (11, 12),
        'Ring_Proximal': (13, 14), 'Ring_Intermediate': (14, 15), 'Ring_Distal': (15, 16),
        'Pinky_Proximal': (17, 18), 'Pinky_Intermediate': (18, 19), 'Pinky_Distal': (19, 20)
    }
    
    calib_data = {'lengths': {}, 'meta_vectors': {}}
    
    print(f"Calibrazione su {len(calib_df)} frame...")
    # Calcolo lunghezze falangi
    for bone_name, (idx1, idx2) in bones.items():
        lengths = []
        for _, row in calib_df.iterrows():
            p1 = np.array([row[f'LM_{idx1}_X'], row[f'LM_{idx1}_Y'], row[f'LM_{idx1}_Z']])
            p2 = np.array([row[f'LM_{idx2}_X'], row[f'LM_{idx2}_Y'], row[f'LM_{idx2}_Z']])
            lengths.append(np.linalg.norm(p2 - p1))
        calib_data['lengths'][bone_name] = np.mean(lengths)

    # Calcolo vettori metacarpi (dal polso LM_0 alla base delle dita) centrati
    meta_bases = {'Thumb': 1, 'Index': 5, 'Middle': 9, 'Ring': 13, 'Pinky': 17}
    for finger, idx in meta_bases.items():
        vecs = []
        for _, row in calib_df.iterrows():
            wrist = np.array([row['LM_0_X'], row['LM_0_Y'], row['LM_0_Z']])
            base = np.array([row[f'LM_{idx}_X'], row[f'LM_{idx}_Y'], row[f'LM_{idx}_Z']])
            vecs.append(base - wrist)
        calib_data['meta_vectors'][finger] = np.mean(vecs, axis=0)
        
    print("Calibrazione completata. Avvio Ottimizzazione SLSQP...")

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
    output_name = csv_path.replace('.csv', '_IKA_24DoF.csv')
    angles_df.to_csv(output_name, index=False)
    print(f"Dati salvati in: {output_name}")

if __name__ == "__main__":
    FILE_INPUT = "NP_project/recordings/trial_5_Kinematics.csv"
    process_full_kinematics(FILE_INPUT)