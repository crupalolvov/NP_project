import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import torch

# --- 1. MATRICI DI ROTAZIONE RIGOROSE ---
def Rx(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

# --- 2. AUTO-CALIBRAZIONE DA MEDIAPIPE ---
def extract_anatomy_from_csv(csv_path, start_frame, end_frame):
    df = pd.read_csv(csv_path).iloc[start_frame:end_frame]
    num_frames = len(df)
    
    bone_links = {
        'Thumb_Proximal': (1, 2), 'Thumb_Intermediate': (2, 3), 'Thumb_Distal': (3, 4),
        'Index_Proximal': (5, 6), 'Index_Intermediate': (6, 7), 'Index_Distal': (7, 8),
        'Middle_Proximal': (9, 10), 'Middle_Intermediate': (10, 11), 'Middle_Distal': (11, 12),
        'Ring_Proximal': (13, 14), 'Ring_Intermediate': (14, 15), 'Ring_Distal': (15, 16),
        'Pinky_Proximal': (17, 18), 'Pinky_Intermediate': (18, 19), 'Pinky_Distal': (19, 20)
    }
    
    anatomy = {'lengths': {}, 'meta_vectors': {}}
    lms_history = np.zeros((num_frames, 21, 3))
    
    for i in range(num_frames):
        row = df.iloc[i]
        for lm in range(21):
            lms_history[i, lm] = [row[f'LM_{lm}_X'], row[f'LM_{lm}_Y'], row[f'LM_{lm}_Z']]
            
    lms_mean = np.mean(lms_history, axis=0)
    wrist_pos = lms_mean[0]
    lms_mean = lms_mean - wrist_pos # Polso in (0,0,0)
    
    anatomy['meta_vectors']['Thumb'] = lms_mean[1]
    anatomy['meta_vectors']['Index'] = lms_mean[5]
    anatomy['meta_vectors']['Middle'] = lms_mean[9]
    anatomy['meta_vectors']['Ring'] = lms_mean[13]
    anatomy['meta_vectors']['Pinky'] = lms_mean[17]
    
    for bone_name, (idx1, idx2) in bone_links.items():
        anatomy['lengths'][bone_name] = np.linalg.norm(lms_mean[idx2] - lms_mean[idx1])
        
    return anatomy

# --- 3. FORWARD KINEMATICS CON BASE ANATOMICA ROBUSTA ---
class CleanHandFK:
    def __init__(self, anatomy):
        self.anatomy = anatomy
        self.R_splay = {}
        
        # 1. Costruzione di un piano palmare rigoroso e indipendente dalla posa delle dita
        v_idx = self.anatomy['meta_vectors']['Index']
        v_pnk = self.anatomy['meta_vectors']['Pinky']
        v_mid = self.anatomy['meta_vectors']['Middle']
        
        # Asse X globale del palmo (direzione Mignolo -> Indice)
        X_palm = v_idx - v_pnk
        X_palm /= (np.linalg.norm(X_palm) + 1e-8)
        
        # Normale del palmo (Z) perpendicolare al piano X-Medio
        Z_palm = np.cross(X_palm, v_mid)
        Z_palm /= (np.linalg.norm(Z_palm) + 1e-8)
        
        # Asse Y globale del palmo (direzione di estensione)
        Y_palm = np.cross(Z_palm, X_palm)
        Y_palm /= (np.linalg.norm(Y_palm) + 1e-8)
        
        for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            Y_local = self.anatomy['meta_vectors'][finger].copy()
            Y_local /= (np.linalg.norm(Y_local) + 1e-8)
            
            if finger == 'Thumb':
                # Pollice: l'asse di flessione (X) entra nel palmo
                X_local = Z_palm.copy()
                Z_local = np.cross(X_local, Y_local)
                Z_local /= (np.linalg.norm(Z_local) + 1e-8)
                X_local = np.cross(Y_local, Z_local) 
            else:
                # Dita standard
                X_local = np.cross(Y_local, Z_palm)
                if np.linalg.norm(X_local) < 1e-8:
                    X_local = X_palm.copy() # Fallback sul piano del palmo
                X_local /= np.linalg.norm(X_local)
                Z_local = np.cross(X_local, Y_local)
                Z_local /= np.linalg.norm(Z_local)
                
            self.R_splay[finger] = np.column_stack([X_local, Y_local, Z_local])

    def forward(self, q):
        LMs = np.zeros((21, 3))
        # FIX POLSO: Ordine Z-Y-X (Yaw, Pitch, Roll)
        R_wrist = Rz(q[2]) @ Ry(q[1]) @ Rx(q[0])
        LMs[0] = [0, 0, 0]
        
        finger_defs = {
            'Index': (5, 6, 7, 8), 'Middle': (9, 10, 11, 12),
            'Ring': (13, 14, 15, 16), 'Pinky': (17, 18, 19, 20)
        }
        
        for idx, (f_name, lm_idx) in enumerate(finger_defs.items()):
            q_idx = 8 + idx * 4
            v_meta = self.anatomy['meta_vectors'][f_name]
            LMs[lm_idx[0]] = R_wrist @ v_meta
            
            R_base = self.R_splay[f_name]
            
            R_mcp = R_wrist @ R_base @ Rz(q[q_idx+1]) @ Rx(q[q_idx])
            v_prox = np.array([0, self.anatomy['lengths'][f_name+'_Proximal'], 0])
            LMs[lm_idx[1]] = LMs[lm_idx[0]] + (R_mcp @ v_prox)
            
            R_pip = R_mcp @ Rx(q[q_idx+2])
            v_inter = np.array([0, self.anatomy['lengths'][f_name+'_Intermediate'], 0])
            LMs[lm_idx[2]] = LMs[lm_idx[1]] + (R_pip @ v_inter)
            
            R_dip = R_pip @ Rx(q[q_idx+3])
            v_dist = np.array([0, self.anatomy['lengths'][f_name+'_Distal'], 0])
            LMs[lm_idx[3]] = LMs[lm_idx[2]] + (R_dip @ v_dist)

        # Pollice
        v_meta_t = self.anatomy['meta_vectors']['Thumb']
        LMs[1] = R_wrist @ v_meta_t
        R_base_t = self.R_splay['Thumb']
        
        R_cmc = R_wrist @ R_base_t @ Rz(q[4]) @ Rx(q[3]) 
        v_prox_t = np.array([0, self.anatomy['lengths']['Thumb_Proximal'], 0])
        LMs[2] = LMs[1] + (R_cmc @ v_prox_t)
        
        R_mcp_t = R_cmc @ Rz(q[6]) @ Rx(q[5])
        v_inter_t = np.array([0, self.anatomy['lengths']['Thumb_Intermediate'], 0])
        LMs[3] = LMs[2] + (R_mcp_t @ v_inter_t)
        
        R_ip_t = R_mcp_t @ Rx(q[7])
        v_dist_t = np.array([0, self.anatomy['lengths']['Thumb_Distal'], 0])
        LMs[4] = LMs[3] + (R_ip_t @ v_dist_t)
        
        return LMs

# --- 4. IKA CON REGOLARIZZAZIONE BILANCIATA E SOFT-CONSTRAINTS ---
class CleanHandIKA:
    def __init__(self, fk_model):
        self.fk = fk_model
        self.num_dofs = 24
        self.q_current = np.zeros(self.num_dofs)
        
        self.wrist_bounds = [(-1.5, 1.5)] * 3
        self.thumb_bounds = [(-1.0, 1.5), (-1.0, 1.0), (-0.5, 1.5), (-0.5, 0.5), (-0.5, 1.5)]
        self.finger_bounds = [(-0.5, 2.0), (-0.5, 0.5), (-0.2, 2.2), (-0.2, 1.8)]

    def enforce_rigid_skeleton(self, raw_targets):
        """
        Forza i landmark rumorosi di MediaPipe su uno scheletro rigido
        usando le lunghezze esatte della calibrazione anatomica.
        """
        fixed_lms = np.zeros((21, 3))
        fixed_lms[0] = raw_targets[0] # Il polso (origine) rimane fermo

        finger_defs = {
            'Thumb':  ([1, 2, 3, 4],    ['Thumb_Proximal', 'Thumb_Intermediate', 'Thumb_Distal']),
            'Index':  ([5, 6, 7, 8],    ['Index_Proximal', 'Index_Intermediate', 'Index_Distal']),
            'Middle': ([9, 10, 11, 12], ['Middle_Proximal', 'Middle_Intermediate', 'Middle_Distal']),
            'Ring':   ([13, 14, 15, 16],['Ring_Proximal', 'Ring_Intermediate', 'Ring_Distal']),
            'Pinky':  ([17, 18, 19, 20],['Pinky_Proximal', 'Pinky_Intermediate', 'Pinky_Distal'])
        }

        for finger, (lms_idx, bone_names) in finger_defs.items():
            # 1. Sistema il Metacarpo (Dal polso LM 0 alla nocca)
            mcp_idx = lms_idx[0]
            dir_meta = raw_targets[mcp_idx] - raw_targets[0]
            norm_meta = np.linalg.norm(dir_meta)
            if norm_meta > 1e-6:
                dir_meta /= norm_meta
            
            # Recuperiamo la lunghezza reale del metacarpo calcolata in FK
            len_meta = np.linalg.norm(self.fk.anatomy['meta_vectors'][finger])
            fixed_lms[mcp_idx] = fixed_lms[0] + dir_meta * len_meta

            # 2. Sistema le Falangi in cascata
            parent_idx = mcp_idx
            for i, child_idx in enumerate(lms_idx[1:]):
                dir_bone = raw_targets[child_idx] - raw_targets[parent_idx]
                norm_bone = np.linalg.norm(dir_bone)
                if norm_bone > 1e-6:
                    dir_bone /= norm_bone
                else:
                    dir_bone = np.array([0.0, 1.0, 0.0]) # Fallback di sicurezza
                
                # Applica la lunghezza ossea inattaccabile
                len_bone = self.fk.anatomy['lengths'][bone_names[i]]
                fixed_lms[child_idx] = fixed_lms[parent_idx] + dir_bone * len_bone
                
                parent_idx = child_idx # Propaga alla giuntura successiva
                
        return fixed_lms
    

    def solve(self, target_lms):
        # 1. Centriamo il polso
        wrist_pos = target_lms[0]
        centered_targets = target_lms - wrist_pos
        
        # 2. AUTO-SCALING (Normalizziamo la grandezza totale della mano di MediaPipe)
        target_dist = np.linalg.norm(centered_targets[9])
        model_dist = np.linalg.norm(self.fk.anatomy['meta_vectors']['Middle'])
        if target_dist > 1e-6:
            centered_targets = centered_targets / (target_dist / model_dist)
            
        # 3. FILTRO STRUTTURALE (Proiezione sullo scheletro rigido)
        # Questo elimina l'effetto "ossa di gomma" alla radice
        centered_targets = self.enforce_rigid_skeleton(centered_targets)
            
        q_temp = np.copy(self.q_current)
        
        # Riduciamo la resistenza ora che i dati sono puliti e solidi
        lambda_reg = 100.0 

        def wrist_objective(q_wrist):
            q_test = np.copy(q_temp)
            q_test[0:3] = q_wrist
            pred = self.fk.forward(q_test)
            palm_lms = [1, 5, 9, 13, 17]
            diff = (pred[palm_lms] - centered_targets[palm_lms]) * 1000.0
            mse = np.sum(np.linalg.norm(diff, axis=1)**2)
            reg = lambda_reg * np.mean((q_wrist - self.q_current[0:3])**2)
            return mse + reg

        res_wrist = minimize(
            fun=wrist_objective, x0=q_temp[0:3], method='SLSQP', bounds=self.wrist_bounds,
            options={'ftol': 1e-5, 'maxiter': 50}
        )
        q_temp[0:3] = res_wrist.x 

        finger_configs = {
            'Thumb':  (3, 8,   [2, 3, 4], self.thumb_bounds),
            'Index':  (8, 12,  [6, 7, 8], self.finger_bounds),
            'Middle': (12, 16, [10, 11, 12], self.finger_bounds),
            'Ring':   (16, 20, [14, 15, 16], self.finger_bounds),
            'Pinky':  (20, 24, [18, 19, 20], self.finger_bounds)
        }

        for finger, (start_idx, end_idx, lms_idx, bounds) in finger_configs.items():
            def finger_objective(q_finger):
                q_test = np.copy(q_temp)
                q_test[start_idx:end_idx] = q_finger
                pred = self.fk.forward(q_test)
                
                weights = np.ones(len(lms_idx))
                weights[-1] = 2.0 
                
                diff = (pred[lms_idx] - centered_targets[lms_idx]) * 1000.0
                mse = np.sum(weights * np.linalg.norm(diff, axis=1)**2)
                
                # Regolarizzazione temporale (smorzamento jitter)
                reg = lambda_reg * np.mean((q_finger - self.q_current[start_idx:end_idx])**2)
                
                # FIX IDENTIFIABILITY: Accoppiamento tendineo PIP-DIP per stabilizzare il solutore
                bio_constraint = 0.0
                if finger != 'Thumb':
                    # L'ultimo angolo del dito (DIP) deve essere circa 2/3 del precedente (PIP)
                    q_pip = q_finger[2]
                    q_dip = q_finger[3]
                    bio_constraint = 2000.0 * (q_dip - (0.66 * q_pip))**2
                
                return mse + reg + bio_constraint

            res_finger = minimize(
                fun=finger_objective, x0=q_temp[start_idx:end_idx], method='SLSQP',
                bounds=bounds, options={'ftol': 1e-5, 'maxiter': 50}
            )
            q_temp[start_idx:end_idx] = res_finger.x 

        self.q_current = q_temp
        return self.q_current
# # ==========================================
# # TEST 1: LA PROVA DI AUTO-CONSISTENZA
# # ==========================================
# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     KIN_FILE = os.path.join(BASE_DIR, "recordings", "trial_1_Kinematics.csv") 
    
#     # Inizializza l'anatomia
#     anatomy = extract_anatomy_from_csv(KIN_FILE, 540, 590)
#     fk = CleanHandFK(anatomy)
#     ika = CleanHandIKA(fk)
    
#     print("\n--- AVVIO TEST DI AUTO-CONSISTENZA (FK -> IK -> FK) ---")
    
#     # 1. Generiamo un set di angoli (q) biomeccanicamente verosimili per simulare una mano semi-chiusa
#     q_true = np.zeros(24)
#     q_true[0:3] = [0.2, -0.1, 0.0] # Polso leggermente flesso
    
#         # Flettiamo le dita in modo biologicamente coerente
#     for i in range(4): 
#         idx = 8 + (i * 4)
#         q_true[idx]   = 1.2 # MCP Flex 
#         q_true[idx+1] = 0.0 # MCP Abd
#         q_true[idx+2] = 1.0 # PIP Flex
#         q_true[idx+3] = 0.66 # DIP Flex (ESATTAMENTE IL 66% della PIP)
        
#     # 2. Generiamo i landmark PERFETTI da questi angoli tramite la nostra FK
#     lms_perfect = fk.forward(q_true)
    
#     # 3. Chiediamo all'IKA di ritrovare gli angoli partendo solo dai landmark (mano aperta come x0)
#     ika.q_current = np.zeros(24) 
#     q_estimated = ika.solve(lms_perfect)
    
#     # 4. Generiamo la mano ricostruita per calcolare l'errore metrico
#     lms_reconstructed = fk.forward(q_estimated)
    
#     error_mm = np.linalg.norm(lms_perfect - lms_reconstructed, axis=1) * 1000.0
    
#     print(f"\nRisultati Test di Auto-Consistenza:")
#     print(f"Errore Massimo: {np.max(error_mm):.4f} mm")
#     print(f"Errore Medio:   {np.mean(error_mm):.4f} mm")
    
#     if np.mean(error_mm) < 1.0:
#         print("✅ MATEMATICA PERFETTA: Il tuo modello cinematico è inattaccabile. Se l'errore sui dati veri è alto, la colpa è 100% del rumore di MediaPipe.")
#     else:
#         print("❌ FALLIMENTO MATEMATICO: Il solutore IKA non riesce a chiudere la catena nemmeno con dati perfetti.")
# ==========================================
# TEST 2: TRACKING CONTINUO SUI DATI REALI
# ==========================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KIN_FILE = os.path.join(BASE_DIR, "recordings", "trial_3_Kinematics.csv") 
    
    # 1. Inizializza l'anatomia sui frame di calibrazione estesi
    START_CALIB = 540
    END_CALIB = 590
    
    anatomy = extract_anatomy_from_csv(KIN_FILE, START_CALIB, END_CALIB)
    fk = CleanHandFK(anatomy)
    ika = CleanHandIKA(fk)
    
    TEST_FRAME = 1730 
    print(f"\n--- AVVIO TRACKING CONTINUO SUI DATI REALI (Con Auto-Scaling) ---")
    
    df = pd.read_csv(KIN_FILE)
    
    # 2. Inseguiamo la mano frame per frame partendo da fine calibrazione
    for f in range(END_CALIB, TEST_FRAME + 1):
        row = df.iloc[f]
        target_lms = np.zeros((21, 3))
        for i in range(21):
            target_lms[i] = [row[f'LM_{i}_X'], row[f'LM_{i}_Y'], row[f'LM_{i}_Z']]
            
        target_lms = target_lms - target_lms[0] # Centriamo il polso
        
        # Auto-Scaling anche per il calcolo dell'errore (rendiamo il confronto leale)
        target_dist = np.linalg.norm(target_lms[9])
        model_dist = np.linalg.norm(fk.anatomy['meta_vectors']['Middle'])
        if target_dist > 1e-6:
            target_lms = target_lms / (target_dist / model_dist)
            
        # Risolviamo sfruttando la memoria del frame precedente
        q_sol = ika.solve(target_lms)
        
        # Log di controllo
        if f % 100 == 0 or f == TEST_FRAME:
            pred_lms = fk.forward(q_sol)
            error_mm = np.linalg.norm(target_lms - pred_lms, axis=1) * 1000.0
            print(f"Frame {f:04d} | Errore Medio: {np.mean(error_mm):.2f} mm | Errore Max: {np.max(error_mm):.2f} mm")