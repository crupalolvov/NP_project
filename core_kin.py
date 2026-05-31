import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
from scipy.signal import savgol_filter
import torch
import matplotlib.pyplot as plt

def detect_coordinate_format(df):
    """
    Rileva automaticamente se il dataframe contiene world_landmarks (metri) o normalizzati (pixel).
    I world_landmarks hanno il polso all'origine (0,0,0) e assumono valori negativi.
    """
    min_x = df[[f'LM_{i}_X' for i in range(21)]].min().min()
    min_y = df[[f'LM_{i}_Y' for i in range(21)]].min().min()
    if min_x < -0.01 or min_y < -0.01 or df['LM_0_X'].abs().max() < 1e-4:
        return True # Metrico
    return False # Normalizzato

# --- Helper function for Rodrigues' rotation ---
def rotate_vector_by_rodrigues(v, k, theta):
    """
    Rotates vector v around axis k by angle theta using Rodrigues' rotation formula.
    v: vector to rotate
    k: rotation axis (unit vector)
    theta: rotation angle (radians)
    """
    # Ensure k is a unit vector, add epsilon for safety against division by zero if k is near zero
    norm_k = np.linalg.norm(k)
    k_unit = k / (norm_k + 1e-8) if norm_k > 1e-8 else np.array([0.0, 0.0, 1.0]) # Fallback to Z-axis if k is zero
    v_rot = v * np.cos(theta) + np.cross(k_unit, v) * np.sin(theta) + k_unit * np.dot(k_unit, v) * (1 - np.cos(theta))
    return v_rot

#Aggiungo limitazione delle distanze dei punti fra loro, operando dirrettamente sui dati originali!
def preprocess_mediapipe_data(csv_path, anatomy, window_length=15, polyorder=3):
    """
    Pulisce i dati di MediaPipe eliminando outlier temporali 
    e forzando la rigidità ossea SUL CORRETTO PIANO DI ROTAZIONE con limiti RoM.
    """
    # Dizionari estratti dall'anatomia (Stillfried et al.)
    min_flexion_bounds = {
        'Thumb_IP': -0.26,   
        'Index_PIP': -0.09,  
        'Index_DIP': -0.17,  
        'Middle_PIP': -0.09, 
        'Middle_DIP': -0.17, 
        'Ring_PIP': -0.09,   
        'Ring_DIP': -0.17,   
        'Pinky_PIP': -0.09,  
        'Pinky_DIP': -0.17,  
    }

    max_flexion_bounds = {
        'Thumb_IP': 1.54,    
        'Index_PIP': 2.23,   
        'Index_DIP': 1.79,   
        'Middle_PIP': 2.25,  
        'Middle_DIP': 1.96,  
        'Ring_PIP': 2.27,    
        'Ring_DIP': 1.66,    
        'Pinky_PIP': 2.18,   
        'Pinky_DIP': 1.64,   
    }

    print("--- AVVIO PRE-PROCESSING CINEMATICO (PLANARE RIGOROSO CON HARD-CLAMP) ---")
    df = pd.read_csv(csv_path)
    num_frames = len(df)
    
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    is_metric = detect_coordinate_format(df)
    if is_metric:
        SCALE_X = SCALE_Y = SCALE_Z = 1000.0 # Convertiamo i metri in millimetri
        print("--- RILEVATI DATI IN METRI (World Landmarks) -> Scalo uniformemente in millimetri ---")
    else:
        SCALE_X, SCALE_Y, SCALE_Z = 640.0, 480.0, 640.0
        print("--- RILEVATI DATI IN PIXEL (Normalizzati) -> Scalo a 640x480 ---")
    
    raw_lms = np.zeros((num_frames, 21, 3))
    for f in range(num_frames):
        row = df.iloc[f]
        for i in range(21):
            raw_lms[f, i] = [
                row[f'LM_{i}_X'] * SCALE_X, 
                row[f'LM_{i}_Y'] * SCALE_Y, 
                row[f'LM_{i}_Z'] * SCALE_Z
            ]
            
    print(f"1. Filtraggio temporale (Savitzky-Golay)...")
    smoothed_lms = np.zeros_like(raw_lms)
    for i in range(21):
        for axis in range(3):
            smoothed_lms[:, i, axis] = savgol_filter(raw_lms[:, i, axis], window_length, polyorder)
            
    print("2. Proiezione su scheletro rigido e cerniere anatomiche...")
    fixed_lms = np.zeros_like(smoothed_lms)
    
    finger_defs = {
        'Thumb':  ([1, 2, 3, 4],    ['Thumb_Proximal', 'Thumb_Intermediate', 'Thumb_Distal']),
        'Index':  ([5, 6, 7, 8],    ['Index_Proximal', 'Index_Intermediate', 'Index_Distal']),
        'Middle': ([9, 10, 11, 12], ['Middle_Proximal', 'Middle_Intermediate', 'Middle_Distal']),
        'Ring':   ([13, 14, 15, 16],['Ring_Proximal', 'Ring_Intermediate', 'Ring_Distal']),
        'Pinky':  ([17, 18, 19, 20],['Pinky_Proximal', 'Pinky_Intermediate', 'Pinky_Distal'])
    }
    
    
    for f in range(num_frames):
        fixed_lms[f, 0] = smoothed_lms[f, 0]
        # 1. Ricalcoliamo il sistema di riferimento del palmo DINAMICAMENTE
        v_idx_f = smoothed_lms[f, 5] - smoothed_lms[f, 0]
        v_mid_f = smoothed_lms[f, 9] - smoothed_lms[f, 0]
        v_pnk_f = smoothed_lms[f, 17] - smoothed_lms[f, 0]
        
        X_palm_f = v_idx_f - v_pnk_f
        X_palm_f /= (np.linalg.norm(X_palm_f) + 1e-8)
        Z_palm_f = np.cross(X_palm_f, v_mid_f) 
        Z_palm_f /= (np.linalg.norm(Z_palm_f) + 1e-8)

        # --- LOGICA POLLICE DINAMICA ---
        thumb_lms = smoothed_lms[f, 1:5] 
        # Definizione dell'asse flessione del Pollice (X_thumb)
        # Il pollice ruota su un asse ortogonale al piano Z_palm e alla direzione MCP-IP
        v_mcp_ip = thumb_lms[2] - thumb_lms[1]
        X_thumb = np.cross(Z_palm_f, v_mcp_ip)
        X_thumb /= (np.linalg.norm(X_thumb) + 1e-8)
        
        for finger, (lms_idx, bone_names) in finger_defs.items():
            # 2. Ricalcoliamo l'asse della cerniera base per questo specifico frame
            Y_local_f = smoothed_lms[f, lms_idx[0]] - fixed_lms[f, 0]
            Y_local_f /= (np.linalg.norm(Y_local_f) + 1e-8)
            
            if finger == 'Thumb':
                X_local = Z_palm_f.copy() # Pollice: l'asse di flessione è la normale al palmo
                # Ortogonalizziamo per allinearci perfettamente a R_splay di CleanHandFK
                Z_local_t = np.cross(X_local, Y_local_f)
                Z_local_t /= (np.linalg.norm(Z_local_t) + 1e-8)
                X_local = np.cross(Y_local_f, Z_local_t)
                X_local /= (np.linalg.norm(X_local) + 1e-8)
            else:
                X_local = np.cross(Y_local_f, Z_palm_f)
                if np.linalg.norm(X_local) < 1e-8:
                    X_local = X_palm_f.copy()
                X_local /= np.linalg.norm(X_local)
                
            # -- 1. SISTEMIAMO IL METACARPO --
            mcp_idx = lms_idx[0]
            dir_meta = smoothed_lms[f, mcp_idx] - fixed_lms[f, 0]
            norm_meta = np.linalg.norm(dir_meta)
            if norm_meta > 1e-6: dir_meta /= norm_meta
            
            len_meta = np.linalg.norm(anatomy['meta_vectors'][finger])
            fixed_lms[f, mcp_idx] = fixed_lms[f, 0] + (dir_meta * len_meta)

            # -- 2. SISTEMIAMO LE FALANGI (CON VINCOLI DoF DINAMICI) --
            parent_idx = mcp_idx
            current_hinge_axis = X_local 
            prev_bone_dir_normalized = dir_meta.copy() 

            for i, child_idx in enumerate(lms_idx[1:]):
                dir_bone = smoothed_lms[f, child_idx] - smoothed_lms[f, parent_idx] 

                # IDENTIFICHIAMO I GIUNTI A 1-DoF (Cerniere pure - F-E)
                is_1dof_hinge = (finger != 'Thumb' and i > 0) or (finger == 'Thumb' and i == 2)
                
                # --- FASE 1: APPLICAZIONE DEL VINCOLO ---
                if is_1dof_hinge:
                    # Rimuoviamo la deviazione laterale usando l'asse del giunto ATTUALE
                    lateral_component = np.dot(dir_bone, current_hinge_axis)
                    dir_bone = dir_bone - (lateral_component * current_hinge_axis)
                
                # --- FASE 2: NORMALIZZAZIONE DEL VETTORE PULITO ---
                norm_bone = np.linalg.norm(dir_bone) 
                if norm_bone > 1e-6: 
                    dir_bone /= norm_bone
                else: 
                    print(f"ATTENZIONE: dir_bone nullo o quasi nullo al frame {f}, dito {finger}, osso {bone_names[i]}. Usato fallback [0, 1, 0].")
                    dir_bone = np.array([0.0, 1.0, 0.0]) # Fallback

                # --- FASE 2.5: ENFORCEMENT LIMITI DI FLESSIONE (per giunti a 1-DoF) ---
                # --- CLAMPING BIOMECCANICO GLOBALE ---
                if is_1dof_hinge:
                    joint_name_for_bounds = None
                    if finger == 'Thumb' and i == 2: 
                        joint_name_for_bounds = 'Thumb_IP'
                    elif finger != 'Thumb': 
                        if i == 1: joint_name_for_bounds = f'{finger}_PIP'
                        elif i == 2: joint_name_for_bounds = f'{finger}_DIP'
                    
                    if joint_name_for_bounds and joint_name_for_bounds in min_flexion_bounds:
                        min_rad = min_flexion_bounds[joint_name_for_bounds]
                        max_rad = max_flexion_bounds[joint_name_for_bounds]

                        if 'DIP' in joint_name_for_bounds:
                            max_rad *= 1.15
                            
                        flexion_y_axis = np.cross( prev_bone_dir_normalized, current_hinge_axis) # in questo ordine per avere verso Y corretto!
                        flexion_y_axis /= (np.linalg.norm(flexion_y_axis) + 1e-8)
                        
                        x_comp = np.dot(dir_bone, prev_bone_dir_normalized)
                        y_comp = np.dot(dir_bone, flexion_y_axis)
                        current_flexion_angle = np.arctan2(y_comp, x_comp)
                        
                        # Se l'angolo sfora uno dei limiti (iperextensione o iperflessione), forziamolo entro il Range of Motion
                        if current_flexion_angle < min_rad or current_flexion_angle > max_rad:
                            # Invece di clip netta, portiamo il valore verso il limite con un fattore di smorzamento (es. 0.3)
                            target = np.clip(current_flexion_angle, min_rad, max_rad)
                            current_flexion_angle = 0.7 * current_flexion_angle + 0.3 * target
                            
                            dir_bone = (prev_bone_dir_normalized * np.cos(current_flexion_angle)) + (flexion_y_axis * np.sin(current_flexion_angle))
                            dir_bone /= np.linalg.norm(dir_bone)
                                
                # --- FASE 3: TRASPORTO DELL'ASSE ---
                # Utilizziamo Rodrigues per TUTTE le dita, incluso il pollice!
                # Questo propaga l'asse esattamente come le matrici in CleanHandFK.
                rotation_axis = np.cross(prev_bone_dir_normalized, dir_bone)
                norm_rot_axis = np.linalg.norm(rotation_axis)
                if norm_rot_axis > 1e-8:
                    rotation_axis /= norm_rot_axis
                    dot_product = np.clip(np.dot(prev_bone_dir_normalized, dir_bone), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    current_hinge_axis = rotate_vector_by_rodrigues(current_hinge_axis, rotation_axis, angle)

                # --- FASE 4: RICOSTRUZIONE POSIZIONALE ---
                len_bone = anatomy['lengths'][bone_names[i]]
                fixed_lms[f, child_idx] = fixed_lms[f, parent_idx] + dir_bone * len_bone
                
                # --- FASE 5: AGGIORNAMENTO VARIABILI DI STATO ---
                parent_idx = child_idx 
                prev_bone_dir_normalized = dir_bone.copy() # È fondamentale salvare il vettore PULITO
                
    print("Pre-processing completato. Dati pronti per IKA.")
    
    # --- Salvataggio CSV ---
    preprocessed_data = []
    for f in range(num_frames):
        row_dict = {}
        if 'Timestamp_LSL' in df.columns:
            row_dict['Timestamp_LSL'] = df.iloc[f]['Timestamp_LSL']
        for i in range(21):
            row_dict[f'LM_{i}_X'] = fixed_lms[f, i, 0] / SCALE_X
            row_dict[f'LM_{i}_Y'] = fixed_lms[f, i, 1] / SCALE_Y
            row_dict[f'LM_{i}_Z'] = fixed_lms[f, i, 2] / SCALE_Z
        preprocessed_data.append(row_dict)
        
    df_preprocessed = pd.DataFrame(preprocessed_data)
    out_csv = csv_path.replace('.csv', '_preprocessed.csv')
    df_preprocessed.to_csv(out_csv, index=False)
    print(f"Dati preprocessati normalizzati salvati in: {os.path.basename(out_csv)}")
    
    return fixed_lms    


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
    
    # Dimensioni dell'immagine usata per la cattura per convertire in pixel
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    is_metric = detect_coordinate_format(df)
    if is_metric:
        SCALE_X = SCALE_Y = SCALE_Z = 1000.0
    else:
        SCALE_X, SCALE_Y, SCALE_Z = 640.0, 480.0, 640.0

    for i in range(num_frames):
        row = df.iloc[i]
        for lm in range(21):
            lms_history[i, lm] = [
                row[f'LM_{lm}_X'] * SCALE_X, 
                row[f'LM_{lm}_Y'] * SCALE_Y, 
                row[f'LM_{lm}_Z'] * SCALE_Z
            ]
            
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
            
            R_mcp = R_wrist @ R_base @ Rx(q[q_idx]) @ Rz(q[q_idx+1])
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
        
        R_cmc = R_wrist @ R_base_t @ Rx(q[3]) @ Rz(q[4]) 
        v_prox_t = np.array([0, self.anatomy['lengths']['Thumb_Proximal'], 0])
        LMs[2] = LMs[1] + (R_cmc @ v_prox_t)
        
        R_mcp_t = R_cmc @ Rx(q[5]) @ Rz(q[6])
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
        
        # 2. AUTO-SCALING
        target_dist = np.linalg.norm(centered_targets[9])
        model_dist = np.linalg.norm(self.fk.anatomy['meta_vectors']['Middle'])
        if target_dist > 1e-6:
            centered_targets = centered_targets / (target_dist / model_dist)
            
        # 3. FILTRO STRUTTURALE INTERNO
        rigid_targets = self.enforce_rigid_skeleton(centered_targets)
            
        q_temp = np.copy(self.q_current)
        lambda_reg = 100.0 

        # =================================================================
        # FASE 1: POLSO (Soglia Rescue in PIXEL)
        # =================================================================
        def wrist_objective(q_wrist):
            q_test = np.copy(q_temp)
            q_test[0:3] = q_wrist
            pred = self.fk.forward(q_test)
            palm_lms = [1, 5, 9, 13, 17]
            diff = pred[palm_lms] - rigid_targets[palm_lms]
            mse = np.sum(np.linalg.norm(diff, axis=1)**2)
            reg = lambda_reg * np.mean((q_wrist - self.q_current[0:3])**2)
            return mse + reg

        res_wrist = minimize(
            fun=wrist_objective, x0=q_temp[0:3], method='SLSQP', bounds=self.wrist_bounds,
            options={'ftol': 1e-5, 'maxiter': 50}
        )
        
        # NUOVA SOGLIA: ~400 equivale a circa 9 pixel di errore medio per nocca.
        if res_wrist.fun > 400.0:
            res_wrist_rescue = minimize(
                fun=wrist_objective, x0=np.zeros(3), method='SLSQP', bounds=self.wrist_bounds,
                options={'ftol': 1e-5, 'maxiter': 50}
            )
            if res_wrist_rescue.fun < res_wrist.fun:
                res_wrist = res_wrist_rescue

        q_temp[0:3] = res_wrist.x 

        # =================================================================
        # FASE 2 & 3: DITA (Con Rescue Indipendente)
        # =================================================================
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
                
                diff = pred[lms_idx] - rigid_targets[lms_idx]
                mse = np.sum(weights * np.linalg.norm(diff, axis=1)**2)
                
                reg = lambda_reg * np.mean((q_finger - self.q_current[start_idx:end_idx])**2)
                
                bio_constraint = 0.0
                if finger != 'Thumb':
                    q_pip = q_finger[2]
                    q_dip = q_finger[3]
                    bio_constraint = 2000.0 * (q_dip - (0.66 * q_pip))**2
                
                return mse + reg + bio_constraint

            res_finger = minimize(
                fun=finger_objective, x0=q_temp[start_idx:end_idx], method='SLSQP',
                bounds=bounds, options={'ftol': 1e-5, 'maxiter': 50}
            )
            
            # NUOVO RESCUE PER LE DITA: ~300 equivale a circa 8-9 pixel di errore per falange
            if res_finger.fun > 300.0:
                res_finger_rescue = minimize(
                    fun=finger_objective, x0=np.zeros(len(bounds)), method='SLSQP',
                    bounds=bounds, options={'ftol': 1e-5, 'maxiter': 50}
                )
                if res_finger_rescue.fun < res_finger.fun:
                    res_finger = res_finger_rescue

            q_temp[start_idx:end_idx] = res_finger.x 

        self.q_current = q_temp
        return self.q_current


# --- 4. SOLUTORE IK ROBOTICO (Damped Least Squares Jacobian) ---
class JacobianIKA:
    def __init__(self, fk_model):
        self.fk = fk_model
        self.num_dofs = 24
        self.q_current = np.zeros(self.num_dofs)
        self.lambda_dls = 0.5  # Fattore di smorzamento: impedisce alla matematica di esplodere vicino alle singolarità
        
        # Pesi Biomeccanici: Diamo priorità assoluta al posizionamento del palmo, poi alle punte.
        # Le articolazioni intermedie si adatteranno fluidamente lungo la catena.
        weights = np.ones(21)
        palm_idx = [0, 1, 5, 9, 13, 17]
        tips_idx = [4, 8, 12, 16, 20]
        weights[palm_idx] = 10.0  # Il polso è la fondazione, deve combaciare perfettamente
        weights[tips_idx] = 3.0   # Le punte guidano la direzione
        self.W = np.diag(np.repeat(weights, 3)) # Creiamo la matrice diagonale 63x63

    def get_jacobian(self, q):
        # Calcolo del Jacobiano Numerico (Derivate parziali per ogni grado di libertà)
        delta = 1e-4
        J = np.zeros((63, self.num_dofs))
        f0 = self.fk.forward(q).flatten()
        
        for i in range(self.num_dofs):
            q_step = np.copy(q)
            q_step[i] += delta
            f_step = self.fk.forward(q_step).flatten()
            J[:, i] = (f_step - f0) / delta
            
        return J, f0

    def solve(self, target_lms, iterations=20):
        # 1. Centratura
        wrist_pos = target_lms[0]
        centered_targets = target_lms - wrist_pos
        
        # 2. Auto-scaling sulle proporzioni fisse
        target_dist = np.linalg.norm(centered_targets[9])
        model_dist = np.linalg.norm(self.fk.anatomy['meta_vectors']['Middle'])
        if target_dist > 1e-6:
            centered_targets = centered_targets / (target_dist / model_dist)
            
        target_flat = centered_targets.flatten()
        q = np.copy(self.q_current)
        
        # 3. Ottimizzazione DLS (Levenberg-Marquardt Approach)
        for step in range(iterations):
            # Calcoliamo dove siamo e come muoverci (Jacobiano)
            J, current_pos_flat = self.get_jacobian(q)
            error = target_flat - current_pos_flat
            
            # Se l'errore è irrilevante, interrompiamo il calcolo per risparmiare tempo
            if np.mean(np.abs(error)) < 0.5: 
                break
            
            # Applichiamo i pesi biomeccanici
            e_weighted = self.W @ error
            J_weighted = self.W @ J
            
            # Calcolo della pseudo-inversa smorzata: (J^T * J + lambda^2 * I)^-1 * J^T
            J_T = J_weighted.T
            H = J_T @ J_weighted + (self.lambda_dls**2) * np.eye(self.num_dofs)
            
            # La magia della robotica: delta_q contiene i radianti esatti per abbattere l'errore
            delta_q = np.linalg.inv(H) @ J_T @ e_weighted
            
            # Aggiorniamo gli angoli della mano (con un passo di 0.8 per evitare oscillazioni)
            q = q + 0.8 * delta_q 
            
            # Reset di sicurezza: se al primo iteratore la mano è capovolta (errore enorme),
            # azzeriamo il polso per permettere al Jacobiano di trovare la discesa giusta.
            if step == 0 and np.mean(np.abs(e_weighted)) > 150.0:
                q[0:3] = 0.0 

        self.q_current = q
        return q
# ==========================================
# TEST 2: TRACKING CONTINUO SUI DATI REALI
# ==========================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KIN_FILE = os.path.join(BASE_DIR, "recordings", "trial_1_Kinematics.csv") 
    
    START_CALIB = 1530
    END_CALIB = 1700
    
    CALIB_FILE = os.path.join(BASE_DIR, "hand_calibration.pt")
    
    # 1. Estraiamo o carichiamo l'anatomia
    if os.path.exists(CALIB_FILE):
        print(f"Caricamento calibrazione anatomica esistente da: {os.path.basename(CALIB_FILE)}")
        anatomy = torch.load(CALIB_FILE, weights_only=False)
    else:
        print("Calcolo della calibrazione anatomica dal file CSV...")
        anatomy = extract_anatomy_from_csv(KIN_FILE, START_CALIB, END_CALIB)
        torch.save(anatomy, CALIB_FILE)
        print(f"Calibrazione anatomica salvata in: {os.path.basename(CALIB_FILE)}")
    
    # 2. PULIZIA TOTALE: passiamo il CSV e riceviamo un array perfetto in pixel
    cleaned_landmarks_array = preprocess_mediapipe_data(KIN_FILE, anatomy)
    df_orig = pd.read_csv(KIN_FILE)
    
    fk = CleanHandFK(anatomy)
    #ika = CleanHandIKA(fk)
    ika = JacobianIKA(fk) # Proviamo anche il solutore Jacobiano su dati puliti, per vedere se riesce a migliorare ulteriormente l'errore (dovrebbe essere più fluido ma con errore medio simile)
    
    num_frames = len(cleaned_landmarks_array)
    print(f"\n--- AVVIO TRACKING CONTINUO SUI DATI REALI (Dati Puliti) ---")
    print(f"Elaborazione di {num_frames} frame. Potrebbe richiedere qualche minuto...")
    
    mean_errors = np.zeros(num_frames)
    max_errors = np.zeros(num_frames)
    
    ika_pred_data = []
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    
    is_metric = detect_coordinate_format(df_orig)
    if is_metric:
        SCALE_X = SCALE_Y = SCALE_Z = 1000.0
    else:
        SCALE_X, SCALE_Y, SCALE_Z = 640.0, 480.0, 640.0
    
    # Inseguiamo la mano frame per frame processandoli tutti
    for f in range(num_frames):
        # PRENDIAMO I DATI DAL VETTORE PULITO E PROIETTATO!
        target_lms = cleaned_landmarks_array[f].copy()
        
        wrist_pos = target_lms[0].copy()
        target_lms = target_lms - wrist_pos # Centriamo il polso
        
        # L'Auto-Scaling esterno è stato rimosso perché i dati puliti 
        # garantiscono già le perfette lunghezze metriche e le corrette proporzioni in pixel.
            
        # Risolviamo!
        q_sol = ika.solve(target_lms)
        
        # Log di controllo
        pred_lms = fk.forward(q_sol)
        error_px = np.linalg.norm(target_lms - pred_lms, axis=1)
        
        # Riportiamo la mano nella posizione originale nello schermo per l'animazione
        pred_lms_abs = pred_lms + wrist_pos
        
        # Prepariamo la riga per il nuovo CSV convertendo di nuovo in scala normalizzata [0-1]
        row_dict = {}
        if 'Timestamp_LSL' in df_orig.columns:
            row_dict['Timestamp_LSL'] = df_orig.iloc[f]['Timestamp_LSL']
        for i in range(21):
            row_dict[f'LM_{i}_X'] = pred_lms_abs[i, 0] / SCALE_X
            row_dict[f'LM_{i}_Y'] = pred_lms_abs[i, 1] / SCALE_Y
            row_dict[f'LM_{i}_Z'] = pred_lms_abs[i, 2] / SCALE_Z
        ika_pred_data.append(row_dict)
        
        mean_errors[f] = np.mean(error_px)
        max_errors[f] = np.max(error_px)
        
        if f % 100 == 0:
            print(f"Frame {f:04d}/{num_frames} | Errore Medio: {mean_errors[f]:.2f} px | Errore Max: {max_errors[f]:.2f} px")
            
    print(f"\nElaborazione completata. Errore Medio Globale: {np.mean(mean_errors):.2f} px")

    # Salvataggio del nuovo CSV
    df_ika_pred = pd.DataFrame(ika_pred_data)
    out_csv = KIN_FILE.replace('.csv', '_IKA_predicted_lms.csv')
    df_ika_pred.to_csv(out_csv, index=False)
    print(f"Predizioni IKA salvate per l'animazione in: {os.path.basename(out_csv)}\n")

    # Plot dell'errore nel tempo
    plt.figure(figsize=(12, 6))
    frames = np.arange(num_frames)
    plt.plot(frames, mean_errors, label='Errore Medio (px)', color='blue')
    plt.plot(frames, max_errors, label='Errore Massimo per frame (px)', color='red', alpha=0.3)
    plt.axhline(y=np.mean(mean_errors), color='green', linestyle='--', label=f'Media Globale ({np.mean(mean_errors):.2f} px)')
    
    plt.title(f'Errore di Ricostruzione IKA nel tempo (Dati Pre-processati)\nFile: {os.path.basename(KIN_FILE)}')
    plt.xlabel('Frame')
    plt.ylabel('Errore (pixel)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


# FUNZIONE FINALE PER FEATURE EXTRACTION
def process_full_kinematics_core(csv_path):
    print("Avvio Pipeline Preprocessing + IKA (Jacobian)...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CALIB_FILE = os.path.join(BASE_DIR, "hand_calibration.pt")
    
    if os.path.exists(CALIB_FILE):
        print(f"Caricamento calibrazione anatomica da: {os.path.basename(CALIB_FILE)}")
        anatomy = torch.load(CALIB_FILE, weights_only=False)
    else:
        print(f"Errore: File di calibrazione '{CALIB_FILE}' non trovato!")
        return None
    
    # 1. Preprocessing (Pulizia e Rigidità)
    cleaned_landmarks_array = preprocess_mediapipe_data(csv_path, anatomy)
    df_orig = pd.read_csv(csv_path)
    
    # 2. IKA
    fk = CleanHandFK(anatomy)
    ika = JacobianIKA(fk)
    
    num_frames = len(cleaned_landmarks_array)
    angles_data = []
    
    print(f"\n--- AVVIO TRACKING IKA SUI DATI PREPROCESSATI ---")
    
    for f in range(num_frames):
        target_lms = cleaned_landmarks_array[f].copy()
        wrist_pos = target_lms[0].copy()
        target_lms = target_lms - wrist_pos # Centriamo il polso
        
        q_sol = ika.solve(target_lms)
        
        row_dict = {}
        if 'Timestamp_LSL' in df_orig.columns:
            row_dict['Timestamp_LSL'] = df_orig.iloc[f]['Timestamp_LSL']
            
        for i in range(24):
            row_dict[f'DoF_{i}'] = np.degrees(q_sol[i]) # Feature extractor si aspetta gradi
            
        angles_data.append(row_dict)
        
        if f % 100 == 0 and f > 0:
            print(f"Processati {f:04d}/{num_frames} frame per IKA...")
            
    df_angles = pd.DataFrame(angles_data)
    out_csv = csv_path.replace('.csv', '_core_IKA_24DoF.csv')
    df_angles.to_csv(out_csv, index=False)
    print(f"Dati IKA (Gradi) salvati in: {os.path.basename(out_csv)}\n")
    return out_csv