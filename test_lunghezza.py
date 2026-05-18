import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- 1. ESTRAZIONE ANATOMIA (Il nostro "Righello") ---
def extract_anatomy_from_csv(df, start_frame, end_frame, IMG_WIDTH, IMG_HEIGHT):
    df_calib = df.iloc[start_frame:end_frame]
    num_frames = len(df_calib)
    
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
        row = df_calib.iloc[i]
        for lm in range(21):
            lms_history[i, lm] = [
                row[f'LM_{lm}_X'] * IMG_WIDTH, 
                row[f'LM_{lm}_Y'] * IMG_HEIGHT, 
                row[f'LM_{lm}_Z'] * IMG_WIDTH
            ]
            
    lms_mean = np.mean(lms_history, axis=0)
    wrist_pos = lms_mean[0]
    lms_mean = lms_mean - wrist_pos 
    
    anatomy['meta_vectors']['Thumb'] = lms_mean[1]
    anatomy['meta_vectors']['Index'] = lms_mean[5]
    anatomy['meta_vectors']['Middle'] = lms_mean[9]
    anatomy['meta_vectors']['Ring'] = lms_mean[13]
    anatomy['meta_vectors']['Pinky'] = lms_mean[17]
    
    for bone_name, (idx1, idx2) in bone_links.items():
        anatomy['lengths'][bone_name] = np.linalg.norm(lms_mean[idx2] - lms_mean[idx1])
        
    return anatomy, bone_links

# --- 2. LA NOSTRA CURA: BONE-LENGTH ENFORCEMENT ---
def enforce_rigid_skeleton_on_sequence(raw_sequence_px, anatomy):
    num_frames = raw_sequence_px.shape[0]
    fixed_lms = np.zeros_like(raw_sequence_px)

    finger_defs = {
        'Thumb':  ([1, 2, 3, 4],    ['Thumb_Proximal', 'Thumb_Intermediate', 'Thumb_Distal']),
        'Index':  ([5, 6, 7, 8],    ['Index_Proximal', 'Index_Intermediate', 'Index_Distal']),
        'Middle': ([9, 10, 11, 12], ['Middle_Proximal', 'Middle_Intermediate', 'Middle_Distal']),
        'Ring':   ([13, 14, 15, 16],['Ring_Proximal', 'Ring_Intermediate', 'Ring_Distal']),
        'Pinky':  ([17, 18, 19, 20],['Pinky_Proximal', 'Pinky_Intermediate', 'Pinky_Distal'])
    }

    for f in range(num_frames):
        fixed_lms[f, 0] = raw_sequence_px[f, 0] # Il polso rimane dove l'ha visto MediaPipe

        for finger, (lms_idx, bone_names) in finger_defs.items():
            # Forza Metacarpo
            mcp_idx = lms_idx[0]
            dir_meta = raw_sequence_px[f, mcp_idx] - fixed_lms[f, 0]
            norm_meta = np.linalg.norm(dir_meta)
            if norm_meta > 1e-6: dir_meta /= norm_meta
            
            len_meta = np.linalg.norm(anatomy['meta_vectors'][finger])
            fixed_lms[f, mcp_idx] = fixed_lms[f, 0] + dir_meta * len_meta

            # Forza Falangi a cascata
            parent_idx = mcp_idx
            for i, child_idx in enumerate(lms_idx[1:]):
                dir_bone = raw_sequence_px[f, child_idx] - raw_sequence_px[f, parent_idx]
                norm_bone = np.linalg.norm(dir_bone)
                if norm_bone > 1e-6: dir_bone /= norm_bone
                else: dir_bone = np.array([0.0, 1.0, 0.0]) 
                
                len_bone = anatomy['lengths'][bone_names[i]]
                fixed_lms[f, child_idx] = fixed_lms[f, parent_idx] + dir_bone * len_bone
                parent_idx = child_idx
                
    return fixed_lms

# --- 3. ANALISI VARIANZA SUI DATI CURATI ---
def analyze_fixed_variance(csv_path, bone_to_plot=None):
    print(f"--- ANALISI VARIANZA DOPO PROIEZIONE RIGIDA ---")
    df = pd.read_csv(csv_path)
    num_frames = len(df)
    
    IMG_WIDTH = 640
    IMG_HEIGHT = 480

    # 1. Calcolo Anatomia (Fase di Calibrazione)
    anatomy, bone_links = extract_anatomy_from_csv(df, 540, 590, IMG_WIDTH, IMG_HEIGHT)
    
    # 2. Estrazione dati grezzi in pixel
    raw_sequence_px = np.zeros((num_frames, 21, 3))
    for f in range(num_frames):
        row = df.iloc[f]
        for i in range(21):
            raw_sequence_px[f, i] = [
                row[f'LM_{i}_X'] * IMG_WIDTH, 
                row[f'LM_{i}_Y'] * IMG_HEIGHT, 
                row[f'LM_{i}_Z'] * IMG_WIDTH
            ]
            
    # 2.5 FILTRAGGIO SAVITZKY-GOLAY
    print("Applicazione del filtro temporale di Savitzky-Golay...")
    window_length = 15
    polyorder = 3
    smoothed_sequence_px = np.zeros_like(raw_sequence_px)
    for i in range(21):
        for axis in range(3):
            smoothed_sequence_px[:, i, axis] = savgol_filter(raw_sequence_px[:, i, axis], window_length, polyorder)

    # 3. APPLICHIAMO LA CURA (Proiezione sulle sfere)
    print("Applicazione del vincolo sferico su tutti i frame...")
    fixed_sequence_px = enforce_rigid_skeleton_on_sequence(smoothed_sequence_px, anatomy)

    # 4. Calcoliamo la varianza sui dati NUOVI
    # Aggiungiamo anche i metacarpi per coerenza con il tuo output precedente
    meta_links = {
        'Thumb_Metacarpal': (0, 1), 'Index_Metacarpal': (0, 5), 
        'Middle_Metacarpal': (0, 9), 'Ring_Metacarpal': (0, 13), 'Pinky_Metacarpal': (0, 17)
    }
    all_links = {**meta_links, **bone_links}
    bone_lengths_history = {bone: np.zeros(num_frames) for bone in all_links.keys()}

    print("Calcolo delle lunghezze ossee SUI DATI CURATI...")
    for f in range(num_frames):
        for bone_name, (idx1, idx2) in all_links.items():
            p1 = fixed_sequence_px[f, idx1]
            p2 = fixed_sequence_px[f, idx2]
            length_px = np.linalg.norm(p2 - p1)
            bone_lengths_history[bone_name][f] = length_px

    # 5. Stampa dei risultati
    print(f"\n{'Segmento Osseo':<20} | {'Media (px)':<10} | {'Std Dev (px)':<12} | {'Max Variazione %':<15}")
    print("-" * 65)
    
    total_max_variation = 0.0
    
    for bone_name, lengths in bone_lengths_history.items():
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        
        variation_pct = ((max_len - min_len) / mean_len) * 100 if mean_len > 0 else 0
        if variation_pct > total_max_variation:
            total_max_variation = variation_pct
            
        print(f"{bone_name:<20} | {mean_len:>10.2f} | {std_len:>12.6f} | {variation_pct:>13.2f}%")
        
    print("-" * 65)
    print(f"\nANALISI COMPLETATA.")
    if total_max_variation < 0.1:
        print("✅ SUCCESSO ASSOLUTO: La fluttuazione è stata annientata. Lo scheletro è ora 100% rigido!")
    else:
        print("Qualcosa non va, c'è ancora varianza.")

    # 6. Salvataggio del nuovo CSV per l'animazione
    output_filename = csv_path.replace('.csv', '_test_lunghezze.csv')
    print(f"\nSalvataggio del nuovo dataset con scheletro rigido in: {os.path.basename(output_filename)}")
    
    # Riportiamo le coordinate da Pixel a Normalizzate (0.0 - 1.0) 
    # affinché animate_raw_kinematics.py possa visualizzarle correttamente senza modifiche.
    df_fixed = df.copy()
    for i in range(21):
        df_fixed[f'LM_{i}_X'] = fixed_sequence_px[:, i, 0] / IMG_WIDTH
        df_fixed[f'LM_{i}_Y'] = fixed_sequence_px[:, i, 1] / IMG_HEIGHT
        df_fixed[f'LM_{i}_Z'] = fixed_sequence_px[:, i, 2] / IMG_WIDTH
        
    df_fixed.to_csv(output_filename, index=False)

    # Generazione del grafico temporale se richiesto
    if bone_to_plot and bone_to_plot in bone_lengths_history:
        print(f"\nGenerazione del grafico per '{bone_to_plot}'...")
        plt.figure(figsize=(12, 6))
        lengths = bone_lengths_history[bone_to_plot]
        frames = np.arange(len(lengths))
        mean_len = np.mean(lengths)
        
        plt.plot(frames, lengths, label=f'Lunghezza (Dati Curati) {bone_to_plot}', color='blue')
        plt.axhline(y=mean_len, color='red', linestyle='--', label=f'Media ({mean_len:.2f} px)')
        
        plt.title(f'Andamento temporale della lunghezza ossea (Dati Curati): {bone_to_plot}\nFile: {os.path.basename(csv_path)}')
        plt.xlabel('Frame')
        plt.ylabel('Lunghezza (pixel)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()
    elif bone_to_plot:
        print(f"\nAttenzione: Il segmento '{bone_to_plot}' non esiste nella topologia.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KIN_FILE = os.path.join(BASE_DIR, "recordings", "trial_3_Kinematics.csv") 
    analyze_fixed_variance(KIN_FILE, bone_to_plot="Middle_Distal")