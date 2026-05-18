import numpy as np
import pandas as pd
import os

def analyze_bone_length_variance(csv_path):
    print(f"--- ANALISI VARIANZA LUNGHEZZE OSSEE (MediaPipe) ---")
    print(f"File in analisi: {os.path.basename(csv_path)}\n")
    
    df = pd.read_csv(csv_path)
    num_frames = len(df)
    
    # Definizione della topologia ossea (i link tra i landmark)
    bone_links = {
        # Pollice
        'Thumb_Metacarpal': (0, 1),
        'Thumb_Proximal': (1, 2), 
        'Thumb_Intermediate': (2, 3), 
        'Thumb_Distal': (3, 4),
        # Indice
        'Index_Metacarpal': (0, 5),
        'Index_Proximal': (5, 6), 
        'Index_Intermediate': (6, 7), 
        'Index_Distal': (7, 8),
        # Medio
        'Middle_Metacarpal': (0, 9),
        'Middle_Proximal': (9, 10), 
        'Middle_Intermediate': (10, 11), 
        'Middle_Distal': (11, 12),
        # Anulare
        'Ring_Metacarpal': (0, 13),
        'Ring_Proximal': (13, 14), 
        'Ring_Intermediate': (14, 15), 
        'Ring_Distal': (15, 16),
        # Mignolo
        'Pinky_Metacarpal': (0, 17),
        'Pinky_Proximal': (17, 18), 
        'Pinky_Intermediate': (18, 19), 
        'Pinky_Distal': (19, 20)
    }
    
    # Inizializziamo un dizionario per raccogliere le lunghezze frame per frame
    bone_lengths_history = {bone: np.zeros(num_frames) for bone in bone_links.keys()}
    
    # Estrazione coordinate (vettorizzata per massima velocità)
    for i in range(21):
        df[f'LM_{i}_pos'] = list(zip(df[f'LM_{i}_X'], df[f'LM_{i}_Y'], df[f'LM_{i}_Z']))
    
    print("Calcolo delle lunghezze frame per frame...")
    for index, row in df.iterrows():
        for bone_name, (idx1, idx2) in bone_links.items():
            p1 = np.array(row[f'LM_{idx1}_pos'])
            p2 = np.array(row[f'LM_{idx2}_pos'])
            # Calcolo distanza euclidea in millimetri (assumendo coordinate normalizzate MediaPipe)
            length_mm = np.linalg.norm(p2 - p1) * 1000.0
            bone_lengths_history[bone_name][index] = length_mm

    # Stampa dei risultati statistici
    print(f"{'Segmento Osseo':<20} | {'Media (mm)':<10} | {'Std Dev (mm)':<12} | {'Min (mm)':<8} | {'Max (mm)':<8} | {'Max Variazione %':<15}")
    print("-" * 85)
    
    total_max_variation = 0.0
    worst_bone = ""
    
    for bone_name, lengths in bone_lengths_history.items():
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        
        # Calcolo della variazione percentuale tra il minimo e il massimo
        variation_pct = ((max_len - min_len) / mean_len) * 100 if mean_len > 0 else 0
        
        if variation_pct > total_max_variation:
            total_max_variation = variation_pct
            worst_bone = bone_name
            
        print(f"{bone_name:<20} | {mean_len:>10.2f} | {std_len:>12.2f} | {min_len:>8.2f} | {max_len:>8.2f} | {variation_pct:>13.2f}%")
        
    print("-" * 85)
    print(f"\\nANALISI COMPLETATA:")
    print(f"Segmento peggiore: {worst_bone} con una fluttuazione massima del {total_max_variation:.2f}%")
    
    if total_max_variation > 10.0:
        print("DIAGNOSI CRITICA: Deformazione strutturale elevata. MediaPipe genera un severo effetto 'ossa di gomma'.")
        print("È imperativo implementare la proiezione (Bone-length enforcement) prima dell'IKA.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KIN_FILE = os.path.join(BASE_DIR, "recordings", "trial_1_Kinematics.csv") 
    analyze_bone_length_variance(KIN_FILE)