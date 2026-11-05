import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def detect_coordinate_format(df):
    """
    Rileva automaticamente se il dataframe contiene world_landmarks (metri) o normalizzati (pixel).
    """
    min_x = df[[f'LM_{i}_X' for i in range(21)]].min().min()
    min_y = df[[f'LM_{i}_Y' for i in range(21)]].min().min()
    if min_x < -0.01 or min_y < -0.01 or df['LM_0_X'].abs().max() < 1e-4:
        return True 
    return False 

def analyze_bone_length_variance(csv_path, bone_to_plot=None):
    print(f"--- ANALISI VARIANZA LUNGHEZZE OSSEE (MediaPipe) ---")
    print(f"File in analisi: {os.path.basename(csv_path)}\n")
    
    df = pd.read_csv(csv_path)
    num_frames = len(df)
    
    is_metric = detect_coordinate_format(df)
    if is_metric:
        SCALE_X = SCALE_Y = SCALE_Z = 1000.0 # Convertiamo i metri in millimetri
        print("--- RILEVATI DATI IN METRI (World Landmarks) -> Scalo uniformemente in millimetri ---")
    else:
        SCALE_X, SCALE_Y, SCALE_Z = 640.0, 480.0, 640.0
        print("--- RILEVATI DATI IN PIXEL (Normalizzati) -> Scalo a 640x480 ---")

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
    
    print("Calcolo delle lunghezze ossee frame per frame...")
    for index, row in df.iterrows():
        for bone_name, (idx1, idx2) in bone_links.items():
            # Estrai coordinate originali
            p1_norm = np.array([row[f'LM_{idx1}_X'], row[f'LM_{idx1}_Y'], row[f'LM_{idx1}_Z']])
            p2_norm = np.array([row[f'LM_{idx2}_X'], row[f'LM_{idx2}_Y'], row[f'LM_{idx2}_Z']])

            # Applica i fattori di scala. Per Z, usiamo la larghezza come fattore di scala nei dati normalizzati
            # per mantenere le proporzioni reali, come suggerito da MediaPipe.
            p1_scaled = np.array([p1_norm[0] * SCALE_X, p1_norm[1] * SCALE_Y, p1_norm[2] * SCALE_Z])
            p2_scaled = np.array([p2_norm[0] * SCALE_X, p2_norm[1] * SCALE_Y, p2_norm[2] * SCALE_Z])

            # Calcolo distanza euclidea
            length_px = np.linalg.norm(p2_scaled - p1_scaled)
            bone_lengths_history[bone_name][index] = length_px

    # Stampa dei risultati statistici
    print(f"\n{'Segmento Osseo':<20} | {'Media':<10} | {'Std Dev':<12} | {'Min [Frame]':<18} | {'Max [Frame]':<18} | {'Max Variazione %':<15}")
    print("-" * 105)
    
    total_max_variation = 0.0
    worst_bone = ""
    
    for bone_name, lengths in bone_lengths_history.items():
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        
        min_idx = np.argmin(lengths)
        max_idx = np.argmax(lengths)
        
        # Calcolo della variazione percentuale tra il minimo e il massimo
        variation_pct = ((max_len - min_len) / mean_len) * 100 if mean_len > 0 else 0
        
        if variation_pct > total_max_variation:
            total_max_variation = variation_pct
            worst_bone = bone_name
            
        print(f"{bone_name:<20} | {mean_len:>10.2f} | {std_len:>12.2f} | {min_len:>8.2f} [fr:{min_idx:04d}] | {max_len:>8.2f} [fr:{max_idx:04d}] | {variation_pct:>13.2f}%")
        
    print("-" * 105)
    print(f"\nANALISI COMPLETATA:")
    print(f"Segmento peggiore: {worst_bone} con una fluttuazione massima del {total_max_variation:.2f}%")
    
    if total_max_variation > 10.0:
        print("\nDIAGNOSI CRITICA: Deformazione strutturale elevata. MediaPipe genera un severo effetto 'ossa di gomma'.")
        print("Questo conferma che le coordinate di MediaPipe non garantiscono lunghezze ossee costanti.")
        print("L'uso di un modello cinematico con lunghezze fisse (come in IKA.py o core_kin.py) è fondamentale per correggere questo artefatto.")
        
    # Generazione del grafico temporale se richiesto
    if bone_to_plot and bone_to_plot in bone_lengths_history:
        print(f"\nGenerazione del grafico per '{bone_to_plot}'...")
        plt.figure(figsize=(12, 6))
        lengths = bone_lengths_history[bone_to_plot]
        frames = np.arange(len(lengths))
        mean_len = np.mean(lengths)
        
        plt.plot(frames, lengths, label=f'Lunghezza {bone_to_plot}', color='blue')
        plt.axhline(y=mean_len, color='red', linestyle='--', label=f'Media ({mean_len:.2f} px)')
        
        plt.title(f'Andamento temporale della lunghezza ossea: {bone_to_plot}\nFile: {os.path.basename(csv_path)}')
        plt.xlabel('Frame')
        plt.ylabel('Lunghezza (unità)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()
    elif bone_to_plot:
        print(f"\nAttenzione: Il segmento '{bone_to_plot}' non esiste nella topologia.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KIN_FILE = os.path.join(BASE_DIR, "recordings", "trial_6_Kinematics.csv") 
    analyze_bone_length_variance(KIN_FILE, bone_to_plot="Middle_Distal")