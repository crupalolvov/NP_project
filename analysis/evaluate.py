import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import os

def calculate_mpcc(true_angles_path, pred_angles_path):
    """
    Calcola il Mean Pearson Correlation Coefficient (MPCC) sui 24 gradi di libertà,
    allineando temporalmente i segnali tramite interpolazione ed escludendo i primi 0.78s.
    """
    # Caricamento dei dati
    df_true = pd.read_csv(true_angles_path)
    df_pred = pd.read_csv(pred_angles_path)
    
    # Estrazione dei timestamp
    t_true = df_true['Timestamp_LSL'].values
    t_pred = df_pred['Timestamp_LSL'].values
    
    # 1. Interpolazione dei segnali reali (~30 Hz variabile) sui timestamp delle predizioni (80 Hz)
    dof_cols = [f'DoF_{i}' for i in range(24)]
    true_angles_raw = df_true[dof_cols].values
    
    interpolator = interp1d(t_true, true_angles_raw, axis=0, bounds_error=False, 
                            fill_value=(true_angles_raw[0], true_angles_raw[-1]))
    true_angles_interpolated = interpolator(t_pred)
    
    pred_cols = [f'Pred_DoF_{i}' for i in range(24)]
    pred_angles = df_pred[pred_cols].values
    
    # 2. Esclusione dei primi 0.78 secondi (64 campioni iniziali a 80 Hz)
    # come specificato nella sezione II.E del paper.
    EXCLUDE_SAMPLES = 64
    true_angles_clean = true_angles_interpolated[EXCLUDE_SAMPLES:]
    pred_angles_clean = pred_angles[EXCLUDE_SAMPLES:]
    
    pcc_values = []
    
    # Calcolo del PCC per ciascuno dei 24 gradi di libertà (DoFs)
    for i in range(24):
        true_signal = true_angles_clean[:, i]
        pred_signal = pred_angles_clean[:, i]
        
        # Evita divisioni per zero se un segnale è completamente piatto
        if np.std(true_signal) > 0 and np.std(pred_signal) > 0:
            pcc, _ = pearsonr(true_signal, pred_signal)
            pcc_values.append(pcc)
        else:
            pcc_values.append(0.0)
                
    mpcc = np.mean(pcc_values)
    pcc_median = np.median(pcc_values)
    pcc_t1 = np.percentile(pcc_values, 33.33)
    pcc_t2 = np.percentile(pcc_values, 66.67)
    
    return mpcc, pcc_t1, pcc_t2, pcc_median

def calculate_wfd(true_lms_path, pred_lms_path):
    """
    Calcola la Mean Distance (MD) basata sulla Weighted Fingertip Distance (WFD)
    per Pollice, Indice e Medio, allineando temporalmente i segnali ed escludendo i primi 0.78s.
    Entrambi i set di coordinate vengono traslati con il polso all'origine (0,0,0) ad ogni frame
    per confrontare la posa intrinseca ed annullare offset spaziali assoluti della webcam.
    """
    # Caricamento dei dati
    df_true = pd.read_csv(true_lms_path)
    df_pred = pd.read_csv(pred_lms_path)
    
    t_true = df_true['Timestamp_LSL'].values
    t_pred = df_pred['Timestamp_LSL'].values
    
    # Definizione corretta delle dita e dei landmark MediaPipe:
    # LM_4 = Thumb tip, LM_8 = Index tip, LM_12 = Middle tip
    # Includiamo anche il polso LM_0 per la traslazione all'origine
    fingertips = {
        'thumb': ['LM_4_X', 'LM_4_Y', 'LM_4_Z'],
        'index': ['LM_8_X', 'LM_8_Y', 'LM_8_Z'],
        'middle': ['LM_12_X', 'LM_12_Y', 'LM_12_Z']
    }
    
    # Raccogliamo le colonne delle dita e del polso
    all_coord_cols = ['LM_0_X', 'LM_0_Y', 'LM_0_Z']
    for finger, coords in fingertips.items():
        all_coord_cols.extend(coords)
        
    true_coords_raw = df_true[all_coord_cols].values
    
    # 1. Interpolazione dei landmark reali sui timestamp delle predizioni
    interpolator = interp1d(t_true, true_coords_raw, axis=0, bounds_error=False, 
                            fill_value=(true_coords_raw[0], true_coords_raw[-1]))
    true_coords_interpolated = interpolator(t_pred)
    
    pred_coords = df_pred[all_coord_cols].values
    
    # 2. Esclusione dei primi 0.78 secondi (64 campioni iniziali a 80 Hz)
    EXCLUDE_SAMPLES = 64
    true_coords_clean = true_coords_interpolated[EXCLUDE_SAMPLES:]
    pred_coords_clean = pred_coords[EXCLUDE_SAMPLES:]
    
    wfd_per_frame = []
    
    # Calcoliamo la distanza per ciascun frame
    num_frames = len(true_coords_clean)
    for i in range(num_frames):
        # Polso di riferimento (LM_0) per questo frame
        wrist_true = true_coords_clean[i, 0:3]
        wrist_pred = pred_coords_clean[i, 0:3]
        
        distances = []
        for idx, (finger, coords) in enumerate(fingertips.items()):
            col_start = 3 + idx * 3
            # Coordinate assolute
            p_true_raw = true_coords_clean[i, col_start:col_start+3]
            p_pred_raw = pred_coords_clean[i, col_start:col_start+3]
            
            # Traslazione con polso all'origine (0,0,0)
            p_true = p_true_raw - wrist_true
            p_pred = p_pred_raw - wrist_pred
            
            # Distanza euclidea (in metri)
            dist_meters = np.linalg.norm(p_true - p_pred)
            # Riconversione in millimetri (mm)
            dist_mm = dist_meters * 1000.0
            distances.append(dist_mm)
            
        wfd_per_frame.append(np.mean(distances))
        
    md = np.mean(wfd_per_frame)
    wfd_median = np.median(wfd_per_frame)
    wfd_t1 = np.percentile(wfd_per_frame, 33.33)
    wfd_t2 = np.percentile(wfd_per_frame, 66.67)
    
    return md, wfd_t1, wfd_t2, wfd_median

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Valuta per default i trial di test non visti dall'addestramento (6, 9, 15)
    TRIALS_TO_EVALUATE = [6, 9, 15]
    
    print("="*60)
    print("      REPORT DI VALUTAZIONE RPC-NET SUI DATASET DI TEST")
    print("="*60)
    
    for trial in TRIALS_TO_EVALUATE:
        print(f"\nValuto Trial {trial}...")
        
        TRUE_ANGLES = os.path.join(BASE_DIR, "recordings", f"trial_{trial}_Kinematics_core_IKA_24DoF.csv")
        PRED_ANGLES = os.path.join(BASE_DIR, f"trial_{trial}_predicted_kinematics_angles.csv")
        
        TRUE_LMS = os.path.join(BASE_DIR, "recordings", f"trial_{trial}_Kinematics_preprocessed.csv")
        PRED_LMS = os.path.join(BASE_DIR, f"trial_{trial}_predicted_kinematics_lms.csv")
        
        # Check files existence
        if not os.path.exists(PRED_ANGLES) or not os.path.exists(PRED_LMS):
            print(f"   [AVVISO]: File di predizione per Trial {trial} non trovati. Salto...")
            continue
            
        if not os.path.exists(TRUE_ANGLES) or not os.path.exists(TRUE_LMS):
            print(f"   [AVVISO]: File Ground Truth per Trial {trial} non trovati. Salto...")
            continue
            
        print("-"*50)
        try:
            mpcc, pcc_t1, pcc_t2, pcc_med = calculate_mpcc(TRUE_ANGLES, PRED_ANGLES)
            print("1. ANALISI ANGOLARE (MPCC):")
            print(f"   MPCC Medio   : {mpcc * 100:.2f} %")
            print(f"   Mediana      : {pcc_med * 100:.2f} %")
            print(f"   Tertili      : (T1: {pcc_t1*100:.1f}%, T2: {pcc_t2*100:.1f}%)")
        except Exception as e:
            print(f"   Errore nel calcolo MPCC: {e}")
            
        try:
            md, wfd_t1, wfd_t2, wfd_med = calculate_wfd(TRUE_LMS, PRED_LMS)
            print("\n2. ANALISI SPAZIALE (MD):")
            print(f"   Mean Distance: {md:.2f} mm")
            print(f"   Mediana      : {wfd_med:.2f} mm")
            print(f"   Tertili      : (T1: {wfd_t1:.1f} mm, T2: {wfd_t2:.1f} mm)")
        except Exception as e:
            print(f"   Errore nel calcolo MD: {e}")
            
        print("-"*50)
    print("\nValutazione completata.")