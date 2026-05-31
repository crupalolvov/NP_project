import pandas as pd
import numpy as np
import os
import json
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. KINEMATIC PROCESSING HELPERS (Stessa logica della PCA) ---
def process_trial_kinematics(csv_path, annotations=None):
    df = pd.read_csv(csv_path)
    time_kin = df['Timestamp_LSL'].values
    dof_cols = [f'DoF_{i}' for i in range(24)]
    raw_angles = df[dof_cols].values
    
    start_time = time_kin[0]
    time_sec = time_kin - start_time
    mask_rest = np.zeros_like(time_sec, dtype=bool)
    
    if annotations is not None:
        for entry in annotations:
            if entry['gesture'] == 'Rest':
                mask_rest = mask_rest | ((time_sec >= entry['start_sec']) & (time_sec <= entry['end_sec']))
    
    if not np.any(mask_rest):
        q_rest = np.mean(raw_angles[:100, :], axis=0)
    else:
        q_rest = np.mean(raw_angles[mask_rest, :], axis=0)
        
    centered = raw_angles - q_rest
    centered = (centered + 180.0) % 360.0 - 180.0
    norm_angles = (centered + 150.0) / 240.0
    
    return norm_angles, time_kin

def process_predicted_kinematics(csv_path):
    df = pd.read_csv(csv_path)
    time_kin = df['Timestamp_LSL'].values
    pred_cols = [f'Pred_DoF_{i}' for i in range(24)]
    norm_angles = df[pred_cols].values
    return norm_angles, time_kin

# --- 2. METRICS EVALUATION CORE ---
def evaluate_trial_metrics(true_angles, time_true, pred_angles, time_pred, trial_name):
    """
    Calcola R, R^2 e RMSE per ciascun DoF e ritorna un DataFrame formattato.
    Allinea temporalmente i segnali tramite interpolazione ed esclude i primi 0.78s.
    """
    # 1. Interpolazione dei segnali reali (~30 Hz variabile) sui timestamp delle predizioni (80 Hz)
    interpolator = interp1d(time_true, true_angles, axis=0, bounds_error=False, 
                            fill_value=(true_angles[0], true_angles[-1]))
    true_angles_interpolated = interpolator(time_pred)
    
    # 2. Esclusione dei primi 0.78 secondi (64 campioni iniziali a 80 Hz)
    EXCLUDE_SAMPLES = 64
    y_true = true_angles_interpolated[EXCLUDE_SAMPLES:]
    y_pred = pred_angles[EXCLUDE_SAMPLES:]
    
    metrics_list = []
    
    for i in range(24):
        true_dof = y_true[:, i]
        pred_dof = y_pred[:, i]
        
        # Pearson Correlation (R)
        # Se il segnale è completamente piatto (varianza zero), pearsonr restituisce NaN. Gestiamo l'eccezione.
        if np.std(true_dof) < 1e-6 or np.std(pred_dof) < 1e-6:
            r_val = 0.0
        else:
            r_val, _ = pearsonr(true_dof, pred_dof)
            
        # Coefficient of Determination (R^2)
        r2_val = r2_score(true_dof, pred_dof)
        
        # Root Mean Square Error (RMSE)
        rmse_val = np.sqrt(mean_squared_error(true_dof, pred_dof))
        
        metrics_list.append({
            'DoF': f'DoF_{i}',
            'R (Pearson)': r_val,
            'R^2 Score': r2_val,
            'RMSE': rmse_val
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    
    # Calcolo riga delle medie globali
    mean_r = df_metrics['R (Pearson)'].mean()
    mean_r2 = df_metrics['R^2 Score'].mean()
    mean_rmse = df_metrics['RMSE'].mean()
    
    mean_row = pd.DataFrame([{
        'DoF': 'OVERALL MEAN',
        'R (Pearson)': mean_r,
        'R^2 Score': mean_r2,
        'RMSE': mean_rmse
    }])
    
    df_metrics = pd.concat([df_metrics, mean_row], ignore_index=True)
    
    # Formattazione per la stampa a schermo
    df_formatted = df_metrics.copy()
    df_formatted['R (Pearson)'] = df_formatted['R (Pearson)'].apply(lambda x: f"{x*100:.2f}%")
    df_formatted['R^2 Score'] = df_formatted['R^2 Score'].apply(lambda x: f"{x:.3f}")
    df_formatted['RMSE'] = df_formatted['RMSE'].apply(lambda x: f"{x*100:.2f}%") # RMSE come percentuale del range
    
    return df_metrics, df_formatted

# --- 3. EXECUTION FLOW ---
def run_metrics_analysis():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print("="*70)
    print("  TEMPORAL REGRESSION METRICS (R, R^2, RMSE)")
    print("="*70)
    
    # Caricamento Labels
    labels_file = os.path.join(BASE_DIR, "labels.json")
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = {}
        
    os.makedirs(os.path.join(BASE_DIR, "eval"), exist_ok=True)
        
    # Analisi Trial 6
    true_6_file = os.path.join(BASE_DIR, "recordings", "trial_6_Kinematics_core_IKA_24DoF.csv")
    pred_6_file = os.path.join(BASE_DIR, "trial_6_predicted_kinematics_angles.csv")
    
    if os.path.exists(true_6_file) and os.path.exists(pred_6_file):
        print("\n--- Analisi Trial 6 ---")
        norm_true_6, time_true_6 = process_trial_kinematics(true_6_file, labels.get('trial_6', []))
        norm_pred_6, time_pred_6 = process_predicted_kinematics(pred_6_file)
        
        df_raw_6, df_fmt_6 = evaluate_trial_metrics(norm_true_6, time_true_6, norm_pred_6, time_pred_6, "Trial 6")
        print(df_fmt_6.to_string(index=False))
        df_raw_6.to_csv(os.path.join(BASE_DIR, "eval", "metrics_trial_6.csv"), index=False)

    # Analisi Trial 9
    true_9_file = os.path.join(BASE_DIR, "recordings", "trial_9_Kinematics_core_IKA_24DoF.csv")
    pred_9_file = os.path.join(BASE_DIR, "trial_9_predicted_kinematics_angles.csv")
    
    if os.path.exists(true_9_file) and os.path.exists(pred_9_file):
        print("\n--- Analisi Trial 9 ---")
        norm_true_9, time_true_9 = process_trial_kinematics(true_9_file, labels.get('trial_9', []))
        norm_pred_9, time_pred_9 = process_predicted_kinematics(pred_9_file)
        
        df_raw_9, df_fmt_9 = evaluate_trial_metrics(norm_true_9, time_true_9, norm_pred_9, time_pred_9, "Trial 9")
        print(df_fmt_9.to_string(index=False))
        df_raw_9.to_csv(os.path.join(BASE_DIR, "eval", "metrics_trial_9.csv"), index=False)
        
    print("\nFile CSV esportati con successo nella cartella 'eval/'")
    print("="*70)

if __name__ == "__main__":
    run_metrics_analysis()