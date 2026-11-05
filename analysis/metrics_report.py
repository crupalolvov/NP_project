import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score

DOF_NAMES = [
    "Wrist Flex/Ext", "Wrist Uln/Rad Dev", "Wrist Pro/Sup",
    "Thumb CMC F/E", "Thumb CMC A/A", "Thumb MCP F/E", "Thumb MCP A/A", "Thumb IP F/E",
    "Index MCP F/E", "Index MCP A/A", "Index PIP F/E", "Index DIP F/E",
    "Middle MCP F/E", "Middle MCP A/A", "Middle PIP F/E", "Middle DIP F/E",
    "Ring MCP F/E", "Ring MCP A/A", "Ring PIP F/E", "Ring DIP F/E",
    "Pinky MCP F/E", "Pinky MCP A/A", "Pinky PIP F/E", "Pinky DIP F/E"
]

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

# --- 1.1 LANDMARK PROCESSING HELPERS ---
def process_lms(csv_path):
    df = pd.read_csv(csv_path)
    time_kin = df['Timestamp_LSL'].values
    
    lm_cols = [f'LM_{i}_{axis}' for i in range(21) for axis in ['X', 'Y', 'Z']]
    # Cerca di estrarre le colonne LM_... se non presenti usa le 63 colonne che seguono il Timestamp
    if all(c in df.columns for c in lm_cols):
        lms = df[lm_cols].values
    else:
        lms = df.drop(columns=['Timestamp_LSL']).iloc[:, :63].values
    return lms, time_kin

def evaluate_lm_metrics(true_lms, time_true, pred_lms, time_pred):
    # Interpolazione per allineare temporalmente i segnali
    interpolator = interp1d(time_true, true_lms, axis=0, bounds_error=False, 
                            fill_value=(true_lms[0], true_lms[-1]))
    true_lms_interpolated = interpolator(time_pred)
    
    EXCLUDE_SAMPLES = 64
    
    # Controllo fattore di scala (metri vs millimetri)
    scale_factor = 1.0 if np.max(np.abs(pred_lms)) > 10.0 else 1000.0
    
    y_true = true_lms_interpolated[EXCLUDE_SAMPLES:] * scale_factor
    y_pred = pred_lms[EXCLUDE_SAMPLES:] * scale_factor
    
    # Reshape in matrici 3D (Frames x 21 Landmark x 3 Assi XYZ)
    y_true_3d = y_true.reshape(-1, 21, 3)
    y_pred_3d = y_pred.reshape(-1, 21, 3)
    
    # Centratura sul Polso (Landmark 0) per isolare la posa intrinseca
    y_true_centered = y_true_3d - y_true_3d[:, 0:1, :]
    y_pred_centered = y_pred_3d - y_pred_3d[:, 0:1, :]
    
    # Calcolo della distanza euclidea vettoriale (Frame x 21)
    distances = np.linalg.norm(y_true_centered - y_pred_centered, axis=2)
    
    # -----------------------------------------------------------------
    # MAPPATURA DELLE PUNTE (Fingertips)
    # 4 = Pollice, 8 = Indice, 12 = Medio, 16 = Anulare, 20 = Mignolo
    # -----------------------------------------------------------------
    
    # 1. WFD (Weighted Fingertip Distance): Errore delle 3 dita funzionali / 3
    wfd = np.mean(distances[:, [4, 8, 12]])
    
    # 2. MFD (Mean Fingertip Distance): Errore di TUTTE le 5 dita / 5
    mfd = np.mean(distances[:, [4, 8, 12, 16, 20]])
    
    return wfd, mfd

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
            'DoF': f'DoF_{i} ({DOF_NAMES[i]})',
            'R (Pearson)': r_val,
            'R^2 Score': r2_val,
            'RMSE': rmse_val
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    
    # --- NUOVO: Trova il joint migliore e peggiore basandosi sulla correlazione di Pearson (R) ---
    best_joint_idx = df_metrics['R (Pearson)'].idxmax()
    worst_joint_idx = df_metrics['R (Pearson)'].idxmin()
    
    best_joint_info = df_metrics.loc[best_joint_idx]
    worst_joint_info = df_metrics.loc[worst_joint_idx]
    
    # Calcolo riga delle medie globali
    mean_r = df_metrics['R (Pearson)'].mean()
    mean_r2 = df_metrics['R^2 Score'].mean()
    mean_rmse = df_metrics['RMSE'].mean()
    
    # Weighted Fingertip (WF) metrics: esclude Mignolo e Anulare (DoF 0-15)
    wf_mean_r = df_metrics.loc[:15, 'R (Pearson)'].mean()
    wf_mean_r2 = df_metrics.loc[:15, 'R^2 Score'].mean()
    wf_mean_rmse = df_metrics.loc[:15, 'RMSE'].mean()

    mean_row = pd.DataFrame([{
        'DoF': 'OVERALL MEAN',
        'R (Pearson)': mean_r,
        'R^2 Score': mean_r2,
        'RMSE': mean_rmse
    }, {
        'DoF': 'WF MEAN (Wrist+Thumb+Idx+Mid)',
        'R (Pearson)': wf_mean_r,
        'R^2 Score': wf_mean_r2,
        'RMSE': wf_mean_rmse
    }])
    
    df_metrics_with_mean = pd.concat([df_metrics, mean_row], ignore_index=True)
    
    # Formattazione per la stampa a schermo
    df_formatted = df_metrics_with_mean.copy()
    df_formatted['R (Pearson)'] = df_formatted['R (Pearson)'].apply(lambda x: f"{x*100:.2f}%")
    df_formatted['R^2 Score'] = df_formatted['R^2 Score'].apply(lambda x: f"{x:.3f}")
    df_formatted['RMSE'] = df_formatted['RMSE'].apply(lambda x: f"{x*100:.2f}%") # RMSE come percentuale del range
    
    return df_metrics_with_mean, df_formatted, best_joint_info, worst_joint_info, wf_mean_r, y_true, y_pred, time_pred[EXCLUDE_SAMPLES:]

def plot_kinematic_tracking(y_true, y_pred, time_array, df_metrics, trial_name, trial_labels=None, start_time=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    GESTURE_COLORS = {
        "Rest": "#7f7f7f", "Closed Hand": "#e41a1c", "Pistol": "#377eb8",
        "Open Hand": "#4daf4a", "OK": "#ff7f0e", "Rock On": "#984ea3"
    }
    
    # Conversione in gradi
    y_true_deg = (y_true * 240.0) - 150.0
    y_pred_deg = (y_pred * 240.0) - 150.0
    
    # Normalizzazione asse temporale rispetto al Ground Truth (allinea con labels.json)
    if start_time is None:
        start_time = time_array[0]
    time_rel = time_array - start_time
    
    fig, axes = plt.subplots(4, 6, figsize=(24, 12), sharex=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)
    
    for i, ax in enumerate(axes.flat):
        if i >= 24:
            break
            
        # Disegno delle bande di background per i gesti
        if trial_labels:
            for entry in trial_labels:
                color = GESTURE_COLORS.get(entry['gesture'], '#cccccc')
                ax.axvspan(entry['start_sec'], entry['end_sec'], facecolor=color, alpha=0.15)
                
        ax.plot(time_rel, y_pred_deg[:, i], color='#d95f02', label='Estimate', linewidth=1.5)
        ax.plot(time_rel, y_true_deg[:, i], color='#1f78b4', linestyle='--', label='Truth', linewidth=1.5)
        
        r_val = df_metrics.loc[i, 'R (Pearson)']
        rmse_val = df_metrics.loc[i, 'RMSE'] * 240.0
        
        ax.set_title(f'DoF_{i}: {DOF_NAMES[i]}\nPCC: {r_val:.2f} | RMSE: {rmse_val:.1f}°', fontsize=10)
        
        if i >= 18:
            ax.set_xlabel('Time (s)')
        if i % 6 == 0:
            ax.set_ylabel('Angle (°)')
            
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Aggiunta etichette dei gesti alla legenda globale
    if trial_labels:
        added_gestures = []
        for entry in trial_labels:
            gest = entry['gesture']
            if gest not in added_gestures:
                color = GESTURE_COLORS.get(gest, '#cccccc')
                patch = mpatches.Patch(color=color, alpha=0.3, label=gest)
                handles.append(patch)
                labels.append(gest)
                added_gestures.append(gest)
                
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    
    os.makedirs(os.path.join(BASE_DIR, "eval"), exist_ok=True)
    plt.savefig(os.path.join(BASE_DIR, "eval", f"tracking_{trial_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_overall_error(y_true, y_pred, time_array, trial_name, trial_labels=None, start_time=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    GESTURE_COLORS = {
        "Rest": "#7f7f7f", "Closed Hand": "#e41a1c", "Pistol": "#377eb8",
        "Open Hand": "#4daf4a", "OK": "#ff7f0e", "Rock On": "#984ea3"
    }
    
    # Conversione in gradi
    y_true_deg = (y_true * 240.0) - 150.0
    y_pred_deg = (y_pred * 240.0) - 150.0
    
    # Calcolo MAE e RMSE medio sui 24 DoF per ogni frame temporale
    mae_over_time = np.mean(np.abs(y_true_deg - y_pred_deg), axis=1)
    rmse_over_time = np.sqrt(np.mean((y_true_deg - y_pred_deg)**2, axis=1))
    
    # Normalizzazione asse temporale
    if start_time is None:
        start_time = time_array[0]
    time_rel = time_array - start_time
    
    plt.figure(figsize=(12, 4))
    
    if trial_labels:
        added_labels = set()
        for entry in trial_labels:
            gest = entry['gesture']
            color = GESTURE_COLORS.get(gest, '#cccccc')
            label = gest if gest not in added_labels else ""
            added_labels.add(gest)
            plt.axvspan(entry['start_sec'], entry['end_sec'], facecolor=color, alpha=0.15, label=label)
            
    plt.plot(time_rel, mae_over_time, color='#e41a1c', label='Mean Absolute Error (MAE)', linewidth=1.5)
    plt.plot(time_rel, rmse_over_time, color='#377eb8', label='Root Mean Square Error (RMSE)', linewidth=1.5, alpha=0.6)
    
    mean_mae = np.mean(mae_over_time)
    plt.axhline(y=mean_mae, color='#e41a1c', linestyle='--', label=f'Global Average MAE: {mean_mae:.2f}°')
    
    plt.title(f'Overall Kinematic Error Over Time (Avg across 24 DoFs) - {trial_name}', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (°)')
    # Legenda spostata all'esterno per non coprire i dati
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(os.path.join(BASE_DIR, "eval"), exist_ok=True)
    plt.savefig(os.path.join(BASE_DIR, "eval", f"overall_error_{trial_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

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

        df_raw_6, df_fmt_6, best_6, worst_6, wf_pcc_6, y_true_6, y_pred_6, time_6 = evaluate_trial_metrics(norm_true_6, time_true_6, norm_pred_6, time_pred_6, "Trial 6")
        print(df_fmt_6.to_string(index=False))
        print("\n  --- HIGHLIGHTS (Trial 6) ---")
        print(f"  Miglior Joint (R): {best_6['DoF']} con R = {best_6['R (Pearson)']*100:.2f}%")
        print(f"  Peggior Joint (R): {worst_6['DoF']} con R = {worst_6['R (Pearson)']*100:.2f}%")
        print(f"  Weighted Fingertip PCC: {wf_pcc_6*100:.2f}%")
        df_raw_6.to_csv(os.path.join(BASE_DIR, "eval", "metrics_trial_6.csv"), index=False)
        
        labels_6 = labels.get('trial_6', [])
        start_t_6 = time_true_6[0]
        plot_kinematic_tracking(y_true_6, y_pred_6, time_6, df_raw_6, "trial_6", trial_labels=labels_6, start_time=start_t_6)
        plot_overall_error(y_true_6, y_pred_6, time_6, "trial_6", trial_labels=labels_6, start_time=start_t_6)

    # Analisi LMs Trial 6
    true_6_lm_file = os.path.join(BASE_DIR, "recordings", "trial_6_Kinematics_preprocessed.csv")
    if not os.path.exists(true_6_lm_file):
        true_6_lm_file = os.path.join(BASE_DIR, "trial_6_Kinematics_preprocessed.csv")
    pred_6_lm_file = os.path.join(BASE_DIR, "trial_6_predicted_kinematics_lms.csv")
    
    if os.path.exists(true_6_lm_file) and os.path.exists(pred_6_lm_file):
        print("\n  --- DISTANZE LANDMARKS (Trial 6) ---")
        true_lms_6, time_true_lm_6 = process_lms(true_6_lm_file)
        pred_lms_6, time_pred_lm_6 = process_lms(pred_6_lm_file)
        wfd_6, gmd_6 = evaluate_lm_metrics(true_lms_6, time_true_lm_6, pred_lms_6, time_pred_lm_6)
        print(f"  Weighted Fingertip Distance (Pollice, Indice, Medio): {wfd_6:.2f} mm")
        print(f"  Global Mean Distance: {gmd_6:.2f} mm")

    # Analisi Trial 9
    true_9_file = os.path.join(BASE_DIR, "recordings", "trial_9_Kinematics_core_IKA_24DoF.csv")
    pred_9_file = os.path.join(BASE_DIR, "trial_9_predicted_kinematics_angles.csv")
    
    if os.path.exists(true_9_file) and os.path.exists(pred_9_file):
        print("\n--- Analisi Trial 9 ---")
        norm_true_9, time_true_9 = process_trial_kinematics(true_9_file, labels.get('trial_9', []))
        norm_pred_9, time_pred_9 = process_predicted_kinematics(pred_9_file)

        df_raw_9, df_fmt_9, best_9, worst_9, wf_pcc_9, y_true_9, y_pred_9, time_9 = evaluate_trial_metrics(norm_true_9, time_true_9, norm_pred_9, time_pred_9, "Trial 9")
        print(df_fmt_9.to_string(index=False))
        print("\n  --- HIGHLIGHTS (Trial 9) ---")
        print(f"  Miglior Joint (R): {best_9['DoF']} con R = {best_9['R (Pearson)']*100:.2f}%")
        print(f"  Peggior Joint (R): {worst_9['DoF']} con R = {worst_9['R (Pearson)']*100:.2f}%")
        print(f"  Weighted Fingertip PCC: {wf_pcc_9*100:.2f}%")
        df_raw_9.to_csv(os.path.join(BASE_DIR, "eval", "metrics_trial_9.csv"), index=False)
        
        labels_9 = labels.get('trial_9', [])
        start_t_9 = time_true_9[0]
        plot_kinematic_tracking(y_true_9, y_pred_9, time_9, df_raw_9, "trial_9", trial_labels=labels_9, start_time=start_t_9)
        plot_overall_error(y_true_9, y_pred_9, time_9, "trial_9", trial_labels=labels_9, start_time=start_t_9)
        
    # Analisi LMs Trial 9
    true_9_lm_file = os.path.join(BASE_DIR, "recordings", "trial_9_Kinematics_preprocessed.csv")
    if not os.path.exists(true_9_lm_file):
        true_9_lm_file = os.path.join(BASE_DIR, "trial_9_Kinematics_preprocessed.csv")
    pred_9_lm_file = os.path.join(BASE_DIR, "trial_9_predicted_kinematics_lms.csv")
    
    if os.path.exists(true_9_lm_file) and os.path.exists(pred_9_lm_file):
        print("\n  --- DISTANZE LANDMARKS (Trial 9) ---")
        true_lms_9, time_true_lm_9 = process_lms(true_9_lm_file)
        pred_lms_9, time_pred_lm_9 = process_lms(pred_9_lm_file)
        wfd_9, gmd_9 = evaluate_lm_metrics(true_lms_9, time_true_lm_9, pred_lms_9, time_pred_lm_9)
        print(f"  Weighted Fingertip Distance (Pollice, Indice, Medio): {wfd_9:.2f} mm")
        print(f"  Global Mean Distance: {gmd_9:.2f} mm")

    print("\nFile CSV esportati con successo nella cartella 'eval/'")
    print("="*70)

if __name__ == "__main__":
    run_metrics_analysis()