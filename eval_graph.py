import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from RPC_Net import RPCNet_Exact

def plot_joint_errors(y_true: np.ndarray, y_pred: np.ndarray, dataset_name="", output_dir="eval"):
    """
    Calcola e plotta l'RMSE, il MAE e la Correlazione di Pearson per ogni Grado di Libertà.
    I valori vengono convertiti da [0, 1] a Gradi Reali.
    """
    # Calcolo RMSE e MAE in unità normalizzate [0, 1]
    rmse_per_joint = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    mae_per_joint = np.mean(np.abs(y_true - y_pred), axis=0)

    # Conversione dell'errore in GRADI REALI (Normalizzazione basata su Range di 240°)
    rmse_deg = rmse_per_joint * 240.0
    mae_deg = mae_per_joint * 240.0
    
    # Calcolo Correlazione di Pearson per ogni joint
    correlations = []
    for i in range(24):
        # Controllo della deviazione standard per evitare divisioni per zero se il target è statico
        if np.std(y_true[:, i]) > 1e-6 and np.std(y_pred[:, i]) > 1e-6:
            corr = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)

    # Stampe di diagnostica globale
    print(f"[{dataset_name}] Errore Medio Assoluto (MAE) globale: {np.mean(mae_deg):.2f}°")
    print(f"[{dataset_name}] Errore Quadratico Medio (RMSE) globale: {np.mean(rmse_deg):.2f}°")
    print(f"[{dataset_name}] Correlazione media (Pearson): {np.nanmean(correlations):.3f}")

    # Configurazione del grafico a barre
    dof_indices = np.arange(24)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_rmse = ax.bar(dof_indices - width/2, rmse_deg, width, label='RMSE (Gradi)', color='#1f77b4')
    bars_mae = ax.bar(dof_indices + width/2, mae_deg, width, label='MAE (Gradi)', color='#ff7f0e')

    ax.set_xlabel('Indice Grado di Libertà (DoF)')
    ax.set_ylabel('Errore Assoluto (Gradi)')
    ax.set_title(f'Errore di Predizione Biomeccanica per ogni Joint - {dataset_name}')
    ax.set_xticks(dof_indices)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Aggiunta del valore in gradi sopra ogni barra RMSE per immediata leggibilità
    for bar in bars_rmse:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}°', ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"joint_errors_{dataset_name.replace('.pt', '').replace('.csv', '').replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Grafico RMSE salvato in: {save_path}")

def plot_trajectory(y_true: np.ndarray, y_pred: np.ndarray, dof_indices: list, dataset_name="", output_dir="eval"):
    """
    Plotta l'andamento temporale di multipli DoF in subplots per analizzare 
    la coerenza del timing e la presenza di crosstalk/sinergie.
    """
    num_plots = len(dof_indices)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3.5 * num_plots), sharex=True)
    
    if num_plots == 1:
        axes = [axes]
        
    for ax, dof_idx in zip(axes, dof_indices):
        ax.plot(y_true[:, dof_idx], label='Reale (Ground Truth)', color='green', linewidth=2)
        ax.plot(y_pred[:, dof_idx], label='Predetto (RPC-Net)', color='red', linestyle='dashed', linewidth=2)
        ax.set_title(f'Traiettoria nel Tempo - DoF {dof_idx}')
        ax.set_ylabel('Normalizzato [0, 1]')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)
        
    axes[-1].set_xlabel('Campioni Temporali (~80 Hz)')
    fig.suptitle(f'Analisi delle Sinergie Articolari ({dataset_name})', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"trajectory_synergy_{dataset_name.replace('.pt', '').replace('.csv', '').replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Grafico traiettorie salvato in: {save_path}")

def evaluate_tensors_directly(tensor_file, model_file, output_dir):
    dataset_name = os.path.basename(tensor_file)
    print(f"\n--- Valutazione Diretta: {dataset_name} ---")
    try:
        dataset = torch.load(tensor_file, weights_only=False)
        X_emg, X_ang, Y_target = dataset['X_emg'], dataset['X_ang'], dataset['Y_target']
        
        model = RPCNet_Exact(in_emg=512, in_ang=192)
        model.load_state_dict(torch.load(model_file, weights_only=True))
        model.eval()
        
        print("Calcolo delle predizioni in corso...")
        with torch.no_grad():
            Y_pred = model(X_emg, X_ang)
            
        plot_joint_errors(Y_target.numpy(), Y_pred.numpy(), dataset_name, output_dir)
        
        # Plot multi-articolare per valutare la distinzione delle dita
        # DoF 0: Polso (o Base) | DoF 8: Indice | DoF 16: Anulare
        plot_trajectory(Y_target.numpy(), Y_pred.numpy(), dof_indices=[0, 8, 16], dataset_name=dataset_name, output_dir=output_dir)
        
    except FileNotFoundError as e:
        print(f"Errore: {e}. Assicurati che i file esistano.")

def evaluate_offline_inference(pred_csv_file, target_tensor_file, output_dir):
    dataset_name = "Offline Inference CSV"
    print(f"\n--- Valutazione da CSV: {dataset_name} ---")
    try:
        print(f"Caricamento predizioni da {pred_csv_file}...")
        df_pred = pd.read_csv(pred_csv_file)
        pred_cols = [f'Pred_DoF_{i}' for i in range(24)]
        y_pred_full = df_pred[pred_cols].values
        
        print(f"Caricamento target reali da {target_tensor_file}...")
        test_data = torch.load(target_tensor_file, weights_only=False)
        y_true = test_data['Y_target'].numpy()
        
        # Rimozione del buffer passivo (0.78s) introdotto dall'inizializzazione nell'inferenza offline
        WINDOW_SIZE = 64
        y_pred = y_pred_full[WINDOW_SIZE:]
        
        # Allineamento in caso di lievi discrepanze nella lunghezza degli array
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        print(f"Dati allineati temporaneamente. Campioni validi da confrontare: {min_len}")
        
        plot_joint_errors(y_true, y_pred, dataset_name, output_dir)
        plot_trajectory(y_true, y_pred, dof_indices=[0, 8, 16], dataset_name=dataset_name, output_dir=output_dir)
        
    except FileNotFoundError as e:
        print(f"Errore: {e}. Assicurati di aver generato '{pred_csv_file}' e '{target_tensor_file}'.")

# --- ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_MODELLO = os.path.join(BASE_DIR, "rpc_net_weights.pth")
    EVAL_DIR = os.path.join(BASE_DIR, "eval")
    
    # =========================================================================
    # MODALITÀ 1: Valutazione diretta sui Tensori (Train, Val e Test)
    # =========================================================================
    datasets_to_evaluate = ["train_tensors.pt", "val_tensors.pt", "test_tensors.pt"]
    
    for ds_name in datasets_to_evaluate:
        file_dataset = os.path.join(BASE_DIR, ds_name)
        if os.path.exists(file_dataset):
            evaluate_tensors_directly(file_dataset, FILE_MODELLO, EVAL_DIR)
    
    # =========================================================================
    # MODALITÀ 2: Valutazione del file CSV prodotto da inference.py
    # =========================================================================
    # Decommenta queste righe se desideri valutare il file esportato dall'inferenza
    # FILE_PRED_CSV = os.path.join(BASE_DIR, "predicted_kinematics_angles.csv")
    # FILE_TARGET_PT = os.path.join(BASE_DIR, "test_tensors.pt")
    # if os.path.exists(FILE_PRED_CSV) and os.path.exists(FILE_TARGET_PT):
    #     evaluate_offline_inference(FILE_PRED_CSV, FILE_TARGET_PT, EVAL_DIR)