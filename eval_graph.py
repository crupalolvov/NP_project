import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from RPC_Net import RPCNet_Exact

def plot_joint_errors(y_true: np.ndarray, y_pred: np.ndarray, dataset_name=""):
    """
    Calcola e plotta l'RMSE per ogni Grado di Libertà.
    y_true, y_pred: array numpy di forma (n_campioni, 24)
    """
    # Calcolo RMSE per ogni colonna (DoF)
    rmse_per_joint = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
    
    # Calcolo Errore Medio Assoluto (MAE) come metrica secondaria
    mae_per_joint = np.mean(np.abs(y_true - y_pred), axis=0)

    # Configurazione del grafico
    dof_indices = np.arange(24)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_rmse = ax.bar(dof_indices - width/2, rmse_per_joint, width, label='RMSE', color='#1f77b4')
    bars_mae = ax.bar(dof_indices + width/2, mae_per_joint, width, label='MAE', color='#ff7f0e')

    ax.set_xlabel('Indice Grado di Libertà (DoF)')
    ax.set_ylabel('Errore (Unità Normalizzate)')
    ax.set_title(f'Errore di Predizione per ogni Joint - {dataset_name}')
    ax.set_xticks(dof_indices)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Aggiunta del valore sopra ogni barra RMSE per chiarezza
    for bar in bars_rmse:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.show()

def plot_trajectory(y_true: np.ndarray, y_pred: np.ndarray, dof_index: int = 0, dataset_name=""):
    """
    Plotta l'andamento temporale di un singolo DoF per confrontare visivamente
    le predizioni della rete rispetto ai valori reali.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:, dof_index], label='Reale (Ground Truth)', color='green', linewidth=2)
    plt.plot(y_pred[:, dof_index], label='Predetto (RPC-Net)', color='red', linestyle='dashed', linewidth=2)
    plt.title(f'Confronto Traiettoria nel Tempo - DoF {dof_index} ({dataset_name})')
    plt.xlabel('Campioni / Time steps (~80 Hz)')
    plt.ylabel('Angolo Normalizzato [0, 1]')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

def evaluate_tensors_directly(tensor_file, model_file):
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
            
        plot_joint_errors(Y_target.numpy(), Y_pred.numpy(), dataset_name)
        plot_trajectory(Y_target.numpy(), Y_pred.numpy(), dof_index=8, dataset_name=dataset_name)
        
    except FileNotFoundError as e:
        print(f"Errore: {e}. Assicurati che i file esistano.")

def evaluate_offline_inference(pred_csv_file, target_tensor_file):
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
        
        WINDOW_SIZE = 64
        y_pred = y_pred_full[WINDOW_SIZE:]
        
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        print(f"Dati allineati! Campioni validi da confrontare: {min_len}")
        
        plot_joint_errors(y_true, y_pred, dataset_name)
        plot_trajectory(y_true, y_pred, dof_index=8, dataset_name=dataset_name)
        
    except FileNotFoundError as e:
        print(f"Errore: {e}. Assicurati di aver generato '{pred_csv_file}' e '{target_tensor_file}'.")

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_MODELLO = os.path.join(BASE_DIR, "rpc_net_weights.pth")
    
    # =========================================================================
    # MODALITÀ 1: Valutazione diretta sui Tensori (Train, Val e Test)
    # =========================================================================
    datasets_to_evaluate = ["train_tensors.pt", "val_tensors.pt", "test_tensors.pt"]
    
    for ds_name in datasets_to_evaluate:
        file_dataset = os.path.join(BASE_DIR, ds_name)
        if os.path.exists(file_dataset):
            evaluate_tensors_directly(file_dataset, FILE_MODELLO)
    
    # =========================================================================
    # MODALITÀ 2: Valutazione del file CSV prodotto da inference.py
    # =========================================================================
    # FILE_PRED_CSV = os.path.join(BASE_DIR, "predicted_kinematics_offline.csv")
    # FILE_TARGET_PT = os.path.join(BASE_DIR, "test_tensors.pt")
    # evaluate_offline_inference(FILE_PRED_CSV, FILE_TARGET_PT)