import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def plot_joint_errors(y_true: np.ndarray, y_pred: np.ndarray):
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
    ax.set_title('Errore di Predizione per ogni Joint (Trial 3)')
    ax.set_xticks(dof_indices)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Aggiunta del valore sopra ogni barra RMSE per chiarezza
    for bar in bars_rmse:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.show()

def plot_trajectory(y_true: np.ndarray, y_pred: np.ndarray, dof_index: int = 0):
    """
    Plotta l'andamento temporale di un singolo DoF per confrontare visivamente
    le predizioni della rete rispetto ai valori reali.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(y_true[:, dof_index], label='Reale (Ground Truth)', color='green', linewidth=2)
    plt.plot(y_pred[:, dof_index], label='Predetto (RPC-Net)', color='red', linestyle='dashed', linewidth=2)
    plt.title(f'Confronto Traiettoria nel Tempo - DoF {dof_index}')
    plt.xlabel('Campioni / Time steps (~80 Hz)')
    plt.ylabel('Angolo Normalizzato [0, 1]')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    FILE_PRED = "NP_project/predicted_kinematics_offline.csv"
    FILE_TARGET = "NP_project/test_tensors.pt"  # Target Ground Truth generato dal Trial 3
    
    try:
        print(f"Caricamento predizioni da {FILE_PRED}...")
        df_pred = pd.read_csv(FILE_PRED)
        pred_cols = [f'Pred_DoF_{i}' for i in range(24)]
        y_pred_full = df_pred[pred_cols].values
        
        print(f"Caricamento target reali da {FILE_TARGET}...")
        test_data = torch.load(FILE_TARGET)
        y_true = test_data['Y_target'].numpy()
        
        # Allineamento temporale (salta il buffer iniziale di 64 samples dell'inferenza)
        WINDOW_SIZE = 64
        y_pred = y_pred_full[WINDOW_SIZE:]
        
        # Allineamento di sicurezza per le lunghezze
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        print(f"Dati allineati! Campioni validi da confrontare: {min_len}")
        
        # 1. Grafico a barre dell'errore globale (RMSE e MAE per DoF)
        plot_joint_errors(y_true, y_pred)
        
        # 2. Grafico temporale: modifica 'dof_index' per ispezionare altri giunti!
        # Esempio: 0,1,2 = Polso | 8 = MCP Indice
        plot_trajectory(y_true, y_pred, dof_index=8)
        
    except FileNotFoundError as e:
        print(f"Errore: {e}. Assicurati di aver generato '{FILE_PRED}' e '{FILE_TARGET}'.")