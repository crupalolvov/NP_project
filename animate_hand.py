import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from RPC_Net import RPCNet_Exact
from IKA import HandIKA

def get_calibration_data(csv_path):
    """Estrae i dati di calibrazione (lunghezze falangi e metacarpi) dal CSV Cinematico originale"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CALIB_FILE = os.path.join(BASE_DIR, "hand_calibration.pt")
    
    if not os.path.exists(CALIB_FILE):
        print(f"Errore: File di calibrazione statica '{CALIB_FILE}' non trovato!")
        return None
    
    print("Caricamento della calibrazione statica (hand_calibration.pt)...")
    return torch.load(CALIB_FILE, weights_only=False)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    KIN_CSV_FILE = os.path.join(BASE_DIR, "trial_3_Kinematics.csv")
    TENSOR_FILE = os.path.join(BASE_DIR, "test_tensors.pt")
    MODEL_FILE = os.path.join(BASE_DIR, "rpc_net_weights.pth")

    if not os.path.exists(KIN_CSV_FILE) or not os.path.exists(TENSOR_FILE) or not os.path.exists(MODEL_FILE):
        print("Errore: File mancanti.")
        print(f"Assicurati che {KIN_CSV_FILE}, {TENSOR_FILE} e {MODEL_FILE} esistano nella cartella.")
        return

    print("1. Estrazione calibrazione anatomica...")
    calib_data = get_calibration_data(KIN_CSV_FILE)
    if calib_data is None:
        return
        
    hand_ika = HandIKA(calib_data)

    print("2. Caricamento tensori di test e calcolo predizioni...")
    dataset = torch.load(TENSOR_FILE, weights_only=False)
    X_emg, X_ang, Y_target = dataset['X_emg'], dataset['X_ang'], dataset['Y_target']
    
    model = RPCNet_Exact(in_emg=512, in_ang=192)
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        Y_pred = model(X_emg, X_ang)
        
    y_true = Y_target.numpy()
    y_pred = Y_pred.numpy()

    # Denormalizzazione [0, 1] -> Gradi -> Radianti (richiesti dalla Forward Kinematics)
    y_true_rad = np.radians(y_true * 240.0 - 150.0)
    y_pred_rad = np.radians(y_pred * 240.0 - 150.0)

    print("3. Inizializzazione animazione 3D...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Connessioni tra i landmark (Segmenti dello scheletro)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Pollice
        (0, 5), (5, 6), (6, 7), (7, 8),        # Indice
        (0, 9), (9, 10), (10, 11), (11, 12),   # Medio
        (0, 13), (13, 14), (14, 15), (15, 16), # Anulare
        (0, 17), (17, 18), (18, 19), (19, 20)  # Mignolo
    ]

    # Inizializza linee vuote per ground truth (verde) e predizioni (rosso tratteggiato)
    lines_true = [ax.plot([], [], [], color='green', linewidth=3, label='Reale' if i==0 else "")[0] for i in range(len(connections))]
    lines_pred = [ax.plot([], [], [], color='red', linestyle='dashed', linewidth=3, label='Predetto (RPC-Net)' if i==0 else "")[0] for i in range(len(connections))]
    
    ax.set_title('Confronto Cinematica 3D (Trial 3 - Test)')
    ax.legend()
    # Limiti spazio 3D (~15 cm in ogni direzione attorno al polso)
    ax.set_xlim([-0.15, 0.15])
    ax.set_ylim([-0.15, 0.15])
    ax.set_zlim([-0.15, 0.15])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    step = 2 # Salta un frame ogni 2 per un'animazione fluida (~40 FPS)
    def update(frame):
        lm_true = hand_ika.forward_kinematics(y_true_rad[frame * step])
        lm_pred = hand_ika.forward_kinematics(y_pred_rad[frame * step])
        for i, (p1, p2) in enumerate(connections):
            lines_true[i].set_data(np.array([lm_true[p1, 0], lm_true[p2, 0]]), np.array([lm_true[p1, 1], lm_true[p2, 1]]))
            lines_true[i].set_3d_properties(np.array([lm_true[p1, 2], lm_true[p2, 2]]))
            lines_pred[i].set_data(np.array([lm_pred[p1, 0], lm_pred[p2, 0]]), np.array([lm_pred[p1, 1], lm_pred[p2, 1]]))
            lines_pred[i].set_3d_properties(np.array([lm_pred[p1, 2], lm_pred[p2, 2]]))
        return lines_true + lines_pred

    ani = animation.FuncAnimation(fig, update, frames=len(y_true_rad)//step, interval=25, blit=False)
    plt.show()

if __name__ == "__main__":
    main()