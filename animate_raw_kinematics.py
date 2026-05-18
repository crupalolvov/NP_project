import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Connessioni anatomiche della mano secondo MediaPipe
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pollice
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indice
    (0, 9), (9, 10), (10, 11), (11, 12),   # Medio
    (0, 13), (13, 14), (14, 15), (15, 16), # Anulare
    (0, 17), (17, 18), (18, 19), (19, 20), # Mignolo
    (5, 9), (9, 13), (13, 17)              # Palmo
]

def visualize_raw_csv(csv_path, step=2):
    print(f"Caricamento dati da {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)
    
    # Estraiamo solo un frame ogni 'step' per velocizzare l'animazione (~40 fps)
    df = df.iloc[::step].reset_index(drop=True)
    
    num_frames = len(df)
    print(f"Fotogrammi da animare: {num_frames}")
    
    frames_data = np.zeros((num_frames, 21, 3))
    for i in range(21):
        frames_data[:, i, 0] = df[f'LM_{i}_X'].values
        frames_data[:, i, 1] = df[f'LM_{i}_Y'].values
        frames_data[:, i, 2] = df[f'LM_{i}_Z'].values

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Limiti: X, Y in MediaPipe sono normalizzati da 0 a 1.
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # INVERTITO: Nei video Y=0 è il bordo superiore!
    ax.set_zlim(-0.2, 0.2)
    ax.set_xlabel('X (Larghezza Video)')
    ax.set_ylabel('Y (Altezza Video)')
    ax.set_zlabel('Z (Profondità stimata MP)')
    ax.set_title(f"Raw MediaPipe Kinematics: {os.path.basename(csv_path)}")

    scatter = ax.scatter([], [], [], c='red', s=30, alpha=0.8)
    lines = [ax.plot([], [], [], c='blue', linewidth=2.5)[0] for _ in range(len(CONNECTIONS))]

    def update(frame_idx):
        data = frames_data[frame_idx]
        # Aggiorna i punti
        scatter._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
        # Aggiorna i segmenti (ossa)
        for line, (i, j) in zip(lines, CONNECTIONS):
            line.set_data([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]])
            line.set_3d_properties([data[i, 2], data[j, 2]])
        return [scatter] + lines

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=30, blit=False)
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(BASE_DIR, "recordings", "trial_3_Kinematics.csv")
    if os.path.exists(CSV_FILE):
        visualize_raw_csv(CSV_FILE, step=2)
    else:
        print(f"File non trovato: {CSV_FILE}")