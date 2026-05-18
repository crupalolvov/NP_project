import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
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

def visualize_raw_csv(csv_path, csv_path2=None, step=1):
    print(f"Caricamento dati da {os.path.basename(csv_path)}...")
    df1 = pd.read_csv(csv_path)
    
    # Estraiamo solo un frame ogni 'step' per velocizzare l'animazione (~40 fps)
    df1 = df1.iloc[::step].reset_index(drop=True)
    
    num_frames = len(df1)
    
    frames_data1 = np.zeros((num_frames, 21, 3))
    for i in range(21):
        frames_data1[:, i, 0] = df1[f'LM_{i}_X'].values
        frames_data1[:, i, 1] = df1[f'LM_{i}_Y'].values
        frames_data1[:, i, 2] = df1[f'LM_{i}_Z'].values

    frames_data2 = None
    if csv_path2 and os.path.exists(csv_path2):
        print(f"Caricamento dati secondari da {os.path.basename(csv_path2)}...")
        df2 = pd.read_csv(csv_path2)
        df2 = df2.iloc[::step].reset_index(drop=True)
        num_frames = min(num_frames, len(df2))
        frames_data2 = np.zeros((len(df2), 21, 3))
        for i in range(21):
            frames_data2[:, i, 0] = df2[f'LM_{i}_X'].values
            frames_data2[:, i, 1] = df2[f'LM_{i}_Y'].values
            frames_data2[:, i, 2] = df2[f'LM_{i}_Z'].values

    print(f"Fotogrammi da animare: {num_frames}")

    # ================= FIGURA 1 =================
    fig1 = plt.figure(figsize=(8, 8))
    fig1.canvas.manager.set_window_title(f"Animazione 1: {os.path.basename(csv_path)}")
    fig1.subplots_adjust(bottom=0.2) # Creiamo spazio in basso per lo slider
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # Limiti: X, Y in MediaPipe sono normalizzati da 0 a 1.
    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 0)  # INVERTITO: Nei video Y=0 è il bordo superiore!
    ax1.set_zlim(-0.2, 0.2)
    ax1.set_xlabel('X (Larghezza Video)')
    ax1.set_ylabel('Y (Altezza Video)')
    ax1.set_zlabel('Z (Profondità stimata MP)')
    ax1.set_title(f"Kinematics: {os.path.basename(csv_path)}")

    scatter1 = ax1.scatter([], [], [], c='red', s=30, alpha=0.8)
    lines1 = [ax1.plot([], [], [], c='blue', linewidth=2.5)[0] for _ in range(len(CONNECTIONS))]

    # ================= FIGURA 2 =================
    fig2 = None
    scatter2 = None
    lines2 = None
    if frames_data2 is not None:
        fig2 = plt.figure(figsize=(8, 8))
        fig2.canvas.manager.set_window_title(f"Animazione 2: {os.path.basename(csv_path2)}")
        ax2 = fig2.add_subplot(111, projection='3d')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(1, 0)
        ax2.set_zlim(-0.2, 0.2)
        ax2.set_xlabel('X (Larghezza Video)')
        ax2.set_ylabel('Y (Altezza Video)')
        ax2.set_zlabel('Z (Profondità stimata MP)')
        ax2.set_title(f"Kinematics: {os.path.basename(csv_path2)}")

        scatter2 = ax2.scatter([], [], [], c='green', s=30, alpha=0.8)
        lines2 = [ax2.plot([], [], [], c='orange', linewidth=2.5)[0] for _ in range(len(CONNECTIONS))]

    def update(frame_idx):
        idx = int(frame_idx)
        data1 = frames_data1[idx]
        # Aggiorna Figura 1
        scatter1._offsets3d = (data1[:, 0], data1[:, 1], data1[:, 2])
        for line, (i, j) in zip(lines1, CONNECTIONS):
            line.set_data([data1[i, 0], data1[j, 0]], [data1[i, 1], data1[j, 1]])
            line.set_3d_properties([data1[i, 2], data1[j, 2]])
            
        # Aggiorna Figura 2
        if fig2 is not None:
            data2 = frames_data2[idx]
            scatter2._offsets3d = (data2[:, 0], data2[:, 1], data2[:, 2])
            for line, (i, j) in zip(lines2, CONNECTIONS):
                line.set_data([data2[i, 0], data2[j, 0]], [data2[i, 1], data2[j, 1]])
                line.set_3d_properties([data2[i, 2], data2[j, 2]])
            try:
                fig2.canvas.draw_idle()
            except:
                pass
            
        return [scatter1] + lines1

    # Aggiunta Slider e Pulsante solo su Figura 1
    ax_slider = fig1.add_axes([0.15, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)
    fig1.slider = slider  # Previene la garbage collection

    ax_button = fig1.add_axes([0.85, 0.05, 0.1, 0.03])
    btn_play = Button(ax_button, 'Pause')
    fig1.btn_play = btn_play

    is_playing = [True]

    def on_slider_change(val):
        update(val)
        fig1.canvas.draw_idle()

    slider.on_changed(on_slider_change)

    def toggle_play(event):
        is_playing[0] = not is_playing[0]
        if is_playing[0]:
            btn_play.label.set_text('Pause')
            ani.event_source.start()
        else:
            btn_play.label.set_text('Play')
            ani.event_source.stop()
        fig1.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    def frame_generator():
        while True:
            yield (slider.val + 1) % num_frames

    def animate(frame_idx):
        slider.set_val(frame_idx)
        return [scatter1] + lines1

    ani = animation.FuncAnimation(fig1, animate, frames=frame_generator, interval=30, blit=False, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE_1 = os.path.join(BASE_DIR, "recordings", "trial_3_Kinematics.csv")
    CSV_FILE_2 = os.path.join(BASE_DIR, "recordings", "trial_3_Kinematics_IKA_predicted_lms.csv")
    #CSV_FILE_2 = os.path.join(BASE_DIR, "recordings", "trial_3_Kinematics_preprocessed.csv")

    if os.path.exists(CSV_FILE_1):
        visualize_raw_csv(CSV_FILE_1, csv_path2=CSV_FILE_2, step=2)
    else:
        print(f"File non trovato: {CSV_FILE_1}")