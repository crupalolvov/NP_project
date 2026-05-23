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

def visualize_raw_csv(csv_path, csv_path2=None, csv_path3=None, step=1):
    csv_paths = [csv_path, csv_path2, csv_path3]
    labels = ["Original", "Processed", "Third"]
    
    frames_data_list = []
    titles = []
    
    num_frames = float('inf')
    for path, label in zip(csv_paths, labels):
        if path and os.path.exists(path):
            print(f"Caricamento dati da {os.path.basename(path)}...")
            df = pd.read_csv(path)
            df = df.iloc[::step].reset_index(drop=True)
            num_frames = min(num_frames, len(df))
            
            data = np.zeros((len(df), 21, 3))
            for i in range(21):
                data[:, i, 0] = df[f'LM_{i}_X'].values
                data[:, i, 1] = df[f'LM_{i}_Y'].values
                data[:, i, 2] = df[f'LM_{i}_Z'].values
            frames_data_list.append(data)
            titles.append(f"{label}: {os.path.basename(path)}")
            
    num_plots = len(frames_data_list)
    if num_plots == 0:
        print("Nessun dato da visualizzare.")
        return
        
    print(f"Fotogrammi da animare: {num_frames}")

    # ================= SETUP FIGURE =================
    fig1 = plt.figure(figsize=(6 * num_plots, 8))
    fig1.canvas.manager.set_window_title("Animazione Kinematics")
    fig1.subplots_adjust(bottom=0.2) # Creiamo spazio in basso per lo slider
    
    axes = []
    scatters = []
    lines_list = []
    
    colors_scatter = ['red', 'green', 'purple']
    colors_lines = ['blue', 'orange', 'cyan']

    for i in range(num_plots):
        ax = fig1.add_subplot(1, num_plots, i + 1, projection='3d')
        
        # Limiti: X, Y in MediaPipe sono normalizzati da 0 a 1.
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0) # INVERTITO: Nei video Y=0 è il bordo superiore!
        ax.set_zlim(-0.2, 0.2)
        ax.set_box_aspect([1, 1, 1]) # Forza le proporzioni ad essere identiche
        ax.set_xlabel('X (Larghezza Video)')
        ax.set_ylabel('Y (Altezza Video)')
        ax.set_zlabel('Z (Profondità stimata MP)')
        ax.set_title(titles[i])
        
        scatter = ax.scatter([], [], [], c=colors_scatter[i], s=30, alpha=0.8)
        lines = [ax.plot([], [], [], c=colors_lines[i], linewidth=2.5)[0] for _ in range(len(CONNECTIONS))]
        
        axes.append(ax)
        scatters.append(scatter)
        lines_list.append(lines)

    if num_plots > 1:
        def sync_axes(event):
            if event.inaxes in axes:
                source_ax = event.inaxes
                for ax in axes:
                    if ax != source_ax:
                        needs_update = False
                        if ax.elev != source_ax.elev or ax.azim != source_ax.azim:
                            ax.view_init(elev=source_ax.elev, azim=source_ax.azim)
                            needs_update = True
                        if hasattr(ax, 'dist') and hasattr(source_ax, 'dist') and ax.dist != source_ax.dist:
                            ax.dist = source_ax.dist
                            needs_update = True
                        if needs_update:
                            fig1.canvas.draw_idle()
            
        fig1.canvas.mpl_connect('motion_notify_event', sync_axes)
        fig1.canvas.mpl_connect('scroll_event', sync_axes) # Sincronizza anche lo zoom

    def update(frame_idx):
        idx = int(frame_idx)
        drawn_artists = []
        for i in range(num_plots):
            data = frames_data_list[i][idx]
            scatters[i]._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
            for line, (u, v) in zip(lines_list[i], CONNECTIONS):
                line.set_data([data[u, 0], data[v, 0]], [data[u, 1], data[v, 1]])
                line.set_3d_properties([data[u, 2], data[v, 2]])
            drawn_artists.append(scatters[i])
            drawn_artists.extend(lines_list[i])
        return drawn_artists

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
        drawn_artists = []
        for i in range(num_plots):
            drawn_artists.append(scatters[i])
            drawn_artists.extend(lines_list[i])
        return drawn_artists

    ani = animation.FuncAnimation(fig1, animate, frames=frame_generator, interval=30, blit=False, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE_1 = os.path.join(BASE_DIR, "recordings", "trial_3_Kinematics.csv") 
    CSV_FILE_2 = os.path.join(BASE_DIR, "recordings", "trial_3_Kinematics_preprocessed.csv")
    CSV_FILE_3 = os.path.join(BASE_DIR, "predicted_kinematics_lms.csv")


    if os.path.exists(CSV_FILE_1):
        visualize_raw_csv(CSV_FILE_1, csv_path2=CSV_FILE_2, csv_path3=CSV_FILE_3, step=2)
    else:
        print(f"File non trovato: {CSV_FILE_1}")