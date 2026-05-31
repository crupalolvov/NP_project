import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pollice
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indice
    (0, 9), (9, 10), (10, 11), (11, 12),   # Medio
    (0, 13), (13, 14), (14, 15), (15, 16), # Anulare
    (0, 17), (17, 18), (18, 19), (19, 20), # Mignolo
    (5, 9), (9, 13), (13, 17)              # Base del palmo
]

def detect_coordinate_format(df):
    min_x = df[[f'LM_{i}_X' for i in range(21)]].min().min()
    min_y = df[[f'LM_{i}_Y' for i in range(21)]].min().min()
    if min_x < -0.01 or min_y < -0.01 or df['LM_0_X'].abs().max() < 1e-4:
        return True 
    return False 

def load_and_resample_synced(path, target_time):
    df = pd.read_csv(path)
    if 'Timestamp_LSL' in df.columns:
        t_original = df['Timestamp_LSL'].values
    else:
        t_original = np.arange(len(df)) * (1/80.0)
        
    data_resampled = np.zeros((len(target_time), 21, 3))
    for i in range(21):
        f_x = interp1d(t_original, df[f'LM_{i}_X'].values, kind='linear', bounds_error=False, fill_value="extrapolate")
        f_y = interp1d(t_original, df[f'LM_{i}_Y'].values, kind='linear', bounds_error=False, fill_value="extrapolate")
        f_z = interp1d(t_original, df[f'LM_{i}_Z'].values, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        data_resampled[:, i, 0] = f_x(target_time)
        data_resampled[:, i, 1] = f_y(target_time)
        data_resampled[:, i, 2] = f_z(target_time)
        
    return data_resampled

def visualize_synced_csv(csv_paths, target_fps=80.0):
    valid_paths = [p for p in csv_paths if p and os.path.exists(p)]
    if not valid_paths:
        return
        
    all_starts = []
    all_ends = []
    for p in valid_paths:
        df = pd.read_csv(p)
        if 'Timestamp_LSL' in df.columns:
            all_starts.append(df['Timestamp_LSL'].iloc[0])
            all_ends.append(df['Timestamp_LSL'].iloc[-1])
        else:
            all_starts.append(0)
            all_ends.append((len(df)-1)/80.0)
        
    t_start = max(all_starts) 
    t_end = min(all_ends)     
    
    if t_end <= t_start: return
        
    target_time = np.arange(t_start, t_end, 1.0 / target_fps)
    num_frames = len(target_time)

    frames_data_list = []
    titles = []
    labels_all = ["Original", "Processed", "Predicted"]
    
    first_df = pd.read_csv(valid_paths[0])
    is_metric = detect_coordinate_format(first_df)
    
    offset_step = 0.25 if is_metric else 1.0

    # Caricamento e allineamento spaziale (Zeroing)
    for i, (p, label) in enumerate(zip(valid_paths, labels_all)):
        data = load_and_resample_synced(p, target_time)
        
        # Centriamo tutta l'animazione basandoci sulla posizione iniziale del polso.
        # In questo modo si annullano eventuali offset generati dalla rete neurale.
        wrist_origin = data[0, 0, :].copy() 
        for f in range(num_frames):
            data[f] -= wrist_origin           # Trasla in 0,0,0
            data[f][:, 0] += (i * offset_step) # Applica la spaziatura orizzontale perfetta
            
        frames_data_list.append(data)
        titles.append(f"{label}: {os.path.basename(p)}")

    num_plots = len(frames_data_list)
    colors_scatter = ['#e41a1c', '#4daf4a', '#984ea3']
    colors_lines = ['#b30000', '#006d2c', '#54278f']

    # ================= SETUP FIGURA OTTIMIZZATO =================
    # Usiamo una figura molto larga (16:6) per accogliere l'asse X allungato
    fig = plt.figure(figsize=(16, 6))
    fig.canvas.manager.set_window_title("Kinematics - Vista Anatomica Centrata")
    
    # Riduciamo drasticamente i margini bianchi spingendoli oltre i bordi della finestra
    fig.subplots_adjust(left=-0.1, right=1.1, bottom=0.15, top=1.1)
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    if is_metric:
        x_min, x_max = -0.15, offset_step * (num_plots - 1) + 0.15
        y_min, y_max = -0.20, 0.20
        z_min, z_max = -0.20, 0.20
        ax.set_xlabel('X (Metri)')
        ax.set_ylabel('Y (Metri)')
        ax.set_zlabel('Z (Metri)')
    else:
        x_min, x_max = -0.2, offset_step * (num_plots - 1) + 0.5
        y_min, y_max = -0.5, 0.5
        z_min, z_max = -0.2, 0.2

    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(y_max, y_min)
    ax.set_zlim3d(z_min, z_max)
    
    # Manteniamo le proporzioni corrette nello spazio
    ax.set_box_aspect((abs(x_max - x_min), abs(y_max - y_min), abs(z_max - z_min)))
    
    ax.view_init(elev=25, azim=-55)
    
    # Zoom forzato della camera (default 10). Più è basso, più il grafico riempie lo schermo
    ax.dist = 6.5 

    scatters = []
    lines_list = []
    palms = []

    marker_sizes = [60] + [30]*4 + [30]*4 + [30]*4 + [30]*4 + [30]*4
    marker_sizes[4] = marker_sizes[8] = marker_sizes[12] = marker_sizes[16] = marker_sizes[20] = 15

    for i in range(num_plots):
        # I dati sono già stati traslati e spaziati durante il pre-processamento
        init_data = frames_data_list[i][0]
        
        # Nodi
        scatter = ax.scatter(init_data[:, 0], init_data[:, 1], init_data[:, 2], 
                             c=colors_scatter[i], s=marker_sizes, alpha=0.9, label=labels_all[i])
        
        # Linee
        lines = []
        for u, v in CONNECTIONS:
            line = ax.plot([init_data[u, 0], init_data[v, 0]], 
                           [init_data[u, 1], init_data[v, 1]], 
                           [init_data[u, 2], init_data[v, 2]], 
                           c=colors_lines[i], linewidth=3, solid_capstyle='round')[0]
            lines.append(line)
        
        # Palmo
        palm_verts = [init_data[0], init_data[5], init_data[9], init_data[13], init_data[17]]
        palm = Poly3DCollection([palm_verts], alpha=0.25, facecolor=colors_scatter[i], edgecolor='none')
        ax.add_collection3d(palm)
        
        scatters.append(scatter)
        lines_list.append(lines)
        palms.append(palm)
        
    # Posizioniamo la legenda in un punto che non intralcia
    ax.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    def update(frame_idx):
        idx = int(frame_idx)
        drawn_artists = []
        for i in range(num_plots):
            data = frames_data_list[i][idx] # Il dato è già pronto all'uso
            
            scatters[i]._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
            
            for line, (u, v) in zip(lines_list[i], CONNECTIONS):
                line.set_data([data[u, 0], data[v, 0]], [data[u, 1], data[v, 1]])
                line.set_3d_properties([data[u, 2], data[v, 2]])
                
            palm_verts = [data[0], data[5], data[9], data[13], data[17]]
            palms[i].set_verts([palm_verts])
            
            drawn_artists.append(scatters[i])
            drawn_artists.extend(lines_list[i])
            drawn_artists.append(palms[i])
            
        slider.valtext.set_text(f"{idx * (1.0/target_fps):.2f} s")
        return drawn_artists

    # Controlli UI 
    ax_slider = fig.add_axes([0.15, 0.05, 0.60, 0.03])
    slider = Slider(ax_slider, 'Time', 0, num_frames - 1, valinit=0, valstep=1)
    
    ax_button = fig.add_axes([0.80, 0.05, 0.1, 0.03])
    btn_play = Button(ax_button, 'Pause')
    is_playing = [True]

    slider.on_changed(lambda val: (update(val), fig.canvas.draw_idle()))

    def toggle_play(event=None):
        is_playing[0] = not is_playing[0]
        if is_playing[0]:
            btn_play.label.set_text('Pause')
            fig.ani.event_source.start()
        else:
            btn_play.label.set_text('Play')
            fig.ani.event_source.stop()
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    def on_key(event):
        if event.key == 'right':
            if is_playing[0]: toggle_play()
            slider.set_val((slider.val + 1) % num_frames)
        elif event.key == 'left':
            if is_playing[0]: toggle_play()
            slider.set_val((slider.val - 1) % num_frames)
            
    fig.canvas.mpl_connect('key_press_event', on_key)

    def frame_generator():
        while True:
            yield (slider.val + 1) % num_frames

    def animate(frame_idx):
        slider.set_val(frame_idx)
        drawn_artists = []
        for i in range(num_plots):
            drawn_artists.append(scatters[i])
            drawn_artists.extend(lines_list[i])
            drawn_artists.append(palms[i])
        return drawn_artists

    fig.ani = animation.FuncAnimation(fig, animate, frames=frame_generator, 
                                      interval=1000/target_fps, blit=False, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRIAL = 15
    
    CSV_FILE_1 = os.path.join(BASE_DIR, "recordings", f"trial_{TRIAL}_Kinematics.csv") 
    CSV_FILE_2 = os.path.join(BASE_DIR, "recordings", f"trial_{TRIAL}_Kinematics_preprocessed.csv")
    CSV_FILE_3 = os.path.join(BASE_DIR, f"trial_{TRIAL}_predicted_kinematics_lms.csv")

    if os.path.exists(CSV_FILE_1):
        visualize_synced_csv([CSV_FILE_1, CSV_FILE_2, CSV_FILE_3], target_fps=25.0)
    else:
        print(f"Errore: File originale {CSV_FILE_1} non trovato.")