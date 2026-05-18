import sys
import os
import numpy as np
import pandas as pd
import torch
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from RPC_Net import RPCNet_Exact
from IKA import HandIKA

def get_calibration_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CALIB_FILE = os.path.join(BASE_DIR, "hand_calibration.pt")
    if not os.path.exists(CALIB_FILE):
        print(f"Errore: File di calibrazione statica '{CALIB_FILE}' non trovato!")
        return None
    print("Caricamento della calibrazione statica (hand_calibration.pt)...")
    return torch.load(CALIB_FILE, weights_only=False)


class HandAnimationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NP_project - Monitor Cinematica 3D")
        self.resize(1024, 768)

        # Configurazione percorsi
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.tensor_file = os.path.join(self.base_dir, "test_tensors.pt")
        self.model_file = os.path.join(self.base_dir, "rpc_net_weights.pth")

        # Parametri stato
        self.current_frame = 0
        self.is_playing = False

        # Caricamento dati e inizializzazione IKA
        calib_data = get_calibration_data()
        if calib_data is None: sys.exit(1)
        self.hand_ika = HandIKA(calib_data)

        self.load_and_predict()
        self.init_ui()
        
        # Timer per l'animazione (40 Hz circa)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

    def load_and_predict(self):
        print("Caricamento modelli e calcolo predizioni...")
        if not os.path.exists(self.tensor_file) or not os.path.exists(self.model_file):
            QtWidgets.QMessageBox.critical(self, "Errore", "File .pt o .pth mancanti!")
            sys.exit(1)

        dataset = torch.load(self.tensor_file, weights_only=False)
        X_emg, X_ang, Y_target = dataset['X_emg'], dataset['X_ang'], dataset['Y_target']
        
        model = RPCNet_Exact(in_emg=512, in_ang=192)
        model.load_state_dict(torch.load(self.model_file, weights_only=True))
        model.eval()

        with torch.no_grad():
            Y_pred = model(X_emg, X_ang)

        # Pre-calcolo delle posizioni 3D per massimizzare la fluidità dello slider
        self.y_true_rad = np.radians(Y_target.numpy() * 240.0 - 150.0)
        self.y_pred_rad = np.radians(Y_pred.numpy() * 240.0 - 150.0)
        self.num_frames = len(self.y_true_rad)

        # Verifica dinamica dei dati caricati
        rom = np.degrees(np.max(self.y_true_rad, axis=0) - np.min(self.y_true_rad, axis=0))
        print(f"Range di movimento rilevato (Gradi): Max {np.max(rom):.1f}°, Min {np.min(rom):.1f}°")
        if np.max(rom) < 5.0:
            print("⚠️ ATTENZIONE: I dati originali sembrano quasi statici (< 5 gradi di movimento).")
        else:
            print("✅ Dati dinamici caricati correttamente.")

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Visualizzatore 3D (Matplotlib integrato in PyQt5)
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas, stretch=1)
        
        # Configurazione degli assi e della prospettiva
        self.ax.set_xlim(-0.2, 0.2)
        self.ax.set_ylim(-0.2, 0.2)
        self.ax.invert_yaxis() # Invertito: nei video la Y scende verso il basso
        self.ax.set_zlim(-0.2, 0.2)
        self.ax.set_box_aspect([1, 1, 1]) # Mantiene le proporzioni cubiche per non deformare la mano
        self.ax.set_xlabel('X (Metri)')
        self.ax.set_ylabel('Y (Metri)')
        self.ax.set_zlabel('Z (Metri)')
        self.ax.set_title('Monitor Cinematica 3D - RPC-Net')

        # Connessioni landmark (come nel vecchio script)
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
            (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        # Inizializzazione oggetti grafici per la mano REALE e PREDETTA
        self.scatter_true = self.ax.scatter([], [], [], c='green', s=40, label='Reale (Ground Truth)', alpha=0.7)
        self.lines_true = [self.ax.plot([], [], [], c='green', linewidth=2.5, alpha=0.5)[0] for _ in self.connections]

        self.scatter_pred = self.ax.scatter([], [], [], c='red', s=40, label='Predetto (RPC-Net)', alpha=0.9)
        self.lines_pred = [self.ax.plot([], [], [], c='red', linewidth=2.5, linestyle='--')[0] for _ in self.connections]
        
        self.ax.legend(loc='upper right')

        # Controlli (Slider e Pulsanti)
        ctrl_layout = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        ctrl_layout.addWidget(self.btn_play)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.num_frames - 1)
        self.slider.valueChanged.connect(self.update_animation)
        ctrl_layout.addWidget(self.slider)

        self.lbl_frame = QtWidgets.QLabel(f"Frame: 0 / {self.num_frames}")
        ctrl_layout.addWidget(self.lbl_frame)
        
        layout.addLayout(ctrl_layout)
        
        # Disegna il primo frame
        self.update_animation(0)

    def update_animation(self, frame_idx):
        self.current_frame = frame_idx
        self.lbl_frame.setText(f"Frame: {frame_idx} / {self.num_frames}")
        
        # Calcolo Forward Kinematics
        lm_true = self.hand_ika.forward_kinematics(self.y_true_rad[frame_idx])
        lm_pred = self.hand_ika.forward_kinematics(self.y_pred_rad[frame_idx])

        # Debug: verifica se le coordinate cambiano nel tempo
        if frame_idx % 50 == 0:
            print(f"Frame {frame_idx}: Posizione Indice Reale {lm_true[8][0]:.4f}, Predetta {lm_pred[8][0]:.4f}")

        # Update Punti (Scatter 3D)
        self.scatter_true._offsets3d = (lm_true[:, 0], lm_true[:, 1], lm_true[:, 2])
        self.scatter_pred._offsets3d = (lm_pred[:, 0], lm_pred[:, 1], lm_pred[:, 2])

        # Update Linee (Ossa)
        for i, (p1, p2) in enumerate(self.connections):
            # Mano Reale
            self.lines_true[i].set_data([lm_true[p1, 0], lm_true[p2, 0]], [lm_true[p1, 1], lm_true[p2, 1]])
            self.lines_true[i].set_3d_properties([lm_true[p1, 2], lm_true[p2, 2]])
            
            # Mano Predetta
            self.lines_pred[i].set_data([lm_pred[p1, 0], lm_pred[p2, 0]], [lm_pred[p1, 1], lm_pred[p2, 1]])
            self.lines_pred[i].set_3d_properties([lm_pred[p1, 2], lm_pred[p2, 2]])

        self.canvas.draw_idle()

    def toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start(30) # ~33 FPS
            self.btn_play.setText("Pause")
        self.is_playing = not self.is_playing

    def next_frame(self):
        if self.current_frame < self.num_frames - 1:
            self.slider.setValue(self.current_frame + 1)
        else:
            self.toggle_play() # Stop alla fine

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = HandAnimationApp()
    window.show()
    sys.exit(app.exec_())