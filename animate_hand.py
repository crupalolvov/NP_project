import sys
import os
import numpy as np
import pandas as pd
import torch
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
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

        # Visualizzatore 3D (pyqtgraph opengl)
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=0.3, elevation=30, azimuth=45)
        self.view.setBackgroundColor('k')
        # Griglia di riferimento
        grid = gl.GLGridItem()
        grid.scale(0.1, 0.1, 0.1)
        self.view.addItem(grid)
        layout.addWidget(self.view, stretch=1)

        # Legenda Colori
        legend_layout = QtWidgets.QHBoxLayout()
        legend_layout.addWidget(QtWidgets.QLabel("<b style='color:green;'>■ Reale (Ground Truth)</b>"))
        legend_layout.addWidget(QtWidgets.QLabel("<b style='color:red;'>■ Predetto (RPC-Net)</b>"))
        legend_layout.addWidget(QtWidgets.QLabel("<b style='color:yellow;'>■ Polso (Origine)</b>"))
        legend_layout.addStretch()
        layout.addLayout(legend_layout)

        # Connessioni landmark (come nel vecchio script)
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
            (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        # Inizializzazione oggetti grafici per la mano REALE (Verde)
        self.points_true = gl.GLScatterPlotItem(size=8, pxMode=True)
        self.lines_true = [gl.GLLinePlotItem(width=3, antialias=True, mode='lines') for _ in self.connections]
        self.view.addItem(self.points_true)
        for line in self.lines_true: self.view.addItem(line)

        # Inizializzazione oggetti grafici per la mano PREDETTA (Rossa)
        self.points_pred = gl.GLScatterPlotItem(size=6, pxMode=True)
        self.lines_pred = [gl.GLLinePlotItem(width=2, antialias=True, mode='lines') for _ in self.connections]
        self.view.addItem(self.points_pred)
        for line in self.lines_pred: self.view.addItem(line)

        # Marcatori speciali per il polso (Landmark 0)
        self.wrist_marker = gl.GLScatterPlotItem(size=15, pxMode=True)
        self.view.addItem(self.wrist_marker)

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

        # Update Punti (Scatter)
        # Colore landmark: verde per reale, rosso per predetto
        colors_true = np.array([[0, 1, 0, 0.8]] * 21)
        colors_pred = np.array([[1, 0, 0, 0.7]] * 21)
        
        self.points_true.setData(pos=lm_true, color=colors_true)
        self.points_pred.setData(pos=lm_pred, color=colors_pred)

        # Update Marcatori Polso (Landmark 0) - Lo rendiamo giallo e grande
        wrist_pos = np.array([lm_true[0]])
        self.wrist_marker.setData(pos=wrist_pos, color=np.array([[1, 1, 0, 1]]))

        # Update Linee (Ossa)
        for i, (p1, p2) in enumerate(self.connections):
            # Mano Reale
            pts_t = np.array([lm_true[p1], lm_true[p2]])
            self.lines_true[i].setData(pos=pts_t, color=(0, 1, 0, 1))
            
            # Mano Predetta
            pts_p = np.array([lm_pred[p1], lm_pred[p2]])
            self.lines_pred[i].setData(pos=pts_p, color=(1, 0, 0, 0.5))

    def toggle_play(self):
        if self.is_playing:
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start(25) # ~40 FPS
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