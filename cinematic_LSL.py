import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylsl import StreamInfo, StreamOutlet
from angles_kalman import SimpleKalmanFilter

# ==========================================
# 1. SETUP LAB STREAMING LAYER (LSL)
# ==========================================
# 21 landmark * 3 coordinate (x, y, z) = 63 canali continui
LSL_CHANNELS = 63
LSL_FPS = 0 # 0 indica un rate irregolare (dipende dai frame elaborati dalla webcam)
info = StreamInfo('MediaPipe_Kinematics', 'Kinematics', LSL_CHANNELS, LSL_FPS, 'float32', 'mediapipe_hand_01')
outlet = StreamOutlet(info)

# ==========================================
# 2. SETUP MEDIAPIPE HAND LANDMARKER
# ==========================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Variabili globali per mantenere lo stato
latest_landmarks = None
latest_filtered_landmarks = None

# Inizializzazione dei Filtri di Kalman per le coordinate dei 21 landmark
Q = 1e-1  # Process variance (AUMENTATA per renderlo più reattivo)
R = 1e-2  # Measurement variance (DIMINUITA per fidarsi di più del sensore)
kf_x = [SimpleKalmanFilter(Q, R) for _ in range(21)]
kf_y = [SimpleKalmanFilter(Q, R) for _ in range(21)]
kf_z = [SimpleKalmanFilter(Q, R) for _ in range(21)]
is_first_frame = True

# Lista per salvare i dati per il CSV
kinematics_data = []
kinematics_raw_data = []  # Lista per salvare i dati grezzi per il grafico

def update_result(result, output_image, timestamp_ms):
    """Callback asincrona: estrae le coordinate 3D, le filtra via Kalman e le invia via LSL."""
    global latest_landmarks, latest_filtered_landmarks, is_first_frame, kinematics_data
    
    if result.hand_landmarks:
        # Per questo setup, estraiamo i dati della prima mano rilevata
        latest_landmarks = result.hand_landmarks[0]
        
        # LSL richiede il timestamp in secondi
        lsl_timestamp = timestamp_ms / 1000.0 
        
        flattened_coords = []
        flattened_raw_coords = []
        current_filtered = []
        
        for i, lm in enumerate(latest_landmarks):
            if is_first_frame:
                kf_x[i].estimated_value = lm.x
                kf_y[i].estimated_value = lm.y
                kf_z[i].estimated_value = lm.z
                
            filtered_x = kf_x[i].update(lm.x)
            filtered_y = kf_y[i].update(lm.y)
            filtered_z = kf_z[i].update(lm.z)
            
            flattened_coords.extend([filtered_x, filtered_y, filtered_z])
            flattened_raw_coords.extend([lm.x, lm.y, lm.z])
            current_filtered.append((filtered_x, filtered_y, filtered_z))
            
        if is_first_frame:
            is_first_frame = False
            
        latest_filtered_landmarks = current_filtered
        
        # Push del campione in rete
        outlet.push_sample(flattened_coords, lsl_timestamp)
        
        # Salvataggio per il CSV e per il plot
        kinematics_data.append([lsl_timestamp] + flattened_coords)
        kinematics_raw_data.append([lsl_timestamp] + flattened_raw_coords)

# Configurazione (Assicurati di avere il file .task scaricato nella directory)
# Link per il download: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_result,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ==========================================
# 3. UTILITIES PER CINEMATICA (FASE 2 OFFLINE)
# ==========================================
def calculate_joint_angle(p1, p2, p3):
    """
    Calcola l'angolo 3D tra tre punti (es. nocca, articolazione, punta).
    Restituisce l'angolo in gradi utilizzando il prodotto scalare.
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
        
    cosine_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# ==========================================
# 4. MAIN LOOP OPENCV
# ==========================================
cap = cv2.VideoCapture(0)
prev_frame_time = 0

print("Avvio streaming LSL 'MediaPipe_Kinematics'...")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Timestamp ad alta precisione per MediaPipe e LSL
        frame_timestamp_ms = int(time.perf_counter() * 1000)
        
        # Esecuzione inferenza asincrona (attiva la callback 'update_result')
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Rendering essenziale per feedback visivo
        if latest_landmarks:
            h, w = frame.shape[:2]
            # Disegna i punti GREZZI in ROSSO (Raw)
            for lm in latest_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                
        if latest_filtered_landmarks:
            h, w = frame.shape[:2]
            # Disegna i punti FILTRATI in VERDE (Kalman)
            for (fx, fy, fz) in latest_filtered_landmarks:
                px, py = int(fx * w), int(fy * h)
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        # Calcolo e rendering FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        cv2.putText(frame, f"FPS: {int(fps)} | LSL Streaming", (10, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Rosso: Raw MediaPipe | Verde: Kalman", (10, 60), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('MediaPipe Kinematics LSL', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Streaming terminato.")

print("Salvataggio dei dati in CSV...")
if kinematics_data:
    columns = ["Timestamp_LSL"]
    for i in range(21):
        columns.extend([f"LM_{i}_X", f"LM_{i}_Y", f"LM_{i}_Z"])
        
    # Salvataggio dati filtrati
    df_kin_filtered = pd.DataFrame(kinematics_data, columns=columns)
    csv_output_path_filtered = "dataset_trial1_Kinematics.csv"  # Manteniamo questo nome per compatibilità con l'altro script
    df_kin_filtered.to_csv(csv_output_path_filtered, index=False)
    print(f"File CSV generato (FILTRATO): {csv_output_path_filtered}")
    
    # Salvataggio dati grezzi
    df_kin_raw = pd.DataFrame(kinematics_raw_data, columns=columns)
    csv_output_path_raw = "dataset_trial1_Kinematics_Raw.csv"
    df_kin_raw.to_csv(csv_output_path_raw, index=False)
    print(f"File CSV generato (GREZZO): {csv_output_path_raw}")
    
    # === GRAFICO CONFRONTO RAW VS KALMAN ===
    def plot_kinematics_comparison(raw_data, filtered_data, landmark_idx=8, coord_name='X'):
        # landmark_idx 8: Punta dell'indice. Index_offset in base a XYZ: (X:0, Y:1, Z:2)
        coord_offset = {'X': 0, 'Y': 1, 'Z': 2}[coord_name]
        col_index = 1 + (landmark_idx * 3) + coord_offset
        
        timestamps = [row[0] for row in raw_data]
        raw_vals = [row[col_index] for row in raw_data]
        filtered_vals = [row[col_index] for row in filtered_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, raw_vals, label=f'Raw LM {landmark_idx} ({coord_name})', alpha=0.5, color='red', linestyle='--')
        plt.plot(timestamps, filtered_vals, label=f'Kalman Filtered LM {landmark_idx} ({coord_name})', color='blue', linewidth=2)
        
        plt.title(f'Coordinate comparison: Landmark {landmark_idx} - Axis {coord_name}')
        plt.xlabel('Timestamp (s)')
        plt.ylabel('Coordinate Value')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

    print("Generazione del grafico di confronto (LM_8_X)...")
    plot_kinematics_comparison(kinematics_raw_data, kinematics_data, landmark_idx=8, coord_name='X')

else:
    print("Nessun dato registrato.")