# WRIST MUST BE STEADY IN SPACE!! (ok rotations, but no translations)

import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import datetime
from pylsl import StreamInfo, StreamOutlet

# 1. SETUP LAB STREAMING LAYER (LSL)
# 21 landmark * 3 coordinate (x, y, z) = 63 canali continui
LSL_CHANNELS = 63
LSL_FPS = 0 # 0 indica un rate irregolare (dipende dai frame elaborati dalla webcam)
info = StreamInfo('MediaPipe_Kinematics', 'Kinematics', LSL_CHANNELS, LSL_FPS, 'float32', 'mediapipe_hand_01')
outlet = StreamOutlet(info)

# 2. SETUP MEDIAPIPE HAND LANDMARKER
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Variabili globali per mantenere lo stato
latest_landmarks = None
is_recording = False
recording_data = []
recording_start_time = 0
status_message = ""
status_message_display_time = 0

def update_result(result, output_image, timestamp_ms):
    global latest_landmarks, is_recording, recording_data
    
    if result.hand_landmarks:
        # Per questo setup, estraiamo i dati della prima mano rilevata
        latest_landmarks = result.hand_landmarks[0]
        
        # LSL richiede il timestamp in secondi
        lsl_timestamp = timestamp_ms / 1000.0 
        
        # Flattening delle coordinate: da oggetti a una lista 1D di 63 float
        # Ordine: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        flattened_coords = []
        for lm in latest_landmarks:
            flattened_coords.extend([lm.x, lm.y, lm.z])
            
        # Push del campione in rete
        outlet.push_sample(flattened_coords, lsl_timestamp)

        # Se la registrazione è attiva, salva i dati
        if is_recording:
            recording_data.append([lsl_timestamp] + flattened_coords)

# Configurazione (file .task scaricato nella directory)
# Link per il download: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_result,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def save_recording():
    """Salva i dati cinematici registrati in un file CSV."""
    global recording_data, status_message, status_message_display_time
    if not recording_data:
        status_message = "Nessun dato da salvare."
        status_message_display_time = time.time()
        return

    # Creazione delle colonne per il DataFrame
    columns = ["Timestamp_LSL"]
    for i in range(21):  # 21 landmark
        columns.extend([f"LM_{i}_X", f"LM_{i}_Y", f"LM_{i}_Z"])
    
    df = pd.DataFrame(recording_data, columns=columns)

    # Generazione del nome file e salvataggio
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kinematics_recording_{timestamp_str}.csv"
    df.to_csv(filename, index=False)
    
    status_message = f"Salvato in: {filename}"
    status_message_display_time = time.time()
    recording_data = []

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

# 4. MAIN LOOP OPENCV
# ==========================================
cap = cv2.VideoCapture(1)
prev_frame_time = 0

print("Avvio streaming LSL 'MediaPipe_Kinematics'...")
print("Premi 'R' per avviare/fermare la registrazione.")
print("Premi 'Q' per uscire.")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            break

        frame = cv2.flip(frame, 1)

        # Gestione input da tastiera
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                recording_data = []
                recording_start_time = time.time()
                status_message = "REC INIZIATA"
                status_message_display_time = time.time()
            else:
                save_recording()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Timestamp ad alta precisione per MediaPipe e LSL
        frame_timestamp_ms = int(time.perf_counter() * 1000)
        
        # Esecuzione inferenza asincrona (attiva la callback 'update_result')
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Rendering essenziale per feedback visivo
        if latest_landmarks:
            h, w = frame.shape[:2]
            for lm in latest_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

        # Calcolo e rendering FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        cv2.putText(frame, f"FPS: {int(fps)} | LSL Streaming", (10, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

        # Visualizzazione dello stato della registrazione e del cronometro
        if is_recording:
            elapsed_time = time.time() - recording_start_time
            timer_text = f"REC ● {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
            cv2.putText(frame, timer_text, (10, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Mostra un messaggio di stato (es. "Salvato" o "Premi R")
            if time.time() - status_message_display_time < 4:
                cv2.putText(frame, status_message, (10, 60), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Premi 'R' per registrare", (10, 60), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('MediaPipe Kinematics LSL', frame)

cap.release()
cv2.destroyAllWindows()
print("Streaming terminato.")

# Se si esce con la registrazione attiva, salva i dati raccolti
if is_recording and recording_data:
    print("Salvataggio della registrazione residua in corso...")
    save_recording()