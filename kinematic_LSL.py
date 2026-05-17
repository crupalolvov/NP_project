import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import datetime
import threading
from pylsl import StreamInfo, StreamOutlet

# ==========================================
# 1. SETUP LAB STREAMING LAYER (LSL)
# ==========================================
LSL_CHANNELS = 63
LSL_FPS = 0 
info = StreamInfo('MediaPipe_Kinematics', 'Kinematics', LSL_CHANNELS, LSL_FPS, 'float32', 'mediapipe_hand_01')
outlet = StreamOutlet(info)

# ==========================================
# 2. MEDIAPIPE & GLOBAL STATE
# ==========================================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

data_lock = threading.Lock()
latest_landmarks = None
is_recording = False
recording_data = []
recording_start_time = 0
status_message = ""
status_message_display_time = 0

def update_result(result, output_image, timestamp_ms):
    global latest_landmarks, is_recording, recording_data
    
    if result.hand_landmarks:
        with data_lock:
            latest_landmarks = result.hand_landmarks[0]
            lsl_timestamp = timestamp_ms / 1000.0 
            
            flattened_coords = []
            for lm in latest_landmarks:
                flattened_coords.extend([lm.x, lm.y, lm.z])
                
            outlet.push_sample(flattened_coords, lsl_timestamp)

            if is_recording:
                recording_data.append([lsl_timestamp] + flattened_coords)
    else:
        with data_lock:
            latest_landmarks = None

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_result,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def save_recording():
    global recording_data, status_message, status_message_display_time
    if not recording_data:
        status_message = "Nessun dato da salvare."
        status_message_display_time = time.time()
        return

    columns = ["Timestamp_LSL"]
    for i in range(21):
        columns.extend([f"LM_{i}_X", f"LM_{i}_Y", f"LM_{i}_Z"])
    
    df = pd.DataFrame(recording_data, columns=columns)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kinematics_recording_{timestamp_str}.csv"
    df.to_csv(filename, index=False)
    
    status_message = f"Salvato in: {filename}"
    status_message_display_time = time.time()
    recording_data = []

# ==========================================
# 3. CONFIGURAZIONE HARDWARE (AVFOUNDATION MAC)
# ==========================================
# Usiamo il backend nativo per macOS (CAP_AVFOUNDATION)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Su Mac, ridurre la risoluzione è il modo più efficace per stabilizzare i frame al secondo
# e ridurre il flickering dovuto al processing parallelo
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Rimossi i comandi CAP_PROP_EXPOSURE e AUTOFOCUS che su Mac causano instabilità o crash intermedi

prev_frame_time = 0

print("Avvio streaming LSL 'MediaPipe_Kinematics'...")
print("Premi 'R' per registrare. Premi 'Q' per uscire.")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            print("Errore: Impossibile ricevere frame dalla webcam.")
            break

        frame = cv2.flip(frame, 1)
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
        
        # Uso di time.perf_counter() per alta precisione temporale coerente su Unix
        frame_timestamp_ms = int(time.perf_counter() * 1000)
        
        try:
            landmarker.detect_async(mp_image, frame_timestamp_ms)
        except Exception as e:
            # Cattura eventuali desincronizzazioni di timestamp
            continue

        with data_lock:
            if latest_landmarks:
                h, w = frame.shape[:2]
                for lm in latest_landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        cv2.putText(frame, f"FPS: {int(fps)} | LSL Streaming", (10, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

        if is_recording:
            elapsed_time = time.time() - recording_start_time
            timer_text = f"REC * {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
            cv2.putText(frame, timer_text, (10, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)
        else:
            if time.time() - status_message_display_time < 4:
                cv2.putText(frame, status_message, (10, 60), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Premi 'R' per registrare", (10, 60), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('MediaPipe Kinematics LSL', frame)

cap.release()
cv2.destroyAllWindows()
print("Streaming terminato.")

if is_recording and recording_data:
    save_recording()