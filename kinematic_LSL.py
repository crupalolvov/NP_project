import cv2
import mediapipe as mp
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

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

def update_result(result, output_image, timestamp_ms):
    """Callback asincrona: estrae le coordinate 3D e le invia via LSL."""
    global latest_landmarks
    
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
cap = cv2.VideoCapture(1)
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
            for lm in latest_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

        # Calcolo e rendering FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        cv2.putText(frame, f"FPS: {int(fps)} | LSL Streaming", (10, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Kinematics LSL', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Streaming terminato.")