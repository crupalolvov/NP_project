import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylsl import StreamInfo, StreamOutlet
from angles_kalman_offline import SimpleKalmanFilter

# 21 landmark * 3 coordinates (x, y, z) = 63 channels
LSL_CHANNELS = 63
LSL_FPS = 0 
info = StreamInfo('MediaPipe_Kinematics', 'Kinematics', LSL_CHANNELS, LSL_FPS, 'float32', 'mediapipe_hand_01')
outlet = StreamOutlet(info)

# MEDIAPIPE SETUP
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_landmarks = None
latest_filtered_landmarks = None

# Kalman Filters Initialization
Q = 1e-1  # Process variance 
R = 1e-2  # Measurement variance 
kf_x = [SimpleKalmanFilter(Q, R) for _ in range(21)]
kf_y = [SimpleKalmanFilter(Q, R) for _ in range(21)]
kf_z = [SimpleKalmanFilter(Q, R) for _ in range(21)]
is_first_frame = True

kinematics_data = []
kinematics_raw_data = []  

def update_result(result, output_image, timestamp_ms):
    global latest_landmarks, latest_filtered_landmarks, is_first_frame, kinematics_data
    
    if result.hand_landmarks:
        latest_landmarks = result.hand_landmarks[0]
        
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
        
        outlet.push_sample(flattened_coords, lsl_timestamp)
        
        kinematics_data.append([lsl_timestamp] + flattened_coords)
        kinematics_raw_data.append([lsl_timestamp] + flattened_raw_coords)

# Link per il download: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=update_result,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# UTILITIES - offline angles computation
def calculate_joint_angle(p1, p2, p3):
   
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
        
    cosine_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

cap = cv2.VideoCapture(0)
prev_frame_time = 0


with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: 
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        frame_timestamp_ms = int(time.perf_counter() * 1000)
        
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        if latest_landmarks:
            h, w = frame.shape[:2]
            for lm in latest_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1) 
                
        if latest_filtered_landmarks:
            h, w = frame.shape[:2]
            for (fx, fy, fz) in latest_filtered_landmarks:
                px, py = int(fx * w), int(fy * h)
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1) 

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
        prev_frame_time = new_frame_time
        
        cv2.putText(frame, f"FPS: {int(fps)} | LSL Streaming", (10, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Red: Raw MediaPipe | Green: Kalman", (10, 60), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('MediaPipe Kinematics LSL', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Streaming terminato.")

if kinematics_data:
    columns = ["Timestamp_LSL"]
    for i in range(21):
        columns.extend([f"LM_{i}_X", f"LM_{i}_Y", f"LM_{i}_Z"])
        
    df_kin_filtered = pd.DataFrame(kinematics_data, columns=columns)
    csv_output_path_filtered = "dataset_trial1_Kinematics.csv"  
    df_kin_filtered.to_csv(csv_output_path_filtered, index=False)
    
    df_kin_raw = pd.DataFrame(kinematics_raw_data, columns=columns)
    csv_output_path_raw = "dataset_trial1_Kinematics_Raw.csv"
    df_kin_raw.to_csv(csv_output_path_raw, index=False)
    
    def plot_kinematics_comparison(raw_data, filtered_data, landmark_idx=8, coord_name='X'):
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

    plot_kinematics_comparison(kinematics_raw_data, kinematics_data, landmark_idx=8, coord_name='X')

else:
    print("No data recorded.")
