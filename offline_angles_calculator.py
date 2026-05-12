import pandas as pd
import numpy as np
import os

def calculate_angle_3d(p1, p2, p3):
    "Angle computation (p1, p2, p3)"

    v1 = p1 - p2
    v2 = p3 - p2
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
        
    cosine_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def process_kinematics(csv_input_path):
    if not os.path.exists(csv_input_path):
        print(f"Error: {csv_input_path} file doesn't exist.")
        return
        
    df = pd.read_csv(csv_input_path)
    
    # (Joint_Name, id_previous_point, id_joint_vertex, id_following_point)
    # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    joints = [
        ("Thumb_CMC", 0, 1, 2),
        ("Thumb_MCP", 1, 2, 3),
        ("Thumb_IP",  2, 3, 4),

        ("Index_MCP", 0, 5, 6),
        ("Index_PIP", 5, 6, 7),
        ("Index_DIP", 6, 7, 8),

        ("Middle_MCP", 0, 9, 10),
        ("Middle_PIP", 9, 10, 11),
        ("Middle_DIP", 10, 11, 12),

        ("Ring_MCP", 0, 13, 14),
        ("Ring_PIP", 13, 14, 15),
        ("Ring_DIP", 14, 15, 16),

        ("Pinky_MCP", 0, 17, 18),
        ("Pinky_PIP", 17, 18, 19),
        ("Pinky_DIP", 18, 19, 20),
    ]
    
    angles_data = []
    
    for index, row in df.iterrows():
        row_angles = {'Timestamp_LSL': row['Timestamp_LSL']}
        
        for joint_name, idx1, idx2, idx3 in joints:
            # (X, Y, Z) coordinates for the 3 landmarks
            p1 = np.array([row[f'LM_{idx1}_X'], row[f'LM_{idx1}_Y'], row[f'LM_{idx1}_Z']])
            p2 = np.array([row[f'LM_{idx2}_X'], row[f'LM_{idx2}_Y'], row[f'LM_{idx2}_Z']])
            p3 = np.array([row[f'LM_{idx3}_X'], row[f'LM_{idx3}_Y'], row[f'LM_{idx3}_Z']])
            
            row_angles[joint_name] = calculate_angle_3d(p1, p2, p3)
            
        angles_data.append(row_angles)
        
    angles_df = pd.DataFrame(angles_data)
    
    csv_output_path = csv_input_path.replace('_Kinematics.csv', '_Angles.csv')
    angles_df.to_csv(csv_output_path, index=False)

if __name__ == "__main__":
    process_kinematics("dataset_trial1_Kinematics.csv")
