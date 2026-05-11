import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# FILTRO DI KALMAN (1D)

class SimpleKalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_value = initial_value
        self.error_covariance = 1.0

    def update(self, measurement):
        # Predizione
        priori_estimate = self.estimated_value
        priori_error_covariance = self.error_covariance + self.process_variance

        # Kalman Gain
        kalman_gain = priori_error_covariance / (priori_error_covariance + self.measurement_variance)

        # Aggiornamento
        self.estimated_value = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.error_covariance = (1 - kalman_gain) * priori_error_covariance

        return self.estimated_value

# UTILITIES 

def calculate_angle_3d(p1, p2, p3):
    """
    Calcola l'angolo 3D tra tre punti (p1, p2, p3).
    p2 è il vertice dell'angolo (il giunto).
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
        
    cosine_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)


def plot_kalman_comparison(df, joint_name):
    """Genera un grafico sovrapposto tra dati grezzi e filtrati."""
    plt.figure(figsize=(12, 6))
    
    # Se la colonna del tempo base non c'è, usiamo l'indice
    x_axis = df['Timestamp_LSL'] if 'Timestamp_LSL' in df.columns else df.index
    
    plt.plot(x_axis, df[f'{joint_name}_Raw'], 
             label='Raw Data (MediaPipe)', alpha=0.5, color='red', linestyle='--')
    
    plt.plot(x_axis, df[f'{joint_name}_Kalman'], 
             label='Kalman Filtered', color='blue', linewidth=2)
    
    plt.title(f'Angles comparison: {joint_name}')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Angles (Degrees)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def process_kinematics(csv_input_path, joint_to_plot="Index_PIP"):
    if not os.path.exists(csv_input_path):
        print(f"Errore: Il file {csv_input_path} non esiste.")
        return
        
    print(f"Caricamento dati da: {csv_input_path}")
    df = pd.read_csv(csv_input_path)
    
    joints = [
        ("Thumb_CMC", 0, 1, 2), ("Thumb_MCP", 1, 2, 3), ("Thumb_IP",  2, 3, 4),
        ("Index_MCP", 0, 5, 6), ("Index_PIP", 5, 6, 7), ("Index_DIP", 6, 7, 8),
        ("Middle_MCP", 0, 9, 10), ("Middle_PIP", 9, 10, 11), ("Middle_DIP", 10, 11, 12),
        ("Ring_MCP", 0, 13, 14), ("Ring_PIP", 13, 14, 15), ("Ring_DIP", 14, 15, 16),
        ("Pinky_MCP", 0, 17, 18), ("Pinky_PIP", 17, 18, 19), ("Pinky_DIP", 18, 19, 20),
    ]
    
    # SETUP PARAMETERS
    Q = 1e-2  # Varianza del processo (reattività ai movimenti veri)
    R = 1e-1  # Varianza della misurazione (capacità di filtrare il rumore)
    
    kalman_filters = {
        joint_name: SimpleKalmanFilter(process_variance=Q, measurement_variance=R) 
        for joint_name, _, _, _ in joints
    }
    
    angles_data = []
    
    
    for index, row in df.iterrows():
        row_angles = {'Timestamp_LSL': row['Timestamp_LSL']}
        
        for joint_name, idx1, idx2, idx3 in joints:
            p1 = np.array([row[f'LM_{idx1}_X'], row[f'LM_{idx1}_Y'], row[f'LM_{idx1}_Z']])
            p2 = np.array([row[f'LM_{idx2}_X'], row[f'LM_{idx2}_Y'], row[f'LM_{idx2}_Z']])
            p3 = np.array([row[f'LM_{idx3}_X'], row[f'LM_{idx3}_Y'], row[f'LM_{idx3}_Z']])
            
            raw_angle = calculate_angle_3d(p1, p2, p3)
            
            # Initialization 
            if index == 0:
                kalman_filters[joint_name].estimated_value = raw_angle
                
            # Filtering
            smoothed_angle = kalman_filters[joint_name].update(raw_angle)
            
            # Save data
            row_angles[f"{joint_name}_Raw"] = raw_angle
            row_angles[f"{joint_name}_Kalman"] = smoothed_angle
            
        angles_data.append(row_angles)
        
    # 
    angles_df = pd.DataFrame(angles_data)
    csv_output_path = csv_input_path.replace('_Kinematics.csv', '_Angles_Kalman.csv')
    angles_df.to_csv(csv_output_path, index=False)
    
    # Plot
    if f"{joint_to_plot}_Raw" in angles_df.columns:
        print(f"Generazione del grafico per l'articolazione: {joint_to_plot}...")
        plot_kalman_comparison(angles_df, joint_to_plot)
    else:
        print(f"Impossibile creare il grafico: il giunto '{joint_to_plot}' non esiste.")

if __name__ == "__main__":

    FILE_INPUT_NAME = "dataset_trial1_Kinematics.csv"
    
    VISUALIZED_JOINT = "Thumb_MCP" 
    
    process_kinematics(FILE_INPUT_NAME, joint_to_plot=VISUALIZED_JOINT)