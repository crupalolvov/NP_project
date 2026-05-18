import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def check_tensor_quality(tensor_file):
    print(f"--- Analisi Qualità Dati: {os.path.basename(tensor_file)} ---")
    if not os.path.exists(tensor_file):
        print(f"Errore: File {tensor_file} non trovato.")
        return

    # Carica i dati dal tensore
    dataset = torch.load(tensor_file, weights_only=False)
    Y_target = dataset['Y_target'].numpy()
    
    num_samples, num_dofs = Y_target.shape
    print(f"Campioni totali: {num_samples}")
    print(f"Gradi di Libertà (DoF): {num_dofs}")

    # Calcola il Range di Movimento (RoM) normalizzato per ogni DoF
    rom_per_dof = np.max(Y_target, axis=0) - np.min(Y_target, axis=0)
    
    # Riportiamo il RoM in gradi (sapendo che la formula inversa è: q = norm * 240 - 150)
    rom_degrees = rom_per_dof * 240.0
    
    print("\nRange di Movimento (RoM) per ogni DoF (in Gradi reali):")
    static_joints = 0
    for i in range(num_dofs):
        print(f"DoF {i:02d}: {rom_degrees[i]:.2f}°")
        if rom_degrees[i] < 1.0:
            static_joints += 1

    print("\n--- SOMMARIO ---")
    if static_joints == num_dofs:
        print("❌ ATTENZIONE CRITICA: TUTTI i giunti sono fermi. L'IKA ha fallito di nuovo.")
    elif static_joints > 0:
        print(f"⚠️ ATTENZIONE: {static_joints} giunti su {num_dofs} sono fermi (< 1 grado di movimento).")
    else:
        print("✅ SUCCESSO: Tutti i giunti mostrano movimento! I dati sono ottimi.")

    # Disegna la traiettoria del polso (DoF 0) e dell'indice (DoF 8)
    plt.figure(figsize=(10, 5))
    plt.plot(Y_target[:, 0] * 240.0 - 150.0, label='DoF 0 (Polso Flex/Ext)')
    plt.plot(Y_target[:, 8] * 240.0 - 150.0, label='DoF 8 (Indice MCP)')
    plt.title(f"Traiettoria nel Tempo - {os.path.basename(tensor_file)}")
    plt.xlabel('Campioni Temporali (~80 Hz)')
    plt.ylabel('Angolo (Gradi)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    files_to_check = [
        "train_tensors.pt", 
        "val_tensors.pt", 
        "test_tensors.pt"
    ]
    
    for filename in files_to_check:
        file_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(file_path):
            check_tensor_quality(file_path)