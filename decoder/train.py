import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
import json

# Importa l'architettura esatta dal tuo file locale
from RPC_Net import RPCNet_Exact 

def train_model(train_file, val_file, epochs=50, batch_size=10):
    # 1. Caricamento dei dati di TRAIN
    print(f"Caricamento dei tensori di TRAIN da {train_file}...")
    train_data = torch.load(train_file, weights_only=False)
    train_dataset = TensorDataset(train_data['X_emg'], train_data['X_ang'], train_data['Y_target'])
    
    # 2. Caricamento dei dati di VALIDATION
    print(f"Caricamento dei tensori di VAL da {val_file}...")
    val_data = torch.load(val_file, weights_only=False)
    val_dataset = TensorDataset(val_data['X_emg'], val_data['X_ang'], val_data['Y_target'])
    
    # --- CARICAMENTO IPERPARAMETRI ---
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_hyperparameters.json")
    lr = 1e-4
    eps = 1e-3
    weight_decay = 0.0
    
    if os.path.exists(json_path):
        print(f"Caricamento iperparametri ottimizzati da '{os.path.basename(json_path)}'...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            hps = data.get("hyperparameters", {})
            lr = hps.get("lr", lr)
            eps = hps.get("eps", eps)
            weight_decay = hps.get("weight_decay", weight_decay)
            batch_size = int(hps.get("batch_size", batch_size))
    else:
        print("File iperparametri non trovato. Uso i valori di default (RPC-Net paper).")

    # Creazione dei DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Inizializzazione della rete
    print("Inizializzazione del modello RPC-Net...")
    model = RPCNet_Exact(in_emg=512, in_ang=192)
    criterion = nn.MSELoss()
    
    # Inizializzazione Ottimizzatore con parametri dinamici
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay, betas=(0.9, 0.99))

    # 4. Training Loop
    print(f"Avvio addestramento ({epochs} epoche)...")
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_emg, batch_ang, batch_target in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_emg, batch_ang)
            loss = criterion(predictions, batch_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        
        # -- VALIDATION LOOP --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_emg, val_ang, val_target in val_loader:
                val_preds = model(val_emg, val_ang)
                val_loss += criterion(val_preds, val_target).item()

                #print(f"   [Debug] Predizioni - Min: {val_preds.min().item():.4f}, Max: {val_preds.max().item():.4f} | Target - Min: {val_target.min().item():.4f}, Max: {val_target.max().item():.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoca {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpc_net_weights.pth")
        # -- SALVATAGGIO DEL MIGLIOR MODELLO --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Nuovo miglior modello trovato e salvato! (Val Loss: {best_val_loss:.6f})")

    print(f"Addestramento completato! I pesi del miglior modello sono in '{model_save_path}'.")

    # 5. Plot della convergenza
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red', marker='x')
    plt.title('Convergenza della Funzione di Loss (MSE)')
    plt.xlabel('Epoca')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_convergence.png")
    plt.savefig(plot_path)
    print(f"Grafico della convergenza salvato in '{plot_path}'.")
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_TRAIN = os.path.join(BASE_DIR, "train_tensors.pt") 
    FILE_VAL = os.path.join(BASE_DIR, "val_tensors.pt")
    train_model(FILE_TRAIN, FILE_VAL)