import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Importa l'architettura esatta dal tuo file locale
from RPC_Net import RPCNet_Exact 

def train_model(train_file, val_file, epochs=3, batch_size=10):
    # 1. Caricamento dei dati di TRAIN
    print(f"Caricamento dei tensori di TRAIN da {train_file}...")
    train_data = torch.load(train_file)
    train_dataset = TensorDataset(train_data['X_emg'], train_data['X_ang'], train_data['Y_target'])
    
    # 2. Caricamento dei dati di VALIDATION
    print(f"Caricamento dei tensori di VAL da {val_file}...")
    val_data = torch.load(val_file)
    val_dataset = TensorDataset(val_data['X_emg'], val_data['X_ang'], val_data['Y_target'])
    
    # Creazione dei DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Inizializzazione della rete
    print("Inizializzazione del modello RPC-Net...")
    model = RPCNet_Exact(in_emg=512, in_ang=192)
    criterion = nn.MSELoss()
    
    # Iperparametri hard-coded dal protocollo
    optimizer = optim.Adam(model.parameters(), lr=1e-5, eps=1e-3, betas=(0.9, 0.99))

    # 4. Training Loop
    print("Avvio addestramento (3 epoche)...")
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
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoca {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 5. Salvataggio dei pesi
    torch.save(model.state_dict(), "NP_project/rpc_net_weights.pth")
    print("Addestramento completato! Pesi salvati in 'NP_project/rpc_net_weights.pth'.")

if __name__ == "__main__":
    FILE_TRAIN = "NP_project/train_tensors.pt" 
    FILE_VAL = "NP_project/val_tensors.pt"
    train_model(FILE_TRAIN, FILE_VAL)