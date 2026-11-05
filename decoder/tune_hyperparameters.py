import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json
import optuna
from optuna.trial import TrialState
import optuna.visualization as vis

# Importazione dell'architettura locale
from RPC_Net import RPCNet_Exact 

# Configurazione costanti e percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_TRAIN = os.path.join(BASE_DIR, "train_tensors.pt") 
FILE_VAL = os.path.join(BASE_DIR, "val_tensors.pt")
JSON_OUTPUT_PATH = os.path.join(BASE_DIR, "best_hyperparameters.json")

# Variabili globali per l'allocazione efficiente della memoria
TRAIN_DATASET = None
VAL_DATASET = None
EPOCHS_PER_TRIAL = 15   # Numero massimo di epoche per singolo tentativo

def load_datasets():
    """Carica i tensori di addestramento e validazione in memoria globale."""
    global TRAIN_DATASET, VAL_DATASET
    print("Caricamento dei tensori di Train e Validation...")
    
    train_data = torch.load(FILE_TRAIN, weights_only=False)
    TRAIN_DATASET = TensorDataset(train_data['X_emg'], train_data['X_ang'], train_data['Y_target'])
    
    val_data = torch.load(FILE_VAL, weights_only=False)
    VAL_DATASET = TensorDataset(val_data['X_emg'], val_data['X_ang'], val_data['Y_target'])
    print("Data loading completato con successo.")

def objective(trial):
    """Funzione obiettivo ottimizzata con accelerazione MPS per Mac."""
    print(f"\n---> Avvio Trial {trial.number}...")
    
    # Configurazione del Device (Usa la GPU del Mac se disponibile)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Spazio di ricerca
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    eps = trial.suggest_float("eps", 1e-8, 1e-3, log=True)

    print(f"     Parametri: lr={lr:.2e}, wd={weight_decay:.2e}, batch={batch_size}, eps={eps:.2e} | Device: {device}")

    # 2. DataLoader
    train_loader = DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(VAL_DATASET, batch_size=batch_size, shuffle=False)

    # 3. Modello spostato su GPU/MPS
    model = RPCNet_Exact(in_emg=512, in_ang=192).to(device)
    model = torch.jit.script(model)  # Compilazione JIT per prestazioni ottimali su MPS e PARALLELE
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps, betas=(0.9, 0.99))

    best_trial_val_loss = float('inf')
    
    # RIDOTTO: 15 epoche sono più che sufficienti per stimare la bontà dei parametri
    EPOCHS_SEARCH = 15 

    # 4. Training Loop
    for epoch in range(EPOCHS_SEARCH):
        model.train()
        for batch_emg, batch_ang, batch_target in train_loader:
            # Spostiamo i singoli batch sul device corrente (MPS)
            batch_emg = batch_emg.to(device)
            batch_ang = batch_ang.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_emg, batch_ang)
            loss = criterion(predictions, batch_target)
            loss.backward()
            optimizer.step()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_emg, val_ang, val_target in val_loader:
                val_emg = val_emg.to(device)
                val_ang = val_ang.to(device)
                val_target = val_target.to(device)
                
                val_preds = model(val_emg, val_ang)
                val_loss += criterion(val_preds, val_target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_trial_val_loss:
            best_trial_val_loss = avg_val_loss

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"     [Trial {trial.number}] Epoca {epoch+1}/{EPOCHS_SEARCH} | Val Loss: {avg_val_loss:.6f}")

        # 5. Pruning attivo prima (dopo 3 trial invece di 5)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            print(f"     [!] Trial {trial.number} potato all'epoca {epoch+1} per scarse prestazioni.")
            raise optuna.exceptions.TrialPruned()

    print(f"---> Fine Trial {trial.number} | Miglior Val Loss: {best_trial_val_loss:.6f}")
    return best_trial_val_loss

def save_results(study):
    """Esporta i migliori iperparametri in JSON e genera i report grafici."""
    print("\n--- SALVATAGGIO DEI RISULTATI ---")
    
    # 1. Esportazione Iperparametri in JSON
    best_trial = study.best_trial
    output_data = {
        "study_name": study.study_name,
        "best_validation_loss": best_trial.value,
        "best_trial_number": best_trial.number,
        "hyperparameters": best_trial.params
    }
    
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Iperparametri ottimali salvati correttamente in: {JSON_OUTPUT_PATH}")

    # 2. Generazione e salvataggio dei grafici analitici in formato HTML
    print("Generazione dei grafici analitici interattivi...")
    
    # Grafico 1: Storia dell'ottimizzazione
    fig_history = vis.plot_optimization_history(study)
    path_history = os.path.join(BASE_DIR, "optuna_optimization_history.html")
    fig_history.write_html(path_history)
    
    # Grafico 2: Importanza degli iperparametri (fondamentale per l'analisi di sensibilità)
    fig_importance = vis.plot_param_importances(study)
    path_importance = os.path.join(BASE_DIR, "optuna_param_importances.html")
    fig_importance.write_html(path_importance)
    
    # Grafico 3: Slice plot (relazione monodimensionale parametro-loss)
    fig_slice = vis.plot_slice(study)
    path_slice = os.path.join(BASE_DIR, "optuna_slice_plot.html")
    fig_slice.write_html(path_slice)

    print(f"Grafici salvati nella directory corrente:\n - {path_history}\n - {path_importance}\n - {path_slice}")

if __name__ == "__main__":
    # Verifica preventiva dell'esistenza dei file sorgente
    if not (os.path.exists(FILE_TRAIN) and os.path.exists(FILE_VAL)):
        print(f"Errore: Tensori di addestramento o validazione non trovati in {BASE_DIR}")
        exit(1)
        
    # Caricamento preliminare dei dati
    load_datasets()
    
    DB_PATH = f"sqlite:///{os.path.join(BASE_DIR, 'optuna_study.db')}"

    # Definizione dello studio con campionamento TPE e potatura mediana
    study = optuna.create_study(
        direction="minimize", 
        study_name="RPC_Net_Hyperparameter_Optimization",
        storage=DB_PATH,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=6)
    )
    
    print("\nAvvio del processo di ottimizzazione bayesiana con Optuna...")
    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("\nOttimizzazione interrotta manualmente dall'utente. Generazione parziale dei report...")

    # Stampa delle statistiche finali a terminale
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n================== RENDICONTO DELLO STUDIO ==================")
    print(f" Trial totali eseguiti: {len(study.trials)}")
    print(f" Trial completati: {len(complete_trials)}")
    print(f" Trial potati precocemente (Pruned): {len(pruned_trials)}")
    print(f" Minima Validation Loss registrata: {study.best_trial.value:.6f}")
    print("=============================================================")

    # Esecuzione del salvataggio dei dati e dei grafici
    save_results(study)
    