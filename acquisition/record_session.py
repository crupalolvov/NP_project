from pylsl import resolve_byprop, StreamInlet
import pandas as pd
import time
import threading
import sys

try:
    from pynput import keyboard
except ImportError:
    print("ERRORE: La libreria 'pynput' è richiesta per il controllo da tastiera.")
    print("Esegui questo comando nel terminale per installarla:\n  pip install pynput")
    sys.exit(1)

def record_synchronized_data():
    print("Ricerca dei flussi LSL sulla rete locale in corso...")
    
    # 1. Risoluzione dei flussi tramite il nome assegnato negli script di trasmissione
    emg_streams = resolve_byprop('name', 'OTB_S64_EMG')
    kin_streams = resolve_byprop('name', 'MediaPipe_Kinematics')
    
    if not emg_streams or not kin_streams:
        print("Errore critico: Impossibile trovare entrambi i flussi.")
        print("Assicurarsi che gli script del Sessantaquattro e di MediaPipe siano in esecuzione.")
        return

    print("Flussi individuati con successo. Inizializzazione degli Inlet...")
    
    # 2. Creazione degli Inlet per ricevere i dati
    inlet_emg = StreamInlet(emg_streams[0])
    inlet_kin = StreamInlet(kin_streams[0])
    
    # Variabili di stato globale per il ciclo e la registrazione
    is_recording = False
    running = True
    trial_counter = 1
    data_lock = threading.Lock()
    
    emg_data, emg_timestamps = [], []
    kin_data, kin_timestamps = [], []
    
    def save_data(output_prefix, e_d, e_t, k_d, k_t):
        # Formattazione e salvataggio dei dati EMG
        if e_d:
            num_emg_ch = len(e_d[0])
            emg_cols = [f"EMG_{i+1}" for i in range(num_emg_ch)]
            df_emg = pd.DataFrame(e_d, columns=emg_cols)
            df_emg.insert(0, "Timestamp_LSL", e_t)
            df_emg.to_csv(f"{output_prefix}_EMG.csv", index=False)
        
        # Formattazione e salvataggio dei dati Cinematici
        if k_d:
            num_kin_ch = len(k_d[0])
            kin_cols = []
            for i in range(num_kin_ch // 3):
                kin_cols.extend([f"LM_{i}_X", f"LM_{i}_Y", f"LM_{i}_Z"])
            
            df_kin = pd.DataFrame(k_d, columns=kin_cols)
            df_kin.insert(0, "Timestamp_LSL", k_t)
            df_kin.to_csv(f"{output_prefix}_Kinematics.csv", index=False)
        
        print(f"Salvataggio completato:")
        if e_d: print(f"- {output_prefix}_EMG.csv")
        if k_d: print(f"- {output_prefix}_Kinematics.csv")

    def on_press(key):
        nonlocal is_recording, running, trial_counter
        try:
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                if char == 'r':
                    if not is_recording:
                        # Avvia la registrazione
                        with data_lock:
                            emg_data.clear()
                            emg_timestamps.clear()
                            kin_data.clear()
                            kin_timestamps.clear()
                            is_recording = True
                        print(f"\n[REC] 🔴 INIZIO REGISTRAZIONE: trial_{trial_counter}")
                        print("Eseguire i task motori. Premi nuovamente 'R' per terminare.")
                    else:
                        # Termina la registrazione
                        with data_lock:
                            is_recording = False
                            # Copiamo i dati per liberare le variabili subito
                            e_d, e_t = list(emg_data), list(emg_timestamps)
                            k_d, k_t = list(kin_data), list(kin_timestamps)
                        
                        print(f"\n[REC] ⏹️ FINE REGISTRAZIONE: trial_{trial_counter}")
                        print("Esportazione dei dataset CSV in corso...")
                        save_data(f"trial_{trial_counter}", e_d, e_t, k_d, k_t)
                        
                        trial_counter += 1
                        print(f"\n--- PRONTO PER IL PROSSIMO TRIAL ---")
                        print(f"Premi 'R' per avviare la registrazione del trial_{trial_counter}. Premi 'Q' per uscire.")
                        
                elif char == 'q':
                    running = False
                    return False # Ferma il thread che ascolta la tastiera
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("\n--- SISTEMA PRONTO ---")
    print("Premi il tasto 'R' per avviare la registrazione del trial_1.")
    print("Premi il tasto 'Q' per chiudere il programma.")
    
    # 3. Ciclo di acquisizione asincrona
    while running:
        # Svuotiamo sempre il buffer LSL (anche se non stiamo registrando)
        # Questo è critico per evitare di accumulare "dati vecchi" prima di premere R
        chunk_emg, timestamps_emg = inlet_emg.pull_chunk(timeout=0.0)
        chunk_kin, timestamps_kin = inlet_kin.pull_chunk(timeout=0.0)
        
        if is_recording:
            with data_lock:
                if timestamps_emg:
                    emg_data.extend(chunk_emg)
                    emg_timestamps.extend(timestamps_emg)
                if timestamps_kin:
                    kin_data.extend(chunk_kin)
                    kin_timestamps.extend(timestamps_kin)
                    
        time.sleep(0.01)
        
    print("\nChiusura programma...")

if __name__ == '__main__':
    record_synchronized_data()