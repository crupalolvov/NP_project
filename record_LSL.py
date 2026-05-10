from pylsl import resolve_byprop, StreamInlet
import pandas as pd
import time

def record_synchronized_data(duration_sec=60, output_prefix="session_01"):
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
    
    # Strutture dati per il salvataggio in memoria RAM durante l'acquisizione
    emg_data, emg_timestamps = [], []
    kin_data, kin_timestamps = [], []
    
    print(f"\n--- INIZIO REGISTRAZIONE ({duration_sec} secondi) ---")
    print("Eseguire i task motori (es. prese ADL, estensione dita)...")
    
    start_time = time.time()
    
    # 3. Ciclo di acquisizione asincrona
    while time.time() - start_time < duration_sec:
        # Recupero dati EMG (usiamo pull_chunk per non perdere dati ad alta frequenza)
        chunk_emg, timestamps_emg = inlet_emg.pull_chunk(timeout=0.0)
        if timestamps_emg:
            emg_data.extend(chunk_emg)
            emg_timestamps.extend(timestamps_emg)
            
        # Recupero dati Cinematica
        chunk_kin, timestamps_kin = inlet_kin.pull_chunk(timeout=0.0)
        if timestamps_kin:
            kin_data.extend(chunk_kin)
            kin_timestamps.extend(timestamps_kin)
            
        # Breve pausa per evitare il blocco del thread e ridurre il carico CPU
        time.sleep(0.01)

    print("\n--- REGISTRAZIONE COMPLETATA ---")
    print("Formattazione ed esportazione dei dataset in corso...")
    
    # 4. Formattazione e salvataggio dei dati EMG
    if emg_data:
        num_emg_ch = len(emg_data[0])
        emg_cols = [f"EMG_{i+1}" for i in range(num_emg_ch)]
        df_emg = pd.DataFrame(emg_data, columns=emg_cols)
        df_emg.insert(0, "Timestamp_LSL", emg_timestamps)
        df_emg.to_csv(f"{output_prefix}_EMG.csv", index=False)
    
    # 5. Formattazione e salvataggio dei dati Cinematici
    if kin_data:
        num_kin_ch = len(kin_data[0])
        kin_cols = []
        # Ricostruzione dinamica delle etichette (X, Y, Z per i 21 landmark)
        for i in range(num_kin_ch // 3):
            kin_cols.extend([f"LM_{i}_X", f"LM_{i}_Y", f"LM_{i}_Z"])
        
        df_kin = pd.DataFrame(kin_data, columns=kin_cols)
        df_kin.insert(0, "Timestamp_LSL", kin_timestamps)
        df_kin.to_csv(f"{output_prefix}_Kinematics.csv", index=False)
    
    print(f"Salvataggio eseguito con successo.")
    print(f"- File EMG generato: {output_prefix}_EMG.csv")
    print(f"- File Cinematica generato: {output_prefix}_Kinematics.csv")

if __name__ == '__main__':
    # È possibile modificare la durata e il prefisso per ogni diverso trial del dataset
    record_synchronized_data(duration_sec=30, output_prefix="dataset_trial1")