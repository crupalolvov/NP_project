# 🦾 NP_project: HD-EMG & Kinematics Acquisition Pipeline

Benvenuto in **NP_project** (Neural Prostheses Project). Questa repository contiene una suite software avanzata per l'acquisizione sincronizzata, il filtraggio in tempo reale e l'elaborazione offline di segnali EMG a media densità (MD-EMG) e dati cinematici della mano. 

Il sistema è concepito per la ricerca nell'ambito delle **Protesi Neurali**, con l'obiettivo ultimo di fornire dataset multimodali di altissima qualità necessari per l'addestramento di modelli di **regressione continua**. Tali modelli permetteranno di decodificare i pattern mioelettrici e tradurli in coordinate spaziali 3D (o angoli articolari), garantendo un controllo proporzionale e naturale degli attuatori bionici / prototipi virtuali.

---

## 🎯 Architettura del Sistema

L'infrastruttura di acquisizione sfrutta il protocollo **LSL (Lab Streaming Layer)** per gestire il time-stamping ad alta precisione e risolvere nativamente il problema della sincronizzazione tra flussi di dati eterogenei e a frequenze di campionamento asimmetriche (es. EMG a 2000 Hz vs Webcam a ~30 Hz).

La pipeline si articola nei seguenti moduli core:

### 1. 📡 Acquisizione e Filtraggio HD-EMG (`Read64_32ch.py` & `communication_sessantaquattro.py`)
Questi moduli gestiscono l'interfacciamento TCP/IP a basso livello con il dispositivo **OTBioelettronica Sessantaquattro**. Implementano una robusta elaborazione del segnale digitale in tempo reale:
- **Protocollo di Comunicazione:** Decodifica esatta dei pacchetti binari del dispositivo per l'estrazione di 32 canali attivi.
- **Filtraggio IIR:** Implementazione di un passa-banda (20-450 Hz) in cascata strutturato in *Second-Order Sections* (SOS) per garantire stabilità matematica , unito a una batteria di filtri Notch (50, 100, 150 Hz) per l'abbattimento delle interferenze di rete.
- **Interfaccia Grafica (GUI):** Visualizzazione multi-thread ad alte prestazioni basata su `PyQt5` e `pyqtgraph`. Include una vista *Multiplot* unificata e viste a *Canale Singolo* (attivabili premendo `S`).
- **Streaming LSL:** Immette nella rete locale i segnali bioelettrici puliti (`OTB_S64_EMG`).

> *Nota:* Oltre a `Read64_32ch.py`, la suite include `Read64_2ch.py` (per setup a canali ridotti).

### 2. 📷 Estrazione Cinematica (`kinematic_LSL.py`)
Sfrutta la Computer Vision e i modelli predittivi di **Google MediaPipe** per l'inferenza spaziale:
- **Acquisizione:** Elabora il feed video della webcam in tempo reale (asincrono).
- **Estrazione:** Identifica le topologie della mano ricavando le coordinate 3D dei 21 landmark anatomici.
- **Streaming LSL:** Trasmette in rete un array in flattening di 63 feature (X, Y, Z per ogni giunto) sotto l'identificativo `MediaPipe_Kinematics`.

### 3. ⏱️ Data Logging e Sincronizzazione (`record_LSL.py`)
Nodo di archiviazione centrale responsabile della generazione dei dataset:
- **Aggancio degli Stream:** Si connette dinamicamente agli stream EMG e Cinematici sulla rete LSL.
- **Pulling Asincrono:** Recupera "chunks" di dati garantendo il non-blocco dei thread (essenziale ad alte frequenze).
- **Strutturazione Dataset:** Concatena i dati su DataFrame e li esporta generando coppie di file `.csv` (`_EMG` e `_Kinematics`), condividendo la medesima base temporale ad alta risoluzione (`Timestamp_LSL`).


### 4. 🗃️ Estrazione Feature e Creazione Tensori (`feature_ext.py`)
Elabora i dataset LSL grezzi generando i tensori PyTorch finali (Z-score standardized) pronti per l'addestramento.
- **Processing EMG:** Sottrazione della DC offset per singolo canale, filtraggio IIR passa-banda (20-450 Hz) + Notch (50, 100, 150 Hz), rettificazione e finestratura scorrevole per estrazione RMS (finestra 100ms, step 12.5ms → decima la frequenza a ~80 Hz).
- **Processing Cinematica:** Interpolazione lineare della cinematica per combaciare rigorosamente con l'asse temporale decimato dell'EMG (80 Hz). Estrazione automatica della cinematica inversa (IKA) tramite `core_kin.py` se il dato d'ingresso è spaziale. Calcolo rest-angles e normalizzazione `[-150°, 90°] → [0, 1]`.
- **Struttura Tensoriale:** Implementa il mapping storico per network densi. Da una finestra di 64 campioni (0.8s) applica sottocampionamento asimmetrico: 1:4 per l'EMG (genera array piatto 16 step x 32 canali = 512) e 1:8 per gli angoli storici (8 step x 24 DoF = 192). Esporta statistiche globali (media/varianza) ed i file `.pt` (`train_tensors.pt`, `val_tensors.pt`, `test_tensors.pt`).

### 5. 🔍 Ottimizzazione Iperparametri (`RPC_optuna.py`)
Implementa la ricerca Bayesiana degli iperparametri (HPO) della rete sfruttando il framework **Optuna**.
- **Spazio di Ricerca (Search Space):** Ottimizzazione continua di *Learning Rate*, *Weight Decay*, *Batch Size* (categorico: 16, 32, 64, 128) e coefficiente *Epsilon* dell'ottimizzatore Adam.
- **Efficienza Computazionale:** Utilizza `MedianPruner` per fermare precocemente le configurazioni stocastiche che presentano una perdita di validazione peggiore rispetto alla mediana dei trial storici.
- **Output:** I parametri ottimali vengono iniettati in `best_hyperparameters.json` (che la rete caricherà in override). Genera automaticamente un report analitico interattivo in HTML per valutare la sensibilità del modello (`optuna_optimization_history.html`, `optuna_param_importances.html`, `optuna_slice_plot.html`).

### 6. 🧠 Training RPC-Net (`train_RPC.py`)
Inizializza, addestra e valida l'architettura neurale RPC-Net (Regressione Proporzionale Continua).
- **Architettura:** Implementazione hard-coded dell'architettura parallela esatta (`RPCNet_Exact` definita in `RPC_Net.py`). Concatena i due rami di embedding (EMG ed Angoli passati) in un layer fully connected finale a 24 testine (per ogni DoF).
- **Protocollo di Addestramento:** Caricamento dei tensori in memoria, auto-acquisizione dell'HPS in formato JSON. Implementa *Early Best Checkpoint* per sovrascrivere `rpc_net_weights.pth` minimizzando la Validation Loss globale (MSE).
- **Output:** Stampa la loss cross-epocale a video e salva la curva di apprendimento `loss_convergence.png`.

### 7. 🔮 Inferenza e Decodifica Offline (`inference.py`)
Script di test diagnostico. Simula il comportamento causale della RPC-Net in ambiente real-time scorrendo array storici ad altissima risoluzione temporale su trial unseen.
- **Domain Adaptation (Z-Score Locale):** Normalizza i buffer EMG utilizzando la media e deviazione standard del trial target anziché le globali estratte in fase di train. Questo stabilizza la predizione contro le severe fluttuazioni di conduttanza elettrodo-pelle (shift di impedenza tra sessioni/sensori).
- **Smoothing Causale:** Post-elaborazione delle traiettorie angolari predette utilizzando un filtro passa-basso ricorsivo unidirezionale Butterworth (4° ordine).
- **Decodifica Forward Kinematics (FK):** Attraverso la topologia ossea calibrata (`hand_calibration.pt`), inverte i 24 angoli articolari riconvertendoli in 21 Landmark Metrici 3D nello spazio globale (`_lms.csv`).

### 8. 🎞️ Animazione e Analisi Visuale 3D (`animate_raw_kinematics.py`)
Strumento di debug per la diagnostica della cinematica umana rigida, predetta e cruda (Ground Truth vs Predetto).
- **Rendering Multiplo:** Ingesta simultaneamente multipli `.csv` tridimensionali. Resampla il tutto in sincronia perfetta a una frequenza bersaglio comune (es. 25-80 Hz).
- **Compensazione Offset Spaziale:** Applica una traslazione forzata, azzerando le coordinate del polso all'origine per tutti i modelli, sovrapponendoli per separazione orizzontale in un'unica griglia isometrica di comparazione visiva.
- **Animazione Interattiva:** Gestisce polilinee, triangolazioni e nodi sparsi in `matplotlib.animation` sfruttando keyframe temporali per il monitoraggio analitico delle discrepanze o violazioni ROM.

---

## Requisiti e Installazione

L'ambiente richiede Python 3.10+. Per installare l'intero stack di dipendenze matematiche, di visione e di rete:

```bash
pip install -r requirements.txt
```

**Nota sul modello di Visione:**
Lo script di cinematica richiede il modello pre-addestrato di MediaPipe. È necessario scaricare il file `hand_landmarker.task` e posizionarlo nella root directory del progetto. 
> 📥 Download hand_landmarker.task

---

## 🛠️ Guida all'Utilizzo

Per registrare un trial di acquisizione (es. durante l'esecuzione di task motori o Activities of Daily Living - ADL), segui l'ordine di esecuzione corretto:

1. **Avviare il flusso HD-EMG:**
   Assicurati che lo script di interfacciamento con il dispositivo EMG (es. `Read64_32ch.py`) sia attivo e stia streammando su LSL con il nome `OTB_S64_EMG`.
   
   > 💡 **Scorciatoie e Debug in `Read64_32ch.py`:**
   > - **Tasto `P`**: Attiva o disattiva in tempo reale i filtri digitali (Preprocessing). All'avvio i filtri sono disattivati di default.
   > - **Tasto `R`**: Avvia e ferma una registrazione diretta dei segnali EMG su file CSV locale. Utile per test rapidi o debug senza dover avviare l'intera pipeline di sincronizzazione LSL.

2. **Avviare lo streaming Cinematico:**
   ```bash
   python kinematic_LSL.py
   ```
   *Si aprirà una finestra OpenCV per il feedback visivo. Verrà creato l'Outlet LSL `MediaPipe_Kinematics`.*

3. **Avviare la registrazione dei dati:**
   In un altro terminale, lancia il logger:
   ```bash
   python record_LSL.py
   ```
   Lo script cercherà automaticamente i due flussi nella rete locale. Una volta agganciati, è possibile avviare e fermare la registrazione premendo il tasto `R`. I dati verranno salvati al termine di ogni registrazione.

4. **Addestramento del Modello e Inferenza:**
   Una volta acquisiti i trial, è possibile procedere con l'addestramento della rete neurale.

   a. **Generazione dei Tensori:**
   Lancia lo script `feature_ext.py`. Questo script processerà tutti i trial nella cartella `recordings/`, eseguirà l'estrazione delle feature (RMS per EMG, IKA per cinematica), li dividerà in set di training, validazione e test, e salverà i tensori finali (`train_tensors.pt`, `val_tensors.pt`, `test_tensors.pt`).
   ```bash
   python feature_ext.py
   ```

   b. **(Opzionale) Ottimizzazione Iperparametri:**
   Per trovare i migliori iperparametri per la rete, esegui `RPC_optuna.py`. Questo avvierà una ricerca Bayesiana e salverà la configurazione ottimale in `best_hyperparameters.json`.
   ```bash
   python RPC_optuna.py
   ```

   c. **Addestramento Rete:**
   Avvia il training con `train_RPC.py`. Lo script caricherà automaticamente i tensori e gli iperparametri (se presenti) e salverà i pesi del modello migliore (`rpc_net_weights.pth`).
   ```bash
   python train_RPC.py
   ```

   d. **Esecuzione Inferenza Offline:**
   Per testare il modello addestrato su un trial non visto (es. `trial_6`), modifica la variabile `TRIAL_TEST` in `inference.py` e lancialo. Lo script utilizzerà il file `_EMG_RMS.csv` (generato da `feature_ext.py`) per produrre le cinematiche predette. Questo genererà un file `_predicted_kinematics_lms.csv` che può essere visualizzato con `animate_raw_kinematics.py`.
   ```bash
   python inference.py
   ```

---

## 📊 Struttura dei Dati Esportati

Al termine della registrazione, lo script genererà due file per ogni trial:

- `[prefisso]_EMG.csv`:
  - `Timestamp_LSL`: Timestamp di rete ad alta precisione.
  - `EMG_1` ... `EMG_32` (o fino a `EMG_64` in base alla configurazione hardware): Campioni elettromiografici in microVolt ($\mu V$).

- `[prefisso]_Kinematics.csv`:
  - `Timestamp_LSL`: Timestamp di rete ad alta precisione.
  - `LM_0_X`, `LM_0_Y`, `LM_0_Z` ... `LM_20_Z`: Coordinate spaziali **in metri** (world landmarks) per la ricostruzione topologica della mano (polso, nocche, falangi).

*Sviluppato per un progetto in ambito Neural Prostheses presso la Scuola Superiore Sant'Anna & Università di Pisa.*
Supervisor: Prof. Silvestro Micera, Dr. Elena Losanno, Dr. Vincent Mendez
PhD Co-supervisor: Firman Isma Serdana

Alumni:
Filippelli Andrea - Frascella Simona - Galella Michele Ivo 
