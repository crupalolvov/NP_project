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

### 4. 🧮 Analisi Cinematica Offline (`angles_kalman.py`)
Modulo dedicato al calcolo della cinematica articolare e all'ottimizzazione del segnale visivo:
- **Trigonometria Vettoriale:** Ricostruisce i vettori tridimensionali a partire dalle coordinate spaziali (falangi/metacarpi) derivando gli angoli di giunzione (es. CMC, MCP, PIP, DIP) tramite prodotto scalare.
- **Filtro di Kalman (1D):** Implementa un filtro di Kalman univariato per mitigare in modo ottimale il rumore di misurazione e lo *jittering* intrinseco dei modelli di visione artificiale, fornendo un angolo "smoothed" altamente fedele all'effettiva biomeccanica della mano.
- **Output Visivo e Analitico:** Produce grafici comparativi (Raw vs. Kalman) ed esporta i dati arricchiti in `_Angles_Kalman.csv`, ideali come label per i modelli di Machine/Deep Learning.

---

## Requisiti e Installazione

L'ambiente richiede Python 3.10+. Per installare l'intero stack di dipendenze matematiche, di visione e di rete:

```bash
pip install mediapipe pylsl opencv-python pandas numpy scipy PyQt5 pyqtgraph matplotlib
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
   Lo script cercherà automaticamente i due flussi nella rete locale. Una volta agganciati, inizierà il pull dei dati e salverà al termine i dataset in locale.

---

## 📊 Struttura dei Dati Esportati

Al termine della registrazione, lo script genererà due file per ogni trial:

- `[prefisso]_EMG.csv`:
  - `Timestamp_LSL`: Timestamp di rete ad alta precisione.
  - `EMG_1` ... `EMG_32` (o fino a `EMG_64` in base alla configurazione hardware): Campioni elettromiografici in microVolt ($\mu V$).

- `[prefisso]_Kinematics.csv`:
  - `Timestamp_LSL`: Timestamp di rete ad alta precisione.
  - `LM_0_X`, `LM_0_Y`, `LM_0_Z` ... `LM_20_Z`: Coordinate spaziali per la ricostruzione topologica della mano (polso, nocche, falangi).

*Sviluppato per un progetto in ambito Neural Prostheses presso la Scuola Superiore Sant'Anna & Università di Pisa.*
Supervisor: Prof. Silvestro Micera, Dr. Elena Losanno, Dr. Vincent Mendez
Co-supervisors: Dr. Elena Losanno, Dr. Vincent Mendez
PhD Co-supervisor: Firman Isma Serdana

Alumni:
Filippelli Andrea - Frascella Simona - Galella Michele Ivo 
