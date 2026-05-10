# 🦾 NP_project: HD-EMG Regressor using MediaPipe Landmarks

Benvenuto in **NP_project** (Neural Prostheses Project), una pipeline avanzata per l'acquisizione, la sincronizzazione e l'elaborazione di segnali biologici e cinematici. 

L'obiettivo finale del progetto è l'addestramento di un modello di **regressione continua** capace di decodificare i segnali **HD-EMG** (High-Density Electromyography) e mapparli in tempo reale sulle coordinate spaziali 3D delle articolazioni della mano, un task fondamentale per il controllo proporzionale delle **Protesi Neurali**.

---

## 🎯 Architettura del Sistema

Il sistema si affida al protocollo **LSL (Lab Streaming Layer)** per garantire una sincronizzazione temporale (time-stamping) ad altissima precisione tra flussi di dati eterogenei e a frequenze di campionamento asimmetriche.

La repository è attualmente divisa in due moduli core:

### 1. 📷 Estrazione Cinematica Real-Time (`cinematic_LSL.py`)
Utilizza la computer vision e i modelli di deep learning di **Google MediaPipe** per il tracciamento markerless della mano.
- **Input:** Feed video asincrono dalla webcam.
- **Elaborazione:** Estrazione dei 21 landmark della mano.
- **Output (LSL):** Streaming in rete di **63 canali** continui (Coordinate X, Y, Z per i 21 landmark) di tipo `float32`.
- Include utility matematiche per il calcolo offline degli angoli articolari tramite prodotto scalare su vettori 3D.

### 2. ⏱️ Data Logger Sincronizzato (`record_LSL.py`)
Un logger asincrono progettato per acquisire contemporaneamente stream multipli e generare dataset strutturati, pronti per il training di algoritmi di Machine Learning.
- **Stream EMG:** Intercetta il flusso `OTB_S64_EMG` (es. amplificatore OTBioelettronica Sessantaquattro).
- **Stream Cinematica:** Intercetta il flusso `MediaPipe_Kinematics`.
- **Output:** Generazione di file `.csv` formattati con colonne esplicite (es. `LM_0_X`, `EMG_1`) e fusi su un asse temporale comune unificato generato da LSL (`Timestamp_LSL`).

---

## 🚀 Requisiti e Setup

Assicurati di avere Python 3.8+ installato. Le dipendenze principali possono essere installate via `pip`:

```bash
pip install mediapipe pylsl opencv-python pandas numpy
```

**Nota sul modello di Visione:**
Lo script di cinematica richiede il modello pre-addestrato di MediaPipe. Devi scaricare il file `hand_landmarker.task` e posizionarlo nella root directory del progetto. 
> 📥 Download hand_landmarker.task

---

## 🛠️ Guida all'Utilizzo

Per registrare un trial di acquisizione (es. durante l'esecuzione di task motori o Activities of Daily Living - ADL), segui l'ordine di esecuzione corretto:

1. **Avviare il flusso HD-EMG:**
   Assicurati che il software/script di interfacciamento con il dispositivo EMG sia attivo e stia streammando su LSL con il nome `OTB_S64_EMG`.

2. **Avviare lo streaming Cinematico:**
   ```bash
   python cinematic_LSL.py
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
  - `EMG_1` ... `EMG_64`: Campioni elettromiografici grezzi.

- `[prefisso]_Kinematics.csv`:
  - `Timestamp_LSL`: Timestamp di rete ad alta precisione.
  - `LM_0_X`, `LM_0_Y`, `LM_0_Z` ... `LM_20_Z`: Coordinate spaziali per la ricostruzione topologica della mano (polso, nocche, falangi).

*Sviluppato per la ricerca in ambito Neural Prostheses presso la Scuola Superiore Sant'Anna.*
