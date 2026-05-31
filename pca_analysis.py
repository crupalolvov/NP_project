import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

# --- 1. NUMPY SIMPLE PCA (ZERO-DEPENDENCY COMPILER) ---
class SimplePCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.explained_variance_ = None
        
    def fit(self, X):
        # Calculate mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Co-variance matrix
        cov = np.cov(X_centered, rowvar=False)
        
        # Eigenvalues and Eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components
        self.components_ = eigenvectors[:, :self.n_components]
        
        # Variance ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / (total_var + 1e-10)
        return self
        
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

# --- 2. KINEMATIC PROCESSING & SEGMENTATION HELPERS ---
def process_trial_kinematics(csv_path, annotations=None):
    """
    Carica ed elabora la cinematica IKA (24 DoF) di un trial.
    Centra gli angoli basandosi sulla rest window e normalizza in [0, 1].
    Ritorna gli angoli normalizzati e i timestamp LSL corrispondenti.
    """
    df = pd.read_csv(csv_path)
    time_kin = df['Timestamp_LSL'].values
    dof_cols = [f'DoF_{i}' for i in range(24)]
    raw_angles = df[dof_cols].values
    
    start_time = time_kin[0]
    time_sec = time_kin - start_time
    mask_rest = np.zeros_like(time_sec, dtype=bool)
    
    if annotations is not None:
        for entry in annotations:
            if entry['gesture'] == 'Rest':
                mask_rest = mask_rest | ((time_sec >= entry['start_sec']) & (time_sec <= entry['end_sec']))
    
    if not np.any(mask_rest):
        print(f"ATTENZIONE: Finestra di 'Rest' non trovata in {os.path.basename(csv_path)}, uso i primi 100 sample.")
        q_rest = np.mean(raw_angles[:100, :], axis=0)
    else:
        q_rest = np.mean(raw_angles[mask_rest, :], axis=0)
        
    # Sottrazione rest window ed eliminazione dei wrapping di 360 gradi
    centered = raw_angles - q_rest
    centered = (centered + 180.0) % 360.0 - 180.0
    
    # Normalizzazione in [0, 1] standard RPC-Net
    norm_angles = (centered + 150.0) / 240.0
    
    return norm_angles, time_kin

def process_predicted_kinematics(csv_path):
    """
    Carica le predizioni degli angoli (già in scala normalizzata [0, 1]).
    """
    df = pd.read_csv(csv_path)
    time_kin = df['Timestamp_LSL'].values
    pred_cols = [f'Pred_DoF_{i}' for i in range(24)]
    norm_angles = df[pred_cols].values
    return norm_angles, time_kin

def segment_trial_by_labels(norm_angles, time_kin, annotations):
    """
    Raggruppa i frame cinematici del Ground Truth per ciascun gesto basandosi sui secondi annotati nel JSON.
    Combina tutti i segmenti dello stesso gesto in un'unica matrice.
    """
    start_time = time_kin[0]
    time_sec = time_kin - start_time
    
    segmented = {}
    for entry in annotations:
        gest = entry['gesture']
        start_sec = entry['start_sec']
        end_sec = entry['end_sec']
        
        mask = (time_sec >= start_sec) & (time_sec <= end_sec)
        if np.any(mask):
            frames = norm_angles[mask]
            if gest not in segmented:
                segmented[gest] = []
            segmented[gest].append(frames)
            
    for gest in list(segmented.keys()):
        segmented[gest] = np.concatenate(segmented[gest], axis=0)
        
    return segmented

def extract_predicted_occurrences(norm_angles, time_kin, annotations):
    """
    Estrae le singole occorrenze (segmenti) delle predizioni per ciascun gesto.
    Ritorna un dizionario {gesture: [lista di array dei frame di ogni singola occorrenza]}.
    """
    start_time = time_kin[0]
    time_sec = time_kin - start_time
    
    occurrences = {}
    for entry in annotations:
        gest = entry['gesture']
        start_sec = entry['start_sec']
        end_sec = entry['end_sec']
        
        mask = (time_sec >= start_sec) & (time_sec <= end_sec)
        if np.any(mask):
            frames = norm_angles[mask]
            if gest not in occurrences:
                occurrences[gest] = []
            occurrences[gest].append(frames)
            
    return occurrences

# --- 3. PLOTTING FUNCTION IN THE STYLE OF FIGURE 5b ---
def generate_postural_synergies_plot(title, true_segments, pred_occurrences, pca_2d, colors_dict, output_path):
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    
    # 1. Plot dei cerchi Ground Truth ed barre di Deviazione Standard passanti per il centroide
    for gest, X_gest_true in true_segments.items():
        color = colors_dict[gest]
        X_pca = pca_2d.transform(X_gest_true)
        
        centroid = np.mean(X_pca, axis=0)
        std = np.std(X_pca, axis=0)
        
        # Plot centroide reale (cerchio vuoto grande con bordo spesso) - reso più piccolo e sottile
        ax.scatter(centroid[0], centroid[1], facecolors='none', edgecolors=color, s=100, linewidths=1.5, zorder=3, label=f"GT: {gest}")
        
        # Sbarre di deviazione standard classiche (cross-bars passanti per il centroide) - rese più sottili e minimali
        ax.errorbar(centroid[0], centroid[1], xerr=std[0], yerr=std[1], fmt='none', ecolor=color, elinewidth=1.0, capsize=3, zorder=2)
        
        # Aggiunta di etichetta testuale vicino al cerchio GT
        ax.text(centroid[0] + 0.008, centroid[1] + 0.008, gest, color=color, fontweight='bold', fontsize=8.5, zorder=5)
        
    # 2. Plot dei diamanti per ogni singola occorrenza predetta (un diamante per ogni segmento nel JSON)
    plotted_pred = set()
    for gest, occurrences_list in pred_occurrences.items():
        color = colors_dict[gest]
        for idx, X_segment_pred in enumerate(occurrences_list):
            X_pca_pred = pca_2d.transform(X_segment_pred)
            centroid_pred = np.mean(X_pca_pred, axis=0)
            
            # Label in legenda solo alla prima occorrenza per evitare voci duplicate
            label_str = f"RPC-Net: {gest}" if gest not in plotted_pred else ""
            # Diamanti predetti resi più piccoli
            ax.scatter(centroid_pred[0], centroid_pred[1], marker='d', color=color, s=80, edgecolor='#ffffff', linewidths=0.6, zorder=4, label=label_str)
            plotted_pred.add(gest)
            
            # Aggiunge un piccolo numero progressivo vicino al diamante se ci sono più occorrenze nello stesso trial
            if len(occurrences_list) > 1:
                ax.text(centroid_pred[0] + 0.006, centroid_pred[1] - 0.008, f"{idx+1}", color='#333333', fontsize=7.0, fontweight='bold', zorder=5)
                
    ax.set_xlabel("Componente Principale 1 (PC1)", fontsize=11, fontweight='500')
    ax.set_ylabel("Componente Principale 2 (PC2)", fontsize=11, fontweight='500')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Posiziona la legenda all'esterno
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

# --- 4. CORE PCA SYNERGIES GENERATOR ---
def run_synergy_analysis():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print("="*80)
    print("  POSTURAL SYNERGIES PCA ANALYSIS - REPLICATING PAPER FIGURE 5 (UPGRADED STYLE)")
    print("="*80)
    
    # 1. Carica le annotazioni manuali dei gesti da labels.json
    labels_file = os.path.join(BASE_DIR, "labels.json")
    if not os.path.exists(labels_file):
        print(f"Errore critico: File '{labels_file}' non trovato!")
        return
        
    with open(labels_file, 'r') as f:
        labels = json.load(f)
        
    print("\n1. Annotazioni dei gesti caricate con successo dal file JSON:")
    for trial_name, segments in labels.items():
        unique_gests = set([s['gesture'] for s in segments])
        print(f"   * {trial_name.upper()}: {len(segments)} segmenti annotati | Gesti unici: {list(unique_gests)}")
        
    # Palette colori bionici ad alta visibilità
    colors_dict = {
        "Rest": "#7f7f7f",           # Grigio
        "Closed Hand": "#e41a1c",    # Rosso
        "Pistol": "#377eb8",         # Blu
        "Open Hand": "#4daf4a",      # Verde
        "OK": "#ff7f0e",             # Arancione
        "Rock On": "#984ea3"         # Viola
    }
        
    # 2. Caricamento della cinematica Ground Truth reale dei trial di test
    print("\n2. Caricamento cinematica Ground Truth (reale) per i Trial 6 e 9...")
    true_6_file = os.path.join(BASE_DIR, "recordings", "trial_6_Kinematics_core_IKA_24DoF.csv")
    true_9_file = os.path.join(BASE_DIR, "recordings", "trial_9_Kinematics_core_IKA_24DoF.csv")
    
    if not os.path.exists(true_6_file) or not os.path.exists(true_9_file):
        print("Errore: I file Ground Truth dei Trial 6 e 9 non sono completi o mancano nella cartella recordings/")
        return
        
    norm_true_6, time_true_6 = process_trial_kinematics(true_6_file, labels.get('trial_6', []))
    norm_true_9, time_true_9 = process_trial_kinematics(true_9_file, labels.get('trial_9', []))
    
    # Concatena per creare lo spazio cinematico globale di riferimento su cui addestrare la PCA
    X_true_all = np.concatenate([norm_true_6, norm_true_9], axis=0)
    print(f"   Matrice cinematica GT totale per addestramento PCA: {X_true_all.shape} (Frame x DoFs)")
    
    # 3. Addestramento PCA (Calcoliamo 11 componenti per mostrare la varianza spiegata cumulativa in Fig 5a)
    pca = SimplePCA(n_components=11)
    pca.fit(X_true_all)
    
    variance_pct = pca.explained_variance_ratio_ * 100
    cum_variance_pct = np.cumsum(variance_pct)
    
    print("\n3. Varianza spiegata dalle prime componenti principali (Fig. 5a):")
    for i in range(11):
        print(f"   PC {i+1:02d}: {variance_pct[i]:.2f}% (Spiegata Cumulata: {cum_variance_pct[i]:.2f}%)")
        
    # Setup SimplePCA per il plot bidimensionale finale (n_components=2)
    pca_2d = SimplePCA(n_components=2)
    pca_2d.fit(X_true_all)
    
    # 4. Caricamento e segmentazione predizioni
    print("\n4. Caricamento e segmentazione delle predizioni RPC-Net...")
    pred_6_file = os.path.join(BASE_DIR, "trial_6_predicted_kinematics_angles.csv")
    pred_9_file = os.path.join(BASE_DIR, "trial_9_predicted_kinematics_angles.csv")
    
    if not os.path.exists(pred_6_file) or not os.path.exists(pred_9_file):
        print("Errore: I file di predizione del Trial 6 o 9 mancano nella root directory!")
        return
        
    norm_pred_6, time_pred_6 = process_predicted_kinematics(pred_6_file)
    norm_pred_9, time_pred_9 = process_predicted_kinematics(pred_9_file)
    
    # Estrazione segmenti Ground Truth e Occorrenze Predizioni per ciascun Trial
    segmented_true_6 = segment_trial_by_labels(norm_true_6, time_true_6, labels['trial_6'])
    occurrences_pred_6 = extract_predicted_occurrences(norm_pred_6, time_pred_6, labels['trial_6'])
    
    segmented_true_9 = segment_trial_by_labels(norm_true_9, time_true_9, labels['trial_9'])
    occurrences_pred_9 = extract_predicted_occurrences(norm_pred_9, time_pred_9, labels['trial_9'])
    
    # Combina per il plot globale (Combined)
    true_combined = {}
    for gest in set(list(segmented_true_6.keys()) + list(segmented_true_9.keys())):
        frames = []
        if gest in segmented_true_6:
            frames.append(segmented_true_6[gest])
        if gest in segmented_true_9:
            frames.append(segmented_true_9[gest])
        true_combined[gest] = np.concatenate(frames, axis=0)
        
    pred_combined = {}
    for gest in set(list(occurrences_pred_6.keys()) + list(occurrences_pred_9.keys())):
        list_occurrences = []
        if gest in occurrences_pred_6:
            list_occurrences.extend(occurrences_pred_6[gest])
        if gest in occurrences_pred_9:
            list_occurrences.extend(occurrences_pred_9[gest])
        pred_combined[gest] = list_occurrences

    # ================= PLOTTING FIGURA 5 (a & b) =================
    eval_dir = os.path.join(BASE_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 5a. Plot della spiegazione della varianza cumulata
    fig_a, ax_a = plt.subplots(figsize=(7, 5))
    ax_a.plot(range(1, 12), cum_variance_pct[:11], marker='o', linewidth=2, color='#1f77b4', label='Varianza Cumulata')
    ax_a.bar(range(1, 12), variance_pct[:11], alpha=0.5, color='#6366f1', label='Varianza Singola')
    
    # Riferimento scientifico cumulata a 2 componenti
    ax_a.axhline(y=cum_variance_pct[1], color='#f43f5e', linestyle='--', alpha=0.8)
    ax_a.text(5, cum_variance_pct[1]-4, f"2 PC spiegano il {cum_variance_pct[1]:.2f}% dei dati", color='#f43f5e', fontweight='bold', fontsize=9)
    
    ax_a.set_xlabel("Componenti Principali (PCs)", fontsize=10)
    ax_a.set_ylabel("Varianza Spiegata (%)", fontsize=10)
    ax_a.set_title("Varianza Spiegata della Cinematica Articolare (Fig. 5a)", fontsize=12, fontweight='bold')
    ax_a.set_xticks(range(1, 12))
    ax_a.grid(True, linestyle=':', alpha=0.5)
    ax_a.legend()
    plt.tight_layout()
    plot_a_path = os.path.join(eval_dir, "pca_explained_variance_5a.png")
    plt.savefig(plot_a_path, dpi=200)
    plt.close()
    
    # 5b. Generazione dei 3 Plot di Sinergia
    print("\n5. Generazione dei grafici di sinergia posturale...")
    
    # 1. Grafico Combinato (Trial 6 + Trial 9)
    plot_combined_path = os.path.join(eval_dir, "pca_postural_synergies_5b.png")
    generate_postural_synergies_plot(
        "Analisi Sinergie Articolari nel PC Postural Space (Combined Fig. 5b)",
        true_combined, pred_combined, pca_2d, colors_dict, plot_combined_path
    )
    print(f"   * Grafico Combinato salvato in: eval/pca_postural_synergies_5b.png")
    
    # 2. Grafico Singolo per Trial 6
    plot_t6_path = os.path.join(eval_dir, "pca_trial_6_postural_synergies.png")
    generate_postural_synergies_plot(
        "Analisi Sinergie Articolari nel PC Postural Space - Trial 6",
        segmented_true_6, occurrences_pred_6, pca_2d, colors_dict, plot_t6_path
    )
    print(f"   * Grafico Trial 6 salvato in  : eval/pca_trial_6_postural_synergies.png")
    
    # 3. Grafico Singolo per Trial 9
    plot_t9_path = os.path.join(eval_dir, "pca_trial_9_postural_synergies.png")
    generate_postural_synergies_plot(
        "Analisi Sinergie Articolari nel PC Postural Space - Trial 9",
        segmented_true_9, occurrences_pred_9, pca_2d, colors_dict, plot_t9_path
    )
    print(f"   * Grafico Trial 9 salvato in  : eval/pca_trial_9_postural_synergies.png")
    
    print("\nRigenerazione dei grafici completata con successo nella cartella 'eval/'.")
    print("="*80)

if __name__ == "__main__":
    run_synergy_analysis()
