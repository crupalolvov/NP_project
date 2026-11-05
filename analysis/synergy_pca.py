"""Postural-synergy analysis (PCA), in the style of Fig. 5 of the RPC-Net paper.

A PCA is fitted on the ground-truth 24-DoF kinematics of the test trials.
Each annotated gesture (``data/labels.json``) is then projected on the first
two principal components: ground truth as centroid +/- std, every predicted
occurrence as a diamond. Good decoding = diamonds close to their circle.

Outputs in ``results/figures``:
    pca_explained_variance_5a.png
    pca_postural_synergies_5b.png              (trials combined)
    pca_trial_<N>_postural_synergies.png       (one per trial)

Usage:
    python -m analysis.synergy_pca [--trials 6 9]
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

import config
from analysis.metrics_report import (GESTURE_COLORS, process_predicted_kinematics,
                                     process_trial_kinematics)


class SimplePCA:
    """Minimal NumPy PCA (eigen-decomposition of the covariance matrix)."""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(X - self.mean_, rowvar=False))
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / (np.sum(eigenvalues) + 1e-10)
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_


def segments_by_gesture(norm_angles, time_kin, annotations):
    """``{gesture: [frames of each annotated occurrence]}``."""
    time_sec = time_kin - time_kin[0]
    segments = {}
    for entry in annotations:
        mask = (time_sec >= entry["start_sec"]) & (time_sec <= entry["end_sec"])
        if np.any(mask):
            segments.setdefault(entry["gesture"], []).append(norm_angles[mask])
    return segments


def plot_postural_synergies(title, true_segments, pred_occurrences, pca_2d, output_path):
    fig, ax = plt.subplots(figsize=(9.5, 8.5))

    # Ground truth: centroid (circle) with standard-deviation bars
    for gest, occurrences in true_segments.items():
        color = GESTURE_COLORS[gest]
        X_pca = pca_2d.transform(np.concatenate(occurrences, axis=0))
        centroid, std = X_pca.mean(axis=0), X_pca.std(axis=0)
        ax.scatter(*centroid, facecolors="none", edgecolors=color, s=100, linewidths=1.5,
                   zorder=3, label=f"GT: {gest}")
        ax.errorbar(*centroid, xerr=std[0], yerr=std[1], fmt="none", ecolor=color,
                    elinewidth=1.0, capsize=3, zorder=2)
        ax.text(centroid[0] + 0.008, centroid[1] + 0.008, gest, color=color,
                fontweight="bold", fontsize=8.5, zorder=5)

    # Predictions: one diamond per annotated occurrence
    for gest, occurrences in pred_occurrences.items():
        color = GESTURE_COLORS[gest]
        for idx, X_seg in enumerate(occurrences):
            c = pca_2d.transform(X_seg).mean(axis=0)
            ax.scatter(*c, marker="d", color=color, s=80, edgecolor="#ffffff", linewidths=0.6,
                       zorder=4, label=f"RPC-Net: {gest}" if idx == 0 else "")
            if len(occurrences) > 1:
                ax.text(c[0] + 0.006, c[1] - 0.008, f"{idx + 1}", color="#333333",
                        fontsize=7.0, fontweight="bold", zorder=5)

    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_explained_variance(variance_pct, output_path):
    cum = np.cumsum(variance_pct)
    n = len(variance_pct)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, n + 1), cum, marker="o", linewidth=2, color="#1f77b4", label="Cumulative variance")
    ax.bar(range(1, n + 1), variance_pct, alpha=0.5, color="#6366f1", label="Individual variance")
    ax.axhline(cum[1], color="#f43f5e", linestyle="--", alpha=0.8)
    ax.text(5, cum[1] - 4, f"2 PCs explain {cum[1]:.2f}% of the data", color="#f43f5e",
            fontweight="bold", fontsize=9)
    ax.set_xlabel("Principal components (PCs)", fontsize=10)
    ax.set_ylabel("Explained variance (%)", fontsize=10)
    ax.set_title("Explained variance of articular kinematics", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, n + 1))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Postural-synergy PCA on the test trials.")
    parser.add_argument("--trials", type=int, nargs="+", default=[6, 9])
    args = parser.parse_args()

    if not config.LABELS_FILE.exists():
        print(f"Error: '{config.LABELS_FILE}' not found!")
        return
    labels = json.loads(config.LABELS_FILE.read_text())

    true_segments, pred_segments, gt_all = {}, {}, []
    for trial in args.trials:
        name = f"trial_{trial}"
        true_file = config.trial_file(trial, "Kinematics_core_IKA_24DoF")
        pred_file = config.prediction_file(trial, "angles")
        if not (true_file.exists() and pred_file.exists() and name in labels):
            print(f"Error: ground truth, predictions or labels missing for {name}.")
            return
        norm_true, time_true = process_trial_kinematics(true_file, labels[name])
        norm_pred, time_pred = process_predicted_kinematics(pred_file)
        gt_all.append(norm_true)
        true_segments[trial] = segments_by_gesture(norm_true, time_true, labels[name])
        pred_segments[trial] = segments_by_gesture(norm_pred, time_pred, labels[name])

    # PCA on the ground-truth kinematic space of all trials
    X_true = np.concatenate(gt_all, axis=0)
    print(f"PCA fitted on {X_true.shape[0]} frames x {X_true.shape[1]} DoF")
    variance_pct = SimplePCA(n_components=11).fit(X_true).explained_variance_ratio_ * 100
    for i, (v, c) in enumerate(zip(variance_pct, np.cumsum(variance_pct))):
        print(f"   PC {i + 1:02d}: {v:.2f}%  (cumulative {c:.2f}%)")
    pca_2d = SimplePCA(n_components=2).fit(X_true)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_explained_variance(variance_pct, config.FIGURES_DIR / "pca_explained_variance_5a.png")

    def merge(per_trial):
        merged = {}
        for segs in per_trial.values():
            for gest, occ in segs.items():
                merged.setdefault(gest, []).extend(occ)
        return merged

    plot_postural_synergies("Articular synergies in PC postural space - combined trials",
                            merge(true_segments), merge(pred_segments), pca_2d,
                            config.FIGURES_DIR / "pca_postural_synergies_5b.png")
    for trial in args.trials:
        plot_postural_synergies(f"Articular synergies in PC postural space - Trial {trial}",
                                true_segments[trial], pred_segments[trial], pca_2d,
                                config.FIGURES_DIR / f"pca_trial_{trial}_postural_synergies.png")
    print(f"Figures saved to {config.rel(config.FIGURES_DIR)}")


if __name__ == "__main__":
    main()
