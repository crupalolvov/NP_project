"""Validate the inverse kinematics: reconstruction error, raw vs preprocessed.

Runs the Jacobian IK on the raw MediaPipe landmarks and on the
rigid-skeleton preprocessed ones, re-applies forward kinematics and measures
the per-landmark distance to the target. Saves the IK landmarks
(``trial_<N>_Kinematics_IKA_predicted_lms.csv``, for animation) and three
figures in ``results/figures``.

Usage:
    python -m analysis.ik_reconstruction_error [--trial 1]
"""
import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from kinematics.hand_kinematics import (CleanHandFK, JacobianIKA, coordinate_scale,
                                        detect_coordinate_format, extract_anatomy_from_csv,
                                        load_landmarks, preprocess_mediapipe_data)


def get_anatomy():
    if config.HAND_CALIBRATION_FILE.exists():
        print(f"Loading hand calibration from {config.HAND_CALIBRATION_FILE.name}")
        return torch.load(config.HAND_CALIBRATION_FILE, weights_only=False)
    print("Computing the hand calibration from the CSV file...")
    start, end = config.CALIBRATION_FRAMES
    anatomy = extract_anatomy_from_csv(str(config.trial_file(config.CALIBRATION_TRIAL, "Kinematics")), start, end)
    torch.save(anatomy, config.HAND_CALIBRATION_FILE)
    return anatomy


def track(ika, fk, landmarks):
    """Run IK frame by frame; returns per-landmark errors and FK landmarks."""
    n = len(landmarks)
    errors = np.zeros((n, 21))
    fk_lms = np.zeros_like(landmarks)
    for f in range(n):
        wrist = landmarks[f, 0].copy()
        target = landmarks[f] - wrist
        pred = fk.forward(ika.solve(target))
        errors[f] = np.linalg.norm(target - pred, axis=1)
        fk_lms[f] = pred + wrist          # back to the original position (for animation)
        if f % 100 == 0:
            print(f"Frame {f:04d}/{n} | mean error {errors[f].mean():.2f}")
    return errors, fk_lms


def main():
    parser = argparse.ArgumentParser(description="IK reconstruction error on one trial.")
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()

    kin_file = config.trial_file(args.trial, "Kinematics")
    anatomy = get_anatomy()
    df_orig = pd.read_csv(kin_file)
    is_metric = detect_coordinate_format(df_orig)
    scale = coordinate_scale(is_metric)
    unit = "mm" if is_metric else "px"

    raw_landmarks = load_landmarks(df_orig, scale)
    cleaned_landmarks = preprocess_mediapipe_data(str(kin_file), anatomy)
    fk = CleanHandFK(anatomy)

    print("\n--- IK on raw landmarks ---")
    errors_raw, _ = track(JacobianIKA(fk), fk, raw_landmarks)
    print(f"Global mean error (raw): {errors_raw.mean():.2f} {unit}")

    print("\n--- IK on preprocessed landmarks ---")
    errors_clean, fk_lms = track(JacobianIKA(fk), fk, cleaned_landmarks)
    print(f"Global mean error (preprocessed): {errors_clean.mean():.2f} {unit}")

    out = pd.DataFrame(fk_lms.reshape(len(df_orig), 63) / np.tile(scale, 21),
                       columns=[f"LM_{i}_{ax}" for i in range(21) for ax in "XYZ"])
    if "Timestamp_LSL" in df_orig.columns:
        out.insert(0, "Timestamp_LSL", df_orig["Timestamp_LSL"].values)
    out_csv = config.trial_file(args.trial, "Kinematics_IKA_predicted_lms")
    out.to_csv(out_csv, index=False)
    print(f"IK landmarks saved for animation: {out_csv.name}\n")

    # ---- Figures ----
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tag = kin_file.stem
    frames = np.arange(len(df_orig))
    mean_raw, mean_clean = errors_raw.mean(axis=1), errors_clean.mean(axis=1)

    # 1. Error over time (preprocessed)
    plt.figure(figsize=(12, 6))
    plt.plot(frames, mean_clean, label=f"Mean error ({unit})", color="blue")
    plt.plot(frames, errors_clean.max(axis=1), label=f"Max error per frame ({unit})", color="red", alpha=0.3)
    plt.axhline(mean_clean.mean(), color="green", linestyle="--",
                label=f"Global mean ({mean_clean.mean():.2f} {unit})")
    plt.title(f"IK reconstruction error over time (preprocessed data)\nFile: {kin_file.name}")
    plt.xlabel("Frame")
    plt.ylabel(f"Error ({unit})")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"error_time_preprocessed_{tag}.png", dpi=300)

    # 2. Error over time (raw vs preprocessed)
    plt.figure(figsize=(12, 6))
    plt.plot(frames, mean_raw, label="Mean error, raw", color="#ff9999", alpha=0.8)
    plt.plot(frames, mean_clean, label="Mean error, preprocessed", color="#99ccff", linewidth=2)
    plt.axhline(mean_raw.mean(), color="#cc0000", linestyle="--", label=f"Raw mean ({mean_raw.mean():.2f} {unit})")
    plt.axhline(mean_clean.mean(), color="#0000cc", linestyle="--",
                label=f"Preprocessed mean ({mean_clean.mean():.2f} {unit})")
    plt.title(f"IK reconstruction error over time: raw vs preprocessed\nFile: {kin_file.name}")
    plt.xlabel("Frame")
    plt.ylabel(f"Error ({unit})")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"error_time_comparison_{tag}.png", dpi=300)

    # 3. Per-landmark box plot with a broken y axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 6), gridspec_kw={"height_ratios": [1, 3]})
    fig.subplots_adjust(hspace=0.05)
    pos_raw = np.arange(21) * 2.0 - 0.4
    pos_clean = np.arange(21) * 2.0 + 0.4
    for ax in (ax1, ax2):
        ax.boxplot(errors_raw, positions=pos_raw, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor="#ff9999", color="#cc0000"),
                   medianprops=dict(color="#cc0000", linewidth=2),
                   flierprops={"marker": "o", "markersize": 3, "alpha": 0.3, "markeredgecolor": "#cc0000"})
        ax.boxplot(errors_clean, positions=pos_clean, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor="#99ccff", color="#0000cc"),
                   medianprops=dict(color="#0000cc", linewidth=2),
                   flierprops={"marker": "o", "markersize": 3, "alpha": 0.3, "markeredgecolor": "#0000cc"})
        ax.grid(True, linestyle=":", alpha=0.7, axis="y")
    ax1.set_ylim(80, 130)
    ax2.set_ylim(0, 30)
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.tick_params(labelbottom=False)
    ax2.xaxis.tick_bottom()
    d = 0.015
    kw = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kw)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    ax2.set_xticks(np.arange(21) * 2.0)
    ax2.set_xticklabels([f"LM_{i}" for i in range(21)], rotation=45)
    ax1.set_title(f"Per-landmark error distribution (raw vs preprocessed)\nFile: {kin_file.name}")
    ax2.set_xlabel("Landmark")
    fig.text(0.04, 0.5, f"Error ({unit})", va="center", rotation="vertical", fontsize=12)
    ax1.legend(handles=[mpatches.Patch(facecolor="#ff9999", edgecolor="#cc0000", label="Raw"),
                        mpatches.Patch(facecolor="#99ccff", edgecolor="#0000cc", label="Preprocessed")],
               loc="upper right")
    plt.savefig(config.FIGURES_DIR / f"boxplot_comparison_{tag}.png", dpi=300, bbox_inches="tight")
    print(f"Figures saved to {config.rel(config.FIGURES_DIR)}")
    plt.show()


if __name__ == "__main__":
    main()
