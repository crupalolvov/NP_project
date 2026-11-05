"""How rigid is the MediaPipe skeleton? Bone-length variability over a trial.

MediaPipe does not enforce constant bone lengths ("rubber bones"). This
diagnostic quantifies the fluctuation of every bone and motivates the
rigid-skeleton preprocessing in ``kinematics.hand_kinematics``.

Usage:
    python -m analysis.mediapipe_bone_variance [--trial 6] [--bone Middle_Distal]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from kinematics.hand_kinematics import coordinate_scale, detect_coordinate_format, load_landmarks

BONE_LINKS = {}
for _finger, _base in (("Thumb", 1), ("Index", 5), ("Middle", 9), ("Ring", 13), ("Pinky", 17)):
    BONE_LINKS[f"{_finger}_Metacarpal"] = (0, _base)
    for _k, _part in enumerate(("Proximal", "Intermediate", "Distal")):
        BONE_LINKS[f"{_finger}_{_part}"] = (_base + _k, _base + _k + 1)


def analyze_bone_length_variance(csv_path, bone_to_plot=None):
    print(f"--- Bone-length variance (MediaPipe): {csv_path.name} ---\n")
    df = pd.read_csv(csv_path)
    is_metric = detect_coordinate_format(df)
    unit = "mm" if is_metric else "px"
    lms = load_landmarks(df, coordinate_scale(is_metric))   # Z scaled by the width, as MediaPipe suggests

    lengths = {bone: np.linalg.norm(lms[:, b] - lms[:, a], axis=1) for bone, (a, b) in BONE_LINKS.items()}

    print(f"{'Bone':<20} | {'Mean':>10} | {'Std':>8} | {'Min [frame]':>18} | {'Max [frame]':>18} | {'Range %':>8}")
    print("-" * 97)
    worst_bone, worst_var = "", 0.0
    for bone, l in lengths.items():
        mean = l.mean()
        variation = (l.max() - l.min()) / mean * 100 if mean > 0 else 0
        if variation > worst_var:
            worst_bone, worst_var = bone, variation
        print(f"{bone:<20} | {mean:>10.2f} | {l.std():>8.2f} | {l.min():>8.2f} [fr:{l.argmin():04d}] | "
              f"{l.max():>8.2f} [fr:{l.argmax():04d}] | {variation:>7.2f}%")
    print("-" * 97)
    print(f"\nWorst bone: {worst_bone}, max fluctuation {worst_var:.2f}%")
    if worst_var > 10.0:
        print("Large structural deformation: MediaPipe does not guarantee constant bone lengths, "
              "so a fixed-length kinematic model is required.")

    if bone_to_plot in lengths:
        l = lengths[bone_to_plot]
        plt.figure(figsize=(12, 6))
        plt.plot(l, label=f"{bone_to_plot} length", color="blue")
        plt.axhline(l.mean(), color="red", linestyle="--", label=f"Mean ({l.mean():.2f} {unit})")
        plt.title(f"Bone length over time: {bone_to_plot}\nFile: {csv_path.name}")
        plt.xlabel("Frame")
        plt.ylabel(f"Length ({unit})")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.show()
    elif bone_to_plot:
        print(f"\nWarning: bone '{bone_to_plot}' does not exist.")


def main():
    parser = argparse.ArgumentParser(description="MediaPipe bone-length variability.")
    parser.add_argument("--trial", type=int, default=6)
    parser.add_argument("--bone", default="Middle_Distal", help=f"one of: {', '.join(BONE_LINKS)}")
    args = parser.parse_args()
    analyze_bone_length_variance(config.trial_file(args.trial, "Kinematics"), args.bone)


if __name__ == "__main__":
    main()
