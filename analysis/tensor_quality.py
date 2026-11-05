"""Sanity check of the training tensors: range of motion of every DoF.

A DoF that never moves (< 1 deg) usually means the inverse kinematics failed.

Usage:
    python -m analysis.tensor_quality [--no-plot]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

import config


def check_tensor_quality(tensor_file, plot=True):
    print(f"--- Data quality: {tensor_file.name} ---")
    Y = torch.load(tensor_file, weights_only=False)["Y_target"].numpy()
    num_samples, num_dofs = Y.shape
    print(f"Samples: {num_samples} | DoF: {num_dofs}")

    rom_degrees = (Y.max(axis=0) - Y.min(axis=0)) * 240.0   # normalized -> degrees
    print("\nRange of motion per DoF (degrees):")
    for i, rom in enumerate(rom_degrees):
        print(f"DoF {i:02d}: {rom:.2f}°")
    static = int(np.sum(rom_degrees < 1.0))
    if static == num_dofs:
        print("\nERROR: every joint is static - inverse kinematics failed.")
    elif static:
        print(f"\nWARNING: {static}/{num_dofs} joints move less than 1 degree.")
    else:
        print("\nOK: every joint shows movement.")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(Y[:, 0] * 240.0 - 150.0, label="DoF 0 (wrist flex/ext)")
        plt.plot(Y[:, 8] * 240.0 - 150.0, label="DoF 8 (index MCP)")
        plt.title(f"Trajectory over time - {tensor_file.name}")
        plt.xlabel("Samples (~80 Hz)")
        plt.ylabel("Angle (degrees)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Range-of-motion check of the tensors.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    for split in ("train", "val", "test"):
        path = config.TENSORS_DIR / f"{split}_tensors.pt"
        if path.exists():
            check_tensor_quality(path, plot=not args.no_plot)
        else:
            print(f"{path.name} not found, skipping.")


if __name__ == "__main__":
    main()
