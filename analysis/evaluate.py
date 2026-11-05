"""Headline metrics of RPC-Net on the unseen test trials.

* **MPCC** - mean Pearson correlation over the 24 DoF (angle space).
* **MD (WFD)** - mean wrist-centred distance of the thumb, index and middle
  fingertips between decoded and ground-truth landmarks (mm).

Signals are aligned by interpolating the ground truth on the prediction
timestamps; the first 0.8 s (64 samples, network warm-up) are excluded, as
in Sec. II.E of the RPC-Net paper.

Usage:
    python -m analysis.evaluate [--trials 6 9 15]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

import config

EXCLUDE_SAMPLES = 64
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12}


def _align(t_true, values, t_pred):
    return interp1d(t_true, values, axis=0, bounds_error=False,
                    fill_value=(values[0], values[-1]))(t_pred)


def calculate_mpcc(true_angles_path, pred_angles_path):
    """Mean / tertiles / median of the per-DoF Pearson correlation."""
    df_true, df_pred = pd.read_csv(true_angles_path), pd.read_csv(pred_angles_path)
    true = _align(df_true["Timestamp_LSL"].values, df_true[[f"DoF_{i}" for i in range(24)]].values,
                  df_pred["Timestamp_LSL"].values)[EXCLUDE_SAMPLES:]
    pred = df_pred[[f"Pred_DoF_{i}" for i in range(24)]].values[EXCLUDE_SAMPLES:]

    pcc = [pearsonr(true[:, i], pred[:, i])[0] if np.std(true[:, i]) > 0 and np.std(pred[:, i]) > 0 else 0.0
           for i in range(24)]
    return np.mean(pcc), np.percentile(pcc, 33.33), np.percentile(pcc, 66.67), np.median(pcc)


def calculate_wfd(true_lms_path, pred_lms_path):
    """Mean / tertiles / median of the weighted fingertip distance (mm)."""
    df_true, df_pred = pd.read_csv(true_lms_path), pd.read_csv(pred_lms_path)
    cols = [f"LM_{j}_{ax}" for j in [0, *FINGERTIPS.values()] for ax in "XYZ"]
    true = _align(df_true["Timestamp_LSL"].values, df_true[cols].values,
                  df_pred["Timestamp_LSL"].values)[EXCLUDE_SAMPLES:].reshape(-1, 4, 3)
    pred = df_pred[cols].values[EXCLUDE_SAMPLES:].reshape(-1, 4, 3)

    # Wrist at the origin in every frame: compare the intrinsic pose only
    true_rel = true[:, 1:] - true[:, :1]
    pred_rel = pred[:, 1:] - pred[:, :1]
    wfd = np.linalg.norm(true_rel - pred_rel, axis=2).mean(axis=1) * 1000.0   # m -> mm
    return np.mean(wfd), np.percentile(wfd, 33.33), np.percentile(wfd, 66.67), np.median(wfd)


def main():
    parser = argparse.ArgumentParser(description="MPCC and fingertip distance on the test trials.")
    parser.add_argument("--trials", type=int, nargs="+", default=config.TRIAL_SPLITS["test"])
    args = parser.parse_args()

    print("=" * 60 + "\n      RPC-NET EVALUATION ON THE TEST TRIALS\n" + "=" * 60)
    for trial in args.trials:
        print(f"\nTrial {trial}\n" + "-" * 50)
        true_angles = config.trial_file(trial, "Kinematics_core_IKA_24DoF")
        true_lms = config.trial_file(trial, "Kinematics_preprocessed")
        pred_angles = config.prediction_file(trial, "angles")
        pred_lms = config.prediction_file(trial, "lms")
        if not (pred_angles.exists() and pred_lms.exists()):
            print("   Predictions not found - run `python -m decoder.inference --trial "
                  f"{trial}` first.")
            continue
        if not (true_angles.exists() and true_lms.exists()):
            print("   Ground-truth files not found, skipping.")
            continue

        mpcc, t1, t2, med = calculate_mpcc(true_angles, pred_angles)
        print("1. Angle space (MPCC)")
        print(f"   Mean    : {mpcc * 100:.2f} %")
        print(f"   Median  : {med * 100:.2f} %")
        print(f"   Tertiles: T1 {t1 * 100:.1f} %, T2 {t2 * 100:.1f} %")

        md, t1, t2, med = calculate_wfd(true_lms, pred_lms)
        print("2. Cartesian space (fingertip distance)")
        print(f"   Mean    : {md:.2f} mm")
        print(f"   Median  : {med:.2f} mm")
        print(f"   Tertiles: T1 {t1:.1f} mm, T2 {t2:.1f} mm")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
