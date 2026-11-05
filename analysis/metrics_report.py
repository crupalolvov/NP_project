"""Per-DoF regression metrics and tracking figures for the test trials.

For every test trial it reports Pearson R, R^2 and RMSE of each of the 24
DoF (ground truth = IK angles, normalized with the trial's rest pose), the
fingertip distances on the decoded landmarks, and saves:

    results/metrics/metrics_trial_<N>.csv
    results/figures/tracking_trial_<N>.png       (24 DoF, gesture-shaded)
    results/figures/overall_error_trial_<N>.png  (MAE / RMSE over time)

Usage:
    python -m analysis.metrics_report [--trials 6 9]
"""
import argparse
import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

import config

DOF_NAMES = [
    "Wrist Flex/Ext", "Wrist Uln/Rad Dev", "Wrist Pro/Sup",
    "Thumb CMC F/E", "Thumb CMC A/A", "Thumb MCP F/E", "Thumb MCP A/A", "Thumb IP F/E",
    "Index MCP F/E", "Index MCP A/A", "Index PIP F/E", "Index DIP F/E",
    "Middle MCP F/E", "Middle MCP A/A", "Middle PIP F/E", "Middle DIP F/E",
    "Ring MCP F/E", "Ring MCP A/A", "Ring PIP F/E", "Ring DIP F/E",
    "Pinky MCP F/E", "Pinky MCP A/A", "Pinky PIP F/E", "Pinky DIP F/E",
]
GESTURE_COLORS = {
    "Rest": "#7f7f7f", "Closed Hand": "#e41a1c", "Pistol": "#377eb8",
    "Open Hand": "#4daf4a", "OK": "#ff7f0e", "Rock On": "#984ea3",
}
EXCLUDE_SAMPLES = 64   # first 0.8 s (network warm-up), as in the paper


# --- Data loading -----------------------------------------------------------
def process_trial_kinematics(csv_path, annotations=None):
    """Ground-truth IK angles normalized like the network targets.

    The rest pose is the mean over all segments annotated as 'Rest'.
    """
    df = pd.read_csv(csv_path)
    time_kin = df["Timestamp_LSL"].values
    raw_angles = df[[f"DoF_{i}" for i in range(24)]].values

    time_sec = time_kin - time_kin[0]
    mask_rest = np.zeros_like(time_sec, dtype=bool)
    for entry in annotations or []:
        if entry["gesture"] == "Rest":
            mask_rest |= (time_sec >= entry["start_sec"]) & (time_sec <= entry["end_sec"])
    q_rest = np.mean(raw_angles[mask_rest] if np.any(mask_rest) else raw_angles[:100], axis=0)

    centered = (raw_angles - q_rest + 180.0) % 360.0 - 180.0
    return (centered + 150.0) / 240.0, time_kin


def process_predicted_kinematics(csv_path):
    df = pd.read_csv(csv_path)
    return df[[f"Pred_DoF_{i}" for i in range(24)]].values, df["Timestamp_LSL"].values


def process_lms(csv_path):
    df = pd.read_csv(csv_path)
    lm_cols = [f"LM_{i}_{ax}" for i in range(21) for ax in "XYZ"]
    if all(c in df.columns for c in lm_cols):
        lms = df[lm_cols].values
    else:
        lms = df.drop(columns=["Timestamp_LSL"]).iloc[:, :63].values
    return lms, df["Timestamp_LSL"].values


# --- Metrics ----------------------------------------------------------------
def evaluate_lm_metrics(true_lms, time_true, pred_lms, time_pred):
    """Wrist-centred fingertip distances in mm.

    Returns ``(wfd, mfd)``: mean distance of thumb/index/middle tips (WFD)
    and of all five fingertips (MFD).
    """
    true_interp = interp1d(time_true, true_lms, axis=0, bounds_error=False,
                           fill_value=(true_lms[0], true_lms[-1]))(time_pred)
    scale = 1.0 if np.max(np.abs(pred_lms)) > 10.0 else 1000.0   # metres -> mm
    y_true = (true_interp[EXCLUDE_SAMPLES:] * scale).reshape(-1, 21, 3)
    y_pred = (pred_lms[EXCLUDE_SAMPLES:] * scale).reshape(-1, 21, 3)
    y_true = y_true - y_true[:, 0:1, :]   # wrist-centred: compare the intrinsic pose
    y_pred = y_pred - y_pred[:, 0:1, :]
    distances = np.linalg.norm(y_true - y_pred, axis=2)            # (frames, 21)
    return np.mean(distances[:, [4, 8, 12]]), np.mean(distances[:, [4, 8, 12, 16, 20]])


def evaluate_trial_metrics(true_angles, time_true, pred_angles, time_pred):
    """R, R^2 and RMSE per DoF on time-aligned signals (first 0.8 s excluded)."""
    true_interp = interp1d(time_true, true_angles, axis=0, bounds_error=False,
                           fill_value=(true_angles[0], true_angles[-1]))(time_pred)
    y_true = true_interp[EXCLUDE_SAMPLES:]
    y_pred = pred_angles[EXCLUDE_SAMPLES:]

    rows = []
    for i in range(24):
        t, p = y_true[:, i], y_pred[:, i]
        r_val = 0.0 if (np.std(t) < 1e-6 or np.std(p) < 1e-6) else pearsonr(t, p)[0]
        rows.append({"DoF": f"DoF_{i} ({DOF_NAMES[i]})",
                     "R (Pearson)": r_val,
                     "R^2 Score": r2_score(t, p),
                     "RMSE": np.sqrt(mean_squared_error(t, p))})
    df = pd.DataFrame(rows)

    best = df.loc[df["R (Pearson)"].idxmax()]
    worst = df.loc[df["R (Pearson)"].idxmin()]
    wf_mean_r = df.loc[:15, "R (Pearson)"].mean()   # wrist + thumb + index + middle
    summary = pd.DataFrame([
        {"DoF": "OVERALL MEAN", **df[["R (Pearson)", "R^2 Score", "RMSE"]].mean().to_dict()},
        {"DoF": "WF MEAN (Wrist+Thumb+Idx+Mid)",
         **df.loc[:15, ["R (Pearson)", "R^2 Score", "RMSE"]].mean().to_dict()},
    ])
    df_all = pd.concat([df, summary], ignore_index=True)

    df_fmt = df_all.copy()
    df_fmt["R (Pearson)"] = df_fmt["R (Pearson)"].map(lambda x: f"{x * 100:.2f}%")
    df_fmt["R^2 Score"] = df_fmt["R^2 Score"].map(lambda x: f"{x:.3f}")
    df_fmt["RMSE"] = df_fmt["RMSE"].map(lambda x: f"{x * 100:.2f}%")   # % of the 240 deg range
    return df_all, df_fmt, best, worst, wf_mean_r, y_true, y_pred, time_pred[EXCLUDE_SAMPLES:]


# --- Figures ----------------------------------------------------------------
def to_degrees(y):
    return y * 240.0 - 150.0


def shade_gestures(ax, trial_labels, with_labels=False):
    seen = set()
    for entry in trial_labels or []:
        gest = entry["gesture"]
        label = gest if with_labels and gest not in seen else ""
        seen.add(gest)
        ax.axvspan(entry["start_sec"], entry["end_sec"],
                   facecolor=GESTURE_COLORS.get(gest, "#cccccc"), alpha=0.15, label=label)


def plot_kinematic_tracking(y_true, y_pred, time_array, df_metrics, trial_name, trial_labels, start_time):
    y_true_deg, y_pred_deg = to_degrees(y_true), to_degrees(y_pred)
    time_rel = time_array - start_time   # aligned with labels.json

    fig, axes = plt.subplots(4, 6, figsize=(24, 12), sharex=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9)
    for i, ax in enumerate(axes.flat):
        shade_gestures(ax, trial_labels)
        ax.plot(time_rel, y_pred_deg[:, i], color="#d95f02", label="Estimate", linewidth=1.5)
        ax.plot(time_rel, y_true_deg[:, i], color="#1f78b4", linestyle="--", label="Truth", linewidth=1.5)
        r_val = df_metrics.loc[i, "R (Pearson)"]
        rmse_deg = df_metrics.loc[i, "RMSE"] * 240.0
        ax.set_title(f"DoF_{i}: {DOF_NAMES[i]}\nPCC: {r_val:.2f} | RMSE: {rmse_deg:.1f}°", fontsize=10)
        if i >= 18:
            ax.set_xlabel("Time (s)")
        if i % 6 == 0:
            ax.set_ylabel("Angle (°)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    for gest in dict.fromkeys(e["gesture"] for e in trial_labels or []):
        handles.append(mpatches.Patch(color=GESTURE_COLORS.get(gest, "#cccccc"), alpha=0.3, label=gest))
        labels.append(gest)
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 0.98))
    plt.savefig(config.FIGURES_DIR / f"tracking_{trial_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_overall_error(y_true, y_pred, time_array, trial_name, trial_labels, start_time):
    err = to_degrees(y_true) - to_degrees(y_pred)
    mae = np.mean(np.abs(err), axis=1)
    rmse = np.sqrt(np.mean(err ** 2, axis=1))
    time_rel = time_array - start_time

    plt.figure(figsize=(12, 4))
    shade_gestures(plt.gca(), trial_labels, with_labels=True)
    plt.plot(time_rel, mae, color="#e41a1c", label="Mean Absolute Error (MAE)", linewidth=1.5)
    plt.plot(time_rel, rmse, color="#377eb8", label="Root Mean Square Error (RMSE)", linewidth=1.5, alpha=0.6)
    plt.axhline(np.mean(mae), color="#e41a1c", linestyle="--", label=f"Global average MAE: {np.mean(mae):.2f}°")
    plt.title(f"Overall kinematic error over time (avg across 24 DoFs) - {trial_name}",
              fontsize=12, fontweight="bold")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (°)")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(config.FIGURES_DIR / f"overall_error_{trial_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


# --- Main -------------------------------------------------------------------
def analyse_trial(trial, labels):
    name = f"trial_{trial}"
    true_file = config.trial_file(trial, "Kinematics_core_IKA_24DoF")
    pred_file = config.prediction_file(trial, "angles")
    trial_labels = labels.get(name, [])

    if true_file.exists() and pred_file.exists():
        print(f"\n--- Trial {trial} ---")
        norm_true, time_true = process_trial_kinematics(true_file, trial_labels)
        norm_pred, time_pred = process_predicted_kinematics(pred_file)
        df_raw, df_fmt, best, worst, wf_pcc, y_true, y_pred, t = evaluate_trial_metrics(
            norm_true, time_true, norm_pred, time_pred)
        print(df_fmt.to_string(index=False))
        print(f"\n  Best joint (R):  {best['DoF']}  R = {best['R (Pearson)'] * 100:.2f}%")
        print(f"  Worst joint (R): {worst['DoF']}  R = {worst['R (Pearson)'] * 100:.2f}%")
        print(f"  Weighted-fingertip PCC: {wf_pcc * 100:.2f}%")
        df_raw.to_csv(config.METRICS_DIR / f"metrics_{name}.csv", index=False)
        plot_kinematic_tracking(y_true, y_pred, t, df_raw, name, trial_labels, time_true[0])
        plot_overall_error(y_true, y_pred, t, name, trial_labels, time_true[0])
    else:
        print(f"\nTrial {trial}: ground truth or predictions missing, skipping angle metrics.")

    true_lm_file = config.trial_file(trial, "Kinematics_preprocessed")
    pred_lm_file = config.prediction_file(trial, "lms")
    if true_lm_file.exists() and pred_lm_file.exists():
        wfd, mfd = evaluate_lm_metrics(*process_lms(true_lm_file), *process_lms(pred_lm_file))
        print(f"\n  Weighted fingertip distance (thumb, index, middle): {wfd:.2f} mm")
        print(f"  Mean fingertip distance (5 fingers): {mfd:.2f} mm")


def main():
    parser = argparse.ArgumentParser(description="Per-DoF metrics on the test trials.")
    parser.add_argument("--trials", type=int, nargs="+", default=[6, 9])
    args = parser.parse_args()

    labels = json.loads(config.LABELS_FILE.read_text()) if config.LABELS_FILE.exists() else {}
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70 + "\n  TEMPORAL REGRESSION METRICS (R, R^2, RMSE)\n" + "=" * 70)
    for trial in args.trials:
        analyse_trial(trial, labels)
    print(f"\nResults saved to {config.rel(config.METRICS_DIR)} and "
          f"{config.rel(config.FIGURES_DIR)}\n" + "=" * 70)


if __name__ == "__main__":
    main()
