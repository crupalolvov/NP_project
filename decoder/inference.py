"""Causal offline decoding of an unseen trial: EMG envelope -> hand pose.

Simulates real-time use of RPC-Net sample by sample:

1. **Local EMG calibration** - the EMG input is z-scored with the statistics
   of the trial itself instead of the training ones, compensating for
   session-to-session changes of the electrode-skin impedance.
2. **Autoregressive loop** - the network's own (smoothed) past predictions
   feed the angle branch, exactly as they would online.
3. **Causal smoothing** - 4th-order low-pass Butterworth (1 Hz) applied
   recursively to the predicted angles.
4. **Forward kinematics** - the 24 predicted angles are mapped back to the
   21 MediaPipe landmarks (metres, wrist at the origin) with the calibrated
   hand skeleton.

Outputs ``results/predictions/trial_<N>_predicted_kinematics_{angles,lms}.csv``.

Usage:
    python -m decoder.inference --trial 15
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, lfilter, lfilter_zi

import config
from decoder.rpc_net import RPCNet
from kinematics.hand_kinematics import CleanHandFK

WINDOW_SIZE = 64            # history length (0.8 s at 80 Hz)
FS_OUTPUT = 80.0            # envelope / prediction rate (Hz)
REST_POSE = 150.0 / 240.0   # normalized value of a zero (rest) angle


def run_offline_inference(emg_csv_path, model_path, output_csv_path, stats_file):
    print(f"Decoding {emg_csv_path.name}...")
    df_emg = pd.read_csv(emg_csv_path)
    timestamps = (df_emg["Timestamp_LSL"].values if "Timestamp_LSL" in df_emg.columns
                  else np.arange(len(df_emg)))
    emg_columns = [c for c in df_emg.columns if "CH_" in c or "EMG_" in c]
    emg_data = df_emg[emg_columns].values
    num_samples = len(emg_data)

    model = RPCNet()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Angle statistics come from the training set...
    train_stats = torch.load(stats_file, weights_only=False)
    ang_mean = train_stats["ang_mean"].numpy()
    ang_std = train_stats["ang_std"].numpy()

    # ...while the EMG ones are computed on this trial (domain adaptation).
    # The network expects 16 time steps x 32 channels, so tile them 16 times.
    emg_mean_local = np.tile(np.mean(emg_data, axis=0), 16)
    emg_std_local = np.tile(np.std(emg_data, axis=0) + 1e-6, 16)

    emg_buffer = np.zeros((WINDOW_SIZE, 32), dtype=np.float32)
    ang_buffer = np.full((WINDOW_SIZE, 24), REST_POSE, dtype=np.float32)

    b, a = butter(4, 1.0 / (FS_OUTPUT / 2.0), btype="low")
    zi = np.array([lfilter_zi(b, a) * REST_POSE for _ in range(24)])

    predicted = []
    start_time = time.time()
    for i in range(num_samples):
        emg_buffer = np.roll(emg_buffer, shift=-1, axis=0)
        emg_buffer[-1, :] = emg_data[i, :]
        if i < WINDOW_SIZE:
            predicted.append(np.full(24, REST_POSE))
            continue

        emg_input = (emg_buffer[::4, :].flatten() - emg_mean_local) / emg_std_local
        ang_input = (ang_buffer[::8, :].flatten() - ang_mean) / ang_std
        with torch.no_grad():
            pred_angles = model(torch.tensor(emg_input, dtype=torch.float32).unsqueeze(0),
                                torch.tensor(ang_input, dtype=torch.float32).unsqueeze(0)).numpy()[0]

        smoothed = np.zeros(24)
        for j in range(24):
            out, zi[j] = lfilter(b, a, [pred_angles[j]], zi=zi[j])
            smoothed[j] = out[0]

        ang_buffer = np.roll(ang_buffer, shift=-1, axis=0)
        ang_buffer[-1, :] = smoothed
        predicted.append(smoothed)

        if i % 1000 == 0:
            print(f"Processed {i}/{num_samples} frames...")

    print(f"Inference completed in {time.time() - start_time:.2f} s.")
    df_out = pd.DataFrame(np.array(predicted), columns=[f"Pred_DoF_{j}" for j in range(24)])
    df_out.insert(0, "Timestamp_LSL", timestamps)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Predicted angles saved to {config.rel(output_csv_path)}")


def decode_predictions_to_csv(pred_csv_path, calibration_file, output_lms_path, stats_file):
    """Convert predicted normalized angles to 3D landmarks via forward kinematics."""
    print("\n--- Forward kinematics (angles -> 3D landmarks) ---")
    if not calibration_file.exists():
        print(f"Error: calibration file '{calibration_file}' not found.")
        return
    df_pred = pd.read_csv(pred_csv_path)
    fk = CleanHandFK(torch.load(calibration_file, weights_only=False))
    q_rest = torch.load(stats_file, weights_only=False)["q_rest"]

    pred_cols = [f"Pred_DoF_{i}" for i in range(24)]
    lms_data = []
    for _, row in df_pred.iterrows():
        q_deg = (row[pred_cols].values * 240.0 - 150.0) + q_rest   # de-normalize, add rest pose
        lms_m = fk.forward(np.radians(q_deg)) / 1000.0             # FK works in mm -> metres
        row_out = {"Timestamp_LSL": row["Timestamp_LSL"]}
        for lm in range(21):                                       # wrist stays at (0, 0, 0)
            row_out[f"LM_{lm}_X"], row_out[f"LM_{lm}_Y"], row_out[f"LM_{lm}_Z"] = lms_m[lm]
        lms_data.append(row_out)

    pd.DataFrame(lms_data).to_csv(output_lms_path, index=False)
    print(f"Decoded landmarks saved to {config.rel(output_lms_path)}\n")


def main():
    parser = argparse.ArgumentParser(description="Offline RPC-Net decoding of one trial.")
    parser.add_argument("--trial", type=int, default=15,
                        help="trial to decode (test trials: 6, 9, 15)")
    args = parser.parse_args()

    emg_rms = config.trial_file(args.trial, "EMG_RMS")
    if not emg_rms.exists():
        print(f"Error: {emg_rms.name} not found - run `python -m decoder.feature_extraction` first.")
        return
    stats = config.TENSORS_DIR / "train_tensors.pt"
    angles_out = config.prediction_file(args.trial, "angles")
    run_offline_inference(emg_rms, config.RPC_NET_WEIGHTS, angles_out, stats)
    decode_predictions_to_csv(angles_out, config.HAND_CALIBRATION_FILE,
                              config.prediction_file(args.trial, "lms"), stats)


if __name__ == "__main__":
    main()
