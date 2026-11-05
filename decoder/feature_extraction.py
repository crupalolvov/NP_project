"""Feature extraction: raw trials -> standardized PyTorch tensors for RPC-Net.

EMG (2 kHz, 32 ch)
    per-channel DC removal -> zero-phase band-pass 20-450 Hz + 50/100/150 Hz
    notches -> rectification, /5 mV -> sliding RMS (100 ms window, 12.5 ms
    step), i.e. an ~80 Hz envelope.
Kinematics (~30 Hz)
    MediaPipe landmarks -> rigid-skeleton preprocessing + Jacobian inverse
    kinematics (24 DoF, ``kinematics.hand_kinematics``) -> rest-pose
    subtraction -> angle wrapping -> normalization [-150 deg, 90 deg] -> [0, 1]
    -> linear interpolation onto the EMG envelope time axis.
Tensors
    from a 64-sample history (0.8 s): EMG sub-sampled 1:4 (16 x 32 = 512
    features), past angles 1:8 (8 x 24 = 192 features); target = the 24
    angles at the next step. Z-score statistics come from the training split.

Outputs ``data/tensors/{train,val,test}_tensors.pt`` and, for every trial,
``trial_<N>_EMG_RMS.csv`` (the envelope used by ``decoder.inference``).

Usage:
    python -m decoder.feature_extraction
"""
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirnotch

import config
from kinematics.hand_kinematics import extract_anatomy_from_csv, process_full_kinematics_core


class BionicFeatureExtractor:
    def __init__(self, fs_emg=config.EMG_FS):
        self.fs_emg = fs_emg
        nyq = fs_emg / 2.0

        # Filters
        self.b_band, self.a_band = butter(4, [20 / nyq, 450 / nyq], btype="band")
        self.notches = [iirnotch(f, 30, fs=fs_emg) for f in (50, 100, 150)]   # EU mains + harmonics

        # Envelope
        self.rms_window = 200   # 100 ms
        self.rms_step = 25      # 12.5 ms -> ~80 Hz envelope
        self.norm_emg = 5000.0  # 5 mV full scale (µV)

        # Network input layout
        self.history_len = 64   # 0.8 s of history at 80 Hz
        self.emg_subsample = 4  # 16 EMG time steps
        self.ang_subsample = 8  # 8 angle time steps

    def process_emg(self, emg_df):
        """Return the RMS envelope ``(windows, channels)`` and its timestamps."""
        print("EMG: DC removal, filtering and RMS envelope...")
        timestamps = emg_df["Timestamp_LSL"].values
        # Older trials use 'CH_' column names, record_session.py uses 'EMG_'
        emg_channels = [c for c in emg_df.columns if "CH_" in c or "EMG_" in c]
        raw_data = emg_df[emg_channels].values

        # Per-channel DC removal (a common average reference was tested and discarded)
        dc_removed = raw_data - np.mean(raw_data, axis=0, keepdims=True)

        filtered = filtfilt(self.b_band, self.a_band, dc_removed, axis=0)
        for b, a in self.notches:
            filtered = filtfilt(b, a, filtered, axis=0)

        rectified = np.abs(filtered) / self.norm_emg

        num_windows = (len(rectified) - self.rms_window) // self.rms_step + 1
        windows = sliding_window_view(rectified, window_shape=self.rms_window, axis=0)[::self.rms_step]
        rms_data = np.sqrt(np.mean(windows ** 2, axis=-1))
        end_indices = np.arange(num_windows) * self.rms_step + self.rms_window - 1
        return rms_data, timestamps[end_indices]

    def process_kinematics(self, kin_df, target_timestamps):
        """Normalize the 24 DoF angles and resample them on ``target_timestamps``."""
        print("Kinematics: rest-angle subtraction and DoF normalization...")
        time_kin = kin_df["Timestamp_LSL"].values
        dof_cols = [c for c in kin_df.columns if "DoF_" in c]
        raw_angles = kin_df[dof_cols].values

        # Rest pose: each trial starts with ~8 s of rest. Skip the first second
        # (settling) and average the following 4 seconds.
        start_time = time_kin[0]
        mask_rest = (time_kin >= start_time + 1.0) & (time_kin <= start_time + 5.0)
        if np.any(mask_rest):
            q_rest = np.mean(raw_angles[mask_rest, :], axis=0)
        else:
            print("WARNING: rest window not found, using the first 100 samples.")
            q_rest = np.mean(raw_angles[:100, :], axis=0)

        centered = raw_angles - q_rest
        centered = (centered + 180.0) % 360.0 - 180.0     # wrap to [-180, 180] (no 360 deg jumps)
        norm_angles = (centered + 150.0) / 240.0          # RPC-Net normalization

        interpolator = interp1d(time_kin, norm_angles, axis=0, bounds_error=False,
                                fill_value=(norm_angles[0], norm_angles[-1]))
        return interpolator(target_timestamps), q_rest

    def create_tensors(self, rms_data, aligned_angles):
        print("Building input/target tensors...")
        num_valid = len(rms_data) - self.history_len
        X_emg = np.zeros((num_valid, 512), dtype=np.float32)
        X_ang = np.zeros((num_valid, 192), dtype=np.float32)
        Y_target = np.zeros((num_valid, 24), dtype=np.float32)

        for i in range(num_valid):
            t_idx = i + self.history_len           # instant to predict
            X_emg[i] = rms_data[i:t_idx:self.emg_subsample, :].flatten()        # (16, 32)
            X_ang[i] = aligned_angles[i:t_idx:self.ang_subsample, :].flatten()  # (8, 24)
            Y_target[i] = aligned_angles[t_idx, :]

        return X_emg, X_ang, np.clip(Y_target, 0.0, 1.0)


def ensure_hand_calibration():
    """Create the anatomical calibration from trial 1 if it does not exist yet."""
    if config.HAND_CALIBRATION_FILE.exists():
        return True
    ref_kin = config.trial_file(config.CALIBRATION_TRIAL, "Kinematics")
    print(f"Hand calibration '{config.HAND_CALIBRATION_FILE.name}' not found.")
    if not ref_kin.exists():
        print(f"Error: {ref_kin.name} is required to compute the calibration.")
        return False
    start, end = config.CALIBRATION_FRAMES
    print(f"Computing calibration from {ref_kin.name} (frames {start}-{end})...")
    anatomy = extract_anatomy_from_csv(str(ref_kin), start, end)
    config.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(anatomy, config.HAND_CALIBRATION_FILE)
    print(f"Hand calibration saved to {config.HAND_CALIBRATION_FILE.name}")
    return True


def load_trial_angles(trial):
    """24-DoF angles of a trial, running IK first if only landmarks exist."""
    kin_file = config.trial_file(trial, "Kinematics")
    ik_file = config.trial_file(trial, "Kinematics_core_IKA_24DoF")
    df_kin_raw = pd.read_csv(kin_file)
    if "LM_0_X" in df_kin_raw.columns:
        if not ik_file.exists():
            print(f"IK file not found: running inverse kinematics on {kin_file.name}...")
            process_full_kinematics_core(str(kin_file))
        return pd.read_csv(ik_file)
    if "DoF_0" in df_kin_raw.columns:
        return df_kin_raw
    raise ValueError(f"Unrecognized kinematics format in {kin_file.name}")


def standardize(X, stats=None):
    """Z-score ``X``; computes the statistics when ``stats`` is None."""
    if stats is None:
        stats = (X.mean(dim=0), X.std(dim=0) + 1e-6)
    mean, std = stats
    return (X - mean) / std, stats


def main():
    if not ensure_hand_calibration():
        return

    extractor = BionicFeatureExtractor()
    train_stats = None
    config.TENSORS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, trial_list in config.TRIAL_SPLITS.items():
        print(f"\n{'=' * 50}\n SPLIT: {split_name.upper()} - trials {trial_list}\n{'=' * 50}")
        all_X_e, all_X_a, all_Y = [], [], []
        last_q_rest = None

        for trial in trial_list:
            print(f"\n--- Trial {trial} ---")
            emg_file = config.trial_file(trial, "EMG")
            if not emg_file.exists() or not config.trial_file(trial, "Kinematics").exists():
                print(f"Trial {trial} missing, skipping.")
                continue

            df_emg = pd.read_csv(emg_file)
            try:
                df_kin = load_trial_angles(trial)
            except ValueError as e:
                print(f"Error: {e}")
                continue

            dur_emg = df_emg["Timestamp_LSL"].iloc[-1] - df_emg["Timestamp_LSL"].iloc[0]
            dur_kin = df_kin["Timestamp_LSL"].iloc[-1] - df_kin["Timestamp_LSL"].iloc[0]
            print(f"Duration EMG: {dur_emg:.2f} s | kinematics: {dur_kin:.2f} s")

            rms_data, rms_timestamps = extractor.process_emg(df_emg)
            aligned_angles, q_rest = extractor.process_kinematics(df_kin, rms_timestamps)
            X_e, X_a, Y = extractor.create_tensors(rms_data, aligned_angles)
            all_X_e.append(X_e)
            all_X_a.append(X_a)
            all_Y.append(Y)
            last_q_rest = q_rest

            # Save the envelope for offline inference
            rms_df = pd.DataFrame(rms_data, columns=[f"CH_{i}" for i in range(rms_data.shape[1])])
            rms_df.insert(0, "Timestamp_LSL", rms_timestamps)
            rms_out = config.trial_file(trial, "EMG_RMS")
            rms_df.to_csv(rms_out, index=False)
            print(f"RMS envelope exported: {rms_out.name}")

        if not all_X_e:
            print(f"No valid data for split '{split_name}'.")
            continue

        X_e = torch.tensor(np.concatenate(all_X_e, axis=0))
        X_a = torch.tensor(np.concatenate(all_X_a, axis=0))
        Y = torch.tensor(np.concatenate(all_Y, axis=0))

        if split_name == "train":
            X_e, emg_stats = standardize(X_e)
            X_a, ang_stats = standardize(X_a)
            train_stats = (emg_stats, ang_stats)
        elif train_stats is None:
            print("WARNING: training statistics not available, using this split's statistics.")
            X_e, emg_stats = standardize(X_e)
            X_a, ang_stats = standardize(X_a)
        else:
            emg_stats, ang_stats = train_stats
            X_e, _ = standardize(X_e, emg_stats)
            X_a, _ = standardize(X_a, ang_stats)

        output_file = config.TENSORS_DIR / f"{split_name}_tensors.pt"
        torch.save({
            "X_emg": X_e, "X_ang": X_a, "Y_target": Y,
            "emg_mean": emg_stats[0], "emg_std": emg_stats[1],
            "ang_mean": ang_stats[0], "ang_std": ang_stats[1],
            "q_rest": last_q_rest,
        }, output_file)
        print(f"\nSplit '{split_name}' saved to {config.rel(output_file)}")
        print(f"EMG: {tuple(X_e.shape)} | angles: {tuple(X_a.shape)} | target: {tuple(Y.shape)}")


if __name__ == "__main__":
    main()
