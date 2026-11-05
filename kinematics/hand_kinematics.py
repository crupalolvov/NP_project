"""Biomechanical hand model: calibration, forward and inverse kinematics.

MediaPipe landmarks are noisy and their bone lengths change from frame to
frame. This module turns them into a consistent 24-DoF joint-angle
representation:

* ``extract_anatomy_from_csv``  - subject-specific bone lengths and
  metacarpal vectors from a steady open-hand segment.
* ``preprocess_mediapipe_data`` - Savitzky-Golay smoothing, projection on a
  rigid skeleton, 1-DoF hinge constraints and soft range-of-motion limits
  (Stillfried et al.).
* ``CleanHandFK``               - forward kinematics, 24 angles -> 21 landmarks.
* ``JacobianIKA``               - damped-least-squares (Levenberg-Marquardt)
  inverse kinematics, 21 landmarks -> 24 angles (used by the pipeline).
* ``CleanHandIKA``              - alternative per-finger SLSQP solver.

DoF layout: 0-2 wrist (roll, pitch, yaw), 3-7 thumb (CMC flex/abd, MCP
flex/abd, IP), then 4 per finger for index, middle, ring and pinky
(MCP flex, MCP abd, PIP, DIP).
"""
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.signal import savgol_filter

import config

FINGER_DEFS = {
    'Thumb':  ([1, 2, 3, 4],     ['Thumb_Proximal', 'Thumb_Intermediate', 'Thumb_Distal']),
    'Index':  ([5, 6, 7, 8],     ['Index_Proximal', 'Index_Intermediate', 'Index_Distal']),
    'Middle': ([9, 10, 11, 12],  ['Middle_Proximal', 'Middle_Intermediate', 'Middle_Distal']),
    'Ring':   ([13, 14, 15, 16], ['Ring_Proximal', 'Ring_Intermediate', 'Ring_Distal']),
    'Pinky':  ([17, 18, 19, 20], ['Pinky_Proximal', 'Pinky_Intermediate', 'Pinky_Distal']),
}

# Range of motion of the 1-DoF hinge joints in radians (Stillfried et al.)
MIN_FLEXION_BOUNDS = {
    'Thumb_IP': -0.26,
    'Index_PIP': -0.09, 'Index_DIP': -0.17,
    'Middle_PIP': -0.09, 'Middle_DIP': -0.17,
    'Ring_PIP': -0.09, 'Ring_DIP': -0.17,
    'Pinky_PIP': -0.09, 'Pinky_DIP': -0.17,
}
MAX_FLEXION_BOUNDS = {
    'Thumb_IP': 1.54,
    'Index_PIP': 2.23, 'Index_DIP': 1.79,
    'Middle_PIP': 2.25, 'Middle_DIP': 1.96,
    'Ring_PIP': 2.27, 'Ring_DIP': 1.66,
    'Pinky_PIP': 2.18, 'Pinky_DIP': 1.64,
}


def detect_coordinate_format(df):
    """True for MediaPipe *world* landmarks (metres), False for normalized image coordinates.

    World landmarks are wrist-centred, so they take negative values.
    """
    min_x = df[[f'LM_{i}_X' for i in range(21)]].min().min()
    min_y = df[[f'LM_{i}_Y' for i in range(21)]].min().min()
    return bool(min_x < -0.01 or min_y < -0.01 or df['LM_0_X'].abs().max() < 1e-4)


def coordinate_scale(is_metric):
    """Scale factors that bring landmarks to mm (metric) or 640x480 pixels."""
    return (1000.0, 1000.0, 1000.0) if is_metric else (640.0, 480.0, 640.0)


def load_landmarks(df, scale):
    """``(frames, 21, 3)`` array of scaled landmark coordinates."""
    cols = [f'LM_{i}_{ax}' for i in range(21) for ax in 'XYZ']
    return df[cols].to_numpy(dtype=float).reshape(len(df), 21, 3) * np.asarray(scale)


def rotate_vector_by_rodrigues(v, k, theta):
    """Rotate ``v`` around axis ``k`` by ``theta`` radians (Rodrigues' formula)."""
    norm_k = np.linalg.norm(k)
    k_unit = k / (norm_k + 1e-8) if norm_k > 1e-8 else np.array([0.0, 0.0, 1.0])
    return (v * np.cos(theta) + np.cross(k_unit, v) * np.sin(theta)
            + k_unit * np.dot(k_unit, v) * (1 - np.cos(theta)))


def preprocess_mediapipe_data(csv_path, anatomy, window_length=15, polyorder=3):
    """Clean MediaPipe landmarks before inverse kinematics.

    Removes temporal outliers and enforces bone rigidity on the correct
    rotation plane, with soft range-of-motion limits. Writes
    ``<csv>_preprocessed.csv`` and returns the cleaned ``(frames, 21, 3)`` array.
    """
    print("--- Kinematic preprocessing (rigid skeleton + RoM clamp) ---")
    df = pd.read_csv(csv_path)
    num_frames = len(df)

    is_metric = detect_coordinate_format(df)
    scale = coordinate_scale(is_metric)
    print("Metric world landmarks detected -> scaling to mm" if is_metric
          else "Normalized landmarks detected -> scaling to 640x480 px")
    raw_lms = load_landmarks(df, scale)

    print("1. Temporal smoothing (Savitzky-Golay)...")
    smoothed_lms = np.zeros_like(raw_lms)
    for i in range(21):
        for axis in range(3):
            smoothed_lms[:, i, axis] = savgol_filter(raw_lms[:, i, axis], window_length, polyorder)

    print("2. Projection on the rigid skeleton and anatomical hinges...")
    fixed_lms = np.zeros_like(smoothed_lms)

    for f in range(num_frames):
        fixed_lms[f, 0] = smoothed_lms[f, 0]
        # Palm reference frame, recomputed at every frame
        v_idx_f = smoothed_lms[f, 5] - smoothed_lms[f, 0]
        v_mid_f = smoothed_lms[f, 9] - smoothed_lms[f, 0]
        v_pnk_f = smoothed_lms[f, 17] - smoothed_lms[f, 0]

        X_palm_f = v_idx_f - v_pnk_f
        X_palm_f /= (np.linalg.norm(X_palm_f) + 1e-8)
        Z_palm_f = np.cross(X_palm_f, v_mid_f)
        Z_palm_f /= (np.linalg.norm(Z_palm_f) + 1e-8)

        for finger, (lms_idx, bone_names) in FINGER_DEFS.items():
            # Base hinge axis for this frame
            Y_local_f = smoothed_lms[f, lms_idx[0]] - fixed_lms[f, 0]
            Y_local_f /= (np.linalg.norm(Y_local_f) + 1e-8)

            if finger == 'Thumb':
                # Thumb: the flexion axis is the palm normal, orthogonalized to
                # match R_splay in CleanHandFK
                X_local = Z_palm_f.copy()
                Z_local_t = np.cross(X_local, Y_local_f)
                Z_local_t /= (np.linalg.norm(Z_local_t) + 1e-8)
                X_local = np.cross(Y_local_f, Z_local_t)
                X_local /= (np.linalg.norm(X_local) + 1e-8)
            else:
                X_local = np.cross(Y_local_f, Z_palm_f)
                if np.linalg.norm(X_local) < 1e-8:
                    X_local = X_palm_f.copy()
                X_local /= np.linalg.norm(X_local)

            # -- Metacarpal: keep the direction, impose the calibrated length --
            mcp_idx = lms_idx[0]
            dir_meta = smoothed_lms[f, mcp_idx] - fixed_lms[f, 0]
            norm_meta = np.linalg.norm(dir_meta)
            if norm_meta > 1e-6:
                dir_meta /= norm_meta
            len_meta = np.linalg.norm(anatomy['meta_vectors'][finger])
            fixed_lms[f, mcp_idx] = fixed_lms[f, 0] + dir_meta * len_meta

            # -- Phalanges, with DoF constraints propagated along the chain --
            parent_idx = mcp_idx
            current_hinge_axis = X_local
            prev_bone_dir = dir_meta.copy()

            for i, child_idx in enumerate(lms_idx[1:]):
                dir_bone = smoothed_lms[f, child_idx] - smoothed_lms[f, parent_idx]

                # Pure flexion-extension hinges (PIP, DIP and thumb IP)
                is_1dof_hinge = (finger != 'Thumb' and i > 0) or (finger == 'Thumb' and i == 2)

                # Step 1: remove the lateral deviation w.r.t. the current hinge axis
                if is_1dof_hinge:
                    dir_bone = dir_bone - np.dot(dir_bone, current_hinge_axis) * current_hinge_axis

                # Step 2: normalize the cleaned direction
                norm_bone = np.linalg.norm(dir_bone)
                if norm_bone > 1e-6:
                    dir_bone /= norm_bone
                else:
                    print(f"WARNING: degenerate bone at frame {f}, {finger} {bone_names[i]}; "
                          f"using fallback [0, 1, 0].")
                    dir_bone = np.array([0.0, 1.0, 0.0])

                # Step 3: soft range-of-motion clamp on the 1-DoF joints
                if is_1dof_hinge:
                    if finger == 'Thumb':
                        joint = 'Thumb_IP'
                    else:
                        joint = f'{finger}_PIP' if i == 1 else f'{finger}_DIP'
                    min_rad = MIN_FLEXION_BOUNDS[joint]
                    max_rad = MAX_FLEXION_BOUNDS[joint]
                    if 'DIP' in joint:
                        max_rad *= 1.15

                    # Cross product in this order gives the correct flexion direction
                    flexion_y_axis = np.cross(prev_bone_dir, current_hinge_axis)
                    flexion_y_axis /= (np.linalg.norm(flexion_y_axis) + 1e-8)
                    flexion = np.arctan2(np.dot(dir_bone, flexion_y_axis), np.dot(dir_bone, prev_bone_dir))

                    if flexion < min_rad or flexion > max_rad:
                        # Pull towards the limit (damped) instead of a hard clip
                        target = np.clip(flexion, min_rad, max_rad)
                        flexion = 0.7 * flexion + 0.3 * target
                        dir_bone = prev_bone_dir * np.cos(flexion) + flexion_y_axis * np.sin(flexion)
                        dir_bone /= np.linalg.norm(dir_bone)

                # Step 4: transport the hinge axis along the chain (as CleanHandFK does)
                rotation_axis = np.cross(prev_bone_dir, dir_bone)
                norm_rot_axis = np.linalg.norm(rotation_axis)
                if norm_rot_axis > 1e-8:
                    rotation_axis /= norm_rot_axis
                    angle = np.arccos(np.clip(np.dot(prev_bone_dir, dir_bone), -1.0, 1.0))
                    current_hinge_axis = rotate_vector_by_rodrigues(current_hinge_axis, rotation_axis, angle)

                # Step 5: rebuild the joint position with the calibrated bone length
                fixed_lms[f, child_idx] = fixed_lms[f, parent_idx] + dir_bone * anatomy['lengths'][bone_names[i]]
                parent_idx = child_idx
                prev_bone_dir = dir_bone.copy()   # must be the *cleaned* direction

    print("Preprocessing complete, data ready for inverse kinematics.")

    out = pd.DataFrame(fixed_lms.reshape(num_frames, 63) / np.tile(scale, 21),
                       columns=[f'LM_{i}_{ax}' for i in range(21) for ax in 'XYZ'])
    if 'Timestamp_LSL' in df.columns:
        out.insert(0, 'Timestamp_LSL', df['Timestamp_LSL'].values)
    out_csv = str(csv_path).replace('.csv', '_preprocessed.csv')
    out.to_csv(out_csv, index=False)
    print(f"Preprocessed landmarks saved to: {out_csv.split('/')[-1]}")
    return fixed_lms


# --- Rotation matrices ------------------------------------------------------
def Rx(theta):
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])


def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])


# --- Anatomical calibration -------------------------------------------------
def extract_anatomy_from_csv(csv_path, start_frame, end_frame):
    """Bone lengths and metacarpal vectors averaged over a steady segment."""
    df = pd.read_csv(csv_path).iloc[start_frame:end_frame]
    bone_links = {}
    for finger, (lms_idx, bone_names) in FINGER_DEFS.items():
        for k, name in enumerate(bone_names):
            bone_links[name] = (lms_idx[k], lms_idx[k + 1])

    lms_mean = load_landmarks(df, coordinate_scale(detect_coordinate_format(df))).mean(axis=0)
    lms_mean = lms_mean - lms_mean[0]   # wrist at the origin

    anatomy = {'lengths': {}, 'meta_vectors': {}}
    for finger, (lms_idx, _) in FINGER_DEFS.items():
        anatomy['meta_vectors'][finger] = lms_mean[lms_idx[0]]
    for bone_name, (idx1, idx2) in bone_links.items():
        anatomy['lengths'][bone_name] = np.linalg.norm(lms_mean[idx2] - lms_mean[idx1])
    return anatomy


# --- Forward kinematics -----------------------------------------------------
class CleanHandFK:
    """24 joint angles (rad) -> 21 landmarks, using a calibrated anatomy."""

    def __init__(self, anatomy):
        self.anatomy = anatomy
        self.R_splay = {}

        # Palm frame that does not depend on the finger pose
        v_idx = anatomy['meta_vectors']['Index']
        v_pnk = anatomy['meta_vectors']['Pinky']
        v_mid = anatomy['meta_vectors']['Middle']

        X_palm = v_idx - v_pnk                      # pinky -> index
        X_palm /= (np.linalg.norm(X_palm) + 1e-8)
        Z_palm = np.cross(X_palm, v_mid)            # palm normal
        Z_palm /= (np.linalg.norm(Z_palm) + 1e-8)
        Y_palm = np.cross(Z_palm, X_palm)           # extension direction
        Y_palm /= (np.linalg.norm(Y_palm) + 1e-8)

        for finger in ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
            Y_local = anatomy['meta_vectors'][finger].copy()
            Y_local /= (np.linalg.norm(Y_local) + 1e-8)

            if finger == 'Thumb':
                # Thumb: the flexion axis (X) points into the palm
                X_local = Z_palm.copy()
                Z_local = np.cross(X_local, Y_local)
                Z_local /= (np.linalg.norm(Z_local) + 1e-8)
                X_local = np.cross(Y_local, Z_local)
            else:
                X_local = np.cross(Y_local, Z_palm)
                if np.linalg.norm(X_local) < 1e-8:
                    X_local = X_palm.copy()
                X_local /= np.linalg.norm(X_local)
                Z_local = np.cross(X_local, Y_local)
                Z_local /= np.linalg.norm(Z_local)

            self.R_splay[finger] = np.column_stack([X_local, Y_local, Z_local])

    def forward(self, q):
        LMs = np.zeros((21, 3))
        R_wrist = Rz(q[2]) @ Ry(q[1]) @ Rx(q[0])    # Z-Y-X order (yaw, pitch, roll)
        lengths = self.anatomy['lengths']

        for idx, finger in enumerate(['Index', 'Middle', 'Ring', 'Pinky']):
            lm_idx = FINGER_DEFS[finger][0]
            q_idx = 8 + idx * 4
            LMs[lm_idx[0]] = R_wrist @ self.anatomy['meta_vectors'][finger]

            R_mcp = R_wrist @ self.R_splay[finger] @ Rx(q[q_idx]) @ Rz(q[q_idx + 1])
            LMs[lm_idx[1]] = LMs[lm_idx[0]] + R_mcp @ np.array([0, lengths[finger + '_Proximal'], 0])
            R_pip = R_mcp @ Rx(q[q_idx + 2])
            LMs[lm_idx[2]] = LMs[lm_idx[1]] + R_pip @ np.array([0, lengths[finger + '_Intermediate'], 0])
            R_dip = R_pip @ Rx(q[q_idx + 3])
            LMs[lm_idx[3]] = LMs[lm_idx[2]] + R_dip @ np.array([0, lengths[finger + '_Distal'], 0])

        # Thumb
        LMs[1] = R_wrist @ self.anatomy['meta_vectors']['Thumb']
        R_cmc = R_wrist @ self.R_splay['Thumb'] @ Rx(q[3]) @ Rz(q[4])
        LMs[2] = LMs[1] + R_cmc @ np.array([0, lengths['Thumb_Proximal'], 0])
        R_mcp_t = R_cmc @ Rx(q[5]) @ Rz(q[6])
        LMs[3] = LMs[2] + R_mcp_t @ np.array([0, lengths['Thumb_Intermediate'], 0])
        R_ip_t = R_mcp_t @ Rx(q[7])
        LMs[4] = LMs[3] + R_ip_t @ np.array([0, lengths['Thumb_Distal'], 0])
        return LMs


# --- Inverse kinematics: per-finger SLSQP with soft constraints --------------
class CleanHandIKA:
    """Alternative IK solver: wrist first, then each finger, with regularization."""

    def __init__(self, fk_model):
        self.fk = fk_model
        self.num_dofs = 24
        self.q_current = np.zeros(self.num_dofs)
        self.wrist_bounds = [(-1.5, 1.5)] * 3
        self.thumb_bounds = [(-1.0, 1.5), (-1.0, 1.0), (-0.5, 1.5), (-0.5, 0.5), (-0.5, 1.5)]
        self.finger_bounds = [(-0.5, 2.0), (-0.5, 0.5), (-0.2, 2.2), (-0.2, 1.8)]

    def enforce_rigid_skeleton(self, raw_targets):
        """Project noisy landmarks on a rigid skeleton with the calibrated lengths."""
        fixed_lms = np.zeros((21, 3))
        fixed_lms[0] = raw_targets[0]
        for finger, (lms_idx, bone_names) in FINGER_DEFS.items():
            mcp_idx = lms_idx[0]
            dir_meta = raw_targets[mcp_idx] - raw_targets[0]
            norm_meta = np.linalg.norm(dir_meta)
            if norm_meta > 1e-6:
                dir_meta /= norm_meta
            fixed_lms[mcp_idx] = fixed_lms[0] + dir_meta * np.linalg.norm(self.fk.anatomy['meta_vectors'][finger])

            parent_idx = mcp_idx
            for i, child_idx in enumerate(lms_idx[1:]):
                dir_bone = raw_targets[child_idx] - raw_targets[parent_idx]
                norm_bone = np.linalg.norm(dir_bone)
                dir_bone = dir_bone / norm_bone if norm_bone > 1e-6 else np.array([0.0, 1.0, 0.0])
                fixed_lms[child_idx] = fixed_lms[parent_idx] + dir_bone * self.fk.anatomy['lengths'][bone_names[i]]
                parent_idx = child_idx
        return fixed_lms

    def solve(self, target_lms):
        centered_targets = target_lms - target_lms[0]

        # Auto-scaling on the wrist -> middle MCP distance
        target_dist = np.linalg.norm(centered_targets[9])
        model_dist = np.linalg.norm(self.fk.anatomy['meta_vectors']['Middle'])
        if target_dist > 1e-6:
            centered_targets = centered_targets / (target_dist / model_dist)
        rigid_targets = self.enforce_rigid_skeleton(centered_targets)

        q_temp = np.copy(self.q_current)
        lambda_reg = 100.0

        # Phase 1: wrist, fitted on the palm landmarks
        def wrist_objective(q_wrist):
            q_test = np.copy(q_temp)
            q_test[0:3] = q_wrist
            pred = self.fk.forward(q_test)
            palm_lms = [1, 5, 9, 13, 17]
            mse = np.sum(np.linalg.norm(pred[palm_lms] - rigid_targets[palm_lms], axis=1) ** 2)
            return mse + lambda_reg * np.mean((q_wrist - self.q_current[0:3]) ** 2)

        opts = {'ftol': 1e-5, 'maxiter': 50}
        res_wrist = minimize(wrist_objective, x0=q_temp[0:3], method='SLSQP',
                             bounds=self.wrist_bounds, options=opts)
        # Rescue from a neutral pose when the fit is poor (~9 px mean error per knuckle)
        if res_wrist.fun > 400.0:
            rescue = minimize(wrist_objective, x0=np.zeros(3), method='SLSQP',
                              bounds=self.wrist_bounds, options=opts)
            if rescue.fun < res_wrist.fun:
                res_wrist = rescue
        q_temp[0:3] = res_wrist.x

        # Phase 2: each finger independently
        finger_configs = {
            'Thumb':  (3, 8,   [2, 3, 4],    self.thumb_bounds),
            'Index':  (8, 12,  [6, 7, 8],    self.finger_bounds),
            'Middle': (12, 16, [10, 11, 12], self.finger_bounds),
            'Ring':   (16, 20, [14, 15, 16], self.finger_bounds),
            'Pinky':  (20, 24, [18, 19, 20], self.finger_bounds),
        }
        for finger, (start_idx, end_idx, lms_idx, bounds) in finger_configs.items():
            def finger_objective(q_finger):
                q_test = np.copy(q_temp)
                q_test[start_idx:end_idx] = q_finger
                pred = self.fk.forward(q_test)
                weights = np.ones(len(lms_idx))
                weights[-1] = 2.0                  # fingertip counts double
                mse = np.sum(weights * np.linalg.norm(pred[lms_idx] - rigid_targets[lms_idx], axis=1) ** 2)
                reg = lambda_reg * np.mean((q_finger - self.q_current[start_idx:end_idx]) ** 2)
                # Biomechanical coupling: DIP ~ 2/3 PIP
                bio = 2000.0 * (q_finger[3] - 0.66 * q_finger[2]) ** 2 if finger != 'Thumb' else 0.0
                return mse + reg + bio

            res_finger = minimize(finger_objective, x0=q_temp[start_idx:end_idx], method='SLSQP',
                                  bounds=bounds, options=opts)
            if res_finger.fun > 300.0:             # ~8-9 px error per phalanx
                rescue = minimize(finger_objective, x0=np.zeros(len(bounds)), method='SLSQP',
                                  bounds=bounds, options=opts)
                if rescue.fun < res_finger.fun:
                    res_finger = rescue
            q_temp[start_idx:end_idx] = res_finger.x

        self.q_current = q_temp
        return self.q_current


# --- Inverse kinematics: damped least squares Jacobian (used by the pipeline) --
class JacobianIKA:
    def __init__(self, fk_model):
        self.fk = fk_model
        self.num_dofs = 24
        self.q_current = np.zeros(self.num_dofs)
        self.lambda_dls = 0.5   # damping: keeps the update bounded near singularities

        # Biomechanical weights: the palm must match first, then the fingertips;
        # intermediate joints adapt along the chain.
        weights = np.ones(21)
        weights[[0, 1, 5, 9, 13, 17]] = 10.0
        weights[[4, 8, 12, 16, 20]] = 3.0
        self.W = np.diag(np.repeat(weights, 3))   # 63 x 63

    def get_jacobian(self, q):
        """Numerical Jacobian of the FK (63 x 24) and the current FK output."""
        delta = 1e-4
        J = np.zeros((63, self.num_dofs))
        f0 = self.fk.forward(q).flatten()
        for i in range(self.num_dofs):
            q_step = np.copy(q)
            q_step[i] += delta
            J[:, i] = (self.fk.forward(q_step).flatten() - f0) / delta
        return J, f0

    def solve(self, target_lms, iterations=20):
        centered_targets = target_lms - target_lms[0]
        target_dist = np.linalg.norm(centered_targets[9])
        model_dist = np.linalg.norm(self.fk.anatomy['meta_vectors']['Middle'])
        if target_dist > 1e-6:
            centered_targets = centered_targets / (target_dist / model_dist)

        target_flat = centered_targets.flatten()
        q = np.copy(self.q_current)   # warm start from the previous frame

        for step in range(iterations):
            J, current = self.get_jacobian(q)
            error = target_flat - current
            if np.mean(np.abs(error)) < 0.5:
                break

            e_w = self.W @ error
            J_w = self.W @ J
            # Damped pseudo-inverse: (J^T J + lambda^2 I)^-1 J^T e
            H = J_w.T @ J_w + (self.lambda_dls ** 2) * np.eye(self.num_dofs)
            q = q + 0.8 * (np.linalg.inv(H) @ J_w.T @ e_w)   # step 0.8 avoids oscillations

            # If the first iteration finds a flipped hand, reset the wrist
            if step == 0 and np.mean(np.abs(e_w)) > 150.0:
                q[0:3] = 0.0

        self.q_current = q
        return q


def load_calibration(calib_file=config.HAND_CALIBRATION_FILE):
    if not calib_file.exists():
        print(f"Error: hand calibration '{calib_file}' not found!")
        return None
    print(f"Loading hand calibration from {calib_file.name}")
    return torch.load(calib_file, weights_only=False)


def process_full_kinematics_core(csv_path, calib_file=config.HAND_CALIBRATION_FILE):
    """Preprocessing + Jacobian IK on a landmark CSV -> ``<csv>_core_IKA_24DoF.csv`` (degrees)."""
    print("Running preprocessing + inverse kinematics (Jacobian)...")
    anatomy = load_calibration(calib_file)
    if anatomy is None:
        return None

    cleaned = preprocess_mediapipe_data(csv_path, anatomy)
    df_orig = pd.read_csv(csv_path)
    ika = JacobianIKA(CleanHandFK(anatomy))

    num_frames = len(cleaned)
    angles = np.zeros((num_frames, 24))
    print("\n--- Inverse kinematics on the preprocessed landmarks ---")
    for f in range(num_frames):
        target = cleaned[f] - cleaned[f, 0]   # wrist at the origin
        angles[f] = np.degrees(ika.solve(target))
        if f % 100 == 0 and f > 0:
            print(f"IK: {f:04d}/{num_frames} frames")

    df_angles = pd.DataFrame(angles, columns=[f'DoF_{i}' for i in range(24)])
    if 'Timestamp_LSL' in df_orig.columns:
        df_angles.insert(0, 'Timestamp_LSL', df_orig['Timestamp_LSL'].values)
    out_csv = str(csv_path).replace('.csv', '_core_IKA_24DoF.csv')
    df_angles.to_csv(out_csv, index=False)
    print(f"24-DoF angles (degrees) saved to: {out_csv.split('/')[-1]}\n")
    return df_angles
