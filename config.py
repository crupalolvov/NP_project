"""Project-wide paths and constants.

Every script resolves files through this module, so the pipeline works no
matter which directory it is launched from. Run scripts from the repository
root as modules, e.g. ``python -m decoder.train``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Data -------------------------------------------------------------------
DATA_DIR = ROOT / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"        # synchronized EMG + kinematics trials
TENSORS_DIR = DATA_DIR / "tensors"              # train/val/test PyTorch tensors
CALIBRATION_DIR = DATA_DIR / "calibration"      # subject-specific hand anatomy
LABELS_FILE = DATA_DIR / "labels.json"          # manual gesture annotations

HAND_CALIBRATION_FILE = CALIBRATION_DIR / "hand_calibration.pt"
CALIBRATION_TRIAL = 1
CALIBRATION_FRAMES = (1530, 1700)   # steady open-hand segment of the calibration trial
CALIBRATION_PHOTO = CALIBRATION_DIR / "hand_open_reference.jpg"

# --- Models -----------------------------------------------------------------
MODELS_DIR = ROOT / "models"
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"   # Google MediaPipe
RPC_NET_WEIGHTS = MODELS_DIR / "rpc_net_weights.pth"
RPC_NET_ONNX = MODELS_DIR / "rpc_net.onnx"
BEST_HYPERPARAMETERS = MODELS_DIR / "best_hyperparameters.json"
OPTUNA_DB = MODELS_DIR / "optuna_study.db"

# --- Results ----------------------------------------------------------------
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# --- Lab Streaming Layer stream names ---------------------------------------
LSL_EMG_STREAM = "OTB_S64_EMG"
LSL_KINEMATICS_STREAM = "MediaPipe_Kinematics"

# --- Dataset ----------------------------------------------------------------
EMG_FS = 2000                       # Sessantaquattro sampling rate (Hz)
N_EMG_CHANNELS = 32
N_DOF = 24                          # degrees of freedom of the hand model
TRIAL_SPLITS = {
    "train": [1, 2, 3, 4, 7, 8, 10, 11, 13],
    "val": [5, 14],
    "test": [6, 9, 15],
}


def trial_file(trial: int, suffix: str) -> Path:
    """Path of a recording file, e.g. ``trial_file(6, "EMG_RMS")``."""
    return RECORDINGS_DIR / f"trial_{trial}_{suffix}.csv"


def prediction_file(trial: int, kind: str) -> Path:
    """Path of a decoder output; ``kind`` is ``"angles"`` or ``"lms"``."""
    return PREDICTIONS_DIR / f"trial_{trial}_predicted_kinematics_{kind}.csv"


def rel(path) -> str:
    """Path relative to the repository root when possible (for log messages)."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
