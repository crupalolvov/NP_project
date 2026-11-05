# Legacy prototypes

Earlier iterations of the pipeline, kept for reference only. They are not maintained and
use the original flat-folder paths, so they will not run as-is.

| File | Superseded by |
|---|---|
| `angles_kalman_offline.py`, `offline_angles_calculator.py`, `kinematics_online_kalman.py` | joint angles from 3-point geometry + 1D Kalman smoothing → `kinematics/hand_kinematics.py` (rigid skeleton + inverse kinematics) |
| `IKA.py`, `test_lunghezza.py` | first inverse-kinematics and bone-length experiments → `kinematics/hand_kinematics.py` |
| `feature_ext_v0.py` | `decoder/feature_extraction.py` |
| `animate_hand.py`, `eval_graph.py` | `analysis/animate_kinematics.py`, `analysis/metrics_report.py` |
| `test_EMG.py` | EMG quality diagnostics → `analysis/emg_spectrum.py` |
| `predicted_kinematics_offline.csv` | output of an early decoder version |
