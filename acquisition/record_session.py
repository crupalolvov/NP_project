"""Synchronized recorder for the EMG and hand-kinematics LSL streams.

Resolves ``OTB_S64_EMG`` and ``MediaPipe_Kinematics`` on the local network,
continuously drains both inlets and, while recording, stores every chunk with
its LSL timestamp. Each recording is saved as a pair of CSV files sharing the
same clock (``Timestamp_LSL``):

    data/recordings/trial_<N>_EMG.csv          Timestamp_LSL, EMG_1..EMG_32 (µV)
    data/recordings/trial_<N>_Kinematics.csv   Timestamp_LSL, LM_0_X..LM_20_Z (m)

Trial numbers continue from the highest one already on disk, so existing
recordings are never overwritten.

Keys:  R  start/stop a trial   Q  quit

Usage:
    python -m acquisition.record_session
"""
import re
import sys
import threading
import time

import pandas as pd
from pylsl import StreamInlet, resolve_byprop

import config

try:
    from pynput import keyboard
except ImportError:
    print("ERROR: the 'pynput' package is required for keyboard control:\n  pip install pynput")
    sys.exit(1)


def next_trial_number():
    """First trial number not yet used in the recordings folder."""
    used = [int(m.group(1)) for p in config.RECORDINGS_DIR.glob("trial_*_EMG.csv")
            if (m := re.match(r"trial_(\d+)_EMG\.csv$", p.name))]
    return max(used, default=0) + 1


def save_trial(trial, emg_data, emg_ts, kin_data, kin_ts):
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    if emg_data:
        df_emg = pd.DataFrame(emg_data, columns=[f"EMG_{i + 1}" for i in range(len(emg_data[0]))])
        df_emg.insert(0, "Timestamp_LSL", emg_ts)
        path = config.trial_file(trial, "EMG")
        df_emg.to_csv(path, index=False)
        saved.append(path)
    if kin_data:
        n_landmarks = len(kin_data[0]) // 3
        cols = [f"LM_{i}_{ax}" for i in range(n_landmarks) for ax in "XYZ"]
        df_kin = pd.DataFrame(kin_data, columns=cols)
        df_kin.insert(0, "Timestamp_LSL", kin_ts)
        path = config.trial_file(trial, "Kinematics")
        df_kin.to_csv(path, index=False)
        saved.append(path)
    print("Saved:" if saved else "Nothing to save.")
    for p in saved:
        print(f"  - {config.rel(p)}")


def record_synchronized_data():
    print("Looking for LSL streams on the local network...")
    emg_streams = resolve_byprop("name", config.LSL_EMG_STREAM)
    kin_streams = resolve_byprop("name", config.LSL_KINEMATICS_STREAM)
    if not emg_streams or not kin_streams:
        print("Error: could not find both streams. Make sure the EMG and MediaPipe "
              "streamers are running.")
        return

    inlet_emg = StreamInlet(emg_streams[0])
    inlet_kin = StreamInlet(kin_streams[0])
    print("Streams found, inlets ready.")

    state = {"recording": False, "running": True, "trial": next_trial_number()}
    lock = threading.Lock()
    emg_data, emg_ts, kin_data, kin_ts = [], [], [], []

    def on_press(key):
        char = getattr(key, "char", None)
        if not char:
            return
        char = char.lower()
        if char == "r":
            if not state["recording"]:
                with lock:
                    for buf in (emg_data, emg_ts, kin_data, kin_ts):
                        buf.clear()
                    state["recording"] = True
                print(f"\n[REC] Recording trial_{state['trial']}... press 'R' again to stop.")
            else:
                with lock:
                    state["recording"] = False
                    snapshot = (list(emg_data), list(emg_ts), list(kin_data), list(kin_ts))
                print(f"\n[REC] Stopped trial_{state['trial']}. Exporting CSV files...")
                save_trial(state["trial"], *snapshot)
                state["trial"] += 1
                print(f"\nReady: press 'R' to record trial_{state['trial']}, 'Q' to quit.")
        elif char == "q":
            state["running"] = False
            return False   # stops the keyboard listener

    keyboard.Listener(on_press=on_press).start()
    print(f"\nReady: press 'R' to record trial_{state['trial']}, 'Q' to quit.")

    while state["running"]:
        # Always drain the LSL buffers, so stale samples never end up in a trial
        chunk_emg, ts_emg = inlet_emg.pull_chunk(timeout=0.0)
        chunk_kin, ts_kin = inlet_kin.pull_chunk(timeout=0.0)
        if state["recording"]:
            with lock:
                if ts_emg:
                    emg_data.extend(chunk_emg)
                    emg_ts.extend(ts_emg)
                if ts_kin:
                    kin_data.extend(chunk_kin)
                    kin_ts.extend(ts_kin)
        time.sleep(0.01)

    print("\nClosing recorder.")


if __name__ == "__main__":
    record_synchronized_data()
