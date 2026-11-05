"""Markerless 3D hand tracking from a webcam, streamed on Lab Streaming Layer.

Google MediaPipe Hand Landmarker extracts the 21 hand landmarks; their
*world* coordinates (metres, wrist-centred) are flattened into 63 values and
pushed on the ``MediaPipe_Kinematics`` LSL stream, time-stamped with the LSL
clock at the moment the frame is captured (not when inference finishes).

Keys:  R  start/stop a local CSV recording   Q  quit

Usage:
    python -m acquisition.hand_tracking_stream [--camera 0]
"""
import argparse
import datetime
import sys
import threading
import time

import cv2
import mediapipe as mp
import pandas as pd
from pylsl import StreamInfo, StreamOutlet, local_clock

import config

N_LANDMARKS = 21
LSL_CHANNELS = N_LANDMARKS * 3   # x, y, z per landmark
FONT = cv2.FONT_HERSHEY_DUPLEX


class HandTracker:
    def __init__(self):
        self.outlet = StreamOutlet(StreamInfo(config.LSL_KINEMATICS_STREAM, "Kinematics",
                                              LSL_CHANNELS, 0, "float32", "mediapipe_hand_01"))
        self.lock = threading.Lock()
        self.latest_landmarks = None     # image coordinates, used for drawing
        self.is_recording = False
        self.recording_data = []
        self.recording_start_time = 0.0
        self.status_message = ""
        self.status_message_time = 0.0

    # Called by MediaPipe on its own thread
    def on_result(self, result, output_image, timestamp_ms):
        with self.lock:
            if not (result.hand_landmarks and result.hand_world_landmarks):
                self.latest_landmarks = None
                return
            self.latest_landmarks = result.hand_landmarks[0]
            world_landmarks = result.hand_world_landmarks[0]   # metric 3D coordinates

            # Capture-time stamp (removes MediaPipe processing latency)
            lsl_timestamp = timestamp_ms / 1000.0
            coords = [c for lm in world_landmarks for c in (lm.x, lm.y, lm.z)]
            self.outlet.push_sample(coords, lsl_timestamp)
            if self.is_recording:
                self.recording_data.append([lsl_timestamp] + coords)

    def set_status(self, message):
        self.status_message = message
        self.status_message_time = time.time()

    def toggle_recording(self):
        with self.lock:
            if not self.is_recording:
                self.recording_data = []
                self.recording_start_time = time.time()
                self.is_recording = True
                self.set_status("Recording started")
                return
            self.is_recording = False
            data, self.recording_data = self.recording_data, []
        self.save_recording(data)

    def save_recording(self, data):
        if not data:
            self.set_status("No data to save.")
            return
        columns = ["Timestamp_LSL"] + [f"LM_{i}_{ax}" for i in range(N_LANDMARKS) for ax in "XYZ"]
        out_dir = config.DATA_DIR / "debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = out_dir / f"kinematics_recording_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        pd.DataFrame(data, columns=columns).to_csv(filename, index=False)
        self.set_status(f"Saved to: {filename.name}")

    def draw_overlay(self, frame, fps):
        with self.lock:
            if self.latest_landmarks:
                h, w = frame.shape[:2]
                for lm in self.latest_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

        cv2.putText(frame, f"FPS: {int(fps)} | LSL Streaming", (10, 30), FONT, 0.6, (0, 255, 0), 1)
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            cv2.putText(frame, f"REC * {int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
                        (10, 60), FONT, 0.6, (0, 0, 255), 1)
        elif time.time() - self.status_message_time < 4:
            cv2.putText(frame, self.status_message, (10, 60), FONT, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Press 'R' to record", (10, 60), FONT, 0.6, (255, 255, 255), 1)


def open_camera(index):
    # AVFoundation is the native (and most stable) backend on macOS
    cap = (cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin"
           else cv2.VideoCapture(index))
    # A lower resolution keeps the frame rate stable while MediaPipe runs in parallel
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    args = parser.parse_args()

    tracker = HandTracker()
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(config.HAND_LANDMARKER_MODEL)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=tracker.on_result,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = open_camera(args.camera)
    prev_frame_time = 0.0
    last_mp_ts = 0

    print(f"Streaming LSL '{config.LSL_KINEMATICS_STREAM}'... Press 'R' to record, 'Q' to quit.")
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            frame_lsl_time = local_clock()   # LSL time at the moment of capture
            if not success:
                print("Error: cannot read frames from the webcam.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                tracker.toggle_recording()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # MediaPipe requires strictly increasing timestamps
            frame_timestamp_ms = max(int(frame_lsl_time * 1000), last_mp_ts + 1)
            last_mp_ts = frame_timestamp_ms
            try:
                landmarker.detect_async(mp_image, frame_timestamp_ms)
            except Exception:
                continue

            now = time.time()
            fps = 1 / (now - prev_frame_time) if now > prev_frame_time else 0
            prev_frame_time = now
            tracker.draw_overlay(frame, fps)
            cv2.imshow("MediaPipe Kinematics LSL", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Streaming stopped.")

    if tracker.is_recording:
        print("Saving the pending recording...")
        tracker.toggle_recording()


if __name__ == "__main__":
    main()
