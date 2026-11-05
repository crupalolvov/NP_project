"""Static hand calibration from a single photo of the open hand.

Alternative to the video-based calibration (``extract_anatomy_from_csv``):
MediaPipe world landmarks of one photo give the bone lengths and metacarpal
vectors, converted to millimetres like the rest of the pipeline. The result
is saved to ``data/calibration/hand_calibration_photo.pt`` (so the default
calibration is never overwritten) and the detected skeleton is shown on the
photo.

Usage:
    python -m kinematics.photo_calibration [--image path.jpg] [--output file.pt]
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

import config
from kinematics.hand_kinematics import FINGER_DEFS

M_TO_MM = 1000.0   # world landmarks are in metres, the hand model works in mm


def main():
    parser = argparse.ArgumentParser(description="Hand calibration from a photo.")
    parser.add_argument("--image", type=Path, default=config.CALIBRATION_PHOTO)
    parser.add_argument("--output", type=Path, default=config.CALIBRATION_DIR / "hand_calibration_photo.pt")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"ERROR: photo '{args.image}' not found.")
        return
    if not config.HAND_LANDMARKER_MODEL.exists():
        print(f"ERROR: model file '{config.HAND_LANDMARKER_MODEL}' not found.")
        return

    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(config.HAND_LANDMARKER_MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        img = cv2.imread(str(args.image))
        results = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    if not results.hand_world_landmarks:
        print("ERROR: no hand detected. Use a photo where the palm is clearly visible.")
        return

    # World landmarks: metric, wrist at the origin -> real proportions
    lms = np.array([[lm.x, lm.y, lm.z] for lm in results.hand_world_landmarks[0]]) * M_TO_MM

    bones = {}
    for lms_idx, bone_names in FINGER_DEFS.values():
        for k, name in enumerate(bone_names):
            bones[name] = (lms_idx[k], lms_idx[k + 1])

    calib = {
        "lengths": {name: float(np.linalg.norm(lms[b] - lms[a])) for name, (a, b) in bones.items()},
        "meta_vectors": {finger: lms[idx[0]] - lms[0] for finger, (idx, _) in FINGER_DEFS.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calib, args.output)
    print(f"Calibration saved to {args.output}")

    # Preview of the detected skeleton
    image_lms = results.hand_landmarks[0]
    h, w, _ = img.shape
    px = [(int(lm.x * w), int(lm.y * h)) for lm in image_lms]
    for idx in (1, 5, 9, 13, 17):
        cv2.line(img, px[0], px[idx], (0, 255, 0), 3)
    for a, b in bones.values():
        cv2.line(img, px[a], px[b], (0, 255, 0), 3)
    for p in px:
        cv2.circle(img, p, 6, (0, 0, 255), -1)
    scale = min(800 / h, 800 / w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow("Calibration (press any key to close)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
