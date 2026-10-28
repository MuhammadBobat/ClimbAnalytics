"""
Step 6: overlay videos (skeleton + z readout) for a few clips, for visual
inspection. Exploratory spike. Not part of the pipeline.

Usage:
    ./.venv/bin/python make_overlays.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = REPO_ROOT / "footage" / "test"
MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_heavy.task"
OUTPUT_DIR = Path(__file__).parent / "output"
MAX_FRAMES = 100

OVERLAY_CLIPS = ["IMG_2412", "IMG_2416", "IMG_2420"]

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]


def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH), delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def process(clip: str):
    video_path = CLIPS_DIR / f"{clip}.MOV"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(OUTPUT_DIR / f"{clip}_blazepose_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    landmarker = make_landmarker()
    frame_idx = 0
    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(frame_idx * 1000 / fps))

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
            for x, y in pts:
                cv2.circle(frame, (x, y), 3, (0, 200, 0), -1)

            if result.pose_world_landmarks:
                wl = result.pose_world_landmarks[0]
                hip_mid_z = (wl[23].z + wl[24].z) / 2
                sh_mid_z = (wl[11].z + wl[12].z) / 2
                torso_lean = sh_mid_z - hip_mid_z
                cv2.putText(frame, f"torso_lean_z={torso_lean:+.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"L_ankle_z-hip={wl[27].z - hip_mid_z:+.3f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"R_ankle_z-hip={wl[28].z - hip_mid_z:+.3f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "NO DETECTION", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    landmarker.close()
    print(f"{clip}: wrote overlay ({frame_idx} frames)")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for clip in OVERLAY_CLIPS:
        process(clip)


if __name__ == "__main__":
    main()
