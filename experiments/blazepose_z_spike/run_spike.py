"""
Step 1: run MediaPipe PoseLandmarker (BlazePose GHUM, heavy model) on the
project's test clips and save per-frame landmarks to CSV.

Exploratory spike. Not part of the pipeline. Not a logged decision.

Scope: matches the first N frames of each clip to the frame range already
covered by the saved ViTPose-L keypoints in validation/output/ (100 frames,
from the D-002a-corrected dataset_index=0 run), so later comparison steps
are apples-to-apples. Does not import any project runtime module - reads
video files directly, nothing else from the main repo.

Usage (from this folder, using its own venv):
    ./.venv/bin/python run_spike.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = REPO_ROOT / "footage" / "test"
VITPOSE_DIR = REPO_ROOT / "validation" / "output"
MODEL_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "output"

MAX_FRAMES = 100  # matches the saved ViTPose-L CSVs' frame range exactly

# Clips with a matching saved ViTPose-L keypoints CSV (confirmed present in
# validation/output/ before running this). Indoor_Bouldering_V3_Rock_Spot has
# a saved CSV but no source video left in footage/test/ - excluded, not
# silently dropped (see REPORT.md).
CLIP_NAMES = ["IMG_2412", "IMG_2416", "IMG_2417", "IMG_2419", "IMG_2420"]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]


def ensure_model() -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "pose_landmarker_heavy.task"
    if not model_path.exists():
        import urllib.request
        print(f"Downloading {MODEL_URL} ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def process_clip(video_path: Path, make_landmarker, writer_image, writer_world) -> dict:
    # A fresh PoseLandmarker per clip: VIDEO running mode requires strictly
    # monotonically increasing timestamps *per landmarker instance*, and each
    # clip's timestamps restart from 0 - reusing one instance across clips
    # throws "Input timestamp must be monotonically increasing." Found by
    # actually running this on a second clip, not anticipated in advance.
    landmarker = make_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    n_detected = 0

    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            n_detected += 1
            lm = result.pose_landmarks[0]
            for idx, name in enumerate(LANDMARK_NAMES):
                writer_image.writerow([
                    video_path.stem, frame_idx, f"{timestamp_ms/1000:.3f}", idx, name,
                    f"{lm[idx].x:.6f}", f"{lm[idx].y:.6f}", f"{lm[idx].z:.6f}", f"{lm[idx].visibility:.4f}",
                ])
            if result.pose_world_landmarks:
                wl = result.pose_world_landmarks[0]
                for idx, name in enumerate(LANDMARK_NAMES):
                    writer_world.writerow([
                        video_path.stem, frame_idx, f"{timestamp_ms/1000:.3f}", idx, name,
                        f"{wl[idx].x:.6f}", f"{wl[idx].y:.6f}", f"{wl[idx].z:.6f}",
                    ])
        frame_idx += 1

    cap.release()
    landmarker.close()
    return {"video": video_path.stem, "frames_read": frame_idx, "frames_detected": n_detected}


def main():
    if not CLIPS_DIR.exists():
        print(f"{CLIPS_DIR} does not exist.", file=sys.stderr)
        sys.exit(1)

    model_path = ensure_model()

    def make_landmarker():
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(options)

    OUTPUT_DIR.mkdir(exist_ok=True)
    image_csv_path = OUTPUT_DIR / "blazepose_image_landmarks.csv"
    world_csv_path = OUTPUT_DIR / "blazepose_world_landmarks.csv"

    with open(image_csv_path, "w", newline="") as f_img, open(world_csv_path, "w", newline="") as f_world:
        writer_image = csv.writer(f_img)
        writer_image.writerow(["video", "frame_idx", "timestamp_s", "landmark_id", "landmark_name", "x", "y", "z", "visibility"])
        writer_world = csv.writer(f_world)
        writer_world.writerow(["video", "frame_idx", "timestamp_s", "landmark_id", "landmark_name", "x", "y", "z"])

        summary = []
        for name in CLIP_NAMES:
            video_path = CLIPS_DIR / f"{name}.MOV"
            if not video_path.exists():
                print(f"WARNING: {video_path} not found, skipping")
                continue
            print(f"Processing {video_path.name} (first {MAX_FRAMES} frames)...")
            stats = process_clip(video_path, make_landmarker, writer_image, writer_world)
            print(f"  {stats['frames_detected']}/{stats['frames_read']} frames with a detected pose")
            summary.append(stats)

    summary_path = OUTPUT_DIR / "extraction_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "frames_read", "frames_detected"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nWrote {image_csv_path}, {world_csv_path}, {summary_path}")


if __name__ == "__main__":
    main()
