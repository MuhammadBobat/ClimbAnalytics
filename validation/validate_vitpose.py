"""
D-002a validation script — ViTPose-L on own footage.

Standalone: deliberately NOT part of the main pipeline module boundaries in
ARCHITECTURE.md §5. This script exists only to answer "does ViTPose-L track
this project's own footage well enough to build on?" before pose_extraction.py
and downstream modules are written. Do not import from this file elsewhere.

For each video in --input-dir, this script:
  1. Detects the climber's bounding box per frame (YOLOv8n, person class only —
     used purely as a box source, not as the YOLOv8-pose fallback; see
     DECISIONS.md D-011).
  2. Runs ViTPose-L (usyd-community/vitpose-plus-large via HuggingFace
     `transformers`; see DECISIONS.md D-011, D-012) on that box to get 17
     COCO keypoints with per-keypoint confidence.
  3. Writes an overlay video (skeleton + bbox, colour-coded by confidence)
     for visual inspection.
  4. Logs every keypoint's per-frame (x, y, confidence) to a CSV.
  5. Flags frames with missing detections or unusually low confidence into a
     second CSV, for quick triage without scrubbing the whole video.

Usage:
    python validation/validate_vitpose.py
    python validation/validate_vitpose.py --input-dir footage/test --max-frames 300

Requirements: see validation/requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------
# QA/visual-inspection defaults — NOT modelling thresholds.
#
# These control what this validation script flags for a human to go look at.
# They are unrelated to DECISIONS.md D-004 (the empirical sweep methodology
# for the four-state classifier's z-score threshold and hysteresis duration,
# and the occlusion tier boundaries). Do not reuse these values as if they
# were D-004 outputs.
# --------------------------------------------------------------------------
DEFAULT_BBOX_DETECTOR_CONF = 0.5       # YOLOv8n person-detection confidence
DEFAULT_KEYPOINT_LOW_CONF = 0.3        # below this, a keypoint is "low confidence"
DEFAULT_FRAME_MEAN_CONF_FLAG = 0.5     # flag frame if mean keypoint confidence drops below this
DEFAULT_MAX_LOW_CONF_KEYPOINTS = 5     # flag frame if this many (of 17) keypoints are low-confidence

# ViTPose++ MoE expert index for the COCO-trained head (0=COCO val, 1=AIC,
# 2=MPII, 3=AP-10K, 4=APT-36K, 5=COCO-WholeBody). Not a modelling choice —
# this is fixed by which expert the checkpoint calls "COCO".
VITPOSE_COCO_DATASET_INDEX = 0

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Standard 17-keypoint COCO order/names, as confirmed against the actual
# loaded ViTPose checkpoint config (DECISIONS.md D-012) — do not assume this
# without re-checking if the checkpoint is ever changed.
COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton edges by COCO17_NAMES index, for overlay drawing only.
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                            # left arm
    (6, 8), (8, 10),                           # right arm
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15),                        # left leg
    (12, 14), (14, 16),                        # right leg
]


def confidence_color(score: float) -> tuple[int, int, int]:
    """BGR colour for a confidence score, for overlay drawing only."""
    if score >= 0.7:
        return (0, 200, 0)      # green
    if score >= DEFAULT_KEYPOINT_LOW_CONF:
        return (0, 200, 255)    # yellow/orange
    return (0, 0, 255)          # red


@dataclass
class PersonBox:
    x: float
    y: float
    w: float
    h: float
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def as_xywh(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]


def select_person_box(
    boxes: list[PersonBox], prev_center: tuple[float, float] | None
) -> PersonBox | None:
    """Single-climber lock-on heuristic from ARCHITECTURE.md §2:
    nearest-to-previous-frame if we already have a lock, else
    largest-confidence-box (covers frame 1, and re-acquisition after a gap).
    """
    if not boxes:
        return None
    if prev_center is None:
        return max(boxes, key=lambda b: b.conf)
    px, py = prev_center
    return min(boxes, key=lambda b: (b.center[0] - px) ** 2 + (b.center[1] - py) ** 2)


def load_models(device: str):
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    from ultralytics import YOLO

    weights_dir = Path(__file__).parent / "weights"
    weights_dir.mkdir(exist_ok=True)

    print("Loading YOLOv8n (person-box detector)...")
    detector = YOLO(str(weights_dir / "yolov8n.pt"))

    print("Loading ViTPose-L (usyd-community/vitpose-plus-large)...")
    pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-large")
    pose_model = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-plus-large"
    ).to(device)
    pose_model.eval()

    return detector, pose_processor, pose_model


def detect_person_boxes(detector, frame_bgr, conf_threshold: float, device: str) -> list[PersonBox]:
    results = detector.predict(
        frame_bgr, classes=[0], conf=conf_threshold, device=device, verbose=False
    )
    boxes_out: list[PersonBox] = []
    if not results:
        return boxes_out
    boxes = results[0].boxes
    if boxes is None:
        return boxes_out
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
        boxes_out.append(PersonBox(x=float(x1), y=float(y1), w=float(x2 - x1), h=float(y2 - y1), conf=float(conf)))
    return boxes_out


def run_pose(pose_processor, pose_model, frame_rgb_pil, box: PersonBox, device: str):
    boxes_arr = np.array([box.as_xywh()], dtype=np.float32)
    inputs = pose_processor(frame_rgb_pil, boxes=[boxes_arr], return_tensors="pt").to(device)
    # vitpose-plus-large is a mixture-of-experts checkpoint (6 expert heads) and
    # requires an explicit dataset_index selecting which expert to route through.
    # 0 = COCO validation expert. Required for "-plus-*" checkpoints only; see
    # DECISIONS.md D-011 and https://huggingface.co/docs/transformers/model_doc/vitpose
    dataset_index = torch.tensor([VITPOSE_COCO_DATASET_INDEX], device=device)
    with torch.no_grad():
        outputs = pose_model(**inputs, dataset_index=dataset_index)
    # threshold=0.0: we want every keypoint logged, even low-confidence ones —
    # this script does its own flagging downstream rather than relying on the
    # library to silently drop points.
    pose_results = pose_processor.post_process_pose_estimation(
        outputs, boxes=[boxes_arr], threshold=0.0
    )
    person = pose_results[0][0]
    keypoints = person["keypoints"].cpu().numpy()   # (17, 2) in original image coords
    scores = person["scores"].cpu().numpy()         # (17,)
    return keypoints, scores


def draw_overlay(frame, box: PersonBox | None, keypoints, scores):
    if box is None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 200), -1)
        cv2.putText(
            frame, "NO DETECTION", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
        )
        return frame

    x1, y1 = int(box.x), int(box.y)
    x2, y2 = int(box.x + box.w), int(box.y + box.h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

    pts = keypoints.astype(int)
    for i, j in SKELETON_EDGES:
        if scores[i] > 0 and scores[j] > 0:
            color = confidence_color(min(scores[i], scores[j]))
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2)
    for idx, (x, y) in enumerate(pts):
        cv2.circle(frame, (int(x), int(y)), 4, confidence_color(scores[idx]), -1)
    return frame


def process_video(
    video_path: Path,
    output_dir: Path,
    detector,
    pose_processor,
    pose_model,
    device: str,
    args,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: could not open {video_path}", file=sys.stderr)
        return {"video": video_path.name, "error": "could not open file"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem = video_path.stem
    overlay_path = output_dir / f"{stem}_overlay.mp4"
    keypoints_csv_path = output_dir / f"{stem}_keypoints.csv"
    flags_csv_path = output_dir / f"{stem}_flags.csv"

    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    kp_file = open(keypoints_csv_path, "w", newline="")
    kp_writer = csv.writer(kp_file)
    kp_writer.writerow([
        "video", "frame_idx", "timestamp_s", "keypoint_id", "keypoint_name",
        "x", "y", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    ])

    flag_file = open(flags_csv_path, "w", newline="")
    flag_writer = csv.writer(flag_file)
    flag_writer.writerow([
        "video", "frame_idx", "timestamp_s", "reason", "mean_confidence",
        "min_confidence", "num_low_conf_keypoints",
    ])

    prev_center: tuple[float, float] | None = None
    frame_idx = 0
    n_missing = 0
    n_flagged = 0
    confidence_sum = 0.0
    confidence_count = 0

    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        timestamp_s = frame_idx / fps

        boxes = detect_person_boxes(detector, frame, args.bbox_conf, device)
        box = select_person_box(boxes, prev_center)

        if box is None:
            n_missing += 1
            flag_writer.writerow([video_path.name, frame_idx, f"{timestamp_s:.3f}", "missing_detection", "", "", ""])
            writer.write(draw_overlay(frame, None, None, None))
            frame_idx += 1
            continue

        prev_center = box.center
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints, scores = run_pose(pose_processor, pose_model, Image.fromarray(frame_rgb), box, device)

        for kp_id, name in enumerate(COCO17_NAMES):
            x, y = keypoints[kp_id]
            kp_writer.writerow([
                video_path.name, frame_idx, f"{timestamp_s:.3f}", kp_id, name,
                f"{x:.2f}", f"{y:.2f}", f"{scores[kp_id]:.4f}",
                f"{box.x:.1f}", f"{box.y:.1f}", f"{box.w:.1f}", f"{box.h:.1f}",
            ])

        mean_conf = float(np.mean(scores))
        min_conf = float(np.min(scores))
        n_low_conf = int(np.sum(scores < DEFAULT_KEYPOINT_LOW_CONF))
        confidence_sum += mean_conf
        confidence_count += 1

        reasons = []
        if mean_conf < DEFAULT_FRAME_MEAN_CONF_FLAG:
            reasons.append("low_mean_confidence")
        if n_low_conf >= DEFAULT_MAX_LOW_CONF_KEYPOINTS:
            reasons.append("many_low_confidence_keypoints")
        if reasons:
            n_flagged += 1
            flag_writer.writerow([
                video_path.name, frame_idx, f"{timestamp_s:.3f}", "+".join(reasons),
                f"{mean_conf:.4f}", f"{min_conf:.4f}", n_low_conf,
            ])

        writer.write(draw_overlay(frame, box, keypoints, scores))

        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.time() - t_start
            print(f"  {video_path.name}: frame {frame_idx} ({frame_idx / elapsed:.1f} fps processing)")

    cap.release()
    writer.release()
    kp_file.close()
    flag_file.close()

    elapsed = time.time() - t_start
    mean_overall_conf = confidence_sum / confidence_count if confidence_count else float("nan")

    print(
        f"  Done: {frame_idx} frames in {elapsed:.1f}s "
        f"({n_missing} missing-detection, {n_flagged} flagged, "
        f"mean confidence {mean_overall_conf:.3f})"
    )
    print(f"  -> {overlay_path}")
    print(f"  -> {keypoints_csv_path}")
    print(f"  -> {flags_csv_path}")

    return {
        "video": video_path.name,
        "total_frames": frame_idx,
        "missing_detection_frames": n_missing,
        "flagged_frames": n_flagged,
        "mean_confidence": mean_overall_conf,
        "processing_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("footage/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation/output"))
    parser.add_argument("--bbox-conf", type=float, default=DEFAULT_BBOX_DETECTOR_CONF,
                         help="YOLOv8n person-detection confidence threshold")
    parser.add_argument("--device", type=str, default=None,
                         help="cpu / mps / cuda — auto-detected if not set")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="limit frames per video, for a quick smoke test")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if not args.input_dir.exists():
        print(f"Input directory {args.input_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    videos = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(
            f"No video files found in {args.input_dir} "
            f"(looked for {sorted(VIDEO_EXTENSIONS)}). "
            "Add 2-3 test clips there and re-run.",
        )
        sys.exit(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        detector, pose_processor, pose_model = load_models(device)
    except Exception as exc:
        print(
            "\nViTPose-L / detector setup failed to load. This is exactly the "
            "'impractical to set up' case D-002 anticipates — do NOT silently "
            "fall back to YOLOv8-pose. Report this error back before deciding "
            "how to proceed:\n",
            file=sys.stderr,
        )
        raise

    print(f"\nFound {len(videos)} video(s) in {args.input_dir}:")
    for v in videos:
        print(f"  - {v.name}")
    print()

    summary_rows = []
    for video_path in videos:
        print(f"Processing {video_path.name}...")
        summary_rows.append(
            process_video(video_path, args.output_dir, detector, pose_processor, pose_model, device, args)
        )
        print()

    summary_path = args.output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "total_frames", "missing_detection_frames",
                        "flagged_frames", "mean_confidence", "processing_seconds", "error"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
