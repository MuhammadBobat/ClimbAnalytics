"""
Hold detection and route/grade assignment. See DECISIONS.md D-017 and
ARCHITECTURE.md §8 for the full design and sources.

Core pipeline module (ARCHITECTURE.md §5), but runs at a different cadence to
the rest of the pipeline: holds are static and the camera is fixed (D-008),
so this runs once per clip on a handful of sampled frames, not once per
frame like pose_extraction.py will.

Requires a fine-tuned single-class ("hold") YOLOv8n checkpoint at
hold_detection_weights/hold_detector.pt -- NOT validation/weights/yolov8n.pt,
which is the person detector used for pose estimation (see D-011). Produced
by training/train_hold_detector.py, which requires a hand-annotated dataset
that does not exist yet (see DECISIONS.md D-017, "Human steps").

Colour-matching uses hold_colours_config.py's GYM_COLOUR_HSV_REFERENCE, which
is a placeholder (all None) until calibrated via
training/calibrate_hold_colours.py -- until then, every detected hold comes
back "unassigned" rather than a guessed colour/grade.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from hold_colours_config import GYM_COLOUR_GRADE_BANDS, GYM_COLOUR_HSV_REFERENCE

# --------------------------------------------------------------------------
# Engineering heuristics -- NOT literature values, NOT D-004 modelling
# thresholds. See ARCHITECTURE.md §8.1/§8.3: sampling count and both merge/
# colour thresholds are explicitly unvalidated, to be checked empirically on
# real footage before relying on them (logged as D-017c once swept).
# --------------------------------------------------------------------------
DEFAULT_N_SAMPLE_FRAMES = 8
DEFAULT_DETECTOR_CONF = 0.5
DEFAULT_MERGE_IOU_THRESHOLD = 0.5
DEFAULT_COLOUR_DISTANCE_THRESHOLD = 30.0  # placeholder scale, meaningless until HSV_REFERENCE is real

HOLD_CLASS_INDEX = 0  # single-class model: "hold" is the only class


@dataclass
class HoldDetection:
    bbox: tuple[float, float, float, float]  # x, y, w, h
    route_label: str  # grade band string, or "unassigned"
    colour_name: str | None
    detector_confidence: float
    colour_distance: float | None


def sample_frame_indices(total_frames: int, n: int) -> list[int]:
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(i * step) for i in range(n)]


def sample_frames(video_path: Path, n: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total, n)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def load_detector(device: str):
    from ultralytics import YOLO

    weights_dir = Path(__file__).parent / "hold_detection_weights"
    checkpoint = weights_dir / "hold_detector.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No fine-tuned hold detector at {checkpoint}. "
            "This requires training/train_hold_detector.py to have been run "
            "against an annotated dataset first -- see DECISIONS.md D-017 "
            "'Human steps'. There is no fallback here: a hold detector "
            "cannot run without its own fine-tuned weights."
        )
    return YOLO(str(checkpoint))


def detect_holds_in_frame(detector, frame_bgr: np.ndarray, conf_threshold: float, device: str):
    results = detector.predict(
        frame_bgr, classes=[HOLD_CLASS_INDEX], conf=conf_threshold, device=device, verbose=False
    )
    boxes_out = []
    if not results or results[0].boxes is None:
        return boxes_out
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
        boxes_out.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
    return boxes_out


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_detections_across_frames(
    all_boxes: list[tuple[float, float, float, float, float]],
    iou_threshold: float,
) -> list[tuple[float, float, float, float, float]]:
    """Greedy NMS-style merge: highest-confidence box per IoU cluster wins.
    See ARCHITECTURE.md §8.1 -- threshold is an unvalidated heuristic."""
    boxes_sorted = sorted(all_boxes, key=lambda b: b[4], reverse=True)
    kept: list[tuple[float, float, float, float, float]] = []
    for box in boxes_sorted:
        if all(_iou(box[:4], k[:4]) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def match_colour(
    crop_bgr: np.ndarray,
    hsv_reference: dict,
    grade_bands: dict,
    distance_threshold: float,
) -> tuple[str, str | None, float | None]:
    """Returns (route_label, colour_name, distance). "unassigned" whenever
    hsv_reference has no real (non-None) entries -- see hold_colours_config.py."""
    known = {c: v for c, v in hsv_reference.items() if v is not None}
    if not known or crop_bgr.size == 0:
        return "unassigned", None, None

    h, w = crop_bgr.shape[:2]
    margin_y, margin_x = int(h * 0.25), int(w * 0.25)
    central = crop_bgr[margin_y : h - margin_y, margin_x : w - margin_x]
    if central.size == 0:
        central = crop_bgr
    hsv = cv2.cvtColor(central, cv2.COLOR_BGR2HSV)
    med_h, med_s, med_v = (float(x) for x in np.median(hsv.reshape(-1, 3), axis=0))

    best_colour, best_dist = None, float("inf")
    for colour, ((h_min, h_max), (s_min, s_max), (v_min, v_max)) in known.items():
        h_mid = ((h_min + h_max) / 2) % 180
        h_dist = min(abs(med_h - h_mid), 180 - abs(med_h - h_mid))
        s_mid, v_mid = (s_min + s_max) / 2, (v_min + v_max) / 2
        dist = (h_dist**2 + (med_s - s_mid) ** 2 + (med_v - v_mid) ** 2) ** 0.5
        if dist < best_dist:
            best_colour, best_dist = colour, dist

    if best_colour is None or best_dist > distance_threshold:
        return "unassigned", None, best_dist if best_colour else None
    return grade_bands.get(best_colour, "unassigned"), best_colour, best_dist


def process_clip(
    video_path: Path, detector, device: str, args
) -> list[HoldDetection]:
    sampled = sample_frames(video_path, args.n_sample_frames)
    if not sampled:
        return []

    all_boxes = []
    for _, frame in sampled:
        all_boxes.extend(detect_holds_in_frame(detector, frame, args.detector_conf, device))

    merged = merge_detections_across_frames(all_boxes, args.merge_iou_threshold)

    representative_frame = sampled[0][1]
    detections = []
    for x1, y1, x2, y2, conf in merged:
        crop = representative_frame[int(y1) : int(y2), int(x1) : int(x2)]
        route_label, colour_name, distance = match_colour(
            crop, GYM_COLOUR_HSV_REFERENCE, GYM_COLOUR_GRADE_BANDS, args.colour_distance_threshold
        )
        detections.append(
            HoldDetection(
                bbox=(x1, y1, x2 - x1, y2 - y1),
                route_label=route_label,
                colour_name=colour_name,
                detector_confidence=conf,
                colour_distance=distance,
            )
        )
    return detections, representative_frame


def draw_overlay(frame: np.ndarray, detections: list[HoldDetection]) -> np.ndarray:
    frame = frame.copy()
    for det in detections:
        x, y, w, h = (int(v) for v in det.bbox)
        color = (0, 200, 0) if det.route_label != "unassigned" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = det.route_label if det.route_label != "unassigned" else "?"
        cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("footage/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("hold_detection_output"))
    parser.add_argument("--n-sample-frames", type=int, default=DEFAULT_N_SAMPLE_FRAMES)
    parser.add_argument("--detector-conf", type=float, default=DEFAULT_DETECTOR_CONF)
    parser.add_argument("--merge-iou-threshold", type=float, default=DEFAULT_MERGE_IOU_THRESHOLD)
    parser.add_argument("--colour-distance-threshold", type=float, default=DEFAULT_COLOUR_DISTANCE_THRESHOLD)
    parser.add_argument("--device", type=str, default=None)
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

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    videos = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in video_extensions)
    if not videos:
        print(f"No video files found in {args.input_dir}.")
        sys.exit(0)

    detector = load_detector(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in videos:
        print(f"Processing {video_path.name}...")
        detections, representative_frame = process_clip(video_path, detector, device, args)

        stem = video_path.stem
        csv_path = args.output_dir / f"{stem}_holds.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "route_label", "colour_name", "detector_confidence", "colour_distance"])
            for det in detections:
                writer.writerow([
                    video_path.name, *[f"{v:.1f}" for v in det.bbox],
                    det.route_label, det.colour_name or "",
                    f"{det.detector_confidence:.4f}",
                    f"{det.colour_distance:.2f}" if det.colour_distance is not None else "",
                ])

        overlay_path = args.output_dir / f"{stem}_holds_overlay.jpg"
        cv2.imwrite(str(overlay_path), draw_overlay(representative_frame, detections))

        print(f"  {len(detections)} holds detected -> {csv_path}, {overlay_path}")


if __name__ == "__main__":
    main()
