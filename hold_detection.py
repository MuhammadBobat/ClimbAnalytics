"""
Hold detection and route/grade assignment. See DECISIONS.md D-017, D-022 and
ARCHITECTURE.md §8 for the full design and sources.

Core pipeline module (ARCHITECTURE.md §5), but runs at a different cadence to
the rest of the pipeline: holds are static and the camera is fixed (D-008),
so this runs once per clip on a handful of sampled frames, not once per
frame like pose_extraction.py will.

Requires a fine-tuned three-class ("hold" / "descent_hold" / "volume")
YOLOv8n-seg instance-segmentation checkpoint at
hold_detection_weights/hold_detector_seg.pt -- NOT validation/weights/yolov8n.pt,
which is the person detector used for pose estimation (see D-011), and NOT
a box-format checkpoint (D-022 superseded the earlier single-class box
design). Produced by training/train_hold_detector.py, which requires a
polygon-annotated dataset that does not exist yet (see DECISIONS.md D-017,
"Human steps").

Class names/order are never assumed -- always read from the loaded model's
own detector.names (see resolve_class_names()), per D-022. Colour matching
runs only for "hold" detections, sampling pixels from inside each
detection's segmentation mask (eroded by a small, unvalidated pixel margin --
see DEFAULT_MASK_ERODE_PX), not from a bounding box. "descent_hold" and
"volume" detections are never colour-matched and never get a route label.

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
DEFAULT_MASK_ERODE_PX = 5  # placeholder, NOT measured -- see DECISIONS.md D-017c once swept


@dataclass
class HoldDetection:
    class_name: str  # "hold" | "descent_hold" | "volume" (D-022)
    bbox: tuple[float, float, float, float]  # x, y, w, h -- derived from the mask's bounding rect
    polygon: list[tuple[float, float]]
    route_label: str  # grade band string, or "unassigned"; always "unassigned" for non-"hold" classes
    colour_name: str | None
    detector_confidence: float
    colour_distance: float | None
    used_mask_erosion_fallback: bool = False  # True if erosion emptied the mask and the un-eroded mask was used instead


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
    checkpoint = weights_dir / "hold_detector_seg.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No fine-tuned hold detector at {checkpoint}. "
            "This requires training/train_hold_detector.py to have been run "
            "against a polygon-annotated dataset first -- see DECISIONS.md D-017 "
            "'Human steps' and D-022. There is no fallback here: a hold detector "
            "cannot run without its own fine-tuned segmentation weights."
        )
    return YOLO(str(checkpoint))


def resolve_class_names(detector) -> dict[int, str]:
    """The ONLY place class names are looked up. Never assume an index order
    (e.g. that 0 == "hold") -- read it from the loaded model itself, per
    D-022. detector.names is the ultralytics convention for this mapping."""
    return dict(detector.names)


def detect_holds_in_frame(
    detector, frame_bgr: np.ndarray, conf_threshold: float, device: str
) -> list[tuple[int, list[tuple[float, float]], tuple[float, float, float, float], float]]:
    """Returns (class_id, polygon_xy, bbox_xyxy, conf) tuples for ALL three
    classes -- no class filter, unlike the old single-class design. bbox is
    derived from the polygon's own bounding rectangle (D-022), not trusted
    from the model's own box output."""
    results = detector.predict(frame_bgr, conf=conf_threshold, device=device, verbose=False)
    out = []
    if not results or results[0].masks is None or results[0].boxes is None:
        return out
    polygons = results[0].masks.xy  # list of (N, 2) arrays, pixel coords
    class_ids = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    for polygon_xy, class_id, conf in zip(polygons, class_ids, confs):
        if len(polygon_xy) == 0:
            continue
        xs, ys = polygon_xy[:, 0], polygon_xy[:, 1]
        bbox_xyxy = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        polygon_list = [(float(x), float(y)) for x, y in polygon_xy]
        out.append((int(class_id), polygon_list, bbox_xyxy, float(conf)))
    return out


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
    all_detections: list[tuple[int, list[tuple[float, float]], tuple[float, float, float, float], float]],
    iou_threshold: float,
) -> list[tuple[int, list[tuple[float, float]], tuple[float, float, float, float], float]]:
    """Greedy NMS-style merge: highest-confidence detection per IoU cluster
    wins. See ARCHITECTURE.md §8.1 -- threshold is an unvalidated heuristic.

    Class-aware (D-022): only detections sharing the same class_id can
    suppress each other. Without this, a higher-confidence "volume"
    detection could wrongly suppress an overlapping "hold" detection -- a
    hold physically mounted on a volume is two separate real objects
    (D-022's own downstream-roles note), not a single ambiguous one, even
    though their boxes overlap heavily.
    """
    by_class: dict[int, list] = {}
    for det in all_detections:
        by_class.setdefault(det[0], []).append(det)

    kept: list = []
    for class_id, dets in by_class.items():
        dets_sorted = sorted(dets, key=lambda d: d[3], reverse=True)
        class_kept: list = []
        for det in dets_sorted:
            if all(_iou(det[2], k[2]) < iou_threshold for k in class_kept):
                class_kept.append(det)
        kept.extend(class_kept)
    return kept


def _polygon_mask(frame_shape: tuple[int, int], polygon_xy: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(polygon_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def match_colour(
    frame_bgr: np.ndarray,
    polygon_xy: list[tuple[float, float]],
    hsv_reference: dict,
    grade_bands: dict,
    distance_threshold: float,
    erode_px: int = DEFAULT_MASK_ERODE_PX,
) -> tuple[str, str | None, float | None, bool]:
    """Returns (route_label, colour_name, distance, used_fallback).
    "unassigned" whenever hsv_reference has no real (non-None) entries --
    see hold_colours_config.py.

    Samples the representative colour from inside the polygon MASK (D-022),
    eroded by erode_px to avoid edge/background bleed -- falls back to the
    un-eroded mask if erosion would empty it (used_fallback=True in that
    case), rather than silently returning no colour. A fixed pixel margin
    can empty a small hold's mask entirely; the caller is expected to count
    and report used_fallback, not swallow it.
    """
    known = {c: v for c, v in hsv_reference.items() if v is not None}
    if not known:
        return "unassigned", None, None, False

    mask = _polygon_mask(frame_bgr.shape, polygon_xy)
    if not mask.any():
        return "unassigned", None, None, False

    kernel = np.ones((max(1, erode_px * 2 + 1),) * 2, dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    used_fallback = not eroded.any()
    sample_mask = mask if used_fallback else eroded

    pixels = frame_bgr[sample_mask]
    hsv_pixels = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    med_h, med_s, med_v = (float(x) for x in np.median(hsv_pixels, axis=0))

    best_colour, best_dist = None, float("inf")
    for colour, ((h_min, h_max), (s_min, s_max), (v_min, v_max)) in known.items():
        h_mid = ((h_min + h_max) / 2) % 180
        h_dist = min(abs(med_h - h_mid), 180 - abs(med_h - h_mid))
        s_mid, v_mid = (s_min + s_max) / 2, (v_min + v_max) / 2
        dist = (h_dist**2 + (med_s - s_mid) ** 2 + (med_v - v_mid) ** 2) ** 0.5
        if dist < best_dist:
            best_colour, best_dist = colour, dist

    if best_colour is None or best_dist > distance_threshold:
        return "unassigned", None, (best_dist if best_colour else None), used_fallback
    return grade_bands.get(best_colour, "unassigned"), best_colour, best_dist, used_fallback


def process_clip(
    video_path: Path, detector, device: str, args
) -> tuple[list[HoldDetection], np.ndarray]:
    sampled = sample_frames(video_path, args.n_sample_frames)
    if not sampled:
        return [], None

    class_names = resolve_class_names(detector)

    all_detections = []
    for _, frame in sampled:
        all_detections.extend(detect_holds_in_frame(detector, frame, args.detector_conf, device))

    merged = merge_detections_across_frames(all_detections, args.merge_iou_threshold)

    representative_frame = sampled[0][1]
    detections = []
    n_fallback = 0
    n_holds = 0
    for class_id, polygon_xy, bbox_xyxy, conf in merged:
        class_name = class_names.get(class_id, f"unknown_class_{class_id}")
        x1, y1, x2, y2 = bbox_xyxy

        if class_name == "hold":
            n_holds += 1
            route_label, colour_name, distance, used_fallback = match_colour(
                representative_frame, polygon_xy, GYM_COLOUR_HSV_REFERENCE,
                GYM_COLOUR_GRADE_BANDS, args.colour_distance_threshold,
            )
            if used_fallback:
                n_fallback += 1
        else:
            # D-022: descent_hold and volume are never colour-matched and
            # never get a route label -- no HSV call at all for these.
            route_label, colour_name, distance, used_fallback = "unassigned", None, None, False

        detections.append(
            HoldDetection(
                class_name=class_name,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                polygon=polygon_xy,
                route_label=route_label,
                colour_name=colour_name,
                detector_confidence=conf,
                colour_distance=distance,
                used_mask_erosion_fallback=used_fallback,
            )
        )

    if n_holds:
        print(
            f"  {n_fallback} of {n_holds} hold detection(s) fell back to the "
            f"un-eroded mask (erosion emptied the mask) -- DEFAULT_MASK_ERODE_PX "
            f"({DEFAULT_MASK_ERODE_PX}px) may be too large for this gym's smaller holds."
        )

    return detections, representative_frame


_CLASS_DRAW_COLOURS = {
    "hold_assigned": (0, 200, 0),      # green
    "hold_unassigned": (0, 0, 255),    # red
    "volume": (128, 128, 128),         # grey
    "descent_hold": (0, 165, 255),     # orange
}


def draw_overlay(frame: np.ndarray, detections: list[HoldDetection]) -> np.ndarray:
    frame = frame.copy()
    for det in detections:
        if det.class_name == "hold":
            colour_key = "hold_assigned" if det.route_label != "unassigned" else "hold_unassigned"
        else:
            colour_key = det.class_name
        color = _CLASS_DRAW_COLOURS.get(colour_key, (255, 255, 255))

        pts = np.array(det.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        x, y = int(det.bbox[0]), int(det.bbox[1])
        label = f"{det.class_name}:{det.route_label}" if det.class_name == "hold" else det.class_name
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
            writer.writerow([
                "video", "class", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "route_label", "colour_name", "detector_confidence", "colour_distance",
            ])
            for det in detections:
                writer.writerow([
                    video_path.name, det.class_name, *[f"{v:.1f}" for v in det.bbox],
                    det.route_label, det.colour_name or "",
                    f"{det.detector_confidence:.4f}",
                    f"{det.colour_distance:.2f}" if det.colour_distance is not None else "",
                ])

        if representative_frame is not None:
            overlay_path = args.output_dir / f"{stem}_holds_overlay.jpg"
            cv2.imwrite(str(overlay_path), draw_overlay(representative_frame, detections))
            print(f"  {len(detections)} detections -> {csv_path}, {overlay_path}")
        else:
            print(f"  {len(detections)} detections -> {csv_path} (no frames sampled, no overlay written)")


if __name__ == "__main__":
    main()
