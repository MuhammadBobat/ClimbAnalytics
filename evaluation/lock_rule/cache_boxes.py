"""
Cache every person box the detector finds, per frame, at a low confidence floor, so lock rules
(DECISIONS.md D-024, D-027) can be replayed offline without re-running the detector.

Same detector call as validation/validate_vitpose.py's detect_person_boxes (YOLOv8n, person class
only, validation/weights/yolov8n.pt, full-resolution frame, device auto-detected the same way); only
the confidence floor differs (default 0.05, D-024's measurement floor instead of 0.5). The call is
repeated here, not imported, because validate_vitpose.py says it must not be imported elsewhere;
evaluation/lock_rule/replay_check.py checks that the cache reproduces its chosen boxes.

Output: validation/box_cache/<clip>_boxes.csv with one row per box (video, frame_idx, box_idx, x, y,
w, h, conf; numbers at full float precision) and one row with box_idx = -1 and empty fields for a
frame with no box at all. A manifest.txt records the run. A folder that already has a manifest.txt is
refused unless --force.

Usage:
    python3 evaluation/lock_rule/cache_boxes.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from validation.analyze_occlusion_gaps import (  # noqa: E402
    MANIFEST_LIBRARIES, _git, _library_version, sha256_file)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
DEFAULT_FLOOR = 0.05
COLUMNS = ["video", "frame_idx", "box_idx", "x", "y", "w", "h", "conf"]


def detect_all(detector, frame_bgr, floor: float, device: str) -> list[tuple]:
    """(x, y, w, h, conf) for every person box with conf >= floor, in the detector's own order."""
    results = detector.predict(frame_bgr, classes=[0], conf=floor, device=device, verbose=False)
    out = []
    if not results or results[0].boxes is None:
        return out
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        out.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(c)))
    return out


def write_rows(writer, video: str, frame_idx: int, boxes: list[tuple]) -> None:
    if not boxes:
        writer.writerow([video, frame_idx, -1, "", "", "", "", ""])
        return
    for i, (x, y, w, h, c) in enumerate(boxes):
        writer.writerow([video, frame_idx, i, repr(x), repr(y), repr(w), repr(h), repr(c)])


def main() -> None:
    import cv2
    import torch
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=Path("footage/test"))
    p.add_argument("--out-dir", type=Path, default=Path("validation/box_cache"))
    p.add_argument("--weights", type=Path, default=Path("validation/weights/yolov8n.pt"))
    p.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    p.add_argument("--device", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if (args.out_dir / "manifest.txt").exists() and not args.force:
        print(f"Refusing to write into {args.out_dir}: it already has a manifest.txt.", file=sys.stderr)
        sys.exit(1)
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    videos = sorted(v for v in args.input_dir.iterdir() if v.suffix.lower() in VIDEO_EXTENSIONS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(args.weights))
    print(f"device {device}, floor {args.floor}, {len(videos)} videos")

    lines = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        n = nboxes = 0
        t0 = time.time()
        with open(args.out_dir / f"{v.stem}_boxes.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(COLUMNS)
            while True:
                ok, frame = cap.read()
                if not ok or (args.max_frames is not None and n >= args.max_frames):
                    break
                boxes = detect_all(detector, frame, args.floor, device)
                write_rows(w, v.name, n, boxes)
                nboxes += len(boxes)
                n += 1
                if n % 200 == 0:
                    print(f"  {v.name}: {n} frames ({n / (time.time() - t0):.1f} fps)", flush=True)
        cap.release()
        lines.append(f"{v.name}: frames {n}, boxes {nboxes}, sha256 {sha256_file(v)}")
        print(f"  {v.name}: {n} frames, {nboxes} boxes", flush=True)

    dirty = _git("status", "--porcelain", "--", "evaluation/lock_rule/cache_boxes.py")
    manifest = [
        "Box cache manifest (evaluation/lock_rule/cache_boxes.py)",
        f"date: {date.today().isoformat()}",
        f"input folder: {args.input_dir.as_posix()}",
        f"detector: YOLOv8n person class, weights {args.weights.as_posix()} sha256 {sha256_file(args.weights)}",
        f"confidence floor: {args.floor} (ultralytics default NMS IoU; validate_vitpose.py uses 0.5)",
        f"device: {device}", f"max frames: {args.max_frames if args.max_frames is not None else 'all'}",
        "", "== Clips", *lines, "", "== Code and environment (at run time)",
        f"git HEAD: {_git('rev-parse', 'HEAD')}",
        "script modified since HEAD: " + ("unknown" if dirty == "unknown" else ("yes" if dirty else "no")),
    ]
    for lib in MANIFEST_LIBRARIES:
        names = ("opencv-python", "opencv-python-headless") if lib == "opencv-python" else (lib,)
        manifest.append(f"{lib}: {_library_version(*names)}")
    (args.out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"Written to {args.out_dir}")


if __name__ == "__main__":
    main()
