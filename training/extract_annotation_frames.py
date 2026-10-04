"""
Pull candidate still frames from own-gym footage for manual annotation in
Roboflow. See DECISIONS.md D-017 ("Human steps") and ARCHITECTURE.md §8.2.

Runnable now -- has no dependency on the hold detector, training data, or
anything else that doesn't exist yet. This is step 1 of the human annotation
workflow: run this, upload the resulting .jpg files to a Roboflow project,
annotate (model-assisted after the first few boxes), export as a YOLO-format
dataset into training/dataset/, then run train_hold_detector.py.

Usage:
    python training/extract_annotation_frames.py
    python training/extract_annotation_frames.py --input-dir footage/test --n-per-clip 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def sample_frame_indices(total_frames: int, n: int) -> list[int]:
    if total_frames <= n:
        return list(range(total_frames))
    step = total_frames / n
    return [int(i * step) for i in range(n)]


def extract_frames(video_path: Path, n: int, output_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total, n)
    count = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = output_dir / f"{video_path.stem}_frame{idx:05d}.jpg"
        cv2.imwrite(str(out_path), frame)
        count += 1
    cap.release()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("footage/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/frames_to_annotate"))
    parser.add_argument("--n-per-clip", type=int, default=15,
                         help="Evenly-spaced frames to pull per clip. More clips/holds visible beats more frames from one clip.")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Input directory {args.input_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    videos = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        print(f"No video files found in {args.input_dir}.")
        sys.exit(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for video_path in videos:
        n = extract_frames(video_path, args.n_per_clip, args.output_dir)
        print(f"{video_path.name}: wrote {n} frames")
        total += n

    print(f"\nTotal: {total} frames in {args.output_dir}")
    print("Next: upload these to a Roboflow project, annotate the 'hold' class, "
          "export as YOLOv8 format into training/dataset/, then run train_hold_detector.py.")


if __name__ == "__main__":
    main()
