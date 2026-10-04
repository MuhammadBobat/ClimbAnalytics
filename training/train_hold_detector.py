"""
Fine-tune a single-class ("hold") YOLOv8n detector on a Roboflow-exported
dataset. See DECISIONS.md D-017, D-017b and ARCHITECTURE.md §8.2.

Not a runtime pipeline module -- one-off training script, same treatment as
validation/'s scripts (ARCHITECTURE.md §8.5 / §6). Do not import this from
hold_detection.py or vice versa.

Requires an annotated dataset (see training/extract_annotation_frames.py and
DECISIONS.md D-017 "Human steps" -- this is a manual Roboflow step, not
something this script can do). Point --data at the data.yaml Roboflow's
export gives you.

Epoch count and other hyperparameters below are starting-point defaults, not
tuned values -- log whatever actually gets used, and why, as DECISIONS.md
D-017b once training actually happens (per CLAUDE.md rule 9: do not report
numbers that weren't produced by an actual run).

Usage:
    python training/train_hold_detector.py --data training/dataset/data.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Path to the Roboflow-exported data.yaml")
    parser.add_argument("--base-weights", type=str, default="yolov8n.pt",
                         help="Pretrained checkpoint to fine-tune from (downloaded by ultralytics if not local)")
    parser.add_argument("--epochs", type=int, default=50,
                         help="Starting-point default for a small single-class dataset -- not tuned, log the real value used as D-017b")
    parser.add_argument("--project-dir", type=Path, default=Path("training/checkpoints"))
    parser.add_argument("--run-name", type=str, default="hold_detector")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} does not exist. This script needs a Roboflow-exported "
            "YOLO-format dataset first -- see DECISIONS.md D-017 'Human steps' "
            "and training/extract_annotation_frames.py."
        )

    from ultralytics import YOLO

    model = YOLO(args.base_weights)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        project=str(args.project_dir),
        name=args.run_name,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"Training finished but no best.pt found at {best_weights} -- check {results.save_dir} manually.")
        return

    target_dir = Path(__file__).parent.parent / "hold_detection_weights"
    target_dir.mkdir(exist_ok=True)
    target_path = target_dir / "hold_detector.pt"
    shutil.copy(best_weights, target_path)
    print(f"\nCopied fine-tuned checkpoint to {target_path}")
    print("hold_detection.py will pick this up automatically.")
    print(f"Log the epoch count used ({args.epochs}) and training outcome as DECISIONS.md D-017b.")


if __name__ == "__main__":
    main()
