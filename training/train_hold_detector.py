"""
Fine-tune a three-class ("hold", "descent_hold", "volume") YOLOv8n-seg
instance-segmentation model on a Roboflow-exported dataset. See
DECISIONS.md D-017, D-017b, D-022 and ARCHITECTURE.md §8.2.

Not a runtime pipeline module -- one-off training script, same treatment as
validation/'s scripts (ARCHITECTURE.md §8.5 / §6). Do not import this from
hold_detection.py or vice versa.

Requires an annotated, polygon-labelled dataset (see
training/extract_annotation_frames.py and DECISIONS.md D-017 "Human steps"
-- this is a manual Roboflow Smart Select step, not something this script
can do). Point --data at the data.yaml Roboflow's segmentation export gives
you. Before training starts, this script runs
training/check_seg_labels.py against the dataset and refuses to start if it
reports any problems (D-022's annotation rule: a box-format line in a
segmentation export is a data error, not a valid label).

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
import subprocess
import sys
from pathlib import Path

# training/ scripts importing from each other is fine -- the "don't import
# across the hold_detection.py / training boundary" rule (ARCHITECTURE.md
# §8.5) is about runtime vs. training, not training-to-training.
sys.path.insert(0, str(Path(__file__).parent))
from check_seg_labels import load_class_names  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Path to the Roboflow-exported data.yaml (segmentation format)")
    parser.add_argument("--base-weights", type=str, default="yolov8n-seg.pt",
                         help="Pretrained segmentation checkpoint to fine-tune from (downloaded by ultralytics if not local)")
    parser.add_argument("--epochs", type=int, default=50,
                         help="Starting-point default for a small three-class dataset -- not tuned, log the real value used as D-017b")
    parser.add_argument("--project-dir", type=Path, default=Path("training/checkpoints"))
    parser.add_argument("--run-name", type=str, default="hold_detector")
    parser.add_argument("--checkpoint-name", type=str, default="hold_detector_seg.pt",
                         help="Output filename under hold_detection_weights/. Defaults to the segmentation "
                              "model's own name -- hold_detector.pt stays reserved for a possible future "
                              "box-model comparison run (D-022). Refuses to overwrite whatever file already "
                              "exists at the target path.")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"{args.data} does not exist. This script needs a Roboflow-exported "
            "YOLO-segmentation dataset first -- see DECISIONS.md D-017 'Human steps' "
            "and training/extract_annotation_frames.py."
        )

    export_dir = args.data.parent
    check_script = Path(__file__).parent / "check_seg_labels.py"
    check_result = subprocess.run(
        [sys.executable, str(check_script), str(export_dir)],
        capture_output=True, text=True,
    )
    print(check_result.stdout)
    if check_result.returncode != 0:
        print(check_result.stderr, file=sys.stderr)
        raise SystemExit(
            f"\nRefusing to start training: training/check_seg_labels.py found problems "
            f"in {export_dir} (see output above). Fix the dataset export and re-run."
        )

    class_names = load_class_names(args.data)
    print(f"Classes from {args.data} (in index order -- never assumed): {class_names}")
    if set(class_names) != {"hold", "descent_hold", "volume"}:
        print(
            f"WARNING: expected exactly {{'hold', 'descent_hold', 'volume'}} (D-022) "
            f"but data.yaml gives {class_names}. Proceeding anyway -- this is a "
            f"dataset-authoring check, not something this script can fix.",
            file=sys.stderr,
        )

    target_dir = Path(__file__).parent.parent / "hold_detection_weights"
    target_dir.mkdir(exist_ok=True)
    target_path = target_dir / args.checkpoint_name
    if target_path.exists():
        raise SystemExit(
            f"{target_path} already exists -- refusing to overwrite it. "
            f"Pass --checkpoint-name to choose a different output filename."
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

    shutil.copy(best_weights, target_path)
    print(f"\nCopied fine-tuned checkpoint to {target_path}")
    print("hold_detection.py will pick this up automatically if the filename matches "
          "its expected default -- check hold_detection.py's load_detector() if you "
          "used a custom --checkpoint-name.")
    print(f"Log the epoch count used ({args.epochs}) and training outcome as DECISIONS.md D-017b.")


if __name__ == "__main__":
    main()
