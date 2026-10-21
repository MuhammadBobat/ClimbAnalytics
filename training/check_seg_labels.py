"""
Label sanity check for the YOLOv8n-seg hold-detector dataset (D-022).

Standalone: not a runtime pipeline module, not imported elsewhere — same
category as the rest of training/. Run it on a Roboflow YOLO-segmentation
export before training, not after.

D-022 switched the hold detector from box detection to instance segmentation
with polygon labels across three classes (hold / descent_hold / volume), and
states explicitly: "A box-type label in an exported segmentation file is a
data error." This script exists because a real export already contained two
5-token (box-format) lines mixed in with polygon lines, which would be
misread as polygon coordinates and silently corrupt training.

A YOLO segmentation label line is:
    class_id x1 y1 x2 y2 x3 y3 ... xn yn
(normalized [0,1] polygon coordinates, at least 3 points). A YOLO
*detection* (box) line is:
    class_id cx cy w h
i.e. exactly 5 tokens — a box line mixed into a segmentation export is
flagged, not silently accepted as a 2-point "polygon."

Checks performed per line, each reported with file + line number:
  - exactly 5 tokens (a box-format line in a segmentation file)
  - an odd number of coordinate tokens (coordinates must come in x,y pairs)
  - fewer than 3 points (not a valid polygon)
  - any coordinate outside [0, 1]
  - a class id not present in data.yaml's class list

Class names and order are read from data.yaml — never hardcoded, per D-022's
own open-questions note that class index order must be read from the export,
not assumed.

Usage:
    python training/check_seg_labels.py <export_dir>
    python training/check_seg_labels.py training/dataset
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "PyYAML is required to read data.yaml (pip install pyyaml). "
        "Not silently skipping class-id validation.",
        file=sys.stderr,
    )
    raise


@dataclass
class LabelProblem:
    file: Path
    line_no: int
    reason: str


def load_class_names(data_yaml_path: Path) -> list[str]:
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if names is None:
        raise ValueError(f"{data_yaml_path} has no 'names' key")
    # Roboflow exports 'names' as either a list or a {index: name} dict —
    # handle both rather than assuming one.
    if isinstance(names, dict):
        return [names[i] for i in sorted(names, key=int)]
    return list(names)


def check_label_file(path: Path, n_classes: int) -> tuple[list[LabelProblem], Counter]:
    problems: list[LabelProblem] = []
    class_counts: Counter = Counter()

    with open(path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()

            if len(tokens) == 5:
                problems.append(LabelProblem(
                    path, line_no,
                    "5 tokens — a box-format (detection) line in a segmentation file",
                ))
                continue

            try:
                class_id = int(tokens[0])
            except ValueError:
                problems.append(LabelProblem(path, line_no, f"non-integer class id {tokens[0]!r}"))
                continue

            coord_tokens = tokens[1:]
            if len(coord_tokens) % 2 != 0:
                problems.append(LabelProblem(
                    path, line_no,
                    f"odd number of coordinate tokens ({len(coord_tokens)}) — coordinates must be x,y pairs",
                ))
                continue

            n_points = len(coord_tokens) // 2
            if n_points < 3:
                problems.append(LabelProblem(path, line_no, f"only {n_points} point(s) — fewer than 3, not a valid polygon"))

            try:
                coords = [float(t) for t in coord_tokens]
            except ValueError:
                problems.append(LabelProblem(path, line_no, "non-numeric coordinate token"))
                coords = []

            for v in coords:
                if v < 0.0 or v > 1.0:
                    problems.append(LabelProblem(path, line_no, f"coordinate {v} outside [0, 1]"))
                    break

            if not (0 <= class_id < n_classes):
                problems.append(LabelProblem(path, line_no, f"class id {class_id} not in data.yaml (expected 0..{n_classes - 1})"))
            else:
                class_counts[class_id] += 1

    return problems, class_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export_dir", type=Path, help="Roboflow YOLO-segmentation export root (contains data.yaml and a labels/ folder)")
    args = parser.parse_args()

    data_yaml_path = args.export_dir / "data.yaml"
    if not data_yaml_path.exists():
        print(f"ERROR: {data_yaml_path} not found.", file=sys.stderr)
        sys.exit(1)

    class_names = load_class_names(data_yaml_path)
    print(f"Classes from {data_yaml_path}: {class_names}")

    label_files = sorted(args.export_dir.rglob("labels/*.txt"))
    if not label_files:
        print(f"No labels/*.txt files found under {args.export_dir}.")
        sys.exit(0)

    all_problems: list[LabelProblem] = []
    total_counts: Counter = Counter()

    for label_file in label_files:
        problems, counts = check_label_file(label_file, len(class_names))
        all_problems.extend(problems)
        total_counts.update(counts)

    print(f"\nChecked {len(label_files)} label file(s).\n")

    print("Per-class instance counts:")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {total_counts.get(idx, 0)}")
    print()

    if all_problems:
        print(f"{len(all_problems)} problem(s) found:\n")
        for p in all_problems:
            print(f"  {p.file}:{p.line_no}: {p.reason}")
        sys.exit(1)
    else:
        print("No problems found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
