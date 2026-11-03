"""
Check of the IMG_2419 exclusion-list range 597-614 (plus 592-596) against the gated run (DECISIONS.md D-024, D-027):
for every frame 592-614, is the box the gated run chose on the climber (IoU >= 0.5 with the author-confirmed reference
box, reference_track_2419.csv), absent (dropped), or on something else? Also records what the legacy run had.
Writes evaluation/lock_rule/output_lock1/window_592_614_check.csv and prints a summary. Exit status 1 if any frame
in the range has a box that is not on the climber.

Usage:
    python3 evaluation/lock_rule/check_window.py [--run-dir validation/output_full_lock1]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compare_gated_run import boxes_of, load_kp  # noqa: E402
from lock_common import load_cache  # noqa: E402
from person_lock import Box, iou  # noqa: E402

EVAL = Path("evaluation/lock_rule/output_full")
OUT = Path("evaluation/lock_rule/output_lock1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("validation/output_full_lock1"))
    args = ap.parse_args()
    n = len(load_cache("IMG_2419"))
    run = boxes_of(load_kp(args.run_dir / "IMG_2419_keypoints.csv"), n)
    leg = boxes_of(load_kp(Path("validation/output_full/IMG_2419_keypoints.csv")), n)
    ref = {}
    with open(EVAL / "reference_track_2419.csv", newline="") as f:
        for r in csv.DictReader(f):
            ref[int(r["frame"])] = Box(float(r["ref_x"]), float(r["ref_y"]), float(r["ref_w"]), float(r["ref_h"]), 1.0)
    rows, counts = [], {"correct": 0, "dropped": 0, "wrong": 0}
    leg_counts = {"correct": 0, "dropped": 0, "wrong": 0}
    for t in range(592, 615):
        def status(b):
            if b is None:
                return "dropped", ""
            v = iou(b, ref[t])
            return ("correct" if v >= 0.5 else "wrong"), f"{v:.3f}"
        s, v = status(run[t])
        ls, lv = status(leg[t])
        counts[s] += 1
        leg_counts[ls] += 1
        rows.append([t, s, v, ls, lv, 597 <= t <= 614])
    with open(OUT / "window_592_614_check.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame", "gated_run", "gated_iou_with_reference", "legacy_run", "legacy_iou_with_reference",
                    "in_exclusion_range_597_614"])
        w.writerows(rows)
    print(f"IMG_2419 592-614 (23 frames), gated run: {counts}; legacy run: {leg_counts}")
    print("dropped frames in the gated run:", [r[0] for r in rows if r[1] == "dropped"])
    sys.exit(1 if counts["wrong"] else 0)


if __name__ == "__main__":
    main()
