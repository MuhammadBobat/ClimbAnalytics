"""
Checks and report for a gated-lock pipeline run (DECISIONS.md D-024, D-027): validation/output_full_lock1/ by default.

1. Replay comparison: the per-frame chosen box (and tier) of the run against the offline replay of the selected
   setting on the cached boxes (validation/box_cache, boxes with conf >= the rescue floor).
2. First-100-frames check: for every frame in 0..99 that the legacy rule tracked (rows in validation/output/ and in
   validation/output_full/), every keypoint row of the run must be IDENTICAL in every column the reference has (exact
   strings; validation/output predates the n_person_boxes column, so that column is only compared with output_full); frames the
   run tracks that legacy did not are counted separately (newly tracked).
3. Per clip: frames, dropped frames and longest untracked run, centre-jump statistics (legacy and gated), height-ratio
   flags, frames changed against legacy (gained, lost, replaced).
4. Pose quality: the share of keypoints at or above 0.5 (all 17, and the 8 pipeline joints: shoulders, wrists, hips,
   ankles) on the newly tracked frames, on the rescue-tier frames and on the 2419 window 592-614, next to each clip's
   overall share in the gated run and in the legacy run.
5. Rejection log: rows of <clip>_lock.csv per clip and check, with the frame ranges.

Writes evaluation/lock_rule/output_lock1/{report.md, replay_comparison.csv, first100_check.csv, per_clip.csv,
pose_quality.csv}. Usage:
    python3 evaluation/lock_rule/compare_gated_run.py [--run-dir validation/output_full_lock1]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lock_common import (  # noqa: E402
    CLIPS, LockParams, height_ratio_flags, jump_stats, load_cache, run_rule, runs_of)
from person_lock import Box  # noqa: E402
from simulate_gaps import params_from_row  # noqa: E402

OUT = Path("evaluation/lock_rule/output_lock1")
EVAL = Path("evaluation/lock_rule/output_full")
PIPELINE_JOINTS = {"left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_hip", "right_hip",
                   "left_ankle", "right_ankle"}


def load_kp(path: Path):
    """frame -> list of row dicts (strings), in file order."""
    out: dict[int, list[dict]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(int(r["frame_idx"]), []).append(r)
    return out


def boxes_of(kp: dict, n: int) -> list[Box | None]:
    out = []
    for t in range(n):
        rows = kp.get(t)
        out.append(None if not rows else Box(float(rows[0]["bbox_x"]), float(rows[0]["bbox_y"]), float(rows[0]["bbox_w"]),
                                           float(rows[0]["bbox_h"]), 0.0))
    return out


def fmt1(b: Box) -> tuple[str, ...]:
    return (f"{b.x:.1f}", f"{b.y:.1f}", f"{b.w:.1f}", f"{b.h:.1f}")


def share(kp: dict, frames, joints=None) -> tuple[int, int]:
    """(keypoints >= 0.5, keypoints counted) over the frames."""
    ok = tot = 0
    for t in frames:
        for r in kp.get(t, []):
            if joints is not None and r["keypoint_name"] not in joints:
                continue
            tot += 1
            ok += float(r["confidence"]) >= 0.5
    return ok, tot


def pct(a: int, b: int) -> str:
    return "" if b == 0 else f"{100 * a / b:.1f}"


def ranges(frames: list[int]) -> str:
    out, i = [], 0
    fr = sorted(frames)
    while i < len(fr):
        j = i
        while j + 1 < len(fr) and fr[j + 1] == fr[j] + 1:
            j += 1
        out.append(str(fr[i]) if i == j else f"{fr[i]}-{fr[j]}")
        i = j + 1
    return ",".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=Path("validation/output_full_lock1"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    label = json.loads((EVAL / "selected_setting.json").read_text())["label"]
    with open(EVAL / "sweep_settings.csv", newline="") as f:
        params = params_from_row(next(r for r in csv.DictReader(f) if r["setting"] == label))

    replay_rows, f100_rows, clip_rows, pq_rows, md = [], [], [], [], []
    lock_lines = []
    for clip in CLIPS:
        cache = load_cache(clip)
        n = len(cache)
        run = load_kp(args.run_dir / f"{clip}_keypoints.csv")
        leg = load_kp(Path("validation/output_full") / f"{clip}_keypoints.csv")
        first100 = load_kp(Path("validation/output") / f"{clip}_keypoints.csv")
        run_boxes, leg_boxes = boxes_of(run, n), boxes_of(leg, n)
        with open(args.run_dir / f"{clip}_lock_frames.csv", newline="") as f:
            tiers = {int(r["frame_idx"]): r["tier"] for r in csv.DictReader(f)}

        # 1. replay comparison (rule on cached boxes with conf >= rescue floor)
        floor = params.conf_rescue or 0.0
        sub = [[b for b in boxes if b.conf >= floor] for boxes in cache]
        chosen, infos = run_rule(sub, params)
        box_mis = pres_mis = tier_mis = both = 0
        first_bad = None
        for t in range(n):
            a, b = chosen[t], run_boxes[t]
            if (a is None) != (b is None):
                pres_mis += 1
                first_bad = first_bad if first_bad is not None else t
            elif a is not None:
                both += 1
                if fmt1(a) != fmt1(b):
                    box_mis += 1
                    first_bad = first_bad if first_bad is not None else t
            if tiers.get(t) != infos[t].tier:
                tier_mis += 1
        replay_rows.append([clip, n, both, box_mis, pres_mis, tier_mis, first_bad])

        # 2. first 100 frames: legacy-tracked frames must be identical
        for name, base in (("validation/output (100 frames)", first100), ("validation/output_full", leg)):
            frames = [t for t in range(min(100, n)) if t in base]
            diff_rows = diff_frames = 0
            for t in frames:
                a, b = run.get(t, []), base[t]
                cols = [c for c in b[0] if c in a[0]] if a and b else []   # validation/output predates n_person_boxes
                diffs = sum(1 for x, y in zip(a, b) if any(x[c] != y[c] for c in cols))
                d = (len(a) != len(b)) or diffs > 0
                diff_frames += d
                diff_rows += diffs
            gained = [t for t in range(min(100, n)) if t in run and t not in base]
            lost = [t for t in range(min(100, n)) if t not in run and t in base]
            f100_rows.append([clip, name, len(frames), diff_frames, diff_rows, ranges(gained), ranges(lost)])

        # 3. per clip statistics
        drops = runs_of([b is None for b in run_boxes])
        ldrops = runs_of([b is None for b in leg_boxes])
        jr, jl = jump_stats(run_boxes), jump_stats(leg_boxes)
        gained = [t for t in range(n) if run_boxes[t] is not None and leg_boxes[t] is None]
        lost = [t for t in range(n) if run_boxes[t] is None and leg_boxes[t] is not None]
        replaced = [t for t in range(n) if run_boxes[t] is not None and leg_boxes[t] is not None
                    and fmt1(run_boxes[t]) != fmt1(leg_boxes[t])]
        clip_rows.append([clip, n, sum(b is None for b in leg_boxes), max((e - s + 1 for s, e in ldrops), default=0),
                          sum(b is None for b in run_boxes), max((e - s + 1 for s, e in drops), default=0),
                          f"{jl.get('max', float('nan')):.4f}", f"{jr.get('max', float('nan')):.4f}",
                          f"{jl.get('p99', float('nan')):.4f}", f"{jr.get('p99', float('nan')):.4f}",
                          jl.get("n_over_0_3", 0), jr.get("n_over_0_3", 0),
                          len(height_ratio_flags(leg_boxes)), len(height_ratio_flags(run_boxes)),
                          len(gained), len(lost), len(replaced), ranges(gained), ranges(lost), ranges(replaced)])

        # 4. pose quality
        rescue = [t for t in range(n) if tiers.get(t) == "rescue" and run_boxes[t] is not None]
        window = [t for t in range(592, 615)] if clip == "IMG_2419" else []
        window_replaced = [t for t in window if t in replaced]
        sets = [("all frames, gated run", sorted(run)), ("all frames, legacy run", sorted(leg)),
                ("newly tracked (legacy had no box)", gained), ("rescue-tier frames", rescue),
                ("2419 window 592-614 frames with a changed box", window_replaced)]
        for name, frames in sets:
            if not frames:
                continue
            kp = leg if name == "all frames, legacy run" else run
            a17, b17 = share(kp, frames)
            a8, b8 = share(kp, frames, PIPELINE_JOINTS)
            pq_rows.append([clip, name, len(frames), pct(a17, b17), b17, pct(a8, b8), b8])

        # 5. rejection log
        with open(args.run_dir / f"{clip}_lock.csv", newline="") as f:
            rej = list(csv.DictReader(f))
        by = {}
        for r in rej:
            by.setdefault(r["check_failed"], []).append(int(r["frame_idx"]))
        lock_lines.append(f"- {clip}: {len(rej)} rejection row(s)" + "".join(
            f"; {k}: {len(v)} frame(s) ({ranges(v)})" for k, v in sorted(by.items())))

    def w(name, header, rows):
        with open(OUT / name, "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(header)
            wr.writerows(rows)
    w("replay_comparison.csv", ["clip", "frames", "frames_with_box_in_both", "box_differs", "box_present_in_only_one",
                                 "tier_differs", "first_differing_frame"], replay_rows)
    w("first100_check.csv", ["clip", "legacy_reference", "frames_tracked_by_legacy_in_first_100", "frames_with_any_row_differing",
                              "rows_differing", "newly_tracked_frames", "frames_legacy_tracked_but_run_not"], f100_rows)
    w("per_clip.csv", ["clip", "frames", "legacy_dropped", "legacy_longest_untracked", "gated_dropped",
                       "gated_longest_untracked", "legacy_jump_max", "gated_jump_max", "legacy_jump_p99", "gated_jump_p99",
                       "legacy_jumps_over_0_3", "gated_jumps_over_0_3", "legacy_hratio_flags", "gated_hratio_flags",
                       "gained", "lost", "replaced", "gained_frames", "lost_frames", "replaced_frames"], clip_rows)
    w("pose_quality.csv", ["clip", "frame_set", "frames", "pct_keypoints_ge_0_5_all17", "n_keypoints_all17",
                           "pct_keypoints_ge_0_5_8_pipeline_joints", "n_keypoints_8"], pq_rows)
    md.append(f"# Gated-lock run report ({args.run_dir.as_posix()}), setting `{label}`\n")
    md.append("## 1. Replay comparison (run vs offline replay of the selected setting)\n")
    md.append("| clip | frames | boxes in both | box differs | box in only one | tier differs | first differing frame |\n|---|---|---|---|---|---|---|")
    md += [f"| {' | '.join(str(x) for x in r)} |" for r in replay_rows]
    md.append("\n## 2. First 100 frames (frames the legacy rule tracked must be identical in every column)\n")
    md.append("| clip | reference | legacy-tracked frames | frames differing | rows differing | newly tracked | legacy-tracked, run not |\n|---|---|---|---|---|---|---|")
    md += [f"| {' | '.join(str(x) for x in r)} |" for r in f100_rows]
    md.append("\n## 3. Per clip\n")
    h = ["clip", "frames", "legacy dropped", "legacy longest run", "gated dropped", "gated longest run", "jump max L/G",
         "jump p99 L/G", "jumps>0.3 L/G", "h-ratio flags L/G", "gained", "lost", "replaced"]
    md.append("| " + " | ".join(h) + " |\n|" + "---|" * len(h))
    for r in clip_rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} / {r[7]} | {r[8]} / {r[9]} | {r[10]} / {r[11]} | "
                  f"{r[12]} / {r[13]} | {r[14]} | {r[15]} | {r[16]} |")
    md.append("\n## 4. Pose quality: share of keypoints at or above 0.5\n")
    md.append("| clip | frame set | frames | % all 17 | % 8 pipeline joints |\n|---|---|---|---|---|")
    md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[5]} |" for r in pq_rows]
    md.append("\n## 5. Rejection log (<clip>_lock.csv)\n")
    md += lock_lines
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
