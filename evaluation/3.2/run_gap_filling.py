"""
Step 3.2 (author's plan label): run occlusion_handling.py on the keypoint CSVs of a pipeline run and write the filled
series, the per-gap events and counts for review. NO tier is validated (DECISIONS.md D-007, D-028 proposed); the
hide-and-fill test of the tier boundaries is step 3.3 and is not built here. Raw data is only read.

Inputs: <input-dir>/*_keypoints.csv (default validation/output_full_lock1), the exclusion list
(validation/frame_exclusions.csv; empty since D-027). Outputs, under evaluation/3.2/<input-folder-name>/:
    <clip>_filled.csv     (gitignored, regenerable) every frame x 17 joints: raw x, y, confidence next to the filled
                          x, y, flag (observed / tier1 / tier2 / tier3 / unavailable), method, sigma_px. The five face
                          points are not handled: flag observed, or unavailable where there is no usable position.
    gap_events.csv        every gap of the 12 body joints: clip, joint, first, last, length, tier by length, filled,
                          method, touches clip start / end, adjacent excluded frame, causes, note
    summary.csv           per clip and ALL: joint-frames by flag (12 body joints), gap events by tier (filled / not)
    comparison_3_1.md     gap events here vs the 3.1 events on the same input, with the definitional differences
    manifest.txt          run provenance and parameters

Usage:
    python3 evaluation/3.2/run_gap_filling.py [--input-dir validation/output_full_lock1]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import metrics  # noqa: E402
import occlusion_handling as oh  # noqa: E402
from validation.analyze_occlusion_gaps import (  # noqa: E402
    MANIFEST_LIBRARIES, _git, _library_version, load_exclusions, sha256_file)

FILLED_COLUMNS = ["video", "frame_idx", "timestamp_s", "keypoint_name", "x_raw", "y_raw", "confidence", "x_filled",
                  "y_filled", "flag", "method", "sigma_px"]
EVENT_COLUMNS = ["clip", "joint", "first", "last", "length", "tier_by_length", "filled", "method", "touches_start",
                 "touches_end", "adjacent_excluded", "causes", "note"]


def raw_rows(path: Path) -> dict[tuple[str, int], dict]:
    with open(path, newline="") as f:
        return {(r["keypoint_name"], int(r["frame_idx"])): r for r in csv.DictReader(f)}


def f4(x):
    return "" if x is None else f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=Path("validation/output_full_lock1"))
    ap.add_argument("--out-dir", type=Path, default=Path("evaluation/3.2"))
    ap.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"))
    ap.add_argument("--eval-3-1", type=Path, default=None, help="3.1 output folder for the comparison (default evaluation/3.1/<input-folder-name>)")
    args = ap.parse_args()
    out = args.out_dir / args.input_dir.resolve().name
    if (out / "summary.csv").exists():
        print(f"Refusing to overwrite {out}/summary.csv", file=sys.stderr)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)
    params = oh.OcclusionParams()
    exclusions = load_exclusions(args.exclusions)

    events_rows, summary_rows = [], []
    totals_flags, totals_events = Counter(), Counter()
    clips = []
    for path in sorted(args.input_dir.glob("*_keypoints.csv")):
        joints = metrics.load_keypoints_csv(path)
        with open(path, newline="") as f:
            clip = next(csv.DictReader(f))["video"]
        clips.append(clip)
        n = len(next(iter(joints.values())).positions)
        joints = metrics.apply_frame_exclusions(joints, {t for t in exclusions.get(clip, set()) if 0 <= t < n})
        filled = oh.fill_gaps(joints, params)
        raw = raw_rows(path)
        ts = next(iter(joints.values())).timestamps
        with open(out / f"{path.stem.replace('_keypoints', '')}_filled.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(FILLED_COLUMNS)
            for t in range(n):
                for kid, name in enumerate(metrics_names(joints)):
                    r = raw.get((name, t))
                    if name in filled:
                        fj = filled[name]
                        p, fl, me, sg = fj.positions[t], fj.flags[t].value, fj.methods[t], fj.sigma_px[t]
                    else:
                        p0 = joints[name].positions[t]
                        p, fl, me, sg = p0, ("observed" if p0 is not None else "unavailable"), None, None
                    w.writerow([clip, t, f"{ts[t]:.3f}", name, r["x"] if r else "", r["y"] if r else "",
                                r["confidence"] if r else "", f4(p[0]) if p else "", f4(p[1]) if p else "", fl,
                                me or "", f4(sg)])
        cnt, ev = Counter(), Counter()
        for name, fj in filled.items():
            cnt.update(fl.value for fl in fj.flags)
            for e in fj.events:
                events_rows.append([clip, e.joint, e.first, e.last, e.length, e.tier.value, e.filled, e.method or "",
                                    e.touches_start, e.touches_end, e.adjacent_excluded, ";".join(e.reasons), e.note])
                ev[(e.tier.value, "filled" if e.filled else "not_filled")] += 1
        summary_rows.append((clip, n, cnt, ev))
        totals_flags.update(cnt)
        totals_events.update(ev)
        print(f"{clip}: {dict(cnt)}")
    # files
    with open(out / "gap_events.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(EVENT_COLUMNS)
        w.writerows(events_rows)
    cols = ["clip", "frames", "body_joint_frames", "observed", "tier1", "tier2", "tier3", "unavailable",
            "events_tier1_filled", "events_tier1_not_filled", "events_tier2_filled", "events_tier2_not_filled",
            "events_tier3_filled", "events_tier3_not_filled"]

    def row(clip, n, cnt, ev):
        return [clip, n, sum(cnt.values()), *(cnt[k] for k in ("observed", "tier1", "tier2", "tier3", "unavailable")),
                *(ev[(t, s)] for t in ("tier1", "tier2", "tier3") for s in ("filled", "not_filled"))]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for c, n, cnt, ev in summary_rows:
            w.writerow(row(c, n, cnt, ev))
        w.writerow(row("ALL", sum(r[1] for r in summary_rows), totals_flags, totals_events))
    write_comparison(out, args.eval_3_1 or Path("evaluation/3.1") / args.input_dir.resolve().name, events_rows)
    dirty = _git("status", "--porcelain", "--", "occlusion_handling.py", "metrics.py", "evaluation/3.2/run_gap_filling.py",
                 "validation/frame_exclusions.csv")
    L = ["Gap-filling run manifest (evaluation/3.2/run_gap_filling.py)", f"date: {date.today().isoformat()}",
         f"input folder: {args.input_dir.as_posix()}",
         "NO tier is validated: boundaries and methods are unvalidated heuristics (D-007); D-028 proposed; hide-and-fill test (3.3) not built",
         "raw data is only read; the filled CSVs carry raw x, y and confidence next to the filled values",
         "", "== Parameters", f"tier boundaries (frames): tier 1 = 1-{oh.TIER1_MAX_FRAMES}, tier 2 = {oh.TIER1_MAX_FRAMES + 1}-{oh.TIER2_MAX_FRAMES}, tier 3 = {oh.TIER2_MAX_FRAMES + 1}+",
         f"occlusion confidence threshold: {metrics.OCCLUSION_CONFIDENCE_THRESHOLD} (metrics.py)", f"OcclusionParams: {params}",
         "joint set: the 12 body joints (shoulders, elbows, wrists, hips, knees, ankles); the five face points are not handled",
         "", "== Exclusion file", *(Path(args.exclusions).read_text().rstrip("\n").split("\n") if Path(args.exclusions).exists() else ["none"]),
         "", "== Code and environment", f"git HEAD: {_git('rev-parse', 'HEAD')}",
         "occlusion_handling.py, metrics.py, this script, exclusions modified since HEAD: " + ("unknown" if dirty == "unknown" else ("yes" if dirty else "no")),
         f"occlusion_handling.py sha256 {sha256_file(Path('occlusion_handling.py'))}", f"scipy: {_library_version('scipy')}"]
    L += [f"{lib}: {_library_version(*(('opencv-python', 'opencv-python-headless') if lib == 'opencv-python' else (lib,)))}" for lib in MANIFEST_LIBRARIES]
    L += ["", "== Clips"] + [f"{c}: keypoint CSV sha256 {sha256_file(args.input_dir / (Path(c).stem + '_keypoints.csv'))}" for c in clips]
    (out / "manifest.txt").write_text("\n".join(L) + "\n")
    print(f"Written to {out}")


def metrics_names(joints):
    order = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
             "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
             "left_ankle", "right_ankle"]
    return [n for n in order if n in joints]


def write_comparison(out: Path, eval31: Path, events_rows) -> None:
    ev31 = eval31 / "events.csv"
    lines = ["# Gap events of the 12 body joints: this run vs the 3.1 events on the same input\n"]
    mine = Counter(r[5] for r in events_rows)
    lines.append(f"This run (a gap = consecutive frames with no usable position, any cause, detector-drop frames included): "
                 f"tier 1 {mine['tier1']}, tier 2 {mine['tier2']}, tier 3 {mine['tier3']}, total {sum(mine.values())}.")
    if ev31.exists():
        body = set(oh.BODY_JOINTS)
        c = Counter()
        for r in csv.DictReader(open(ev31, newline="")):
            if r["joint"] in body:
                n = int(r["length_frames"])
                c["tier1" if n < 5 else ("tier2" if n <= 20 else "tier3")] += 1
        lines.append(f"3.1 events on the same input ({ev31.as_posix()}; events end at detector-drop frames, which are counted separately): "
                     f"tier 1 {c['tier1']}, tier 2 {c['tier2']}, tier 3 {c['tier3']}, total {sum(c.values())}.")
        lines.append("\nDifferences are expected where a detector-drop frame touches a low-confidence run (merged here, split in 3.1) "
                     "and from the whole-frame drops themselves, which are a gap for every body joint here and are not joint-level events in 3.1.")
    else:
        lines.append(f"3.1 events not found at {ev31.as_posix()}.")
    (out / "comparison_3_1.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
