"""
Occlusion-gap statistics — checks D-007 / ARCHITECTURE.md §3's tier boundaries
against real occlusion events in the Phase 1 validation output.

Standalone: deliberately NOT part of the main pipeline module boundaries in
ARCHITECTURE.md §5, same category as validate_vitpose.py. Do not import from
this file elsewhere.

ARCHITECTURE.md §3 defines occlusion as a per-joint condition, keyed to a
per-joint confidence threshold C < 0.5, with three tiers by gap duration:
  Tier 1 (<5 frames, ~0.15s)   — cubic spline interpolation
  Tier 2 (5-20 frames, ~0.15-0.6s) — Kalman filter
  Tier 3 (>20 frames, >0.6s)   — freeze stale, exclude from per-limb metrics
This threshold (0.5) is D-007's own and is NOT the same as this validation
suite's QA thresholds in validate_vitpose.py (0.3 low-conf keypoint, 0.5
frame-mean flag, 5-keypoint count) — those exist to flag frames for a human
to look at, not to define an occlusion event. Do not conflate the two.

This script computes two DISTINCT conditions and never merges them:
  1. Whole-frame drops  — "missing_detection": the person-box detector found
     no climber at all that frame. A lost bounding box is not the same
     failure as one hidden wrist, and D-007's Tier 3 (freeze + substitute)
     assumes partial, not total, limb loss — so this is reported on its own,
     never folded into "all 17 joints occluded."
  2. Joint-level gaps   — per joint, consecutive *detected* frames where that
     joint's own confidence < 0.5.

Known limitation (not fixed by this script): a keypoint with confidence >=
0.5 during a real occlusion (a "confidently wrong" hallucinated position)
would not be caught by this threshold at all. This is a limit of the
statistic itself, not something this script attempts to detect.

A missing_detection frame has no row at all in *_keypoints.csv (confirmed
directly: e.g. IMG_2412_keypoints.csv has no rows for frame_idx 41/42). Row
position in the CSV is therefore NOT the same as frame index — this script
reindexes every series to the full frame_idx range before any run-length
scan, so a gap can never be silently bridged by reading two adjacent rows as
adjacent frames.

Clip scope: every *_keypoints.csv in --input-dir is analysed. All clips are
the author's own footage, filmed on a fixed camera in the same gym.

Clip length: if <input-dir>/summary.csv has a total_frames entry for a clip, the
last frame is taken from it, so dropped frames at the END of a clip are counted.
Without it the last frame is the last frame that has a keypoint row, and any
trailing dropped frames are invisible (the report says so).

Frame exclusions: frames listed in --exclusions (default validation/frame_exclusions.csv,
video,start_frame,end_frame,reason) are treated like dropped frames for the joint-level
counts: no observation, and a run does not continue across them. They are reported
separately from detector drops, and the keypoint data itself is never modified. Use it
for frames where the tracker locked onto the wrong person.

Truncation: a run next to a dropped/excluded frame (or the clip start/end) is cut
there, so its length is a lower bound. Events are flagged for both cases.

Usage:
    python validation/analyze_occlusion_gaps.py
    python validation/analyze_occlusion_gaps.py --input-dir validation/output_full
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
import sys
from pathlib import Path

# D-007's threshold is defined once, in metrics.py (repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import OCCLUSION_CONFIDENCE_THRESHOLD  # noqa: E402

# D-007 / ARCHITECTURE.md §3's tier boundaries, in frames.
TIER_BOUNDARIES_FRAMES = (5, 20)

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class GapEvent:
    start_frame: int
    end_frame: int  # inclusive
    length_frames: int
    length_seconds: float
    touches_boundary: bool  # starts at frame 0 or ends at the clip's last frame
    adjacent_to_unobserved: bool = False  # next to a dropped/excluded frame: length is a lower bound


@dataclass
class ClipGapStats:
    clip: str
    fps: float
    total_frames: int  # full observed frame_idx range, inclusive
    n_excluded_frames: int = 0  # missing_detection frames, excluded from joint-level counts
    n_detected_frames: int = 0  # frames used for joint-level counts (detected, not excluded)
    excluded_frames: list[int] = field(default_factory=list)  # from --exclusions, within the clip
    frame_count_known: bool = False  # True if total_frames came from summary.csv
    missing_detection_events: list[GapEvent] = field(default_factory=list)
    joint_gap_events: dict[str, list[GapEvent]] = field(default_factory=dict)


def load_keypoints_long(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_run_lengths(
    occluded: dict[int, bool], fps: float, last_frame: int
) -> list[GapEvent]:
    """Scan a {frame_idx: is_occluded} map (reindexed, so every integer
    frame_idx from 0..last_frame is a real key) for consecutive-True runs.
    """
    events: list[GapEvent] = []
    run_start = None

    def adjacent(start: int, end: int) -> bool:
        return (start - 1 >= 0 and (start - 1) not in occluded) or (
            end + 1 <= last_frame and (end + 1) not in occluded
        )

    for frame_idx in range(last_frame + 1):
        is_occ = occluded.get(frame_idx, False)
        if is_occ and run_start is None:
            run_start = frame_idx
        elif not is_occ and run_start is not None:
            run_end = frame_idx - 1
            length = run_end - run_start + 1
            events.append(GapEvent(
                start_frame=run_start,
                end_frame=run_end,
                length_frames=length,
                length_seconds=length / fps,
                touches_boundary=(run_start == 0 or run_end == last_frame),
                adjacent_to_unobserved=adjacent(run_start, run_end),
            ))
            run_start = None
    if run_start is not None:
        run_end = last_frame
        length = run_end - run_start + 1
        events.append(GapEvent(
            start_frame=run_start,
            end_frame=run_end,
            length_frames=length,
            length_seconds=length / fps,
            touches_boundary=True,
            adjacent_to_unobserved=adjacent(run_start, run_end),
        ))
    return events


def load_total_frames(input_dir: Path) -> dict[str, int]:
    """{video filename: total_frames} from validate_vitpose.py's summary.csv,
    or {} if it is absent or lacks the column."""
    summary = input_dir / "summary.csv"
    if not summary.exists():
        return {}
    out: dict[str, int] = {}
    with open(summary, newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[r["video"]] = int(r["total_frames"])
            except (KeyError, ValueError):
                continue
    return out


def load_exclusions(path: Path) -> dict[str, set[int]]:
    """{video filename: excluded frame_idx set} from a video,start_frame,end_frame,reason
    CSV; lines starting with '#' are comments. {} if the file does not exist."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    out: dict[str, set[int]] = defaultdict(set)
    for r in csv.DictReader(lines):
        out[r["video"]].update(range(int(r["start_frame"]), int(r["end_frame"]) + 1))
    return dict(out)


def analyze_clip(
    path: Path, total_frames: int | None = None, excluded: set[int] | None = None
) -> ClipGapStats:
    rows = load_keypoints_long(path)
    if not rows:
        raise ValueError(f"{path} has no rows")

    clip = rows[0]["video"]

    # Real per-clip fps from the actual timestamp_s column, not assumed.
    frame_idxs = sorted({int(r["frame_idx"]) for r in rows})
    timestamps = {int(r["frame_idx"]): float(r["timestamp_s"]) for r in rows}
    if len(frame_idxs) >= 2:
        # timestamp_s = frame_idx / fps (validate_vitpose.py), so fps is
        # recoverable directly from any two distinct detected frames.
        f0, f1 = frame_idxs[0], frame_idxs[-1]
        dt = timestamps[f1] - timestamps[f0]
        fps = (f1 - f0) / dt if dt > 0 else 30.0
    else:
        fps = 30.0

    # last_frame: the clip's real frame count when known (summary.csv), so
    # trailing dropped frames are counted; otherwise the last frame with a
    # keypoint row, which cannot see a dropped tail (noted in the report).
    last_frame = max(frame_idxs)
    frame_count_known = total_frames is not None and total_frames - 1 >= last_frame
    if frame_count_known:
        last_frame = total_frames - 1

    # Whole-frame drops: a frame_idx that never appears in the CSV at all is
    # a missing_detection frame (confirmed against validate_vitpose.py: a
    # frame with box is None writes zero keypoint rows for that frame_idx).
    detected_frames = set(frame_idxs)  # every frame the detector found someone in
    missing_map = {f: (f not in detected_frames) for f in range(last_frame + 1)}
    missing_events = find_run_lengths(missing_map, fps, last_frame)

    # Joint-level gaps: per joint, among frames that DO have a detection,
    # confidence < 0.5. A missing_detection frame is NOT a joint-level
    # confidence observation at all — the frame has no row for any joint, so
    # there is no confidence value to threshold. Counting it as "occluded"
    # here would double-count the same whole-frame event once per joint (17
    # times), contradicting the module's own "never merged" claim. Instead,
    # missing_detection frames are left OUT of the joint-level map entirely
    # (excluded from the run-length scan, not marked True or False) — a run
    # does not continue across one, and it is not itself counted as a gap.
    # They're already reported on their own in section 1.
    joint_conf: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        joint_conf[r["keypoint_name"]][int(r["frame_idx"])] = float(r["confidence"])

    # Listed exclusions (wrong-person lock): drop them from the joint-level
    # frame set only. Detector drops above are unaffected and reported as before.
    excluded_in_clip = sorted(f for f in (excluded or set()) if f in detected_frames)
    used_frames = detected_frames - set(excluded_in_clip)
    n_excluded_frames = (last_frame + 1) - len(detected_frames)

    joint_gap_events: dict[str, list[GapEvent]] = {}
    for joint in COCO17_NAMES:
        conf_by_frame = joint_conf.get(joint, {})
        occluded_map = {
            f: (conf_by_frame[f] < OCCLUSION_CONFIDENCE_THRESHOLD)
            for f in used_frames
        }
        joint_gap_events[joint] = find_run_lengths(occluded_map, fps, last_frame)

    return ClipGapStats(
        clip=clip,
        fps=fps,
        total_frames=last_frame + 1,
        n_excluded_frames=n_excluded_frames,
        n_detected_frames=len(used_frames),
        excluded_frames=excluded_in_clip,
        frame_count_known=frame_count_known,
        missing_detection_events=missing_events,
        joint_gap_events=joint_gap_events,
    )


def bucket_events(events: list[GapEvent]) -> dict[str, list[GapEvent]]:
    lo, hi = TIER_BOUNDARIES_FRAMES
    buckets: dict[str, list[GapEvent]] = {"<5 frames": [], "5-20 frames": [], ">20 frames": []}
    for e in events:
        if e.length_frames < lo:
            buckets["<5 frames"].append(e)
        elif e.length_frames <= hi:
            buckets["5-20 frames"].append(e)
        else:
            buckets[">20 frames"].append(e)
    return buckets


def format_bucket_line(label: str, events: list[GapEvent]) -> str:
    cut_off = sum(1 for e in events if e.touches_boundary)
    seconds = [e.length_seconds for e in events]
    sec_range = f"{min(seconds):.3f}-{max(seconds):.3f}s" if seconds else "n/a"
    adj = sum(1 for e in events if e.adjacent_to_unobserved)
    return (
        f"    {label}: {len(events)} event(s) ({sec_range}), {cut_off} touching a clip "
        f"boundary, {adj} next to a dropped/excluded frame"
    )


def _ranges(frames: list[int]) -> str:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in out)


def render_report(all_stats: list[ClipGapStats]) -> str:
    lines: list[str] = []
    lines.append("# Occlusion-gap statistics (real data, D-007 / ARCHITECTURE.md §3)\n")
    lines.append(
        f"Threshold used: per-joint confidence < {OCCLUSION_CONFIDENCE_THRESHOLD} "
        f"(D-007's own threshold, not validate_vitpose.py's QA thresholds).\n"
    )
    lines.append(
        "Whole-frame drops (missing_detection) and joint-level confidence gaps "
        "are reported separately below and never merged into one count.\n"
    )

    lines.append("## Per-clip fps (measured from real timestamps, not assumed)\n")
    for s in all_stats:
        lines.append(f"- {s.clip}: {s.fps:.2f} fps, {s.total_frames} frames observed")
    lines.append("")

    lines.append("## 1. Whole-frame drops (missing_detection) — per clip\n")
    lines.append(
        "These frames are excluded entirely from the joint-level analysis in "
        "section 2/3 below (no confidence value exists for any joint on a "
        "frame with no detection at all) — reported here on their own, never "
        "counted a second time per joint.\n"
    )
    total_missing_events = 0
    for s in all_stats:
        buckets = bucket_events(s.missing_detection_events)
        n = sum(len(v) for v in buckets.values())
        total_missing_events += n
        lines.append(f"- {s.clip}: {n} event(s) total, {s.n_excluded_frames} frame(s) excluded from joint-level counts")
        for label, events in buckets.items():
            if events:
                lines.append(format_bucket_line(label, events))
        if s.excluded_frames:
            lines.append(
                f"    plus {len(s.excluded_frames)} frame(s) excluded by list (not detector drops), "
                f"frames {_ranges(s.excluded_frames)}"
            )
    lines.append("")

    def render_joint_aggregate(stats_subset: list[ClipGapStats], heading: str) -> list[str]:
        out: list[str] = [f"## {heading}\n"]
        agg_buckets: dict[str, list[GapEvent]] = {"<5 frames": [], "5-20 frames": [], ">20 frames": []}
        per_joint_counts: dict[str, int] = defaultdict(int)
        for s in stats_subset:
            for joint, events in s.joint_gap_events.items():
                per_joint_counts[joint] += len(events)
                b = bucket_events(events)
                for label in agg_buckets:
                    agg_buckets[label].extend(b[label])

        total = sum(len(v) for v in agg_buckets.values())
        out.append(f"Total joint-level gap events: {total}\n")
        for label, events in agg_buckets.items():
            out.append(format_bucket_line(label, events))
        out.append("")
        out.append("Per-joint event counts:\n")
        for joint in COCO17_NAMES:
            out.append(f"- {joint}: {per_joint_counts[joint]} event(s)")
        out.append("")
        return out, total

    section2, total_joint_events = render_joint_aggregate(
        all_stats, "2. Joint-level gaps (confidence < 0.5) — all clips"
    )
    lines.extend(section2)
    lines.append(
        "**Provenance:** All clips are the author's own footage, filmed on a "
        "fixed camera in the same gym.\n"
    )

    lines.append("## 3. Per-clip joint-level breakdown\n")
    for s in all_stats:
        clip_total = sum(len(v) for v in s.joint_gap_events.values())
        lines.append(f"- {s.clip}: {clip_total} joint-level event(s) total")
        for joint in COCO17_NAMES:
            events = s.joint_gap_events[joint]
            if events:
                b = bucket_events(events)
                parts = [f"{label}={len(v)}" for label, v in b.items() if v]
                lines.append(f"    {joint}: " + ", ".join(parts))
    lines.append("")

    n_det = sum(s.n_detected_frames for s in all_stats)
    n_total = sum(s.total_frames for s in all_stats)
    lines.append("## Counts and n\n")
    n_listed = sum(len(s.excluded_frames) for s in all_stats)
    lines.append(
        f"- n = {len(all_stats)} clip(s), {n_total} frame(s) in range, "
        f"{n_det} used for joint-level counts (detected, not excluded by list)"
    )
    lines.append(f"- Frames excluded by list: {n_listed}")
    lines.append(f"- Whole-frame-drop events: {total_missing_events}")
    lines.append(f"- Joint-level gap events (17 joints): {total_joint_events}")
    lines.append(f"- Joint-level observations (used frames x 17 joints): {n_det * len(COCO17_NAMES)}")
    longest = max(
        ((e, s.clip, j) for s in all_stats for j, evs in s.joint_gap_events.items() for e in evs),
        key=lambda t: t[0].length_frames, default=None,
    )
    if longest:
        e, clip, joint = longest
        lines.append(
            f"- Longest joint-level gap: {clip} {joint}, frames {e.start_frame}-{e.end_frame} "
            f"({e.length_frames} frames, {e.length_seconds:.3f} s); touches clip boundary: "
            f"{e.touches_boundary}; next to a dropped/excluded frame: {e.adjacent_to_unobserved}"
        )
    lines.append("")
    unknown = [s.clip for s in all_stats if not s.frame_count_known]
    if unknown:
        lines.append(
            "**Limit:** clip length was not available for: " + ", ".join(unknown) +
            ". For these the last frame is the last frame with a keypoint row, "
            "so dropped frames at the very end of the clip are not counted."
        )
        lines.append("")

    lines.append("## Known limitation\n")
    lines.append(
        "A keypoint with confidence >= 0.5 during a real occlusion (a "
        "\"confidently wrong\" hallucinated position) would not be caught by "
        "this threshold at all. This is a limit of the statistic itself, not "
        "something this script attempts to detect."
    )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/output"))
    parser.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"),
                         help="video,start_frame,end_frame,reason CSV of frames to exclude (default if present)")
    parser.add_argument("--output-file", type=Path, default=None,
                         help="defaults to <input-dir>/occlusion_gap_report.md")
    args = parser.parse_args()

    csv_paths = sorted(args.input_dir.glob("*_keypoints.csv"))
    if not csv_paths:
        print(f"No *_keypoints.csv files found in {args.input_dir}.")
        return

    total_frames = load_total_frames(args.input_dir)
    exclusions = load_exclusions(args.exclusions)
    all_stats = []
    for p in csv_paths:
        clip = load_keypoints_long(p)[0]["video"]
        all_stats.append(analyze_clip(p, total_frames.get(clip), exclusions.get(clip)))

    report = render_report(all_stats)
    print(report)

    out_path = args.output_file or (args.input_dir / "occlusion_gap_report.md")
    out_path.write_text(report)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
