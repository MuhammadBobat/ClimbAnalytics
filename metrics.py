"""
Per-joint velocity (ARCHITECTURE.md §4.1), cumulative displacement (§4.2) and
a trunk-centroid CoM PROXY (§4.3, step 2.2).

    v_j(t) = || p_j(t) - p_j(t-1) || / Δt         (§4.1)
    D_j    = Σ_t || p_j(t) - p_j(t-1) ||            (§4.2, full clip or a window)

p_j(t) is defined in §4.1 as "the (smoothed, gap-filled) 2D pixel position" —
that's a stated assumption about the INPUT, not a named smoothing method, so
this module adds no smoothing of its own. Pixel-space Euclidean norm is a
meaningful (if uncalibrated, D-013) motion proxy only because of D-008's
fixed-perpendicular-camera assumption — see DECISIONS.md D-008.

occlusion_handling.py (D-007's three-tier gap-filling) does not exist yet, so
real keypoint series available today (validation/output/*_keypoints.csv) are
NOT actually gap-filled. This module still assumes gap-filled input per the
ARCHITECTURE.md §5 module boundary (that is occlusion_handling.py's job, not
this one's) — when fed the raw, ungapfilled CSVs for testing, a gap (missing
frame, or confidence < 0.5 per D-007's own threshold, see
validation/analyze_occlusion_gaps.py) produces None/NaN at that step rather
than a value that silently bridges the gap. The demo below prints this
explicitly: these are raw, unsmoothed, unfilled numbers, expected to be
jittery, not a final result.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

# D-007's own occlusion threshold (ARCHITECTURE.md §3) — a keypoint below
# this confidence is treated as missing here too, same as a fully absent
# frame. Not this module's own QA threshold; see validate_vitpose.py's header
# comment for why QA thresholds (0.3/0.5) are a different, unrelated thing.
# This is the ONE definition of D-007's threshold in the repo: import it from
# here (validation/analyze_occlusion_gaps.py does), do not redefine it.
OCCLUSION_CONFIDENCE_THRESHOLD = 0.5

# Why a position is None. Kept separate because they are different failures
# (a lost person-box vs. one low-confidence joint) — see
# validation/analyze_occlusion_gaps.py's docstring.
GAP_MISSING_DETECTION = "missing_detection"
GAP_LOW_CONFIDENCE = "low_confidence"
GAP_UNSPECIFIED = "unspecified"  # series built by hand without a reason


@dataclass
class JointSeries:
    """One joint's per-frame position, reindexed to the full frame_idx range.

    positions[i] is None where that frame is a gap (missing detection, or
    confidence below OCCLUSION_CONFIDENCE_THRESHOLD) — never filled in here.
    timestamps[i] is the real timestamp_s for frame i, taken from the data,
    not assumed from a constant fps.
    gap_reasons[i] is None where positions[i] is present, otherwise one of
    GAP_MISSING_DETECTION / GAP_LOW_CONFIDENCE (GAP_UNSPECIFIED if the series
    was built without reasons).
    """
    joint_name: str
    positions: list[tuple[float, float] | None]
    timestamps: list[float]
    gap_reasons: list[str | None] | None = None

    def __post_init__(self) -> None:
        if self.gap_reasons is None:
            self.gap_reasons = [
                None if p is not None else GAP_UNSPECIFIED for p in self.positions
            ]
        elif len(self.gap_reasons) != len(self.positions):
            raise ValueError("gap_reasons must be the same length as positions")


@dataclass
class GapCounts:
    """Frames with no usable position, split by cause (never merged)."""
    missing_detection: int
    low_confidence: int
    unspecified: int

    @property
    def total(self) -> int:
        return self.missing_detection + self.low_confidence + self.unspecified


def count_gap_frames(series: JointSeries) -> GapCounts:
    reasons = series.gap_reasons
    return GapCounts(
        missing_detection=sum(1 for r in reasons if r == GAP_MISSING_DETECTION),
        low_confidence=sum(1 for r in reasons if r == GAP_LOW_CONFIDENCE),
        unspecified=sum(1 for r in reasons if r == GAP_UNSPECIFIED),
    )


@dataclass
class DisplacementResult:
    """§4.2 total plus how many steps contributed / were skipped due to gaps."""
    total: float
    n_steps_counted: int
    n_steps_skipped: int  # a step is skipped if either end is a gap


def load_keypoints_csv(path: Path) -> dict[str, JointSeries]:
    """Load one validate_vitpose.py keypoints CSV into one JointSeries per
    joint, reindexed to the full observed frame_idx range.

    Reindexing is required, not optional: a missing_detection frame has no
    row at all in the CSV (confirmed directly against real output, e.g.
    IMG_2412_keypoints.csv has no rows for frame_idx 41/42), so building a
    series by row position instead of by frame_idx would silently treat the
    frames on either side of a gap as adjacent, and velocity would compute
    straight across it without anyone noticing.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")

    frame_idxs = sorted({int(r["frame_idx"]) for r in rows})
    last_frame = max(frame_idxs)

    # timestamp_s per frame_idx, from the real detected rows only; gap frames
    # get an interpolated timestamp from fps (needed only to report Δt for a
    # step that spans a gap — the position itself still stays None/absent).
    ts_by_frame: dict[int, float] = {}
    for r in rows:
        ts_by_frame[int(r["frame_idx"])] = float(r["timestamp_s"])
    if len(frame_idxs) >= 2:
        f0, f1 = frame_idxs[0], frame_idxs[-1]
        dt_total = ts_by_frame[f1] - ts_by_frame[f0]
        fps = (f1 - f0) / dt_total if dt_total > 0 else 30.0
    else:
        fps = 30.0
    timestamps = [
        ts_by_frame.get(f, f / fps) for f in range(last_frame + 1)
    ]

    by_joint: dict[str, dict[int, tuple[float, float, float]]] = {}
    for r in rows:
        joint = r["keypoint_name"]
        by_joint.setdefault(joint, {})[int(r["frame_idx"])] = (
            float(r["x"]), float(r["y"]), float(r["confidence"]),
        )

    series: dict[str, JointSeries] = {}
    for joint, frame_map in by_joint.items():
        positions: list[tuple[float, float] | None] = []
        reasons: list[str | None] = []
        for f in range(last_frame + 1):
            entry = frame_map.get(f)
            if entry is None:
                positions.append(None)
                reasons.append(GAP_MISSING_DETECTION)
            elif entry[2] < OCCLUSION_CONFIDENCE_THRESHOLD:
                positions.append(None)
                reasons.append(GAP_LOW_CONFIDENCE)
            else:
                positions.append((entry[0], entry[1]))
                reasons.append(None)
        series[joint] = JointSeries(
            joint_name=joint, positions=positions, timestamps=timestamps,
            gap_reasons=reasons,
        )
    return series


def compute_velocity(series: JointSeries) -> list[float | None]:
    """Per-joint velocity, §4.1: v_j(t) = ||p_j(t) - p_j(t-1)|| / Δt.

    v[0] is always None (no preceding frame). v[t] is None wherever p_j(t)
    or p_j(t-1) is a gap — a gap is never bridged by interpolating or by
    skipping ahead to the next observed frame; that would fabricate a value
    across missing data (CLAUDE.md rule 9).
    """
    positions = series.positions
    timestamps = series.timestamps
    out: list[float | None] = [None]
    for t in range(1, len(positions)):
        p_prev, p_curr = positions[t - 1], positions[t]
        if p_prev is None or p_curr is None:
            out.append(None)
            continue
        dt = timestamps[t] - timestamps[t - 1]
        dist = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        out.append(dist / dt if dt > 0 else None)
    return out


def compute_displacement(
    series: JointSeries, start_idx: int = 0, end_idx: int | None = None
) -> DisplacementResult:
    """Cumulative displacement, §4.2: D_j = Σ_t ||p_j(t) - p_j(t-1)|| over
    the full clip or a chosen window [start_idx, end_idx] (inclusive).

    A window [a, b] counts only the steps whose BOTH endpoints lie inside it,
    i.e. t = a+1 .. b. The step entering the window (a-1 -> a) is not counted.

    A gap breaks the sum into segments: the step spanning a gap contributes
    nothing, rather than treating the pre-gap and post-gap positions as if
    they were adjacent frames. Those skipped steps are counted and returned,
    so a low displacement caused by gaps is distinguishable from a still joint.
    """
    positions = series.positions
    if end_idx is None:
        end_idx = len(positions) - 1
    total = 0.0
    counted = 0
    skipped = 0
    for t in range(max(start_idx, 0) + 1, min(end_idx, len(positions) - 1) + 1):
        p_prev, p_curr = positions[t - 1], positions[t]
        if p_prev is None or p_curr is None:
            skipped += 1
            continue
        total += math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        counted += 1
    return DisplacementResult(total=total, n_steps_counted=counted, n_steps_skipped=skipped)


def compute_cumulative_displacement(
    series: JointSeries, start_idx: int = 0, end_idx: int | None = None
) -> float:
    """§4.2 total only; see compute_displacement for window and gap semantics."""
    return compute_displacement(series, start_idx, end_idx).total


# --- Centre of mass (ARCHITECTURE.md §4.3, DECISIONS.md D-005) --------------
#
# PROXY, NOT a segment-mass model. §4.3 allows a simplified proxy if its exact
# weights are logged; the weights below are an engineering heuristic with no
# literature source and are validated only empirically (D-005 follow-up drafted,
# not yet logged in DECISIONS.md). Equal weight on the four trunk joints is
# algebraically (hip_midpoint + shoulder_midpoint) / 2. It ignores the head and
# all limbs, so it is not whole-body CoM and must be called a proxy wherever it
# is reported. 2D pixel space only (D-008, D-013).
#
# The weights table is keyed by COCO-17 joint name so a different table (e.g. a
# cited anthropometric one) can be passed to compute_com without changing it.
TRUNK_CENTROID_PROXY_WEIGHTS: dict[str, float] = {
    "left_shoulder": 1.0,
    "right_shoulder": 1.0,
    "left_hip": 1.0,
    "right_hip": 1.0,
}


def compute_com(
    joints: dict[str, JointSeries],
    weights: dict[str, float] = TRUNK_CENTROID_PROXY_WEIGHTS,
) -> JointSeries:
    """CoM(t) = Σ_i w_i p_i(t) / Σ_i w_i over the joints named in `weights`
    (§4.3). With the default table this is the trunk-centroid PROXY.

    Returns a JointSeries (so compute_velocity / compute_displacement apply
    unchanged). CoM(t) is None whenever ANY weighted joint is a gap at t — no
    renormalising over the remaining joints, which would make the estimate jump
    when a joint drops out. D-007's tiers belong to occlusion_handling.py, which
    does not exist yet, so nothing is filled here.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    missing = [name for name in weights if name not in joints]
    if missing:
        raise KeyError(f"joints not in series: {missing}")

    used = [joints[name] for name in weights]
    n = len(used[0].positions)
    if any(len(j.positions) != n for j in used):
        raise ValueError("all weighted joints must have the same length")
    w_total = sum(weights.values())

    positions: list[tuple[float, float] | None] = []
    reasons: list[str | None] = []
    for t in range(n):
        pts = [j.positions[t] for j in used]
        if any(p is None for p in pts):
            positions.append(None)
            causes = {j.gap_reasons[t] for j in used if j.positions[t] is None}
            # a dropped frame (no person) dominates a single low-confidence joint
            reasons.append(
                GAP_MISSING_DETECTION if GAP_MISSING_DETECTION in causes
                else GAP_LOW_CONFIDENCE if GAP_LOW_CONFIDENCE in causes
                else GAP_UNSPECIFIED
            )
            continue
        x = sum(weights[name] * p[0] for name, p in zip(weights, pts)) / w_total
        y = sum(weights[name] * p[1] for name, p in zip(weights, pts)) / w_total
        positions.append((x, y))
        reasons.append(None)
    return JointSeries(
        joint_name="com_trunk_centroid_proxy", positions=positions,
        timestamps=list(used[0].timestamps), gap_reasons=reasons,
    )


def _demo(path: Path) -> None:
    print(f"metrics.py demo — {path.name}")
    print(
        "NOTE: occlusion_handling.py (D-007) is not implemented yet, so this "
        "is raw, unsmoothed, unfilled ViTPose-L output, not the gap-filled "
        "input §4.1/§4.2 actually assume. A confidence < 0.5 keypoint "
        "(D-007's threshold) is counted as missing here, same as a fully "
        "absent frame. Expect jittery numbers, not a final result.\n"
    )
    all_series = load_keypoints_csv(path)
    for joint_name, series in all_series.items():
        gaps = count_gap_frames(series)
        velocities = compute_velocity(series)
        valid_v = [v for v in velocities if v is not None]
        disp = compute_displacement(series)
        mean_v = sum(valid_v) / len(valid_v) if valid_v else float("nan")
        max_v = max(valid_v) if valid_v else float("nan")
        print(
            f"  {joint_name:16s} n={len(series.positions):3d}  "
            f"dropped_frames={gaps.missing_detection:3d}  "
            f"low_conf={gaps.low_confidence:3d}  "
            f"mean_v={mean_v:8.1f}px/s  max_v={max_v:8.1f}px/s  "
            f"displacement={disp.total:8.1f}px "
            f"({disp.n_steps_skipped} step(s) skipped for gaps)"
        )
    com = compute_com(all_series)
    com_gaps = count_gap_frames(com)
    com_v = [v for v in compute_velocity(com) if v is not None]
    print(
        f"\n  {com.joint_name} (PROXY, not whole-body CoM): "
        f"dropped_frames={com_gaps.missing_detection} low_conf={com_gaps.low_confidence} "
        f"mean_v={sum(com_v) / len(com_v) if com_v else float('nan'):.1f}px/s "
        f"displacement={compute_displacement(com).total:.1f}px"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keypoints-csv", type=Path, default=Path("validation/output/IMG_2412_keypoints.csv"))
    args = parser.parse_args()
    _demo(args.keypoints_csv)
