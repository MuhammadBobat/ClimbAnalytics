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
# Frames removed by a listed exclusion (e.g. a wrong-person lock, D-024); set by
# apply_frame_exclusions, never by the loader.
GAP_EXCLUDED = "excluded_by_list"


@dataclass
class JointSeries:
    """One joint's per-frame position, reindexed to the full frame_idx range.

    positions[i] is None where that frame is a gap (missing detection, or
    confidence below OCCLUSION_CONFIDENCE_THRESHOLD) — never filled in here.
    timestamps[i] is the real timestamp_s for frame i, taken from the data,
    not assumed from a constant fps.
    confidences[i] (optional, D-026) is the raw keypoint confidence at frame i, kept
    even when it is below OCCLUSION_CONFIDENCE_THRESHOLD (positions[i] is then None);
    None where there was no detection, or when the series was built without them.
    gap_reasons[i] is None where positions[i] is present, otherwise one of
    GAP_MISSING_DETECTION / GAP_LOW_CONFIDENCE (GAP_UNSPECIFIED if the series
    was built without reasons).
    """
    joint_name: str
    positions: list[tuple[float, float] | None]
    timestamps: list[float]
    gap_reasons: list[str | None] | None = None
    confidences: list[float | None] | None = None

    def __post_init__(self) -> None:
        if self.confidences is not None and len(self.confidences) != len(self.positions):
            raise ValueError("confidences must be the same length as positions")
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


def apply_frame_exclusions(
    joints: dict[str, JointSeries], excluded: set[int]
) -> dict[str, JointSeries]:
    """Copy of every series with the excluded frames set to None (reason GAP_EXCLUDED).
    The input series are not modified."""
    out = {}
    for name, s in joints.items():
        pos, rs = list(s.positions), list(s.gap_reasons)
        for f in excluded:
            if 0 <= f < len(pos):
                pos[f], rs[f] = None, GAP_EXCLUDED
        out[name] = JointSeries(name, pos, s.timestamps, rs, s.confidences)
    return out


def _combined_gap_reason(causes: set) -> str:
    """Why a value derived from several joints is None, most informative cause first."""
    for r in (GAP_EXCLUDED, GAP_MISSING_DETECTION, GAP_LOW_CONFIDENCE):
        if r in causes:
            return r
    return GAP_UNSPECIFIED


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
        confs: list[float | None] = []
        for f in range(last_frame + 1):
            entry = frame_map.get(f)
            confs.append(None if entry is None else entry[2])
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
            gap_reasons=reasons, confidences=confs,
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
# literature source and are validated only empirically (logged as the D-005
# follow-up, 2026-10-28). Equal weight on the four trunk joints is
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
            reasons.append(_combined_gap_reason(causes))
            continue
        x = sum(weights[name] * p[0] for name, p in zip(weights, pts)) / w_total
        y = sum(weights[name] * p[1] for name, p in zip(weights, pts)) / w_total
        positions.append((x, y))
        reasons.append(None)
    return JointSeries(
        joint_name="com_trunk_centroid_proxy", positions=positions,
        timestamps=list(used[0].timestamps), gap_reasons=reasons,
    )


# --- Segment-model CoM (de Leva 1996, DECISIONS.md D-005 follow-up) ---------------------
#
# Whole-body CoM from 14 body segments, next to the trunk-centroid proxy above (which
# stays as the baseline). Source: de Leva, P. (1996), "Adjustments to Zatsiorsky-
# Seluyanov's segment inertia parameters", J Biomech 29(9):1223-1230, Table 4 (p. 1228):
# segment mass as a fraction of body mass and CM position as a fraction of segment
# length from the proximal end, male and female. Values read from the PDF and checked
# against a secondary data file (all matched); stored here as percent / 100.
#
# Segment endpoints use the COCO-17 keypoints as stand-ins for joint centres (not
# measured to be equal); trunk = MIDS-MIDH row (mid-shoulder-joint to mid-hip-joint),
# shank = KJC-AJC row. Head mass sits at the ear midpoint (the paper's head endpoints,
# vertex and mid-gonion, are not in COCO-17); hand and foot mass sit at the wrist and
# ankle keypoints, so their offset CMs are neglected. 2D pixel space (D-008, D-013).
# A segment needs all of its joints: any gap makes the CoM None (no renormalisation).

SEX_TABLES = ("male", "female", "mean")

# {sex: {segment: mass_fraction | (mass_fraction, cm_fraction_from_proximal_end)}}
_DE_LEVA_1996_TABLE_4 = {
    "male": {
        "head": 0.0694, "trunk": (0.4346, 0.4310), "upper_arm": (0.0271, 0.5772),
        "forearm": (0.0162, 0.4574), "hand": 0.0061, "thigh": (0.1416, 0.4095),
        "shank": (0.0433, 0.4395), "foot": 0.0137,
    },
    "female": {
        "head": 0.0668, "trunk": (0.4257, 0.3782), "upper_arm": (0.0255, 0.5754),
        "forearm": (0.0138, 0.4559), "hand": 0.0056, "thigh": (0.1478, 0.3612),
        "shank": (0.0481, 0.4352), "foot": 0.0129,
    },
}


@dataclass(frozen=True)
class SegmentSpec:
    """One body segment. `proximal` / `distal` are joint names; two names mean their
    midpoint. distal=None means a point mass at `proximal` (cm_fraction must be None)."""
    name: str
    mass_fraction: float
    proximal: tuple[str, ...]
    distal: tuple[str, ...] | None = None
    cm_fraction: float | None = None

    def __post_init__(self) -> None:
        if (self.distal is None) != (self.cm_fraction is None):
            raise ValueError(f"{self.name}: distal and cm_fraction must be given together")

    @property
    def joints(self) -> tuple[str, ...]:
        return self.proximal + (self.distal or ())


def _sex_params(sex: str) -> dict:
    if sex not in SEX_TABLES:
        raise ValueError(f"sex must be one of {SEX_TABLES}, got {sex!r}")
    if sex != "mean":
        return _DE_LEVA_1996_TABLE_4[sex]
    m, f = _DE_LEVA_1996_TABLE_4["male"], _DE_LEVA_1996_TABLE_4["female"]
    out = {}
    for seg in m:  # unweighted mean of the two tables: a logged approximation
        if isinstance(m[seg], tuple):
            out[seg] = tuple((a + b) / 2 for a, b in zip(m[seg], f[seg]))
        else:
            out[seg] = (m[seg] + f[seg]) / 2
    return out


def segment_model_specs(sex: str) -> list[SegmentSpec]:
    """The 14 segments (head, trunk, and left/right arm, forearm, hand, thigh, shank, foot)."""
    p = _sex_params(sex)
    specs = [
        SegmentSpec("head", p["head"], ("left_ear", "right_ear")),
        SegmentSpec("trunk", p["trunk"][0], ("left_shoulder", "right_shoulder"),
                    ("left_hip", "right_hip"), p["trunk"][1]),
    ]
    for side in ("left", "right"):
        j = lambda n: (f"{side}_{n}",)  # noqa: E731
        specs += [
            SegmentSpec(f"{side}_upper_arm", p["upper_arm"][0], j("shoulder"), j("elbow"), p["upper_arm"][1]),
            SegmentSpec(f"{side}_forearm", p["forearm"][0], j("elbow"), j("wrist"), p["forearm"][1]),
            SegmentSpec(f"{side}_hand", p["hand"], j("wrist")),
            SegmentSpec(f"{side}_thigh", p["thigh"][0], j("hip"), j("knee"), p["thigh"][1]),
            SegmentSpec(f"{side}_shank", p["shank"][0], j("knee"), j("ankle"), p["shank"][1]),
            SegmentSpec(f"{side}_foot", p["foot"], j("ankle")),
        ]
    return specs


def segment_model_joints(sex: str = "male") -> list[str]:
    """Every joint the segment model needs (the same for all sex tables): 14 of 17."""
    return sorted({j for s in segment_model_specs(sex) for j in s.joints})


def _point(joints: dict[str, JointSeries], names: tuple[str, ...], t: int) -> tuple[float, float]:
    pts = [joints[n].positions[t] for n in names]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def compute_com_from_segments(
    joints: dict[str, JointSeries], specs: list[SegmentSpec], name: str = "com_segment_model"
) -> JointSeries:
    """CoM(t) = Σ m_s · CM_s(t) / Σ m_s, with CM_s = P_prox + r_s (P_dist − P_prox) for a
    segment with two endpoints and CM_s = P_prox for a point mass. None whenever any
    joint of any segment is a gap at t (no renormalisation), with the gap cause kept."""
    needed = sorted({j for s in specs for j in s.joints})
    missing = [j for j in needed if j not in joints]
    if missing:
        raise KeyError(f"joints not in series: {missing}")
    total_mass = sum(s.mass_fraction for s in specs)
    if total_mass <= 0:
        raise ValueError("segment masses must sum to a positive value")
    series = [joints[j] for j in needed]
    n = len(series[0].positions)
    if any(len(s.positions) != n for s in series):
        raise ValueError("all joints must have the same length")

    positions: list[tuple[float, float] | None] = []
    reasons: list[str | None] = []
    for t in range(n):
        if any(joints[j].positions[t] is None for j in needed):
            positions.append(None)
            reasons.append(_combined_gap_reason(
                {joints[j].gap_reasons[t] for j in needed if joints[j].positions[t] is None}))
            continue
        x = y = 0.0
        for s in specs:
            prox = _point(joints, s.proximal, t)
            if s.distal is None:
                cm = prox
            else:
                dist = _point(joints, s.distal, t)
                cm = (prox[0] + s.cm_fraction * (dist[0] - prox[0]),
                      prox[1] + s.cm_fraction * (dist[1] - prox[1]))
            x += s.mass_fraction * cm[0]
            y += s.mass_fraction * cm[1]
        positions.append((x / total_mass, y / total_mass))
        reasons.append(None)
    return JointSeries(name, positions, list(series[0].timestamps), reasons)


def compute_com_segment_model(joints: dict[str, JointSeries], sex: str) -> JointSeries:
    """Whole-body segment-model CoM, de Leva (1996) Table 4. `sex` is required:
    "male", "female", or "mean" (unweighted average of the two tables, a logged
    approximation — sensitivity checks only). See the block comment above."""
    return compute_com_from_segments(
        joints, segment_model_specs(sex), name=f"com_segment_model_de_leva_1996_{sex}")


# --- Candidate CoM estimators B-E (step 2.2c, DECISIONS.md D-026) -----------------------
#
# Candidates compared next to the proxy (A = compute_com) and the full segment model
# (F = compute_com_segment_model). NONE is adopted; the proxy stays the working CoM
# (D-025). Parameters: de Leva (1996) Table 4 as above. The subsets, the trunk-point
# fit, the confidence gate and the blend are this project's own design, with no
# literature source (D-026). Same missing-joint policy as above (a joint below 0.5 is a
# gap, no renormalising over a missing joint) except E, which drops single limb segments.


@dataclass(frozen=True)
class Renormalisation:
    """What renormalising a reduced segment set did (logged in the 2.2c manifest, D-026)."""
    removed: tuple[str, ...]
    removed_mass: float       # sum of the removed segments' fractions before renormalising
    kept_mass_before: float   # sum of the kept fractions before renormalising
    scale: float              # 1 / kept_mass_before


def renormalised_specs(
    specs: list[SegmentSpec], drop: frozenset[str] | set[str]
) -> tuple[list[SegmentSpec], Renormalisation]:
    """The specs without the segments named in `drop`, mass fractions divided by their sum so
    they total 1. Equivalent to placing the removed mass at the CoM of the kept segments
    (D-026 approximation 3)."""
    unknown = set(drop) - {s.name for s in specs}
    if unknown:
        raise KeyError(f"segments not in specs: {sorted(unknown)}")
    kept = [s for s in specs if s.name not in drop]
    kept_mass = sum(s.mass_fraction for s in kept)
    if kept_mass <= 0:
        raise ValueError("no mass left after dropping segments")
    out = [SegmentSpec(s.name, s.mass_fraction / kept_mass, s.proximal, s.distal, s.cm_fraction)
           for s in kept]
    removed = tuple(s.name for s in specs if s.name in drop)
    removed_mass = sum(s.mass_fraction for s in specs if s.name in drop)
    return out, Renormalisation(removed, removed_mass, kept_mass, 1.0 / kept_mass)


def trunk_cm_fraction(sex: str) -> float:
    """de Leva's trunk CM position as a fraction of shoulder-midpoint -> hip-midpoint (the w of
    candidate B that puts it at the trunk CM; 0.4310 male)."""
    return next(s.cm_fraction for s in segment_model_specs(sex) if s.name == "trunk")


def _no_head_specs(sex: str) -> tuple[list[SegmentSpec], Renormalisation]:
    return renormalised_specs(segment_model_specs(sex), {"head"})


def _trunk_legs_specs(sex: str) -> tuple[list[SegmentSpec], Renormalisation]:
    drop = {"head"} | {f"{side}_{seg}" for side in ("left", "right")
                       for seg in ("upper_arm", "forearm", "hand")}
    return renormalised_specs(segment_model_specs(sex), drop)


def compute_com_trunk_point(
    joints: dict[str, JointSeries], w_hip: float, name: str | None = None
) -> JointSeries:
    """Candidate B: CoM = (1 - w) * shoulder midpoint + w * hip midpoint, 0 <= w <= 1.
    None where any of the four trunk joints is a gap. w = 0.5 is the proxy (A). See D-026."""
    if not 0.0 <= w_hip <= 1.0:
        raise ValueError(f"w_hip must be in [0, 1], got {w_hip}")
    s = (1.0 - w_hip) / 2.0
    h = w_hip / 2.0
    series = compute_com(joints, {"left_shoulder": s, "right_shoulder": s,
                                  "left_hip": h, "right_hip": h})
    series.joint_name = name or f"com_trunk_point_w{w_hip:.4f}"
    return series


def compute_com_segment_model_no_head(joints: dict[str, JointSeries], sex: str) -> JointSeries:
    """Candidate C: the segment model without the head (trunk, both arms, both legs; hands at
    wrists, feet at ankles), mass fractions renormalised to sum to 1. 12 joints needed (no
    ears); None if any is a gap. See D-026."""
    specs, _ = _no_head_specs(sex)
    return compute_com_from_segments(joints, specs, name=f"com_no_head_{sex}")


def compute_com_trunk_legs(joints: dict[str, JointSeries], sex: str) -> JointSeries:
    """Candidate D: trunk and both legs only (thigh, shank, foot at ankle), renormalised.
    8 joints needed (shoulders, hips, knees, ankles); None if any is a gap. See D-026."""
    specs, _ = _trunk_legs_specs(sex)
    return compute_com_from_segments(joints, specs, name=f"com_trunk_legs_{sex}")


def _segment_cm(joints: dict[str, JointSeries], s: SegmentSpec, t: int) -> tuple[float, float]:
    prox = _point(joints, s.proximal, t)
    if s.distal is None:
        return prox
    dist = _point(joints, s.distal, t)
    return (prox[0] + s.cm_fraction * (dist[0] - prox[0]),
            prox[1] + s.cm_fraction * (dist[1] - prox[1]))


def _blended(joints, sex, w_hip, confidence_cutoff):
    if not 0.0 < confidence_cutoff <= 1.0:
        raise ValueError(f"confidence_cutoff must be in (0, 1], got {confidence_cutoff}")
    limbs = [s for s in _no_head_specs(sex)[0] if s.name != "trunk"]
    for s in limbs:
        for j in s.joints:
            if j not in joints:
                raise KeyError(f"joints not in series: {[j]}")
            if joints[j].confidences is None:
                raise ValueError(f"{j}: series carry no confidences (needed by the blended CoM)")
    base = compute_com_trunk_point(joints, w_hip)
    positions: list[tuple[float, float] | None] = []
    reasons: list[str | None] = []
    usage: list[tuple[str, ...]] = []
    for t, b in enumerate(base.positions):
        if b is None:
            positions.append(None)
            reasons.append(base.gap_reasons[t])
            usage.append(())
            continue
        x, y, used = b[0], b[1], []
        for s in limbs:
            # position present (so not a gap or exclusion) and raw confidence >= cut-off
            if all(joints[j].positions[t] is not None and joints[j].confidences[t] is not None
                   and joints[j].confidences[t] >= confidence_cutoff for j in s.joints):
                cm = _segment_cm(joints, s, t)
                x += s.mass_fraction * (cm[0] - b[0])
                y += s.mass_fraction * (cm[1] - b[1])
                used.append(s.name)
        positions.append((x, y))
        reasons.append(None)
        usage.append(tuple(used))
    series = JointSeries(f"com_blended_{sex}_w{w_hip:.4f}_c{confidence_cutoff}", positions,
                         list(base.timestamps), reasons)
    return series, usage


def compute_com_blended(
    joints: dict[str, JointSeries], sex: str, w_hip: float, confidence_cutoff: float
) -> JointSeries:
    """Candidate E: E = B + sum over limb segments (upper arms, forearms, thighs, shanks,
    hands, feet) of m'_s * (CM_s - B), m'_s the head-renormalised mass fractions. A segment is
    used only when every one of its joints has a position and raw confidence >= the cut-off
    (hand, foot: their single joint); otherwise its term is zero, so a lost limb falls back
    toward B. None only where B is None (a trunk joint is a gap; trunk joints keep the
    D-007 0.5 rule). Needs JointSeries.confidences. Cut-off is swept, not chosen. See D-026."""
    return _blended(joints, sex, w_hip, confidence_cutoff)[0]


def blended_limb_usage(
    joints: dict[str, JointSeries], sex: str, w_hip: float, confidence_cutoff: float
) -> list[tuple[str, ...]]:
    """Per frame, the names of the limb segments compute_com_blended used (diagnostic)."""
    return _blended(joints, sex, w_hip, confidence_cutoff)[1]


# Working CoM (DECISIONS.md D-026 addenda; status proposed). The author's decision of 2026-11-26 made it EdLfill_c0.5
# (E-dL at the D-007 cut-off whose limb gate is "the joint has a position": observed or Kalman-filled, never a Tier 3
# frame; depends on D-030, not validated), superseding the 2026-11-01 choice of E-dL with the raw-confidence gate. That
# earlier version is kept as compute_com_working_edl_raw_gate, the documented fallback; 5.1a and the earlier runs used it.
# Chosen on the author's visual review and a blind review, not on accuracy; no ground truth. The trunk-centroid proxy
# (compute_com) stays as the baseline. Existing callers still pass the CoM explicitly.
WORKING_COM_CUTOFF = OCCLUSION_CONFIDENCE_THRESHOLD


def position_gated(joints: dict[str, JointSeries]) -> dict[str, JointSeries]:
    """Copies of the joints whose confidence is 1.0 wherever a position exists (observed or filled) and the original value
    elsewhere, so the blended CoM's gate 'position present and confidence >= cut-off' reduces to 'position present'
    (D-026 addendum 2026-11-26). The originals are not modified. On series with no gap filling a position exists only
    where the raw confidence is at least 0.5, so the result then equals the raw-confidence gate."""
    out = {}
    for name, js in joints.items():
        conf = [1.0 if p is not None else (None if js.confidences is None else js.confidences[i])
                for i, p in enumerate(js.positions)]
        out[name] = JointSeries(js.joint_name, js.positions, js.timestamps, js.gap_reasons, conf)
    return out


def compute_com_working_edl_raw_gate(joints: dict[str, JointSeries], sex: str) -> JointSeries:
    """E-dL at cut-off 0.5 with the RAW-confidence limb gate: the working CoM of 2026-11-01 (D-026 addendum), now the
    documented fallback. compute_com_blended with w = de Leva trunk CM fraction (0.4310 male) and confidence_cutoff =
    WORKING_COM_CUTOFF. `sex` is required, as for the segment model."""
    series = compute_com_blended(joints, sex, trunk_cm_fraction(sex), WORKING_COM_CUTOFF)
    series.joint_name = f"com_working_edl_{sex}"
    return series


def compute_com_working(joints: dict[str, JointSeries], sex: str) -> JointSeries:
    """The working CoM since 2026-11-26 (D-026 addendum, proposed): EdLfill_c0.5, E-dL at cut-off 0.5 whose limb gate is
    'the joint has a position' (observed or Kalman-filled, never a Tier 3 frame, D-030). Feed it the gap-filled joints
    (occlusion_handling.as_joint_series); depends on D-030, not validated. Fallback: compute_com_working_edl_raw_gate."""
    series = compute_com_blended(position_gated(joints), sex, trunk_cm_fraction(sex), WORKING_COM_CUTOFF)
    series.joint_name = f"com_working_edlfill_{sex}"
    return series


# --- Smoothness of the CoM speed profile: LDLJ and SPARC (DECISIONS.md D-006) ---------------
#
# Both are from Balasubramanian, Melendez-Calderon, Roby-Brami & Burdet (2015), "On the analysis
# of movement smoothness", J NeuroEng Rehabil 12:112 (primary PDF read 2026-11-01, D-006 addendum).
# They take a speed profile v(t) over ONE segment (a window with no gaps); choosing segments
# (hold-to-hold, D-018) and a minimum segment duration is not done here. No smoothing is applied
# (the paper reports both measures are noise-sensitive). numpy is imported inside the functions.


def speed_profile(
    series: JointSeries, start_idx: int, end_idx: int
) -> tuple[list[float], list[float]] | None:
    """(timestamps, speeds) of a CoM (or any) series for frames start_idx..end_idx inclusive,
    speed from compute_velocity (speed at frame t is the step t-1 -> t). None if any speed in
    the window is None (a gap is never bridged, CLAUDE.md rule 9) or the window is empty."""
    v = compute_velocity(series)
    if not (0 <= start_idx <= end_idx < len(v)):
        return None
    window = v[start_idx:end_idx + 1]
    if any(x is None for x in window):
        return None
    return list(series.timestamps[start_idx:end_idx + 1]), list(window)


def _ldlj(times, speeds, exponent: int) -> float | None:
    import numpy as np
    t, v = np.asarray(times, dtype=float), np.asarray(speeds, dtype=float)
    if t.ndim != 1 or t.shape != v.shape:
        raise ValueError("times and speeds must be 1-D and the same length")
    if len(t) < 3:
        return None
    if not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing")
    v_peak = float(v.max())
    if v_peak <= 0:
        return None
    # second derivative of speed by two passes of np.gradient (central differences, second-order
    # one-sided at the ends, real timestamps), integral by the trapezoid rule
    d2v = np.gradient(np.gradient(v, t, edge_order=2), t, edge_order=2)
    f = d2v * d2v
    integral = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(t)))
    dj = (t[-1] - t[0]) ** exponent / v_peak ** 2 * integral
    return -math.log(dj) if dj > 0 else None


def compute_ldlj(times, speeds) -> float | None:
    """Log dimensionless jerk of a speed profile, D-006:
        LDLJ = -ln( (t2 - t1)^3 / v_peak^2 * integral_{t1}^{t2} |d2v/dt2|^2 dt ).
    The exponent is 3, not the 5 printed in Balasubramanian et al. (2015) Eq. 2: only 3 makes the
    quantity dimensionless for a speed profile (D-006 correction of 2026-10-28; checked
    numerically in tests/test_smoothness.py). None if undefined (fewer than 3 samples, no
    positive speed, or a zero integral)."""
    return _ldlj(times, speeds, 3)


def _sparc_from_spectrum(omega, vhat, amp_threshold: float, omega_c_max: float) -> float | None:
    """SPARC = -(arc length of the normalised magnitude spectrum on [0, wc]), D-006:
        SPARC = -int_0^wc sqrt( (1/wc)^2 + (dVhat/dw)^2 ) dw,   Vhat = V(w)/V(0),
        wc = min( wc_max, min{ w : Vhat(r) < amp_threshold for all r > w } )   (paper Eq. 4, 5).
    On the grid: w is the frequency of the last point with Vhat >= amp_threshold; the integral is
    the length of the polyline through (w_k / wc, Vhat_k) for the grid points w_k <= wc (exact for
    a piecewise-linear Vhat; no point is interpolated at wc when wc = wc_max falls between points)."""
    import numpy as np
    omega, vhat = np.asarray(omega, dtype=float), np.asarray(vhat, dtype=float)
    above = np.nonzero(vhat >= amp_threshold)[0]
    if len(above) == 0:
        return None
    wc = min(omega_c_max, float(omega[above[-1]]))
    if wc <= 0:
        return None
    sel = omega <= wc
    if int(sel.sum()) < 2:
        return None
    x, y = omega[sel] / wc, vhat[sel]
    return -float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def compute_sparc(times, speeds, *, amp_threshold: float, omega_c_max: float, n_fft: int) -> float | None:
    """Spectral arc length (SPARC) of a speed profile, Balasubramanian et al. (2015), D-006.
    Vhat is the Fourier magnitude spectrum of the speed samples normalised by its DC value.

    The three keyword arguments have NO defaults on purpose, so every call site states them.
    The paper's main text names amp_threshold (V-bar) and omega_c_max (rad/s) as free parameters
    and gives values only for its simulated example; the first-run values are the authors'
    reference-code defaults in SPARC_REFERENCE_DEFAULTS (not project-tuned, D-006 addendum).
    The paper's Appendix A (FFT length) was not available.     n_fft >= len(speeds) (zero-padded). Uniform sampling is assumed; fs = (N-1)/(t_last - t_first).
    omega_c_max above the Nyquist frequency is capped by the grid. None if undefined."""
    import numpy as np
    if amp_threshold <= 0 or omega_c_max <= 0:
        raise ValueError("amp_threshold and omega_c_max must be positive")
    t, v = np.asarray(times, dtype=float), np.asarray(speeds, dtype=float)
    if t.ndim != 1 or t.shape != v.shape:
        raise ValueError("times and speeds must be 1-D and the same length")
    if n_fft < len(v):
        raise ValueError("n_fft must be at least the number of samples")
    if len(v) < 2 or not t[-1] > t[0]:
        return None
    spectrum = np.abs(np.fft.rfft(v, n=n_fft))
    if spectrum[0] <= 0:
        return None
    fs = (len(t) - 1) / (t[-1] - t[0])
    omega = 2 * math.pi * np.arange(len(spectrum)) * fs / n_fft
    return _sparc_from_spectrum(omega, spectrum / spectrum[0], amp_threshold, omega_c_max)


# SPARC reference-code defaults, not project-tuned; swept in 5.1a (D-004 method, D-006 addendum).
# Source (as stated by the author, read 2026-11-01): github.com/siva82kb/SPARC, scripts/smoothness.py,
# commit 3650934: amp_th = 0.05, fc = 10 Hz (omega_c_max = 20 pi rad/s), padlevel = 4 with
# nfft = 2^(ceil(log2 N) + padlevel). Call sites still pass amp_threshold / omega_c_max / n_fft
# explicitly (compute_sparc has no defaults); use sparc_n_fft for n_fft.
SPARC_REFERENCE_DEFAULTS = {"amp_threshold": 0.05, "omega_c_max": 20 * math.pi, "padlevel": 4}


def sparc_n_fft(n_samples: int, padlevel: int = SPARC_REFERENCE_DEFAULTS["padlevel"]) -> int:
    """nfft = 2^(ceil(log2 N) + padlevel), the reference code's FFT length."""
    return 2 ** (math.ceil(math.log2(n_samples)) + padlevel)


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
