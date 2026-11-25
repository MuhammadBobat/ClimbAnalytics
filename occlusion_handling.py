"""
Three-tier occlusion handling (ARCHITECTURE.md §3; DECISIONS.md D-007, D-028 proposed).

Fills gaps in per-joint keypoint series and flags every frame. NONE of the tiers is validated: the tier
boundaries (under 5, 5 to 20, over 20 frames) and the fill methods are unvalidated heuristics (D-007), to be tested by
the hide-and-fill test (step 3.3). The raw input is never modified: results are new objects, the raw series are carried
through untouched.

Input: dict[str, metrics.JointSeries] from metrics.load_keypoints_csv, optionally through
metrics.apply_frame_exclusions. A joint is "observed" at a frame when its position is not None (confidence at or above
metrics.OCCLUSION_CONFIDENCE_THRESHOLD, 0.5, and a detected person box); everything else is a gap, whatever the cause
(low confidence, detector drop). Excluded frames (metrics.GAP_EXCLUDED) are never filled and never bridged.

Output: dict[str, FilledJoint] with a filled copy of the positions and a per-frame Flag (observed, tier1, tier2, tier3,
unavailable). Use as_joint_series to feed metrics.py (Tier 3 positions are left out by default: ARCHITECTURE §3 says a
stale joint is excluded from per-limb metrics).

Tiers (lengths in frames; constants below):
  Tier 1, 1 to 4 frames: per-axis cubic spline (not-a-knot) through observed frames only, tier1_context observed
      frames on each side, at least 2 on each side.
  Tier 2, 5 to 20 frames: constant-velocity Kalman filter with an RTS backward pass (or forward only,
      tier2_causal), noise estimated from the joint's own observed data; limb joints are clamped to the segment-length
      range seen in the same clip (a stand-in for the joint-limit clamp of ARCHITECTURE §3, no angular limits are
      sourced); trunk joints use the affine mirror of the other three trunk joints when all three are observed on every
      frame of the gap.
  Tier 3, over 20 frames, limb joints only: stale: placed at the offset (in torso lengths, in a trunk-aligned frame) it
      had at the nearest observed frame, so it follows the trunk. Trunk joints are the root and are never Tier 3: a trunk
      gap over 20 frames is unavailable, and a limb frame whose trunk is unavailable is unavailable (no circularity).
  Gaps touching a clip start or end, or an excluded frame, have no observed frame on one side: Tier 1 and 2 leave
  them unavailable (no extrapolation); Tier 3 uses the observed side.
OcclusionParams.force_method (default None, behaviour unchanged) forces one method at any gap length; it exists only for the
hide-and-fill evaluation (step 3.3, evaluation/3.3/, D-028 addendum), not for use in the pipeline.
The five face points are not handled. A confident wrong keypoint (confidence >= 0.5) is "observed" and invisible here.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.interpolate import CubicSpline

from metrics import GAP_EXCLUDED, JointSeries

# D-007 boundaries (ARCHITECTURE §3): "< 5 frames", "5-20 frames", "> 20 frames". Unvalidated, step 3.3.
TIER1_MAX_FRAMES = 4
TIER2_MAX_FRAMES = 20

TRUNK = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
ARM_JOINTS = ("left_elbow", "right_elbow", "left_wrist", "right_wrist")
LEG_JOINTS = ("left_knee", "right_knee", "left_ankle", "right_ankle")
BODY_JOINTS = TRUNK + ARM_JOINTS + LEG_JOINTS
# processed proximal to distal so a clamp or a root always uses an earlier result
PROCESSING_ORDER = TRUNK + ("left_elbow", "right_elbow", "left_knee", "right_knee",
                            "left_wrist", "right_wrist", "left_ankle", "right_ankle")
SEGMENT_PARENT = {
    "left_elbow": "left_shoulder", "left_wrist": "left_elbow",
    "right_elbow": "right_shoulder", "right_wrist": "right_elbow",
    "left_knee": "left_hip", "left_ankle": "left_knee",
    "right_knee": "right_hip", "right_ankle": "right_knee",
}


class Flag(str, Enum):
    OBSERVED = "observed"
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    UNAVAILABLE = "unavailable"


USABLE_FLAGS = (Flag.OBSERVED, Flag.TIER1, Flag.TIER2)

# OcclusionParams.force_method -> the tier branch of fill_gaps that runs (step 3.3 hide-and-fill, D-028 addendum)
FORCE_METHODS = {"spline": Flag.TIER1, "kalman": Flag.TIER2, "trunk_offset": Flag.TIER3}


@dataclass(frozen=True)
class OcclusionParams:
    """Every tunable. None of these values is validated or sourced (D-028 proposed)."""
    tier1_context: int = 5            # observed frames used on each side (ARCHITECTURE §3 says 5 to 10)
    min_each_side: int = 2            # observed frames needed on each side for Tier 1 and 2
    tier2_context: int = 10           # observed frames on each side in the Kalman window
    tier2_causal: bool = False        # True: forward filter only (no backward pass)
    tier2_clamp: bool = True          # segment-length clamp for limb joints
    tier2_mirror: bool = True         # affine mirror for trunk joints
    min_acceleration_triples: int = 3  # observed consecutive triples needed to estimate the noise
    force_method: str | None = None   # EVALUATION ONLY (step 3.3, D-028 addendum): None = method by gap length (D-007);
                                      # "spline", "kalman" or "trunk_offset" = use that method for every gap whatever its
                                      # length. Flags still follow the length-based tier; the method column records the truth.


@dataclass(frozen=True)
class GapEvent:
    joint: str
    first: int
    last: int
    length: int
    tier: Flag                         # by length only: TIER1, TIER2 or TIER3
    filled: bool
    method: str | None                 # spline, kalman, kalman_causal, mirror, trunk_offset, or None
    touches_start: bool
    touches_end: bool
    adjacent_excluded: bool
    reasons: tuple[str, ...]
    note: str = ""


@dataclass
class FilledJoint:
    name: str
    raw: JointSeries                   # the input series, same object, never modified
    positions: list[tuple[float, float] | None]
    flags: list[Flag]
    methods: list[str | None]
    sigma_px: list[float | None]       # position standard deviation of a Kalman fill, else None
    events: list[GapEvent] = field(default_factory=list)


def classify_gap(length: int) -> Flag:
    """Tier by gap length in frames (D-007 boundaries)."""
    if length < 1:
        raise ValueError("gap length must be at least 1")
    if length <= TIER1_MAX_FRAMES:
        return Flag.TIER1
    return Flag.TIER2 if length <= TIER2_MAX_FRAMES else Flag.TIER3


def _excluded_frames(series: JointSeries) -> set[int]:
    return {t for t, r in enumerate(series.gap_reasons) if r == GAP_EXCLUDED and series.positions[t] is None}


def find_gaps(series: JointSeries) -> list[tuple[int, int]]:
    """Maximal runs (first, last) of frames with no usable position, whatever the cause. Excluded frames are
    not part of a run (they are walls)."""
    ex = _excluded_frames(series)
    out, start = [], None
    n = len(series.positions)
    for t in range(n + 1):
        gap = t < n and series.positions[t] is None and t not in ex
        if gap and start is None:
            start = t
        elif not gap and start is not None:
            out.append((start, t - 1))
            start = None
    return out


def _observed_context(pos, excluded: set[int], first: int, last: int, k: int):
    """Up to k observed frames before and after the gap, nearest first, stopping at excluded frames."""
    left, i = [], first - 1
    while i >= 0 and len(left) < k and i not in excluded:
        if pos[i] is not None:
            left.append(i)
        i -= 1
    right, i = [], last + 1
    while i < len(pos) and len(right) < k and i not in excluded:
        if pos[i] is not None:
            right.append(i)
        i += 1
    return left[::-1], right


# --- Tier 1 -------------------------------------------------------------------------------------------

def fill_tier1(series: JointSeries, first: int, last: int, params: OcclusionParams):
    """Per-axis not-a-knot cubic spline through the observed frames around the gap. Returns a list of (x, y) for
    frames first..last, or None if either side has fewer than params.min_each_side observed frames."""
    pos, ts = series.positions, series.timestamps
    left, right = _observed_context(pos, _excluded_frames(series), first, last, params.tier1_context)
    if len(left) < params.min_each_side or len(right) < params.min_each_side:
        return None
    idx = left + right
    t = np.array([ts[i] for i in idx])
    sx = CubicSpline(t, [pos[i][0] for i in idx], bc_type="not-a-knot")
    sy = CubicSpline(t, [pos[i][1] for i in idx], bc_type="not-a-knot")
    tt = np.array([ts[i] for i in range(first, last + 1)])
    return [(float(a), float(b)) for a, b in zip(sx(tt), sy(tt))]


# --- Tier 2 -------------------------------------------------------------------------------------------

def estimate_noise(series: JointSeries, params: OcclusionParams):
    """(sigma_d2, sigma_a, sigma_m): robust (MAD) scale of the second difference of the observed positions on
    consecutive observed triples, pooled over x and y; sigma_a = sigma_d2 / dt^2 is used as the process (acceleration)
    noise and sigma_m = sigma_d2 / sqrt(6) as the measurement noise (the white-noise value that would produce that
    second-difference scale; an upper bound). Heuristic, from the joint's own data. None if there are too few triples."""
    pos, ts = series.positions, series.timestamps
    d2, dts = [], []
    for t in range(1, len(pos) - 1):
        if pos[t - 1] is None or pos[t] is None or pos[t + 1] is None:
            continue
        d2.append(pos[t + 1][0] - 2 * pos[t][0] + pos[t - 1][0])
        d2.append(pos[t + 1][1] - 2 * pos[t][1] + pos[t - 1][1])
        dts.append(((ts[t + 1] - ts[t - 1]) / 2.0))
    if len(dts) < params.min_acceleration_triples:
        return None
    med = statistics.median(d2)
    sigma_d2 = max(1.4826 * statistics.median(abs(v - med) for v in d2), 1e-9)
    dt = statistics.median(dts)
    return sigma_d2, sigma_d2 / dt ** 2, sigma_d2 / math.sqrt(6.0)


def _kalman_axis(times, z, observed, sigma_a, sigma_m, causal):
    """Constant-velocity Kalman filter (+ RTS smoother unless causal) on one axis. times, z: arrays over the window,
    observed: bool array. Returns (position, position std) arrays over the window (NaN before the second observed
    sample). State [position, velocity], white-noise-acceleration process noise, initial state from the first two
    observed samples (position = second sample, velocity = their finite difference, with the covariance those two
    noisy samples imply)."""
    n = len(times)
    obs_idx = [i for i in range(n) if observed[i]]
    i1, i2 = obs_idx[0], obs_idx[1]
    dt0 = times[i2] - times[i1]
    r = sigma_m ** 2
    x0 = np.array([z[i2], (z[i2] - z[i1]) / dt0])
    p0 = np.array([[r, r / dt0], [r / dt0, 2 * r / dt0 ** 2]])
    xf = np.full((n, 2), np.nan)
    pf = np.full((n, 2, 2), np.nan)
    xp = np.full((n, 2), np.nan)
    pp = np.full((n, 2, 2), np.nan)
    fs = [None] * n
    xf[i2], pf[i2] = x0, p0
    for k in range(i2 + 1, n):
        dt = times[k] - times[k - 1]
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        fs[k] = f
        xp[k] = f @ xf[k - 1]
        pp[k] = f @ pf[k - 1] @ f.T + q
        if observed[k]:
            s = pp[k][0, 0] + r
            kg = pp[k][:, 0] / s
            xf[k] = xp[k] + kg * (z[k] - xp[k][0])
            pf[k] = pp[k] - np.outer(kg, pp[k][0, :])
        else:
            xf[k], pf[k] = xp[k], pp[k]
    if causal:
        return xf[:, 0], np.sqrt(np.maximum(pf[:, 0, 0], 0.0))
    xs, ps = xf.copy(), pf.copy()
    for k in range(n - 2, i2 - 1, -1):
        c = pf[k] @ fs[k + 1].T @ np.linalg.inv(pp[k + 1])
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
        ps[k] = pf[k] + c @ (ps[k + 1] - pp[k + 1]) @ c.T
    return xs[:, 0], np.sqrt(np.maximum(ps[:, 0, 0], 0.0))


def fill_tier2_kalman(series: JointSeries, first: int, last: int, params: OcclusionParams):
    """Kalman fill of a gap. Returns (positions, sigma) lists for frames first..last, or None (no bracketing observed
    frames, or no noise estimate)."""
    pos, ts = series.positions, series.timestamps
    left, right = _observed_context(pos, _excluded_frames(series), first, last, params.tier2_context)
    if len(left) < params.min_each_side or len(right) < params.min_each_side:
        return None
    noise = estimate_noise(series, params)
    if noise is None:
        return None
    _, sigma_a, sigma_m = noise
    lo, hi = left[0], right[-1]
    frames = range(lo, hi + 1)
    times = np.array([ts[i] for i in frames])
    observed = np.array([pos[i] is not None and i not in _excluded_frames(series) for i in frames])
    out_x, sx = _kalman_axis(times, np.array([pos[i][0] if pos[i] is not None else 0.0 for i in frames]),
                             observed, sigma_a, sigma_m, params.tier2_causal)
    out_y, sy = _kalman_axis(times, np.array([pos[i][1] if pos[i] is not None else 0.0 for i in frames]),
                             observed, sigma_a, sigma_m, params.tier2_causal)
    res, sig = [], []
    for i in range(first, last + 1):
        j = i - lo
        res.append((float(out_x[j]), float(out_y[j])))
        sig.append(float(math.hypot(sx[j], sy[j])))
    return res, sig


def segment_bounds(joints: dict[str, JointSeries]) -> dict[str, tuple[float, float]]:
    """Per distal joint, the (min, max) 2D length of its segment to its parent over frames where both ends are
    observed in the same clip (needs at least 2 such frames)."""
    out = {}
    for child, parent in SEGMENT_PARENT.items():
        if child not in joints or parent not in joints:
            continue
        lens = [math.hypot(c[0] - p[0], c[1] - p[1]) for c, p in zip(joints[child].positions, joints[parent].positions)
                if c is not None and p is not None]
        if len(lens) >= 2:
            out[child] = (min(lens), max(lens))
    return out


def clamp_to_segment(p, parent, bounds):
    """Project p radially so that its distance to parent lies within bounds."""
    lo, hi = bounds
    dx, dy = p[0] - parent[0], p[1] - parent[1]
    d = math.hypot(dx, dy)
    if d == 0:
        return p
    target = min(max(d, lo), hi)
    if target == d:
        return p
    return (parent[0] + dx / d * target, parent[1] + dy / d * target)


def fill_trunk_mirror(joints: dict[str, JointSeries], name: str, first: int, last: int):
    """Affine mirror of a lost trunk joint: its barycentric coordinates with respect to the other three trunk joints
    (taken at the nearest frame, before the gap if possible, where all four are observed) are applied to those three
    joints at each gap frame. Only if the other three are observed on every frame of the gap. Returns positions or None."""
    others = [n for n in TRUNK if n != name]
    if any(n not in joints for n in others) or name not in joints:
        return None
    for t in range(first, last + 1):
        if any(joints[n].positions[t] is None for n in others):
            return None

    def all_observed(t):
        return all(joints[n].positions[t] is not None for n in TRUNK)
    ref = next((t for t in range(first - 1, -1, -1) if all_observed(t)), None)
    if ref is None:
        ref = next((t for t in range(last + 1, len(joints[name].positions)) if all_observed(t)), None)
    if ref is None:
        return None
    p = [np.array(joints[n].positions[ref], dtype=float) for n in others]
    target = np.array(joints[name].positions[ref], dtype=float)
    basis = np.column_stack([p[1] - p[0], p[2] - p[0]])
    if abs(np.linalg.det(basis)) < 1e-9:
        return None
    a, b = np.linalg.solve(basis, target - p[0])
    out = []
    for t in range(first, last + 1):
        q = [np.array(joints[n].positions[t], dtype=float) for n in others]
        r = q[0] + a * (q[1] - q[0]) + b * (q[2] - q[0])
        out.append((float(r[0]), float(r[1])))
    return out


# --- Tier 3 -------------------------------------------------------------------------------------------

def _trunk_frame(positions: dict[str, list], t: int, group: str):
    """(origin, u, w, scale) of the trunk-aligned frame at frame t: origin = shoulder midpoint for arm joints and hip
    midpoint for leg joints, u along hip-midpoint to shoulder-midpoint, w perpendicular, scale = torso length (px).
    None if any trunk joint is unavailable."""
    if any(positions.get(n) is None or positions[n][t] is None for n in TRUNK):
        return None
    pts = [positions[n][t] for n in TRUNK]
    ls, rs, lh, rh = pts
    sm = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hm = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    ax = (sm[0] - hm[0], sm[1] - hm[1])
    s = math.hypot(*ax)
    if s == 0:
        return None
    u = (ax[0] / s, ax[1] / s)
    return (sm if group == "arm" else hm), u, (-u[1], u[0]), s


def fill_tier3(series: JointSeries, name: str, positions: dict[str, list], first: int, last: int):
    """Stale fill of a limb joint (see the module docstring). `positions` holds the working (filled) positions of all
    body joints, trunk first. Returns (positions, note): a list over first..last with None where the trunk is
    unavailable, or (None, note) if no reference frame exists."""
    group = "arm" if name in ARM_JOINTS else "leg"
    pos, ex = series.positions, _excluded_frames(series)

    def reference(frames):
        for i in frames:
            if i in ex:
                return None
            if pos[i] is not None and _trunk_frame(positions, i, group) is not None:
                return i
        return None
    # the nearest observed frame before the gap at which the trunk is available, else the nearest after it
    ref = reference(range(first - 1, -1, -1))
    if ref is None:
        ref = reference(range(last + 1, len(pos)))
    if ref is None:
        return None, "no observed reference frame with an available trunk"
    fr = _trunk_frame(positions, ref, group)
    o, u, w, s = fr
    d = (pos[ref][0] - o[0], pos[ref][1] - o[1])
    cu, cw = (d[0] * u[0] + d[1] * u[1]) / s, (d[0] * w[0] + d[1] * w[1]) / s
    out = []
    for t in range(first, last + 1):
        f = _trunk_frame(positions, t, group)
        if f is None:
            out.append(None)
            continue
        o, u, w, s = f
        out.append((o[0] + s * (cu * u[0] + cw * w[0]), o[1] + s * (cu * u[1] + cw * w[1])))
    return out, f"offset from frame {ref}"


# --- driver -------------------------------------------------------------------------------------------

def fill_gaps(joints: dict[str, JointSeries], params: OcclusionParams = OcclusionParams()) -> dict[str, FilledJoint]:
    """Fill the gaps of the 12 body joints present in `joints` (see the module docstring). The input is not modified."""
    if params.force_method is not None and params.force_method not in FORCE_METHODS:
        raise ValueError(f"force_method must be None or one of {sorted(FORCE_METHODS)}, got {params.force_method!r}")
    names = [n for n in PROCESSING_ORDER if n in joints]
    n_frames = {len(joints[n].positions) for n in names}
    if len(n_frames) > 1:
        raise ValueError("all joints must have the same length")
    work = {n: list(joints[n].positions) for n in names}
    flags = {n: [Flag.OBSERVED if p is not None else Flag.UNAVAILABLE for p in joints[n].positions] for n in names}
    methods = {n: [None] * len(joints[n].positions) for n in names}
    sigmas = {n: [None] * len(joints[n].positions) for n in names}
    events = {n: [] for n in names}
    bounds = segment_bounds(joints) if params.tier2_clamp else {}

    for n in names:
        s = joints[n]
        ex = _excluded_frames(s)
        total = len(s.positions)
        for first, last in find_gaps(s):
            length = last - first + 1
            tier = classify_gap(length)
            branch = FORCE_METHODS[params.force_method] if params.force_method else tier  # method branch; flags use `tier`
            reasons = tuple(sorted({s.gap_reasons[t] for t in range(first, last + 1) if s.gap_reasons[t]}))
            new, method, sig, note = None, None, None, ""
            if branch == Flag.TIER1:
                new, method = fill_tier1(s, first, last, params), "spline"
                note = "" if new else "fewer than the required observed frames on a side (clip edge, excluded frame or other gaps)"
            elif branch == Flag.TIER2:
                if n in TRUNK and params.tier2_mirror:
                    new = fill_trunk_mirror(joints, n, first, last)
                    method = "mirror" if new else None
                if new is None:
                    res = fill_tier2_kalman(s, first, last, params)
                    if res:
                        new, sig = res
                        method = "kalman_causal" if params.tier2_causal else "kalman"
                        if n in bounds and n in SEGMENT_PARENT:
                            par = SEGMENT_PARENT[n]
                            parent_pos = work.get(par)
                            new = [clamp_to_segment(p, parent_pos[first + k], bounds[n])
                                   if parent_pos is not None and parent_pos[first + k] is not None else p
                                   for k, p in enumerate(new)]
                    else:
                        note = "no bracketing observed frames or no noise estimate"
            else:
                if n in TRUNK:
                    note = "root joint: a trunk gap over 20 frames is unavailable (no root-freeze of the root)"
                else:
                    new, note = fill_tier3(s, n, work, first, last)
                    method = "trunk_offset" if new else None
            if new is not None:
                for k, p in enumerate(new):
                    t = first + k
                    if p is None:
                        continue
                    work[n][t] = p
                    flags[n][t] = tier
                    methods[n][t] = method
                    if sig is not None:
                        sigmas[n][t] = sig[k]
            events[n].append(GapEvent(
                n, first, last, length, tier, new is not None and any(p is not None for p in new), method,
                first == 0, last == total - 1, (first - 1 in ex) or (last + 1 in ex), reasons, note))
    return {n: FilledJoint(n, joints[n], work[n], flags[n], methods[n], sigmas[n], events[n]) for n in names}


def as_joint_series(filled: dict[str, FilledJoint], raw_joints: dict[str, JointSeries] | None = None,
                    include_tier3: bool = False) -> dict[str, JointSeries]:
    """JointSeries for metrics.py: filled positions for observed, Tier 1 and Tier 2 frames (and Tier 3 if asked),
    None for unavailable frames and, by default, Tier 3 frames (ARCHITECTURE §3: a stale joint is excluded from per-limb
    metrics). Timestamps, raw gap reasons (kept where no position is given) and raw confidences are carried over. Joints
    not in `filled` (the face points) are passed through from raw_joints unchanged."""
    out: dict[str, JointSeries] = dict(raw_joints or {})
    for n, f in filled.items():
        ok = USABLE_FLAGS + ((Flag.TIER3,) if include_tier3 else ())
        pos = [p if fl in ok else None for p, fl in zip(f.positions, f.flags)]
        reasons = [None if p is not None else f.raw.gap_reasons[t] for t, p in enumerate(pos)]
        out[n] = JointSeries(n, pos, list(f.raw.timestamps), reasons,
                             None if f.raw.confidences is None else list(f.raw.confidences))
    return out


def summarise(filled: dict[str, FilledJoint]) -> dict[str, int]:
    """Frame counts by flag over all filled joints."""
    out = {fl.value: 0 for fl in Flag}
    for f in filled.values():
        for fl in f.flags:
            out[fl.value] += 1
    return out
