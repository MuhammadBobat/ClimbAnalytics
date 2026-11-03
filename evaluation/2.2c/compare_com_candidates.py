"""
Step 2.2c: compare the candidate CoM estimators of DECISIONS.md D-026 on the five own clips.
Reuses metrics.py for every CoM and speed and the 2.2b helpers; nothing is re-implemented.
Raw, unsmoothed, no gap-filling (occlusion_handling.py / 3.2 does not exist yet). The same
exclusion list as 2.2b. NO candidate is chosen or ranked here.

Candidates (labels in the outputs):
  A_proxy         trunk-centroid proxy (equal weights on shoulders and hips), compute_com
  B_trunk_point   (1-w) shoulder midpoint + w hip midpoint; w fitted leave-one-clip-out to
                  minimise the mean Euclidean position difference (px) to F on frames where F is
                  defined, w in [0, 1], frames pooled over the four training clips. Each clip's B
                  uses the w of the fold in which that clip was held out.
  C_no_head       segment model without the head, mass fractions renormalised
  D_trunk_legs    trunk and legs only, renormalised
  E_c<cut>        blend B + sum m'_s (CM_s - B) over 12 limb segments gated on raw confidence >=
                  cut (0.5, 0.6, 0.7, 0.8; not chosen), B at the fitted fold w
  EdL_c<cut>      the same E with w = de Leva trunk CM fraction (0.4310 male)
  F_full          the 14-joint segment model, compute_com_segment_model
Male de Leva (1996) Table 4 throughout.

Definitions used in the outputs:
  shared frames   frames where the candidate and F are both defined
  torso length    shoulder midpoint -> hip midpoint per frame, as in 2.2b
  coverage        frames where the candidate is defined / all frames of the clip
  jitter          J(t) = || p(t+1) - 2 p(t) + p(t-1) || / L(t), torso lengths per frame^2, for
                  each t where p(t-1), p(t), p(t+1) and L(t) are defined. No division by dt^2.
                  It cannot separate estimator noise from real acceleration.
  common support  speed steps / jitter triples where ALL candidates are defined
  defined-to-None frames t where the candidate is defined at t-1 and not at t, per 1,000 frames
  defined runs    number of maximal runs of consecutive defined frames
  percentile      linear interpolation between order statistics (the numpy default)
  ALL             frame-weighted concatenation over clips, as in 2.2b

Outputs, under evaluation/2.2c/<input-folder-name>/:
    <clip>/frames.csv       per frame: torso length and x, y, speed, jitter for every candidate
    summary.csv             per clip and ALL, one row per candidate
    loco_trunk_weight.csv   the fitted w per fold, with training and held-out differences
    triage.csv              dominance triage (rule in TRIAGE_RULE and in the manifest)
    manifest.txt            run provenance, renormalisation log, regression check, triage

Usage:
    python3 evaluation/2.2c/compare_com_candidates.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "2.2b"))

import compare_com_models as cm  # noqa: E402  (2.2b helpers)
from metrics import (  # noqa: E402
    SEX_TABLES,
    JointSeries,
    apply_frame_exclusions,
    blended_limb_usage,
    compute_com,
    compute_com_blended,
    compute_com_segment_model,
    compute_com_segment_model_no_head,
    compute_com_trunk_legs,
    compute_com_trunk_point,
    compute_velocity,
    load_keypoints_csv,
    segment_model_specs,
    trunk_cm_fraction,
    _no_head_specs,
    _trunk_legs_specs,
)
from validation.analyze_occlusion_gaps import (  # noqa: E402
    MANIFEST_LIBRARIES,
    _git,
    _library_version,
    load_exclusions,
    sha256_file,
)

CUTOFFS = (0.5, 0.6, 0.7, 0.8)   # the sweep requested by the author; not chosen here
FIT_TOL = 1e-9                   # golden-section interval width at which the w search stops
RAW_NOTE = "raw, unsmoothed, no gap-filling"

L_A, L_B, L_C, L_D, L_F = "A_proxy", "B_trunk_point", "C_no_head", "D_trunk_legs", "F_full"
E_LABELS = [f"E_c{c}" for c in CUTOFFS]
EDL_LABELS = [f"EdL_c{c}" for c in CUTOFFS]
LABELS = [L_A, L_B, L_C, L_D, *E_LABELS, *EDL_LABELS, L_F]
OVERLAY_LABELS = [x for x in LABELS if x not in EDL_LABELS]   # nine markers


def e_cutoff(label: str) -> float:
    return float(label.split("_c")[1])


# --- loading ----------------------------------------------------------------------------

@dataclass
class ClipData:
    clip: str
    fps: float
    n_frames: int
    n_excluded: int
    joints: dict
    torso: list           # px, or None
    series: dict          # label -> JointSeries
    speeds: dict          # label -> list[float | None]
    jitter: dict          # label -> list[float | None]
    usage: dict           # E label -> list[tuple[str, ...]]
    w_fit: float = float("nan")


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def torso_lengths(joints) -> list:
    n = len(joints["left_shoulder"].positions)
    out = []
    for t in range(n):
        ls, rs = joints["left_shoulder"].positions[t], joints["right_shoulder"].positions[t]
        lh, rh = joints["left_hip"].positions[t], joints["right_hip"].positions[t]
        if None in (ls, rs, lh, rh):
            out.append(None)
        else:
            a, b = _mid(ls, rs), _mid(lh, rh)
            out.append(math.hypot(b[0] - a[0], b[1] - a[1]))
    return out


def load_clip_joints(csv_path: Path, excluded: set | None):
    with open(csv_path, newline="") as f:
        clip = next(csv.DictReader(f))["video"]
    joints = load_keypoints_csv(csv_path)
    n = len(next(iter(joints.values())).positions)
    exc = {f for f in (excluded or set()) if 0 <= f < n}
    joints = apply_frame_exclusions(joints, exc)
    ts = next(iter(joints.values())).timestamps
    fps = (n - 1) / (ts[-1] - ts[0]) if n > 1 and ts[-1] > ts[0] else 30.0
    return clip, joints, n, len(exc), fps


# --- B fit (leave-one-clip-out) ----------------------------------------------------------

def fit_samples(joints, f_series: JointSeries, torso: list) -> list:
    """(shoulder midpoint, hip midpoint, F, torso length) for every frame where F, the trunk
    joints and the torso length are defined."""
    out = []
    for t, f in enumerate(f_series.positions):
        L = torso[t]
        if f is None or not L:
            continue
        s = _mid(joints["left_shoulder"].positions[t], joints["right_shoulder"].positions[t])
        h = _mid(joints["left_hip"].positions[t], joints["right_hip"].positions[t])
        out.append((s, h, f, L))
    return out


def mean_distance(samples: list, w: float, torso_units: bool = False) -> float:
    """Mean over samples of || S + w (H - S) - F ||, in px or in torso lengths."""
    tot = 0.0
    for s, h, f, L in samples:
        d = math.hypot(s[0] + w * (h[0] - s[0]) - f[0], s[1] + w * (h[1] - s[1]) - f[1])
        tot += d / L if torso_units else d
    return tot / len(samples)


def fit_trunk_weight(samples: list, tol: float = FIT_TOL) -> float:
    """w in [0, 1] minimising the mean Euclidean distance (px) of the trunk point to F. The
    objective is convex in w, so a golden-section search is exact to `tol`; the end points are
    also tried so a minimum on the boundary is returned exactly."""
    if not samples:
        raise ValueError("no samples to fit")
    inv = (math.sqrt(5) - 1) / 2
    lo, hi = 0.0, 1.0
    c, d = hi - inv * (hi - lo), lo + inv * (hi - lo)
    fc, fd = mean_distance(samples, c), mean_distance(samples, d)
    while hi - lo > tol:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - inv * (hi - lo)
            fc = mean_distance(samples, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + inv * (hi - lo)
            fd = mean_distance(samples, d)
    best = (lo + hi) / 2
    return min((best, 0.0, 1.0), key=lambda w: mean_distance(samples, w))


def loco_weights(samples_by_clip: dict) -> list:
    """One fold per clip: w fitted on the other clips; training and held-out differences."""
    rows = []
    for held, held_samples in samples_by_clip.items():
        train = [s for c, ss in samples_by_clip.items() if c != held for s in ss]
        w = fit_trunk_weight(train)
        rows.append({
            "held_out_clip": held, "w": w, "n_train": len(train),
            "train_px": mean_distance(train, w), "train_torso": mean_distance(train, w, True),
            "n_heldout": len(held_samples),
            "heldout_px": mean_distance(held_samples, w) if held_samples else None,
            "heldout_torso": mean_distance(held_samples, w, True) if held_samples else None,
        })
    return rows


# --- series, speed, jitter ----------------------------------------------------------------

def build_series(joints, sex: str, w_fit: float) -> tuple[dict, dict]:
    """(label -> JointSeries, E label -> per-frame used limb segments) for all candidates."""
    w_dl = trunk_cm_fraction(sex)
    series = {
        L_A: compute_com(joints),
        L_B: compute_com_trunk_point(joints, w_fit),
        L_C: compute_com_segment_model_no_head(joints, sex),
        L_D: compute_com_trunk_legs(joints, sex),
        L_F: compute_com_segment_model(joints, sex),
    }
    usage = {}
    for c in CUTOFFS:
        series[f"E_c{c}"] = compute_com_blended(joints, sex, w_fit, c)
        usage[f"E_c{c}"] = blended_limb_usage(joints, sex, w_fit, c)
        series[f"EdL_c{c}"] = compute_com_blended(joints, sex, w_dl, c)
        usage[f"EdL_c{c}"] = blended_limb_usage(joints, sex, w_dl, c)
    return series, usage


def jitter_series(series: JointSeries, torso: list) -> list:
    """J(t) = || p(t+1) - 2 p(t) + p(t-1) || / L(t), torso lengths per frame^2 (see header)."""
    pos = series.positions
    out = [None] * len(pos)
    for t in range(1, len(pos) - 1):
        a, b, c, L = pos[t - 1], pos[t], pos[t + 1], torso[t]
        if a is None or b is None or c is None or not L:
            continue
        out[t] = math.hypot(c[0] - 2 * b[0] + a[0], c[1] - 2 * b[1] + a[1]) / L
    return out


def build_clip(csv_path: Path, excluded, sex: str, w_fit: float) -> ClipData:
    clip, joints, n, n_exc, fps = load_clip_joints(csv_path, excluded)
    torso = torso_lengths(joints)
    series, usage = build_series(joints, sex, w_fit)
    return ClipData(
        clip, fps, n, n_exc, joints, torso, series,
        {k: compute_velocity(v) for k, v in series.items()},
        {k: jitter_series(v, torso) for k, v in series.items()}, usage, w_fit)


# --- statistics ----------------------------------------------------------------------------

def percentile(vals: list, q: float) -> float:
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo = int(math.floor(k))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _f(x) -> str:
    return "" if x is None else f"{x:.4f}"


def stats4(vals: list) -> list:
    """mean, median, 95th percentile, max as strings (blank if empty)."""
    if not vals:
        return ["", "", "", ""]
    return [_f(statistics.fmean(vals)), _f(statistics.median(vals)), _f(percentile(vals, 0.95)),
            _f(max(vals))]


def stats4_jitter(vals: list) -> list:
    """mean, median, 95th percentile, rms as strings."""
    if not vals:
        return ["", "", "", ""]
    return [_f(statistics.fmean(vals)), _f(statistics.median(vals)), _f(percentile(vals, 0.95)),
            _f(math.sqrt(statistics.fmean(x * x for x in vals)))]


def continuity(positions: list) -> tuple[int, int]:
    """(defined-to-None transitions, number of defined runs)."""
    to_none = runs = 0
    for t, p in enumerate(positions):
        prev = positions[t - 1] if t else None
        if p is not None and prev is None:
            runs += 1
        if p is None and prev is not None:
            to_none += 1
    return to_none, runs


def position_diffs(c: ClipData, label: str) -> list:
    """(px, torso lengths) to F where both are defined."""
    out = []
    for p, f, L in zip(c.series[label].positions, c.series[L_F].positions, c.torso):
        if p is None or f is None or not L:
            continue
        d = math.hypot(p[0] - f[0], p[1] - f[1])
        out.append((d, d / L))
    return out


def common_steps(c: ClipData, table: dict) -> list:
    return [t for t in range(c.n_frames) if all(table[l][t] is not None for l in LABELS)]


def limb_stats(comps: list, label: str):
    """(mean limb segments used on defined frames, segment use toggles per 1,000 frames)."""
    used_counts, toggles, frames = [], 0, 0
    for c in comps:
        u, pos = c.usage[label], c.series[label].positions
        frames += c.n_frames
        for t in range(c.n_frames):
            if pos[t] is not None:
                used_counts.append(len(u[t]))
                if t and pos[t - 1] is not None:
                    toggles += len(set(u[t]) ^ set(u[t - 1]))
    if not used_counts:
        return "", ""
    return _f(statistics.fmean(used_counts)), _f(toggles / frames * 1000)


SUMMARY_COLUMNS = [
    "clip", "candidate", "n_frames", "n_excluded_by_list", "n_defined", "coverage",
    "n_f_defined_candidate_not", "n_candidate_defined_f_not", "defined_to_none_per_1000_frames",
    "n_defined_runs", "n_shared_with_f",
    "pos_px_mean", "pos_px_median", "pos_px_p95", "pos_px_max",
    "pos_torso_mean", "pos_torso_median", "pos_torso_p95", "pos_torso_max",
    "speed_own_n", "speed_own_mean", "speed_own_median", "speed_own_p95", "speed_own_max",
    "speed_common_n", "speed_common_mean", "speed_common_median", "speed_common_p95",
    "speed_common_max", "speed_vs_f_n", "speed_vs_f_pearson", "speed_vs_f_absdiff_mean",
    "jitter_own_n", "jitter_own_mean", "jitter_own_median", "jitter_own_p95", "jitter_own_rms",
    "jitter_common_n", "jitter_common_mean", "jitter_common_median", "jitter_common_p95",
    "jitter_common_rms", "limb_segments_used_mean", "limb_use_toggles_per_1000_frames",
]


def summary_row(name: str, label: str, comps: list) -> list:
    n_frames = sum(c.n_frames for c in comps)
    defined = [p is not None for c in comps for p in c.series[label].positions]
    f_def = [p is not None for c in comps for p in c.series[L_F].positions]
    to_none = runs = 0
    for c in comps:
        a, b = continuity(c.series[label].positions)
        to_none, runs = to_none + a, runs + b
    diffs = [d for c in comps for d in position_diffs(c, label)]
    own_sp = [v for c in comps for v in c.speeds[label] if v is not None]
    own_j = [v for c in comps for v in c.jitter[label] if v is not None]
    com_sp = [c.speeds[label][t] for c in comps for t in common_steps(c, c.speeds)]
    com_j = [c.jitter[label][t] for c in comps for t in common_steps(c, c.jitter)]
    pairs = [(c.speeds[label][t], c.speeds[L_F][t]) for c in comps for t in range(c.n_frames)
             if c.speeds[label][t] is not None and c.speeds[L_F][t] is not None]
    absd = [abs(a - b) for a, b in pairs]
    limb = limb_stats(comps, label) if label in E_LABELS + EDL_LABELS else ("", "")
    return [
        name, label, n_frames, sum(c.n_excluded for c in comps), sum(defined),
        _f(sum(defined) / n_frames),
        sum(1 for d, f in zip(defined, f_def) if f and not d),
        sum(1 for d, f in zip(defined, f_def) if d and not f),
        _f(to_none / n_frames * 1000), runs, len(diffs),
        *stats4([d[0] for d in diffs]), *stats4([d[1] for d in diffs]),
        len(own_sp), *stats4(own_sp), len(com_sp), *stats4(com_sp),
        len(pairs), _f(cm.pearson(pairs)), _f(statistics.fmean(absd)) if absd else "",
        len(own_j), *stats4_jitter(own_j), len(com_j), *stats4_jitter(com_j), *limb,
    ]


# --- triage (added after the first run) --------------------------------------------------------
#
# Dominance triage on summary.csv. The rule was chosen AFTER the pooled first-run table had been
# seen (not pre-registered); no threshold is involved. It drops nothing by itself that the author
# has not asked for: it only marks candidates, and A and F are never dropped.

NEVER_DROPPED = (L_A, L_F)
# (summary column, +1 = higher is better, -1 = lower is better)
TRIAGE_METRICS = (("coverage", +1), ("defined_to_none_per_1000_frames", -1),
                  ("jitter_common_mean", -1))
TRIAGE_RULE = (
    "A candidate is dominated if another candidate is at least as good on coverage (higher), "
    "defined-to-None transitions per 1,000 frames (lower) and mean jitter on the common support "
    "(lower), and strictly better on at least one of the three, pooled (ALL) and in every one of "
    "the five clips. Values are those written to summary.csv (4 decimals); a missing value never "
    "dominates and is never dominated. No threshold. A and F are never dropped (they are reported "
    "as dominated or not, but kept). The rule was chosen after the pooled first-run table had "
    "been seen; it is not a pre-registered rule."
)


def _metric(row: dict, col: str):
    v = row.get(col, "")
    return float(v) if v not in ("", None) else None


def dominates(a: dict, b: dict) -> bool:
    """True if summary row a is at least as good as b on all three triage metrics and strictly
    better on at least one (one scope: one clip or ALL)."""
    strict = False
    for col, sign in TRIAGE_METRICS:
        x, y = _metric(a, col), _metric(b, col)
        if x is None or y is None:
            return False
        d = sign * (x - y)
        if d < 0:
            return False
        strict = strict or d > 0
    return strict


def triage(rows: list, never_dropped=NEVER_DROPPED) -> list:
    """rows: summary.csv rows as dicts (keys clip, candidate, ...). Per candidate: the other
    candidates dominating it in each scope (ALL first, then the clips in file order), those
    dominating it in every scope, and whether it is dropped."""
    scopes = ["ALL"] + [c for c in dict.fromkeys(r["clip"] for r in rows) if c != "ALL"]
    by = {(r["clip"], r["candidate"]): r for r in rows}
    cands = list(dict.fromkeys(r["candidate"] for r in rows))
    out = []
    for x in cands:
        per_scope = {sc: [y for y in cands if y != x and dominates(by[(sc, y)], by[(sc, x)])]
                     for sc in scopes}
        every = [y for y in cands if all(y in per_scope[sc] for sc in scopes)]
        out.append({"candidate": x, "dominated": bool(every), "dominated_by": every,
                    "protected": x in never_dropped, "dropped": bool(every) and x not in never_dropped,
                    "per_scope": per_scope})
    return out


def survivors(tri: list) -> list:
    return [t["candidate"] for t in tri if not t["dropped"]]


def triage_csv_rows(tri: list) -> tuple[list, list]:
    scopes = list(tri[0]["per_scope"]) if tri else []
    header = ["candidate", "dominated", "dominated_by", "never_dropped", "dropped"] + \
             [f"dominated_by_in_{sc}" for sc in scopes]
    rows = []
    for t in tri:
        rows.append([t["candidate"], "yes" if t["dominated"] else "no", ";".join(t["dominated_by"]),
                     "yes" if t["protected"] else "no", "yes" if t["dropped"] else "no"] +
                    [";".join(t["per_scope"][sc]) for sc in scopes])
    return header, rows


def write_csv(path: Path, header: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def frames_rows(c: ClipData):
    header = ["frame_idx", "timestamp_s", "torso_length_px"]
    for l in LABELS:
        header += [f"{l}_x", f"{l}_y", f"{l}_speed_px_s", f"{l}_jitter_torso"]
    rows = []
    ts = c.series[L_A].timestamps
    for t in range(c.n_frames):
        row = [t, f"{ts[t]:.4f}", _f(c.torso[t])]
        for l in LABELS:
            p = c.series[l].positions[t]
            row += [_f(p[0]) if p else "", _f(p[1]) if p else "", _f(c.speeds[l][t]),
                    _f(c.jitter[l][t])]
        rows.append(row)
    return header, rows


# --- regression check against 2.2b ----------------------------------------------------------

def regression_check(comps: list, ref_dir: Path, expect: dict | None = None) -> tuple[bool, list]:
    """A and F must equal the 2.2b frames.csv columns proxy_x/y and seg_male_x/y exactly (as
    written, 4 decimals), and coverage must equal `expect` {A_proxy: n, F_full: n, frames: n}
    if given. Returns (ok, message lines)."""
    msgs, ok = [], True
    for c in comps:
        path = ref_dir / Path(c.clip).stem / "frames.csv"
        if not path.exists():
            msgs.append(f"{c.clip}: reference {path.as_posix()} not found")
            ok = False
            continue
        with open(path, newline="") as f:
            ref = list(csv.DictReader(f))
        bad = 0
        if len(ref) != c.n_frames:
            msgs.append(f"{c.clip}: {len(ref)} reference frames vs {c.n_frames}")
            ok = False
            continue
        for t, r in enumerate(ref):
            for label, (kx, ky) in ((L_A, ("proxy_x", "proxy_y")), (L_F, ("seg_male_x", "seg_male_y"))):
                p = c.series[label].positions[t]
                got = (_f(p[0]), _f(p[1])) if p else ("", "")
                if got != (r[kx], r[ky]):
                    bad += 1
        msgs.append(f"{c.clip}: {c.n_frames} frames, A and F columns differ on {bad} (frame, candidate) cells")
        ok = ok and bad == 0
    if expect:
        got = {"frames": sum(c.n_frames for c in comps),
               L_A: sum(p is not None for c in comps for p in c.series[L_A].positions),
               L_F: sum(p is not None for c in comps for p in c.series[L_F].positions)}
        for k, v in expect.items():
            same = got[k] == v
            msgs.append(f"coverage {k}: {got[k]} (expected {v}) {'ok' if same else 'MISMATCH'}")
            ok = ok and same
    return ok, msgs


# --- manifest ---------------------------------------------------------------------------------

def renormalisation_log(sex: str) -> list:
    L = []
    for name, (specs, r) in (("C_no_head / E (limbs)", _no_head_specs(sex)),
                             ("D_trunk_legs", _trunk_legs_specs(sex))):
        L.append(f"{name}: removed {', '.join(r.removed)}; removed mass {r.removed_mass:.4f}; "
                 f"kept mass before {r.kept_mass_before:.4f}; scale 1/{r.kept_mass_before:.4f} = "
                 f"{r.scale:.6f}; sum after {sum(s.mass_fraction for s in specs):.6f}")
    return L


def build_manifest(comps, folds, input_dir, footage_dir, exclusions_path, paper, sex, tol,
                   check, today=None, tri=None) -> str:
    ok, msgs = check if check else (None, ["not run"])
    L = [
        "CoM candidate comparison manifest (evaluation/2.2c/compare_com_candidates.py)",
        f"date: {today or date.today().isoformat()}",
        f"input folder: {input_dir.as_posix()}",
        f"data: {RAW_NOTE}",
        "No candidate is chosen or ranked by this run (DECISIONS.md D-026).",
        "", "== Candidates",
        "A_proxy: equal weights on shoulders and hips (compute_com)",
        "B_trunk_point: (1-w) shoulder midpoint + w hip midpoint; w per fold below",
        "C_no_head: segment model without head, renormalised",
        "D_trunk_legs: trunk and legs, renormalised",
        f"E_c<cut>: blended, B at the fold w; cut-offs {', '.join(str(c) for c in CUTOFFS)} (swept, not chosen)",
        f"EdL_c<cut>: blended, w = de Leva trunk CM fraction = {trunk_cm_fraction(sex):.4f} ({sex})",
        "F_full: 14-joint segment model (compute_com_segment_model)",
        f"sex table: {sex}; de Leva (1996), J Biomech 29(9):1223-1230, Table 4; local paper copy "
        f"sha256 {sha256_file(paper) if paper.exists() else 'not found'}",
        "", "== B fit",
        "objective: mean Euclidean distance (px) between the trunk point and F, over frames where F "
        "is defined; w in [0, 1]; frames pooled over the four training clips (frame-weighted); "
        f"golden-section search, interval stop {tol:g}; end points also tried. Torso-length "
        "differences are reported alongside in loco_trunk_weight.csv.",
    ]
    for r in folds:
        L.append(f"fold held out {r['held_out_clip']}: w = {r['w']:.6f} "
                 f"(train n {r['n_train']}, held-out n {r['n_heldout']})")
    L += ["", "== Renormalisation log (mass fractions divided by the kept sum)"] + renormalisation_log(sex)
    L += ["", "== Definitions",
          "shared frames: candidate and F both defined. torso length: shoulder midpoint -> hip "
          "midpoint per frame.",
          "coverage: defined frames / all frames. defined-to-None: frames defined at t-1 and not "
          "at t, per 1,000 frames. defined runs: maximal runs of consecutive defined frames.",
          "jitter J(t) = ||p(t+1) - 2 p(t) + p(t-1)|| / L(t), torso lengths per frame^2, for each t "
          "where p(t-1), p(t), p(t+1) and L(t) are defined; no division by dt^2; cannot separate "
          "estimator noise from real acceleration. common support: all candidates defined.",
          "percentile: linear interpolation between order statistics.",
          "E limb gate: all of a segment's joints have a position and raw confidence >= cut-off "
          "(hand, foot: their single joint); trunk joints keep the D-007 0.5 rule.",
          "", "== Regression check against evaluation/2.2b/output_full (A vs proxy, F vs seg_male)",
          f"result: {'PASS' if ok else ('not run' if ok is None else 'FAIL')}"] + msgs
    if tri is not None:
        keep = survivors(tri)
        L += ["", "== Triage (triage.csv; added after the first run)", TRIAGE_RULE,
              "survivors (default overlay panels are these plus A and F, which are always kept): "
              + ", ".join(keep),
              "dropped: " + (", ".join(f"{t['candidate']} (dominated by {', '.join(t['dominated_by'])})"
                                       for t in tri if t["dropped"]) or "none"),
              "A and F dominated but kept: " + (", ".join(
                  f"{t['candidate']} (by {', '.join(t['dominated_by'])})" for t in tri
                  if t["protected"] and t["dominated"]) or "none"),
              "Triage only marks candidates; nothing is chosen or ranked."]
    L += ["", "== Frame exclusion file"]
    if exclusions_path.exists():
        L += [f"path: {exclusions_path.as_posix()}", "contents (verbatim):",
              exclusions_path.read_text().rstrip("\n")]
    else:
        L.append("none")
    L += ["", "== Code and environment (at run time)", f"git HEAD: {_git('rev-parse', 'HEAD')}"]
    dirty = _git("status", "--porcelain", "--", "metrics.py", "evaluation/2.2c/compare_com_candidates.py",
                 "evaluation/2.2c/make_candidate_overlays.py",
                 "validation/frame_exclusions.csv")
    L.append("metrics/scripts/exclusions modified since HEAD: "
             + ("unknown" if dirty == "unknown" else ("yes" if dirty else "no")))
    for lib in MANIFEST_LIBRARIES:
        names = ("opencv-python", "opencv-python-headless") if lib == "opencv-python" else (lib,)
        L.append(f"{lib}: {_library_version(*names)}")
    L += ["", "== Clips"]
    for c in comps:
        vp = footage_dir / c.clip
        L.append(f"{c.clip}: frames {c.n_frames}, excluded {c.n_excluded}, fps {c.fps:.3f} (from "
                 f"timestamps), sha256 {sha256_file(vp) if vp.exists() else 'video not found'}")
    return "\n".join(L) + "\n"


# --- main -------------------------------------------------------------------------------------

def run(input_dir: Path, exclusions_path: Path, sex: str):
    """(comps, folds) for every *_keypoints.csv in input_dir."""
    exclusions = load_exclusions(exclusions_path)
    loaded = []
    for p in sorted(input_dir.glob("*_keypoints.csv")):
        clip, joints, n, n_exc, fps = load_clip_joints(p, None)
        loaded.append((p, clip))
    samples = {}
    for p, clip in loaded:
        _, joints, *_ = load_clip_joints(p, exclusions.get(clip))
        samples[clip] = fit_samples(joints, compute_com_segment_model(joints, sex),
                                    torso_lengths(joints))
    folds = loco_weights(samples)
    w_of = {r["held_out_clip"]: r["w"] for r in folds}
    comps = [build_clip(p, exclusions.get(clip), sex, w_of[clip]) for p, clip in loaded]
    return comps, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/output_full_lock1"))  # D-027: gated-lock run; earlier input was validation/output_full
    parser.add_argument("--footage-dir", type=Path, default=Path("footage/test"))
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation/2.2c"))
    parser.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"))
    parser.add_argument("--sex", choices=SEX_TABLES, default="male")
    parser.add_argument("--paper", type=Path, default=cm.DEFAULT_PAPER,
                        help="local copy of de Leva (1996), only its SHA-256 is recorded")
    parser.add_argument("--regression-ref", type=Path, default=Path("evaluation/2.2b/output_full_lock1"),
                        help="2.2b output folder whose A and F columns must be reproduced exactly")
    parser.add_argument("--expect-coverage", default="",
                        help="frames,A defined,F defined for the regression check ('' to skip; the earlier "
                             "validation/output_full run gave 5886,5828,5349)")
    args = parser.parse_args()

    if not list(args.input_dir.glob("*_keypoints.csv")):
        print(f"No *_keypoints.csv files in {args.input_dir}.")
        return
    comps, folds = run(args.input_dir, args.exclusions, args.sex)
    out_root = args.out_dir / args.input_dir.resolve().name
    out_root.mkdir(parents=True, exist_ok=True)

    for c in comps:
        folder = out_root / Path(c.clip).stem
        folder.mkdir(parents=True, exist_ok=True)
        header, rows = frames_rows(c)
        write_csv(folder / "frames.csv", header, rows)

    rows = []
    for l in LABELS:
        for c in comps:
            rows.append(summary_row(c.clip, l, [c]))
        rows.append(summary_row("ALL", l, comps))
    write_csv(out_root / "summary.csv", SUMMARY_COLUMNS, rows)
    tri = triage([dict(zip(SUMMARY_COLUMNS, r)) for r in rows])
    write_csv(out_root / "triage.csv", *triage_csv_rows(tri))
    write_csv(out_root / "loco_trunk_weight.csv",
              ["held_out_clip", "w", "n_train_frames", "train_mean_px", "train_mean_torso",
               "n_heldout_frames", "heldout_mean_px", "heldout_mean_torso"],
              [[r["held_out_clip"], f"{r['w']:.6f}", r["n_train"], _f(r["train_px"]),
                _f(r["train_torso"]), r["n_heldout"], _f(r["heldout_px"]), _f(r["heldout_torso"])]
               for r in folds])

    expect = None
    if args.expect_coverage:
        nf, na, nfull = (int(x) for x in args.expect_coverage.split(","))
        expect = {"frames": nf, L_A: na, L_F: nfull}
    check = regression_check(comps, args.regression_ref, expect)
    (out_root / "manifest.txt").write_text(build_manifest(
        comps, folds, args.input_dir, args.footage_dir, args.exclusions, args.paper, args.sex,
        FIT_TOL, check, tri=tri))
    print("regression check:", "PASS" if check[0] else "FAIL")
    for m in check[1]:
        print("  " + m)
    for r in folds:
        print(f"fold {r['held_out_clip']}: w = {r['w']:.6f}")
    print("survivors:", ", ".join(survivors(tri)))
    print(f"Written to {out_root}")
    if not check[0]:
        sys.exit(1)


if __name__ == "__main__":
    main()
