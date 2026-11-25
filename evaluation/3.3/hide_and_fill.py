"""
Step 3.3 (author's plan label): hide-and-fill evaluation of the occlusion tiers (DECISIONS.md D-007, D-028 and its
force_method addendum, D-029). Real stretches of the real clips where a joint is confidently observed are hidden
artificially, filled by calling the real occlusion_handling.fill_gaps (never a re-implementation), and the fill is
compared with the hidden value. NO tier is validated by this: "truth" is the pose model's own confident output
(confidence >= 0.5), not ground truth; confident stretches are easier than real occlusions; the clips are five.

Design (all fixed in this file before the full run; the author approved the plan on 2026-11-03):
  lengths      1 2 3 4 5 6 7 8 10 12 15 18 19 20 21 22 25 30 40 60 (40 and 60 are EXTRAPOLATION beyond the longest real gap, 30,
               kept only to find where each method stops beating the hold baseline)
  trials       per clip and length, a start frame every 25 frames beginning at frame 10; all 12 body joints are hidden one at a
               time; a trial exists only if all 12 body joints are raw-observed on every frame of the hidden span and on the 10
               frames on each side
  batching     several hidden gaps share one fill_gaps call (different joints, at least 60 frames between hidden spans)
  methods      hold (last observed position, pixels); spline; Kalman smoothed (clamp on or off); Kalman causal (clamp on or off);
               trunk mirror on or off (trunk joints); Tier 3 stale offset (limb joints). Every method at every length.
               The segment-length clamp acts only on limb joints and the mirror only on trunk joints, so clamp on/off is
               reported for limb joints and mirror on/off for trunk joints (verified in tests/test_hide_and_fill.py).
  error        Euclidean px between the filled and the hidden position; per trial the mean over the hidden frames, and the mean
               over the central frame(s) ("mid"); also divided by the clip's MEDIAN torso length (shoulder midpoint to hip midpoint,
               median over frames where the four trunk joints are observed)
  smooth/sudden per trial the peak second difference of the TRUE position over the hidden span plus 2 frames each side, divided by the
               clip's median torso length; terciles of that value WITHIN each (gap length, joint), pooled over the clips: lowest
               third = smooth, top third = sudden, middle third unlabelled. Computed from true motion only; the cut points are
               written (cutpoints.csv, manifest) before any fill is run. Fixed before any error was seen; the development smoke
               runs looked at errors for plumbing only and the rule was not changed afterwards.
  crossover    smallest tested length at which the median paired difference (method error minus hold error) is >= 0; clip
               bootstrap (resampling the five clips) is DESCRIPTIVE ONLY, not a significance test

Usage:
    python3 evaluation/3.3/hide_and_fill.py [--input-dir validation/output_full_lock1] [--out-dir evaluation/3.3]
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import metrics  # noqa: E402
import occlusion_handling as oh  # noqa: E402
from metrics import GAP_LOW_CONFIDENCE, JointSeries  # noqa: E402

LENGTHS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 19, 20, 21, 22, 25, 30, 40, 60)
EXTRAPOLATION_LENGTHS = (40, 60)          # longer than any real gap in the five clips (longest is 30)
CONTEXT = 10                              # observed frames required on each side of a hidden span
START_FIRST = CONTEXT
START_STRIDE = 25
MIN_SEPARATION = 60                       # frames of observed data between two hidden spans in one call
MOTION_MARGIN = 2                         # frames either side of the span for the smooth/sudden measure
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
CLASSES = ("all", "smooth", "sudden")
GROUPS = ("limb", "trunk")
JOINT_ORDER = list(oh.PROCESSING_ORDER)


def group_of(joint: str) -> str:
    return "trunk" if joint in oh.TRUNK else "limb"


# name -> (OcclusionParams kwargs, joint groups it is reported for); "hold" is computed here, not by fill_gaps
METHODS: dict[str, dict] = {
    "hold": {"kwargs": None, "groups": ("limb", "trunk")},
    "spline": {"kwargs": dict(force_method="spline"), "groups": ("limb", "trunk")},
    "kalman_smooth": {"kwargs": dict(force_method="kalman", tier2_clamp=True, tier2_mirror=False, tier2_causal=False),
                      "groups": ("limb", "trunk")},                   # limb: clamp on; trunk: mirror off
    "kalman_smooth_noclamp": {"kwargs": dict(force_method="kalman", tier2_clamp=False, tier2_mirror=False, tier2_causal=False),
                              "groups": ("limb",)},                   # limb: clamp off
    "kalman_smooth_mirror": {"kwargs": dict(force_method="kalman", tier2_clamp=True, tier2_mirror=True, tier2_causal=False),
                             "groups": ("trunk",)},                   # trunk: mirror on (needs the other three observed)
    "kalman_causal": {"kwargs": dict(force_method="kalman", tier2_clamp=True, tier2_mirror=False, tier2_causal=True),
                      "groups": ("limb", "trunk")},
    "kalman_causal_noclamp": {"kwargs": dict(force_method="kalman", tier2_clamp=False, tier2_mirror=False, tier2_causal=True),
                              "groups": ("limb",)},
    "trunk_offset": {"kwargs": dict(force_method="trunk_offset"), "groups": ("limb",)},
}
METHOD_ORDER = list(METHODS)


@dataclass(frozen=True)
class Trial:
    clip: str
    joint: str
    length: int
    start: int          # first hidden frame
    torso_px: float     # the clip's median torso length
    motion: float       # peak second difference of the true position / torso_px
    motion_class: str = ""

    @property
    def span(self) -> range:
        return range(self.start, self.start + self.length)


# --- trials ---------------------------------------------------------------------------------------------

def all_observed_mask(joints: dict[str, JointSeries]) -> np.ndarray:
    n = len(next(iter(joints.values())).positions)
    ok = np.ones(n, dtype=bool)
    for name in oh.BODY_JOINTS:
        ok &= np.array([p is not None for p in joints[name].positions])
    return ok


def median_torso_px(joints: dict[str, JointSeries]) -> float:
    """Median over the frames where the four trunk joints are observed of |shoulder midpoint - hip midpoint| (px)."""
    ls, rs, lh, rh = (joints[n].positions for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    d = []
    for a, b, c, e in zip(ls, rs, lh, rh):
        if a is None or b is None or c is None or e is None:
            continue
        d.append(math.hypot((a[0] + b[0]) / 2 - (c[0] + e[0]) / 2, (a[1] + b[1]) / 2 - (c[1] + e[1]) / 2))
    if not d:
        raise ValueError("no frame with all four trunk joints observed")
    return float(np.median(d))


def peak_second_difference(positions, first: int, last: int) -> float:
    """Peak |p[t+1] - 2 p[t] + p[t-1]| (px) over t in first..last (all three frames must be observed)."""
    best = 0.0
    for t in range(first, last + 1):
        a, b, c = positions[t - 1], positions[t], positions[t + 1]
        best = max(best, math.hypot(c[0] - 2 * b[0] + a[0], c[1] - 2 * b[1] + a[1]))
    return best


def make_trials(clip: str, joints: dict[str, JointSeries], lengths=LENGTHS) -> list[Trial]:
    n = len(next(iter(joints.values())).positions)
    ok = all_observed_mask(joints)
    bad = np.concatenate([[0], np.cumsum(~ok)])        # bad[i] = unobserved frames among 0..i-1
    torso = median_torso_px(joints)
    out = []
    for length in lengths:
        s = START_FIRST
        while s + length - 1 + CONTEXT <= n - 1:
            lo, hi = s - CONTEXT, s + length - 1 + CONTEXT
            if bad[hi + 1] - bad[lo] == 0:
                for joint in JOINT_ORDER:
                    peak = peak_second_difference(joints[joint].positions, s - MOTION_MARGIN + 1, s + length - 1 + MOTION_MARGIN - 1)
                    out.append(Trial(clip, joint, length, s, torso, peak / torso))
            s += START_STRIDE
    return out


def tercile_cuts(trials: list[Trial]) -> dict[tuple[int, str], tuple[float, float]]:
    """Per (gap length, joint), pooled over clips: the 1/3 and 2/3 percentiles of the motion measure (true motion only)."""
    groups: dict[tuple[int, str], list[float]] = {}
    for t in trials:
        groups.setdefault((t.length, t.joint), []).append(t.motion)
    return {k: (float(np.percentile(v, 100 / 3)), float(np.percentile(v, 200 / 3))) for k, v in groups.items()}


def classify(trials: list[Trial], cuts) -> list[Trial]:
    out = []
    for t in trials:
        lo, hi = cuts[(t.length, t.joint)]
        cls = "smooth" if t.motion <= lo else ("sudden" if t.motion >= hi else "middle")
        out.append(replace(t, motion_class=cls))
    return out


def schedule_batches(trials: list[Trial]) -> list[list[Trial]]:
    """First-fit: a batch holds at most one hidden gap per joint, the spans at least MIN_SEPARATION frames apart."""
    order = {j: i for i, j in enumerate(JOINT_ORDER)}
    batches: list[list[Trial]] = []
    for t in sorted(trials, key=lambda x: (x.start, order[x.joint])):
        for b in batches:
            if all(o.joint != t.joint and (t.start - (o.start + o.length - 1) - 1 >= MIN_SEPARATION
                                           or o.start - (t.start + t.length - 1) - 1 >= MIN_SEPARATION) for o in b):
                b.append(t)
                break
        else:
            batches.append([t])
    return batches


# --- hiding, filling, scoring -----------------------------------------------------------------------------

def mask_joints(joints: dict[str, JointSeries], batch: list[Trial]) -> dict[str, JointSeries]:
    """Copy of the joints with every trial's hidden span removed (position None, a low-confidence gap, confidence None).
    The input is not modified; nothing about the hidden values survives in the copy."""
    out = {}
    for name, s in joints.items():
        out[name] = JointSeries(name, list(s.positions), s.timestamps, list(s.gap_reasons),
                                None if s.confidences is None else list(s.confidences))
    for t in batch:
        s = out[t.joint]
        for f in t.span:
            s.positions[f] = None
            s.gap_reasons[f] = GAP_LOW_CONFIDENCE
            if s.confidences is not None:
                s.confidences[f] = None
    return out


def fill_batch(joints: dict[str, JointSeries], batch: list[Trial], methods=None, mask=mask_joints):
    """{(trial index in batch, method): list of (x, y) or None per hidden frame} from the real fill_gaps on the masked
    joints (hold: the last observed position before the span). `joints` holds the TRUE values only through `mask`."""
    methods = methods or METHOD_ORDER
    masked = mask(joints, batch)
    out = {}
    for m in methods:
        spec = METHODS[m]
        if spec["kwargs"] is None:
            for i, t in enumerate(batch):
                p = masked[t.joint].positions[t.start - 1]
                out[(i, m)] = [p] * t.length if p is not None else None
            continue
        filled = oh.fill_gaps(masked, oh.OcclusionParams(**spec["kwargs"]))
        for i, t in enumerate(batch):
            ps = [filled[t.joint].positions[f] for f in t.span]
            out[(i, m)] = None if any(p is None for p in ps) else ps
    return out


def score(trial: Trial, joints_true: dict[str, JointSeries], fill):
    """(mean error px, central-frame error px) or (None, None) when the method left a frame unavailable."""
    if fill is None:
        return None, None
    errs = [math.hypot(p[0] - joints_true[trial.joint].positions[f][0], p[1] - joints_true[trial.joint].positions[f][1])
            for p, f in zip(fill, trial.span)]
    mid = [errs[(trial.length - 1) // 2], errs[trial.length // 2]]
    return float(np.mean(errs)), float(np.mean(mid))


_CLIP_CACHE: dict = {}


def load_clip(path: str, exclusions: tuple[int, ...] = ()):
    key = (path, exclusions)
    if key not in _CLIP_CACHE:
        joints = metrics.load_keypoints_csv(Path(path))
        n = len(next(iter(joints.values())).positions)
        _CLIP_CACHE[key] = metrics.apply_frame_exclusions(joints, {t for t in exclusions if 0 <= t < n})
    return _CLIP_CACHE[key]


def run_task(task):
    """(path, exclusions, [(joint, start, length), ...]) -> [(joint, start, length, {method: (mean_px, mid_px)})]. One
    clip, one gap length (any lengths work; batching is per call)."""
    path, exclusions, items = task
    joints = load_clip(path, tuple(exclusions))
    torso = median_torso_px(joints)
    trials = [Trial("", j, L, s, torso, 0.0) for j, s, L in items]
    rows = []
    for batch in schedule_batches(trials):
        fills = fill_batch(joints, batch)
        for i, t in enumerate(batch):
            rows.append((t.joint, t.start, t.length, {m: score(t, joints, fills[(i, m)]) for m in METHOD_ORDER}))
    return rows


# --- analysis -------------------------------------------------------------------------------------------

def crossover(lengths, diffs_by_length) -> int | None:
    """Smallest length in `lengths` whose median paired difference (method - hold) is >= 0; None if there is none."""
    for L in lengths:
        d = diffs_by_length.get(L)
        if d is not None and len(d) and np.median(d) >= 0:
            return L
    return None


def recovers_after(lengths, diffs_by_length, cross) -> bool:
    return cross is not None and any(len(diffs_by_length.get(L, [])) and np.median(diffs_by_length[L]) < 0
                                     for L in lengths if L > cross)


def pstats(a: np.ndarray) -> dict:
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"n": 0, "median": "", "q25": "", "q75": "", "p5": "", "p95": ""}
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return {"n": int(len(a)), "median": q[2], "q25": q[1], "q75": q[3], "p5": q[0], "p95": q[4]}


def read_trials(path: Path):
    import pandas as pd
    return pd.read_csv(path)


def analyse(df, out: Path, lengths=LENGTHS):
    """Write summary_by_length.csv, per_clip.csv, paired_vs_hold.csv, crossover.csv, figures and report.md from the
    trial table. Returns the crossover rows."""
    import pandas as pd
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    clips = sorted(df["clip"].unique())
    summ, perclip, paired, cross_rows = [], [], [], []
    boot = {}
    for gname in GROUPS:
        dg = df[df["group"] == gname]
        for m in METHOD_ORDER:
            if gname not in METHODS[m]["groups"]:
                continue
            for cls in CLASSES:
                dc = dg if cls == "all" else dg[dg["motion_class"] == cls]
                diffs_by_clip: dict[int, dict[str, np.ndarray]] = {}
                diffs_by_len: dict[int, np.ndarray] = {}
                for L in lengths:
                    dl = dc[dc["length"] == L]
                    err = dl[f"{m}_mean_px"].to_numpy(float)
                    mid = dl[f"{m}_mid_px"].to_numpy(float)
                    nrm = err / dl["torso_px"].to_numpy(float)
                    ps, pn, pm = pstats(err), pstats(nrm), pstats(mid)
                    summ.append({"set": gname, "method": m, "class": cls, "length": L, "n_trials": len(dl),
                                 "n_available": ps["n"], **{f"px_{k}": v for k, v in ps.items() if k != "n"},
                                 **{f"torso_{k}": v for k, v in pn.items() if k != "n"},
                                 "mid_px_median": pm["median"]})
                    if cls == "all":
                        for c in clips:
                            dcl = dl[dl["clip"] == c]
                            pc = pstats(dcl[f"{m}_mean_px"].to_numpy(float))
                            perclip.append({"set": gname, "method": m, "clip": c, "length": L, "n_trials": len(dcl),
                                            "n_available": pc["n"], "px_median": pc["median"], "px_q25": pc["q25"],
                                            "px_q75": pc["q75"]})
                    if m != "hold":
                        hold = dl["hold_mean_px"].to_numpy(float)
                        ok = ~np.isnan(err) & ~np.isnan(hold)
                        d = (err - hold)[ok]
                        diffs_by_len[L] = d
                        diffs_by_clip[L] = {c: (err - hold)[ok & (dl["clip"].to_numpy() == c)] for c in clips}
                        paired.append({"set": gname, "method": m, "class": cls, "length": L, "n_paired": int(ok.sum()),
                                       "median_diff_px": float(np.median(d)) if len(d) else "",
                                       "frac_method_better": float(np.mean(d < 0)) if len(d) else "",
                                       "median_diff_torso": float(np.median((d / dl["torso_px"].to_numpy(float)[ok]))) if len(d) else ""})
                if m == "hold":
                    continue
                cr = crossover(lengths, diffs_by_len)
                rec = recovers_after(lengths, diffs_by_len, cr)
                boots = []
                for _ in range(BOOTSTRAP_RESAMPLES):
                    pick = rng.integers(0, len(clips), len(clips))
                    bd = {L: np.concatenate([diffs_by_clip[L][clips[i]] for i in pick]) for L in lengths}
                    c = crossover(lengths, bd)
                    boots.append(math.inf if c is None else c)
                finite = [b for b in boots if b != math.inf]
                srt = sorted(boots)

                def pct(q, srt=srt):
                    v = srt[int(round(q / 100 * (len(srt) - 1)))]
                    return "none" if v == math.inf else v
                cross_rows.append({
                    "set": gname, "method": m, "class": cls,
                    "crossover_length": "none through %d" % max(lengths) if cr is None else cr,
                    "median_diff_positive_again_later_negative": rec,
                    "boot_p5": pct(5), "boot_p50": pct(50), "boot_p95": pct(95),
                    "boot_share_no_crossover": 1 - len(finite) / len(boots)})
    write_csv(out / "summary_by_length.csv", summ)
    write_csv(out / "per_clip.csv", perclip)
    write_csv(out / "paired_vs_hold.csv", paired)
    write_csv(out / "crossover.csv", cross_rows)
    return summ, perclip, paired, cross_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def fmt_cell(r) -> str:
    if r["n_available"] == 0 or r["px_median"] == "":
        return "-"
    return f"{r['px_median']:.1f} [{r['px_q25']:.1f}, {r['px_q75']:.1f}]"


def md_table(summ, gname, cls, methods, lengths):
    idx = {(r["set"], r["method"], r["class"], r["length"]): r for r in summ}
    head = "| length | n trials | " + " | ".join(methods) + " |\n|---|---|" + "---|" * len(methods)
    rows = []
    for L in lengths:
        n = idx[(gname, methods[0], cls, L)]["n_trials"]
        mark = "*" if L in EXTRAPOLATION_LENGTHS else ""
        rows.append(f"| {L}{mark} | {n} | " + " | ".join(fmt_cell(idx[(gname, m, cls, L)]) for m in methods) + " |")
    return head + "\n" + "\n".join(rows)


def write_report(out: Path, summ, cross_rows, lengths, manifest_name="manifest.txt") -> None:
    limb = [m for m in METHOD_ORDER if "limb" in METHODS[m]["groups"]]
    trunk = [m for m in METHOD_ORDER if "trunk" in METHODS[m]["groups"]]
    L = ["# 3.3 hide-and-fill: error against gap length\n",
         "NO tier is validated by this. Truth = the pose model's own confident output (>= 0.5), not ground truth; five clips; "
         "trials overlap (a start every 25 frames) so they are not independent. Errors are px, median [IQR] over trials of the "
         "per-trial mean error over the hidden frames. `*` = 40 and 60 are EXTRAPOLATION (longer than any real gap, longest 30). "
         f"Settings, cut points, clips and provenance: `{manifest_name}`. Torso length for the torso-normalised columns in "
         "`summary_by_length.csv` is the clip's MEDIAN torso length.\n"]
    for cls in CLASSES:
        L.append(f"## Limb joints (8), class: {cls}\n\nkalman_smooth = clamp on, kalman_smooth_noclamp = clamp off.\n")
        L.append(md_table(summ, "limb", cls, limb, lengths) + "\n")
    for cls in CLASSES:
        L.append(f"## Trunk joints (4), class: {cls}\n\nkalman_smooth = mirror off, kalman_smooth_mirror = mirror on. "
                 "**Artificial for production:** the five clips contain no trunk gap, so the mirror result is a hide-and-fill "
                 "result only; the mirror needs the other three trunk joints observed.\n")
        L.append(md_table(summ, "trunk", cls, trunk, lengths) + "\n")
    L.append("## Crossover with the hold-last-position baseline\n\nSmallest tested length at which the median paired "
             "difference (method minus hold) is >= 0. The bootstrap resamples the five clips (1000 times, seed 0) and is "
             "DESCRIPTIVE ONLY: with five clips it is not a significance test.\n")
    L.append("| set | method | class | crossover length | better again later | bootstrap p5 / p50 / p95 | share of resamples with none |\n|---|---|---|---|---|---|---|")
    for r in cross_rows:
        L.append(f"| {r['set']} | {r['method']} | {r['class']} | {r['crossover_length']} | {r['median_diff_positive_again_later_negative']} | "
                 f"{fmtb(r['boot_p5'])} / {fmtb(r['boot_p50'])} / {fmtb(r['boot_p95'])} | {r['boot_share_no_crossover']:.2f} |")
    (out / "report.md").write_text("\n".join(L) + "\n")


def fmtb(v):
    return v if isinstance(v, str) else f"{v:g}"


def make_figures(out: Path, summ, lengths) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = {(r["set"], r["method"], r["class"], r["length"]): r for r in summ}
    for gname in GROUPS:
        methods = [m for m in METHOD_ORDER if gname in METHODS[m]["groups"]]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
        for ax, cls in zip(axes, CLASSES):
            for i, m in enumerate(methods):
                xs = [L for L in lengths if idx[(gname, m, cls, L)]["px_median"] != ""]
                med = [idx[(gname, m, cls, L)]["px_median"] for L in xs]
                lo = [idx[(gname, m, cls, L)]["px_q25"] for L in xs]
                hi = [idx[(gname, m, cls, L)]["px_q75"] for L in xs]
                ax.plot(xs, med, marker="o", ms=3, lw=1.4, ls="--" if m == "hold" else "-", color=f"C{i}", label=m)
                ax.fill_between(xs, lo, hi, color=f"C{i}", alpha=0.12, lw=0)
            for b in (4.5, 20.5):
                ax.axvline(b, color="0.5", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_title(f"{gname} joints, {cls}")
            ax.set_xlabel("hidden gap length (frames); 40 and 60 are extrapolation")
        axes[0].set_ylabel("median error (px), band = IQR")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"error_vs_length_{gname}.png", dpi=130)
        plt.close(fig)


# --- driver ---------------------------------------------------------------------------------------------

def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    from validation.analyze_occlusion_gaps import MANIFEST_LIBRARIES, _library_version, load_exclusions, sha256_file
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=Path("validation/output_full_lock1"))
    ap.add_argument("--out-dir", type=Path, default=Path("evaluation/3.3"))
    ap.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lengths", type=int, nargs="*", default=None, help="development smoke runs only")
    ap.add_argument("--clips", nargs="*", default=None, help="development smoke runs only (clip file stems, e.g. IMG_2412)")
    args = ap.parse_args()
    lengths = tuple(args.lengths) if args.lengths else LENGTHS
    out = args.out_dir / args.input_dir.resolve().name
    if out.exists() and any(out.iterdir()):
        print(f"Refusing to write into the non-empty folder {out}", file=sys.stderr)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)
    exclusions = load_exclusions(args.exclusions)
    t0 = datetime.now()

    # stage 1: trials and cut points, from the TRUE positions only; no fill has been run
    paths = sorted(args.input_dir.glob("*_keypoints.csv"))
    if args.clips:
        paths = [p for p in paths if p.name.replace("_keypoints.csv", "") in args.clips]
    clip_names, trials = {}, []
    for p in paths:
        with open(p, newline="") as f:
            clip = next(csv.DictReader(f))["video"]
        clip_names[p] = clip
        ex = tuple(sorted(exclusions.get(clip, set())))
        trials += make_trials(clip, load_clip(str(p), ex), lengths)
    cuts = tercile_cuts(trials)
    trials = classify(trials, cuts)
    with open(out / "cutpoints.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["length", "joint", "smooth_if_motion_le", "sudden_if_motion_ge", "n_trials"])
        for (L, j), (a, b) in sorted(cuts.items(), key=lambda kv: (kv[0][0], JOINT_ORDER.index(kv[0][1]))):
            w.writerow([L, j, f"{a:.5f}", f"{b:.5f}", sum(1 for t in trials if t.length == L and t.joint == j)])
    stage1_done = datetime.now()
    print(f"stage 1: {len(trials)} trials, cut points written ({out / 'cutpoints.csv'})", flush=True)

    # stage 2: fills
    tasks, meta = [], []
    for p in paths:
        clip = clip_names[p]
        ex = tuple(sorted(exclusions.get(clip, set())))
        for L in lengths:
            items = [(t.joint, t.start, t.length) for t in trials if t.clip == clip and t.length == L]
            if items:
                tasks.append((str(p), ex, items))
                meta.append((clip, L))
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run_task, tasks))
    else:
        results = [run_task(t) for t in tasks]
    cls_of = {(t.clip, t.joint, t.length, t.start): t for t in trials}
    cols = ["clip", "length", "joint", "group", "start", "torso_px", "motion", "motion_class"]
    for m in METHOD_ORDER:
        cols += [f"{m}_mean_px", f"{m}_mid_px"]
    rows = []
    for (clip, L), res in zip(meta, results):
        for joint, start, length, scores in res:
            t = cls_of[(clip, joint, length, start)]
            row = [clip, length, joint, group_of(joint), start, f"{t.torso_px:.3f}", f"{t.motion:.5f}", t.motion_class]
            for m in METHOD_ORDER:
                mean, mid = scores[m]
                row += ["" if mean is None else f"{mean:.4f}", "" if mid is None else f"{mid:.4f}"]
            rows.append(row)
    order = {j: i for i, j in enumerate(JOINT_ORDER)}
    rows.sort(key=lambda r: (r[0], r[1], int(r[4]), order[r[2]]))
    with open(out / "trials.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        w.writerows(rows)
    print(f"stage 2: {len(rows)} trial rows", flush=True)

    # stage 3: analysis
    df = read_trials(out / "trials.csv")
    summ, perclip, paired, cross_rows = analyse(df, out, lengths)
    write_report(out, summ, cross_rows, lengths)
    make_figures(out, summ, lengths)

    # manifest
    dirty = _git("status", "--porcelain", "--", "occlusion_handling.py", "metrics.py", "evaluation/3.3/hide_and_fill.py",
                 "validation/frame_exclusions.csv")
    counts = {}
    for t in trials:
        counts[(t.length)] = counts.get(t.length, 0) + 1
    M = ["Hide-and-fill manifest (evaluation/3.3/hide_and_fill.py)", f"date: {date.today().isoformat()}",
         f"input folder: {args.input_dir.as_posix()}",
         "NO tier is validated: truth is the pose model's own confident output (>= 0.5), not ground truth; D-007 proposed, D-028 proposed, D-029 proposed",
         "", "== Design", f"gap lengths: {list(lengths)} (40 and 60 are EXTRAPOLATION beyond the longest real gap, 30)",
         f"context required observed on each side: {CONTEXT} frames, all 12 body joints observed over the span and context",
         f"starts: first start frame {START_FIRST}, then every {START_STRIDE} frames; every body joint hidden one at a time",
         f"batching: one hidden gap per joint per fill_gaps call, at least {MIN_SEPARATION} frames of observed data between spans",
         f"methods: {METHOD_ORDER}", f"OcclusionParams per method: { {m: v['kwargs'] for m, v in METHODS.items()} }",
         "clamp on/off is reported for limb joints and mirror on/off for trunk joints (the other switch has no effect on them; tested)",
         "error: Euclidean px, per trial the mean over the hidden frames; mid = mean over the central frame(s); torso-normalised columns divide by the clip's MEDIAN torso length (shoulder midpoint to hip midpoint, frames with the four trunk joints observed)",
         f"smooth/sudden: peak second difference of the TRUE position over the hidden span plus {MOTION_MARGIN} frames each side / the clip's median torso length; terciles WITHIN each (gap length, joint), pooled over clips; lowest third smooth, top third sudden, middle unlabelled",
         f"cut points: cutpoints.csv, written at {stage1_done.isoformat(timespec='seconds')} (run started {t0.isoformat(timespec='seconds')}), before any fill was run; the rule was fixed in this file before the full run; development smoke runs looked at errors for plumbing only and the rule was not changed afterwards",
         f"crossover bootstrap: {BOOTSTRAP_RESAMPLES} resamples of the {len(set(t.clip for t in trials))} clips with replacement, seed {BOOTSTRAP_SEED}; descriptive only, not a significance test",
         "", "== Trials per gap length (all joints, all clips)", *(f"{L}: {counts.get(L, 0)}" for L in lengths),
         "", "== Code and environment", f"git HEAD: {_git('rev-parse', 'HEAD')}",
         "occlusion_handling.py, metrics.py, this script, exclusions modified since HEAD: " + ("unknown" if dirty == "unknown" else ("yes" if dirty else "no")),
         f"occlusion_handling.py sha256 {sha256_file(REPO_ROOT / 'occlusion_handling.py')}", f"numpy {np.__version__}"]
    M += [f"{lib}: {_library_version(*(('opencv-python', 'opencv-python-headless') if lib == 'opencv-python' else (lib,)))}" for lib in MANIFEST_LIBRARIES]
    M += ["", "== Clips"] + [f"{clip_names[p]}: keypoint CSV sha256 {sha256_file(p)}" for p in paths]
    M += ["", "== Cut points (smooth if motion <= first, sudden if >= second), see cutpoints.csv for n"] + \
         [f"length {L} {j}: {a:.5f} {b:.5f}" for (L, j), (a, b) in sorted(cuts.items(), key=lambda kv: (kv[0][0], JOINT_ORDER.index(kv[0][1])))]
    (out / "manifest.txt").write_text("\n".join(M) + "\n")
    print(f"Written to {out} in {(datetime.now() - t0).total_seconds():.0f} s")


if __name__ == "__main__":
    main()
