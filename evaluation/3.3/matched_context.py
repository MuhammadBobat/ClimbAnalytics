"""
Step 3.3 follow-up (DECISIONS.md D-029 addendum): spline against smoothed Kalman at MATCHED context.

The first hide-and-fill run (evaluation/3.3/hide_and_fill.py, outputs in evaluation/3.3/output_full_lock1/) found the smoothed
Kalman filter at or below the spline's median error at every length, but the spline used 5 observed frames on each side
(OcclusionParams.tier1_context, default 5) and the Kalman filter 10 (tier2_context, default 10), so method and context were
mixed. This script gives both methods the SAME context, at two sizes (5 and 10 observed frames each side), on the SAME trials
as the first run, by calling the real occlusion_handling.fill_gaps with force_method and the context parameter set.
NO tier is validated; truth is the pose model's own confident output (>= 0.5), not ground truth; five clips.
It changes no tier and no default: only the test sets tier1_context / tier2_context, nothing in occlusion_handling.py.

Fixed in this file BEFORE the full run (the author specified the contexts, lengths, trial set and outputs on 2026-11-25; the
choices below are the assistant's and are not from a literature source):
  trials       exactly the trial set of hide_and_fill.py (its make_trials, cut points and classes are imported, not re-implemented;
               10 observed frames required on each side, so both contexts are fully available on every trial). Check: the trial keys
               and cut points are compared with the first run's trials.csv and cutpoints.csv and the result is written to
               reproduction_check.txt.
  Kalman       the default Kalman of the first run, "kalman_smooth": force_method kalman, RTS-smoothed (not causal), segment-length
               clamp on (it acts on limb joints only), mirror off (it acts on trunk joints only and is off in the default).
  configs      spline_c5, spline_c10, kalman_c5, kalman_c10 (+ hold, the baseline). spline_c5 and kalman_c10 are the first run's
               spline and kalman_smooth settings; the check above also compares their errors with the first run's (expected equal).
  error        as the first run: Euclidean px, per trial the mean over the hidden frames; also in torso lengths.
  paired       per trial, Kalman error minus spline error at the same context (negative = Kalman better); also the context effect,
               c10 minus c5, within each method. Trials where either fill is unavailable are dropped from that comparison (n_paired).
  classes      smooth / sudden as in the first run (cutpoints from true motion only); all = every trial.
  per clip     every table is also given per clip.
  survival     (the assistant's rule, a DESCRIPTIVE label, not a significance test; fixed before the full run): the Kalman advantage
               "survives" at (joint group, length, context) if the pooled median paired difference (Kalman - spline) is below 0 AND the
               Kalman median error is the lower one in at least 4 of the 5 clips (clips with at least one paired trial). "Survives at
               1 to 4 frames" means it does at each of the lengths 1, 2, 3 and 4.
Anything beyond this list that is chosen after seeing results is labelled NOT PRE-REGISTERED in the report or the decision log.

Usage:
    python3 evaluation/3.3/matched_context.py [--input-dir validation/output_full_lock1] \
        [--out-dir evaluation/3.3/output_matched_context_lock1] [--compare-with evaluation/3.3/output_full_lock1]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import occlusion_handling as oh  # noqa: E402

_spec = importlib.util.spec_from_file_location("hide_and_fill", REPO_ROOT / "evaluation" / "3.3" / "hide_and_fill.py")
hf = importlib.util.module_from_spec(_spec)
sys.modules["hide_and_fill"] = hf
_spec.loader.exec_module(hf)

CONTEXTS = (5, 10)
CLASSES = hf.CLASSES
GROUPS = hf.GROUPS
LENGTHS = hf.LENGTHS
EXTRAPOLATION_LENGTHS = hf.EXTRAPOLATION_LENGTHS
SURVIVE_MIN_CLIPS = 4                       # of 5 (assistant's rule, fixed before the full run)
SHORT_LENGTHS = (1, 2, 3, 4)

# config name -> OcclusionParams kwargs (None = hold, computed here)
CONFIGS: dict[str, dict | None] = {"hold": None}
for _c in CONTEXTS:
    CONFIGS[f"spline_c{_c}"] = dict(force_method="spline", tier1_context=_c)
    CONFIGS[f"kalman_c{_c}"] = dict(force_method="kalman", tier2_clamp=True, tier2_mirror=False, tier2_causal=False,
                                    tier2_context=_c)
CONFIG_ORDER = list(CONFIGS)


# --- filling ---------------------------------------------------------------------------------------------

def fill_batch(joints, batch, configs=None, mask=hf.mask_joints):
    """{(trial index, config): list of (x, y) per hidden frame, or None} from the real fill_gaps on the masked joints."""
    configs = configs or CONFIG_ORDER
    masked = mask(joints, batch)
    out = {}
    for name in configs:
        kw = CONFIGS[name]
        if kw is None:
            for i, t in enumerate(batch):
                p = masked[t.joint].positions[t.start - 1]
                out[(i, name)] = [p] * t.length if p is not None else None
            continue
        filled = oh.fill_gaps(masked, oh.OcclusionParams(**kw))
        for i, t in enumerate(batch):
            ps = [filled[t.joint].positions[f] for f in t.span]
            out[(i, name)] = None if any(p is None for p in ps) else ps
    return out


def run_task(task):
    """(path, exclusions, [(joint, start, length), ...]) -> [(joint, start, length, {config: (mean_px, mid_px)})]."""
    path, exclusions, items = task
    joints = hf.load_clip(path, tuple(exclusions))
    torso = hf.median_torso_px(joints)
    trials = [hf.Trial("", j, L, s, torso, 0.0) for j, s, L in items]
    rows = []
    for batch in hf.schedule_batches(trials):
        fills = fill_batch(joints, batch)
        for i, t in enumerate(batch):
            rows.append((t.joint, t.start, t.length, {c: hf.score(t, joints, fills[(i, c)]) for c in CONFIG_ORDER}))
    return rows


# --- analysis --------------------------------------------------------------------------------------------

def _sel(df, group=None, cls="all", length=None, clip=None):
    m = np.ones(len(df), dtype=bool)
    if group:
        m &= (df["group"] == group).to_numpy()
    if cls != "all":
        m &= (df["motion_class"] == cls).to_numpy()
    if length is not None:
        m &= (df["length"] == length).to_numpy()
    if clip is not None:
        m &= (df["clip"] == clip).to_numpy()
    return df[m]


def _q(a):
    return (float(np.percentile(a, 25)), float(np.percentile(a, 50)), float(np.percentile(a, 75))) if len(a) else (np.nan,) * 3


def pair_stats(sub, ctx):
    """Spline vs Kalman at context ctx on the rows of sub (dropping trials where either fill is unavailable)."""
    s, k = sub[f"spline_c{ctx}_mean_px"], sub[f"kalman_c{ctx}_mean_px"]
    ok = (s.notna() & k.notna()).to_numpy()
    s, k, h = s[ok].to_numpy(), k[ok].to_numpy(), sub["hold_mean_px"][ok].to_numpy()
    d = k - s
    torso = sub["torso_px"][ok].to_numpy()
    sq, kq, dq = _q(s), _q(k), _q(d)
    return {"n_trials": int(len(sub)), "n_paired": int(ok.sum()),
            "hold_median": float(np.median(h)) if len(h) else np.nan,
            "spline_q25": sq[0], "spline_median": sq[1], "spline_q75": sq[2],
            "kalman_q25": kq[0], "kalman_median": kq[1], "kalman_q75": kq[2],
            "diff_q25": dq[0], "diff_median": dq[1], "diff_q75": dq[2],
            "diff_median_torso": float(np.median(d / torso)) if len(d) else np.nan,
            "frac_kalman_better": float(np.mean(d < 0)) if len(d) else np.nan}


def context_effect(sub):
    """Within each method, error at context 10 minus error at context 5, paired (negative = more context helps)."""
    out = {}
    for m in ("spline", "kalman"):
        a, b = sub[f"{m}_c5_mean_px"], sub[f"{m}_c10_mean_px"]
        ok = (a.notna() & b.notna()).to_numpy()
        d = b[ok].to_numpy() - a[ok].to_numpy()
        q = _q(d)
        out[m] = {"n_paired": int(ok.sum()), "diff_q25": q[0], "diff_median": q[1], "diff_q75": q[2],
                  "frac_more_context_better": float(np.mean(d < 0)) if len(d) else np.nan}
    return out


def analyse(df, out: Path, lengths=LENGTHS):
    clips = sorted(df["clip"].unique())
    summ, perclip, ctx_rows, surv = [], [], [], []
    for g in GROUPS:
        for cls in CLASSES:
            for L in lengths:
                sub = _sel(df, g, cls, L)
                if not len(sub):
                    continue
                for c in CONTEXTS:
                    ps = pair_stats(sub, c)
                    kl, n_cl = 0, 0
                    for clip in clips:
                        sc = _sel(sub, clip=clip)
                        if not len(sc):
                            continue
                        pc = pair_stats(sc, c)
                        perclip.append({"group": g, "class": cls, "length": L, "context": c, "clip": clip, **pc})
                        if pc["n_paired"]:
                            n_cl += 1
                            kl += int(pc["kalman_median"] < pc["spline_median"])
                    row = {"group": g, "class": cls, "length": L, "context": c, **ps, "clips_with_trials": n_cl,
                           "clips_kalman_lower": kl}
                    row["survives"] = bool(ps["n_paired"] and ps["diff_median"] < 0 and kl >= SURVIVE_MIN_CLIPS)
                    summ.append(row)
                ce = context_effect(sub)
                for m, v in ce.items():
                    ctx_rows.append({"group": g, "class": cls, "length": L, "method": m, **v})
    write_csv(out / "matched_summary.csv", summ)
    write_csv(out / "matched_per_clip.csv", perclip)
    write_csv(out / "context_effect.csv", ctx_rows)
    return summ, perclip, ctx_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def _f(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


def _iqr(r, key):
    return f"{_f(r[key + '_median'])} [{_f(r[key + '_q25'])}, {_f(r[key + '_q75'])}]"


def write_report(out: Path, summ, perclip, ctx_rows, lengths, clips) -> None:
    L = ["# 3.3 follow-up: spline against smoothed Kalman at matched context",
         "",
         "NO tier is validated by this. Truth = the pose model's own confident output (>= 0.5), not ground truth; five clips; trials overlap "
         "(a start every 25 frames) so they are not independent. Errors are px, median [IQR] over trials of the per-trial mean error over the "
         "hidden frames. Paired difference = Kalman minus spline at the SAME context (negative = Kalman better). `*` = 40 and 60 are "
         "EXTRAPOLATION (longer than any real gap, longest 30). Settings, provenance and the pre-run rule: `manifest.txt` and the docstring of "
         "`evaluation/3.3/matched_context.py`. Kalman = the default `kalman_smooth` of the first run (smoothed, clamp on, mirror off).",
         "",
         f"Survival rule (the assistant's, descriptive, fixed before the run, NOT a significance test): the Kalman advantage 'survives' at a "
         f"length and context if the pooled median paired difference is below 0 and the Kalman median is lower in at least {SURVIVE_MIN_CLIPS} of the "
         "5 clips."]
    by = {}
    for r in summ:
        by[(r["group"], r["class"], r["length"], r["context"])] = r
    cl = {}
    for r in perclip:
        cl[(r["group"], r["class"], r["length"], r["context"], r["clip"])] = r
    names = {"limb": "Limb joints (8)", "trunk": "Trunk joints (4)"}
    for g in GROUPS:
        for cls in CLASSES:
            L += ["", f"## {names[g]}, class: {cls}", "",
                  "| length | context | n paired | spline | Kalman | paired diff (Kalman - spline) | Kalman better in | clips Kalman lower | survives |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for n in lengths:
                for c in CONTEXTS:
                    r = by.get((g, cls, n, c))
                    if r is None:
                        continue
                    lab = f"{n}*" if n in EXTRAPOLATION_LENGTHS else str(n)
                    L.append(f"| {lab} | {c} | {r['n_paired']} | {_iqr(r, 'spline')} | {_iqr(r, 'kalman')} | {_iqr(r, 'diff')} | "
                             f"{r['frac_kalman_better']:.0%} | {r['clips_kalman_lower']}/{r['clips_with_trials']} | {'yes' if r['survives'] else 'no'} |")
    L += ["", "## Per clip, lengths 1 to 4 (class: all)", ""]
    for g in GROUPS:
        L += [f"### {names[g]}", "", "| clip | length | context | n paired | spline median | Kalman median | median paired diff | Kalman better in |",
              "|---|---|---|---|---|---|---|---|"]
        for clip in clips:
            for n in SHORT_LENGTHS:
                for c in CONTEXTS:
                    r = cl.get((g, "all", n, c, clip))
                    if r:
                        L.append(f"| {clip} | {n} | {c} | {r['n_paired']} | {_f(r['spline_median'])} | {_f(r['kalman_median'])} | "
                                 f"{_f(r['diff_median'])} | {r['frac_kalman_better']:.0%} |")
        L.append("")
    L += ["## Does the Kalman advantage at 1 to 4 frames survive matched context? (rule-based, class: all)", "",
          "| group | context | survives at lengths 1, 2, 3, 4 | pooled median paired diff at 1, 2, 3, 4 (px) | clips Kalman lower at 1, 2, 3, 4 |", "|---|---|---|---|---|"]
    for g in GROUPS:
        for c in CONTEXTS:
            rs = [by[(g, "all", n, c)] for n in SHORT_LENGTHS if (g, "all", n, c) in by]
            if len(rs) == len(SHORT_LENGTHS):
                L.append(f"| {g} | {c} | {'yes' if all(r['survives'] for r in rs) else 'no'} | "
                         f"{', '.join(_f(r['diff_median']) for r in rs)} | {', '.join(str(r['clips_kalman_lower']) + '/' + str(r['clips_with_trials']) for r in rs)} |")
    L += ["", "## Context effect within each method (context 10 minus context 5, paired; negative = more context helps), class: all", "",
          "| group | length | spline: median diff [IQR], more context better in | Kalman: median diff [IQR], more context better in |", "|---|---|---|---|"]
    ce = {(r["group"], r["class"], r["length"], r["method"]): r for r in ctx_rows}
    for g in GROUPS:
        for n in lengths:
            a, b = ce.get((g, "all", n, "spline")), ce.get((g, "all", n, "kalman"))
            if a and b:
                lab = f"{n}*" if n in EXTRAPOLATION_LENGTHS else str(n)
                L.append(f"| {g} | {lab} | {_iqr(a, 'diff')}, {a['frac_more_context_better']:.0%} | {_iqr(b, 'diff')}, {b['frac_more_context_better']:.0%} |")
    (out / "report.md").write_text("\n".join(L) + "\n")


def make_figure(out: Path, summ, lengths) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, g in zip(axes, GROUPS):
        for c, col in zip(CONTEXTS, ("#1f77b4", "#d62728")):
            rows = [r for r in summ if r["group"] == g and r["class"] == "all" and r["context"] == c]
            xs = [r["length"] for r in rows]
            ax.plot(xs, [r["diff_median"] for r in rows], "-o", ms=3, color=col, label=f"context {c}")
            ax.fill_between(xs, [r["diff_q25"] for r in rows], [r["diff_q75"] for r in rows], color=col, alpha=0.12)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvspan(30, max(lengths) + 2, color="0.9", zorder=0)
        ax.set_xscale("log")
        ax.set_title(f"{g}: Kalman minus spline (px), median and IQR; shaded = extrapolation")
        ax.set_xlabel("hidden length (frames)")
        ax.title.set_fontsize(8)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "paired_diff_vs_length.png", dpi=130)
    plt.close(fig)


# --- reproduction check ------------------------------------------------------------------------------------

def reproduction_check(df, trials, cuts_text: str, compare_dir: Path | None) -> list[str]:
    """Compare with the first run's folder: cut points text, trial keys, and the errors of the settings both runs share."""
    if compare_dir is None or not (compare_dir / "trials.csv").exists():
        return ["not checked: the first run's trials.csv was not available"]
    import pandas as pd
    old = pd.read_csv(compare_dir / "trials.csv")
    L = []
    same_cuts = (compare_dir / "cutpoints.csv").read_text() == cuts_text
    L.append(f"cut points identical to {compare_dir.as_posix()}/cutpoints.csv: {same_cuts}")
    key = ["clip", "length", "joint", "start"]
    ko = set(map(tuple, old[key].to_numpy().tolist()))
    kn = set(map(tuple, df[key].to_numpy().tolist()))
    L.append(f"trial keys identical to the first run's (clip, length, joint, start): {ko == kn} (first run {len(ko)}, this run {len(kn)})")
    m = old.merge(df, on=key, suffixes=("_old", "_new"))
    L.append(f"classes identical: {bool((m['motion_class_old'] == m['motion_class_new']).all())}")
    for new, oldc in (("spline_c5", "spline"), ("kalman_c10", "kalman_smooth"), ("hold", "hold")):
        # a column present in both files (hold) gets the merge suffixes; the others keep their names
        ca = f"{oldc}_mean_px_old" if oldc == "hold" else f"{oldc}_mean_px"
        cb = f"{new}_mean_px_new" if new == "hold" else f"{new}_mean_px"
        a, b = m[ca].to_numpy(float), m[cb].to_numpy(float)
        both = ~np.isnan(a) & ~np.isnan(b)
        L.append(f"{new} against the first run's {oldc}: {int(both.sum())} trials compared, max |difference| "
                 f"{np.max(np.abs(a[both] - b[both])) if both.any() else float('nan'):.6f} px, availability identical: "
                 f"{bool((np.isnan(a) == np.isnan(b)).all())}")
    return L


def main() -> None:
    from validation.analyze_occlusion_gaps import MANIFEST_LIBRARIES, _library_version, load_exclusions, sha256_file
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=Path("validation/output_full_lock1"))
    ap.add_argument("--out-dir", type=Path, default=Path("evaluation/3.3/output_matched_context_lock1"))
    ap.add_argument("--compare-with", type=Path, default=Path("evaluation/3.3/output_full_lock1"))
    ap.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lengths", type=int, nargs="*", default=None, help="development smoke runs only")
    ap.add_argument("--clips", nargs="*", default=None, help="development smoke runs only (clip file stems)")
    args = ap.parse_args()
    lengths = tuple(args.lengths) if args.lengths else LENGTHS
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        print(f"Refusing to write into the non-empty folder {out}", file=sys.stderr)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)
    exclusions = load_exclusions(args.exclusions)
    t0 = datetime.now()

    paths = sorted(args.input_dir.glob("*_keypoints.csv"))
    if args.clips:
        paths = [p for p in paths if p.name.replace("_keypoints.csv", "") in args.clips]
    clip_names, trials = {}, []
    for p in paths:
        with open(p, newline="") as f:
            clip = next(csv.DictReader(f))["video"]
        clip_names[p] = clip
        ex = tuple(sorted(exclusions.get(clip, set())))
        trials += hf.make_trials(clip, hf.load_clip(str(p), ex), lengths)
    cuts = hf.tercile_cuts(trials)
    trials = hf.classify(trials, cuts)
    cut_lines = ["length,joint,smooth_if_motion_le,sudden_if_motion_ge,n_trials"]
    for (L, j), (a, b) in sorted(cuts.items(), key=lambda kv: (kv[0][0], hf.JOINT_ORDER.index(kv[0][1]))):
        cut_lines.append(f"{L},{j},{a:.5f},{b:.5f},{sum(1 for t in trials if t.length == L and t.joint == j)}")
    cuts_text = "\n".join(cut_lines) + "\n"
    (out / "cutpoints.csv").write_text(cuts_text)
    stage1_done = datetime.now()
    print(f"stage 1: {len(trials)} trials", flush=True)

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
    for c in CONFIG_ORDER:
        cols += [f"{c}_mean_px", f"{c}_mid_px"]
    rows = []
    for (clip, L), res in zip(meta, results):
        for joint, start, length, scores in res:
            t = cls_of[(clip, joint, length, start)]
            row = [clip, length, joint, hf.group_of(joint), start, f"{t.torso_px:.3f}", f"{t.motion:.5f}", t.motion_class]
            for c in CONFIG_ORDER:
                mean, mid = scores[c]
                row += ["" if mean is None else f"{mean:.4f}", "" if mid is None else f"{mid:.4f}"]
            rows.append(row)
    order = {j: i for i, j in enumerate(hf.JOINT_ORDER)}
    rows.sort(key=lambda r: (r[0], r[1], int(r[4]), order[r[2]]))
    with open(out / "trials.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        w.writerows(rows)
    print(f"stage 2: {len(rows)} trial rows", flush=True)

    import pandas as pd
    df = pd.read_csv(out / "trials.csv")
    summ, perclip, ctx_rows = analyse(df, out, lengths)
    clips = sorted(df["clip"].unique())
    write_report(out, summ, perclip, ctx_rows, lengths, clips)
    make_figure(out, summ, lengths)
    check = reproduction_check(df, trials, cuts_text, args.compare_with if not (args.lengths or args.clips) else None)
    (out / "reproduction_check.txt").write_text("\n".join(check) + "\n")

    dirty = hf._git("status", "--porcelain", "-uno", "--", "occlusion_handling.py", "metrics.py", "evaluation/3.3/hide_and_fill.py",
                    "evaluation/3.3/matched_context.py", "validation/frame_exclusions.csv")
    counts = {}
    for t in trials:
        counts[t.length] = counts.get(t.length, 0) + 1
    M = ["Matched-context hide-and-fill manifest (evaluation/3.3/matched_context.py)", f"date: {date.today().isoformat()}",
         f"input folder: {args.input_dir.as_posix()}", f"compared with: {args.compare_with.as_posix()} (reproduction_check.txt)",
         "NO tier is validated: truth is the pose model's own confident output (>= 0.5), not ground truth; D-007, D-028, D-029 proposed. "
         "No tier and no default is changed by this run.",
         "", "== Design", f"gap lengths: {list(lengths)} (40 and 60 are EXTRAPOLATION beyond the longest real gap, 30)",
         f"contexts (observed frames on each side given to the spline AND to the Kalman filter): {list(CONTEXTS)}",
         f"trials: those of hide_and_fill.py (make_trials, {hf.CONTEXT} observed frames required each side, start {hf.START_FIRST} then every {hf.START_STRIDE}, all 12 body joints hidden one at a time, batching with at least {hf.MIN_SEPARATION} frames between spans)",
         f"OcclusionParams per config: { {c: v for c, v in CONFIGS.items()} }",
         "Kalman = the first run's kalman_smooth (smoothed, clamp on for limb joints, mirror off for trunk joints)",
         "error: Euclidean px, per trial the mean over the hidden frames; paired difference = Kalman minus spline at the same context",
         f"survival rule (assistant's, descriptive, fixed in the script before the full run): pooled median paired difference < 0 and Kalman median lower in >= {SURVIVE_MIN_CLIPS} of 5 clips",
         f"cut points: cutpoints.csv, written at {stage1_done.isoformat(timespec='seconds')} (run started {t0.isoformat(timespec='seconds')}), before any fill was run",
         "Not pre-registered: nothing in this run was chosen after seeing results; any later choice is labelled in the decision log.",
         "", "== Trials per gap length (all joints, all clips)", *(f"{L}: {counts.get(L, 0)}" for L in lengths),
         "", "== Reproduction check against the first run", *check,
         "", "== Code and environment", f"git HEAD: {hf._git('rev-parse', 'HEAD')}",
         "occlusion_handling.py, metrics.py, hide_and_fill.py, matched_context.py, exclusions modified since HEAD (tracked files): " + ("unknown" if dirty == "unknown" else ("yes" if dirty else "no")),
         f"occlusion_handling.py sha256 {sha256_file(REPO_ROOT / 'occlusion_handling.py')}", f"numpy {np.__version__}"]
    M += [f"{lib}: {_library_version(*(('opencv-python', 'opencv-python-headless') if lib == 'opencv-python' else (lib,)))}" for lib in MANIFEST_LIBRARIES]
    M += ["", "== Clips"] + [f"{clip_names[p]}: keypoint CSV sha256 {sha256_file(p)}" for p in paths]
    (out / "manifest.txt").write_text("\n".join(M) + "\n")
    print(f"Written to {out} in {(datetime.now() - t0).total_seconds():.0f} s")


if __name__ == "__main__":
    main()
