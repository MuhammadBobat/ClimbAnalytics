"""
Step 2.2b: compare the trunk-centroid PROXY CoM (baseline) with the whole-body
segment-model CoM (de Leva 1996, Table 4; DECISIONS.md D-005 follow-ups) on the same clips.

Reuses metrics.py for every number (load_keypoints_csv, compute_com,
compute_com_segment_model, compute_velocity, apply_frame_exclusions); nothing is
re-implemented here. Raw, unsmoothed, no gap-filling (occlusion_handling.py / 3.2 does not
exist yet). Frames in the exclusion list are unusable for both models.

SCOPE OF THE COMPARISON: it runs ONLY on frames where BOTH models are defined. The segment
model needs 14 joints, the proxy only 4, so it is defined on fewer frames; the number of
frames where only the proxy is defined is reported separately (n_only_proxy_defined) and is
not compared. The comparison should be re-run after the occlusion handling (3.2), which
should raise the segment model's coverage. The LDLJ difference is not available until 5.1
exists (column ldlj_difference = n/a).

Sex: the segment model needs a sex table. The primary one is --primary-sex (default male: the
current footage, per the author); the other two ("female", "mean" = unweighted average of
the two tables, a logged approximation) are sensitivity checks only. A later product would
take sex as an input.

Outputs, under evaluation/2.2b/<input-folder-name>/:
    <clip>/frames.csv   per frame: both CoMs, torso length, difference (px and torso lengths)
    <clip>/com_compare.png   (gitignored) CoM x/y and difference in torso lengths vs time
    summary.csv         per clip and sex, plus ALL (pooled), with n
    manifest.txt        run provenance

Usage:
    python3 evaluation/2.2b/compare_com_models.py
    python3 evaluation/2.2b/compare_com_models.py --input-dir validation/output
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

from metrics import (  # noqa: E402
    SEX_TABLES,
    JointSeries,
    apply_frame_exclusions,
    compute_com,
    compute_com_segment_model,
    compute_velocity,
    load_keypoints_csv,
    segment_model_joints,
)
from validation.analyze_occlusion_gaps import (  # noqa: E402
    MANIFEST_LIBRARIES,
    _git,
    _library_version,
    load_exclusions,
    sha256_file,
)

DEFAULT_PAPER = Path.home() / "Documents" / "papers" / "de_leva_1996.pdf"
TABLE_ROWS_USED = (
    "head VERT-MIDG (mass only, at ear midpoint); trunk MIDS-MIDH; upper arm SJC-EJC; "
    "forearm EJC-WJC; hand WJC-MET3 (mass only, at wrist); thigh HJC-KJC; "
    "shank KJC-AJC (alternative-endpoints row); foot HEEL-TTIP (mass only, at ankle)"
)
RAW_NOTE = "raw, unsmoothed, no gap-filling"


@dataclass
class ClipComparison:
    clip: str
    fps: float
    n_frames: int
    n_excluded: int
    proxy: JointSeries
    seg: dict[str, JointSeries]     # sex -> segment-model CoM
    torso_px: list[float | None]    # shoulder midpoint -> hip midpoint, where defined


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def compare_clip(csv_path: Path, excluded: set[int] | None) -> ClipComparison:
    with open(csv_path, newline="") as f:
        clip = next(csv.DictReader(f))["video"]
    joints = load_keypoints_csv(csv_path)
    n = len(next(iter(joints.values())).positions)
    exc = {f for f in (excluded or set()) if 0 <= f < n}
    joints = apply_frame_exclusions(joints, exc)
    proxy = compute_com(joints)
    seg = {sex: compute_com_segment_model(joints, sex) for sex in SEX_TABLES}
    torso = []
    for t in range(n):
        ls, rs = joints["left_shoulder"].positions[t], joints["right_shoulder"].positions[t]
        lh, rh = joints["left_hip"].positions[t], joints["right_hip"].positions[t]
        if None in (ls, rs, lh, rh):
            torso.append(None)
        else:
            a, b = _mid(ls, rs), _mid(lh, rh)
            torso.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    ts = proxy.timestamps
    fps = (n - 1) / (ts[-1] - ts[0]) if n > 1 and ts[-1] > ts[0] else 30.0
    return ClipComparison(clip, fps, n, len(exc), proxy, seg, torso)


def frame_differences(c: ClipComparison, sex: str) -> list[tuple[float, float] | None]:
    """(difference in px, difference in torso lengths) where both CoMs are defined, else None."""
    out = []
    for t in range(c.n_frames):
        p, s, torso = c.proxy.positions[t], c.seg[sex].positions[t], c.torso_px[t]
        if p is None or s is None or not torso:
            out.append(None)
            continue
        d = math.hypot(s[0] - p[0], s[1] - p[1])
        out.append((d, d / torso))
    return out


def speed_pairs(c: ClipComparison, sex: str) -> list[tuple[float, float]]:
    """(proxy speed, segment-model speed) px/s where both are defined on the same step."""
    vp, vs = compute_velocity(c.proxy), compute_velocity(c.seg[sex])
    return [(a, b) for a, b in zip(vp, vs) if a is not None and b is not None]


SUMMARY_COLUMNS = [
    "clip", "sex", "role", "n_frames", "n_excluded_by_list", "n_proxy_defined",
    "n_segment_defined", "n_both_defined", "n_only_proxy_defined", "n_only_segment_defined",
    "pos_diff_px_mean", "pos_diff_px_median", "pos_diff_px_max",
    "pos_diff_torso_mean", "pos_diff_torso_median", "pos_diff_torso_max",
    "n_speed_pairs", "speed_proxy_mean_px_s", "speed_segment_mean_px_s",
    "speed_absdiff_mean_px_s", "speed_absdiff_max_px_s", "ldlj_difference",
]


def _f(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def _stats(vals: list[float]) -> tuple[str, str, str]:
    if not vals:
        return "", "", ""
    return _f(statistics.fmean(vals)), _f(statistics.median(vals)), _f(max(vals))


def summary_row(label: str, sex: str, primary: str, comps: list[ClipComparison]) -> list:
    n_frames = sum(c.n_frames for c in comps)
    n_proxy = sum(sum(p is not None for p in c.proxy.positions) for c in comps)
    n_seg = sum(sum(p is not None for p in c.seg[sex].positions) for c in comps)
    diffs = [d for c in comps for d in frame_differences(c, sex) if d is not None]
    n_both = len(diffs)
    n_only_proxy = sum(
        1 for c in comps for t in range(c.n_frames)
        if c.proxy.positions[t] is not None and c.seg[sex].positions[t] is None)
    n_only_seg = sum(
        1 for c in comps for t in range(c.n_frames)
        if c.proxy.positions[t] is None and c.seg[sex].positions[t] is not None)
    sp = [pair for c in comps for pair in speed_pairs(c, sex)]
    absd = [abs(b - a) for a, b in sp]
    return [
        label, sex, "primary" if sex == primary else "sensitivity",
        n_frames, sum(c.n_excluded for c in comps), n_proxy, n_seg, n_both, n_only_proxy, n_only_seg,
        *_stats([d[0] for d in diffs]), *_stats([d[1] for d in diffs]),
        len(sp), _f(statistics.fmean(a for a, _ in sp)) if sp else "",
        _f(statistics.fmean(b for _, b in sp)) if sp else "",
        _f(statistics.fmean(absd)) if absd else "", _f(max(absd)) if absd else "", "n/a",
    ]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def frames_rows(c: ClipComparison) -> tuple[list[str], list[list]]:
    header = ["frame_idx", "timestamp_s", "proxy_x", "proxy_y", "torso_length_px"]
    for sex in SEX_TABLES:
        header += [f"seg_{sex}_x", f"seg_{sex}_y", f"diff_px_{sex}", f"diff_torso_{sex}"]
    diffs = {sex: frame_differences(c, sex) for sex in SEX_TABLES}
    rows = []
    for t in range(c.n_frames):
        p = c.proxy.positions[t]
        row = [t, f"{c.proxy.timestamps[t]:.4f}", _f(p[0]) if p else "", _f(p[1]) if p else "",
               _f(c.torso_px[t])]
        for sex in SEX_TABLES:
            s, d = c.seg[sex].positions[t], diffs[sex][t]
            row += [_f(s[0]) if s else "", _f(s[1]) if s else "",
                    _f(d[0]) if d else "", _f(d[1]) if d else ""]
        rows.append(row)
    return header, rows


def plot_clip(c: ClipComparison, primary: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = c.proxy.timestamps
    nan = float("nan")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, k, label in ((axes[0], 0, "CoM x (px)"), (axes[1], 1, "CoM y (px, image coordinates, down)")):
        ax.plot(ts, [p[k] if p else nan for p in c.proxy.positions], lw=1, label="trunk-centroid proxy")
        ax.plot(ts, [p[k] if p else nan for p in c.seg[primary].positions], lw=1,
                label=f"segment model ({primary})")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    for sex in SEX_TABLES:
        d = frame_differences(c, sex)
        axes[2].plot(ts, [x[1] if x else nan for x in d], lw=1,
                     label=f"{sex}{' (primary)' if sex == primary else ' (sensitivity)'}")
    axes[2].set_ylabel("|difference| (torso lengths)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{c.clip} — proxy vs segment-model CoM, frames where both are defined — {RAW_NOTE}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_manifest(comps: list[ClipComparison], primary: str, input_dir: Path, footage_dir: Path,
                   exclusions_path: Path, paper: Path, today: str | None = None) -> str:
    n_total = sum(c.n_frames for c in comps)
    n_proxy = sum(sum(p is not None for p in c.proxy.positions) for c in comps)
    n_both = sum(sum(1 for d in frame_differences(c, primary) if d is not None) for c in comps)
    L = [
        "CoM model comparison manifest (evaluation/2.2b/compare_com_models.py)",
        f"date: {today or date.today().isoformat()}",
        f"input folder: {input_dir.as_posix()}",
        "",
        "== Scope",
        f"The comparison runs ONLY on frames where both models are defined: {n_both} of {n_total} "
        f"frames on these {len(comps)} clip(s).",
        f"Frames where only the proxy is defined (not compared): {n_proxy - n_both} "
        f"(proxy defined on {n_proxy}).",
        "Re-run after the occlusion handling (3.2): it should raise the segment model's coverage.",
        "LDLJ difference: not available until 5.1 exists (summary.csv column = n/a).",
        "",
        "== Models",
        "baseline: trunk-centroid proxy, equal weights on shoulders and hips (D-005 follow-up)",
        "segment model: de Leva (1996), J Biomech 29(9):1223-1230, Table 4 (p. 1228)",
        f"table rows used: {TABLE_ROWS_USED}",
        f"required joints ({len(segment_model_joints())}): {', '.join(segment_model_joints())}",
        f"sex tables: primary = {primary}; sensitivity = "
        f"{', '.join(s for s in SEX_TABLES if s != primary)} (mean = unweighted average of the two tables)",
        "local paper copy (outside the repo): de_leva_1996.pdf sha256 "
        + (sha256_file(paper) if paper.exists() else "not found"),
        "",
        "== Frame exclusion file",
    ]
    if exclusions_path.exists():
        L += [f"path: {exclusions_path.as_posix()}", "contents (verbatim):",
              exclusions_path.read_text().rstrip("\n")]
    else:
        L.append("none")
    L += ["", "== Code and environment (at run time)",
          f"git HEAD: {_git('rev-parse', 'HEAD')}"]
    dirty = _git("status", "--porcelain", "--", "metrics.py", "evaluation/2.2b/compare_com_models.py",
                 "validation/frame_exclusions.csv")
    L.append("metrics/script/exclusions modified since HEAD: "
             + ("unknown" if dirty == "unknown" else ("yes" if dirty else "no")))
    for lib in MANIFEST_LIBRARIES:
        names = ("opencv-python", "opencv-python-headless") if lib == "opencv-python" else (lib,)
        L.append(f"{lib}: {_library_version(*names)}")
    L += ["", "== Clips"]
    for c in comps:
        vp = footage_dir / c.clip
        digest = sha256_file(vp) if vp.exists() else "video not found"
        L.append(f"{c.clip}: frames {c.n_frames}, fps {c.fps:.3f} (from timestamps), sha256 {digest}")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/output_full"))
    parser.add_argument("--footage-dir", type=Path, default=Path("footage/test"))
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation/2.2b"))
    parser.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"))
    parser.add_argument("--primary-sex", choices=SEX_TABLES, default="male")
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER,
                        help="local copy of de Leva (1996), only its SHA-256 is recorded")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    csv_paths = sorted(args.input_dir.glob("*_keypoints.csv"))
    if not csv_paths:
        print(f"No *_keypoints.csv files in {args.input_dir}.")
        return
    exclusions = load_exclusions(args.exclusions)
    out_root = args.out_dir / args.input_dir.resolve().name
    out_root.mkdir(parents=True, exist_ok=True)

    comps = []
    for p in csv_paths:
        with open(p, newline="") as f:
            clip = next(csv.DictReader(f))["video"]
        c = compare_clip(p, exclusions.get(clip))
        comps.append(c)
        folder = out_root / Path(clip).stem
        folder.mkdir(parents=True, exist_ok=True)
        header, rows = frames_rows(c)
        write_csv(folder / "frames.csv", header, rows)
        if not args.no_plots:
            plot_clip(c, args.primary_sex, folder / "com_compare.png")

    rows = []
    for sex in SEX_TABLES:
        for c in comps:
            rows.append(summary_row(c.clip, sex, args.primary_sex, [c]))
        rows.append(summary_row("ALL", sex, args.primary_sex, comps))
    write_csv(out_root / "summary.csv", SUMMARY_COLUMNS, rows)
    (out_root / "manifest.txt").write_text(build_manifest(
        comps, args.primary_sex, args.input_dir, args.footage_dir, args.exclusions, args.paper))

    cols = SUMMARY_COLUMNS
    for r in rows:
        if r[1] == args.primary_sex:
            d = dict(zip(cols, r))
            print(f"{d['clip']:12s} [{d['sex']}] frames={d['n_frames']} proxy={d['n_proxy_defined']} "
                  f"segment={d['n_segment_defined']} both={d['n_both_defined']} "
                  f"only_proxy={d['n_only_proxy_defined']} | diff px mean/median/max="
                  f"{d['pos_diff_px_mean']}/{d['pos_diff_px_median']}/{d['pos_diff_px_max']} | "
                  f"torso lengths mean/median/max={d['pos_diff_torso_mean']}/{d['pos_diff_torso_median']}/"
                  f"{d['pos_diff_torso_max']} | speed pairs n={d['n_speed_pairs']} "
                  f"|Δv| mean/max={d['speed_absdiff_mean_px_s']}/{d['speed_absdiff_max_px_s']} px/s")
    print(f"Written to {out_root}")


if __name__ == "__main__":
    main()
