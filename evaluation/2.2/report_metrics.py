"""
Step 2.1 / 2.2 review aid: per-clip overlay, plots and tables for velocity,
displacement and the trunk-centroid CoM PROXY. Built to be WATCHED and read by a
person; nothing in the files interprets the numbers.

Reuses metrics.py for every number (load_keypoints_csv, compute_velocity,
compute_displacement, compute_com) — no metric logic is duplicated here. Everything
is raw and unsmoothed, with no gap-filling (occlusion_handling.py, D-007, does not
exist yet), so a gap is None, never a bridged value. CoM is the D-005 follow-up
trunk-centroid proxy (equal weights on shoulders and hips), not whole-body CoM.

Frames listed in validation/frame_exclusions.csv (wrong-person lock, D-024) are
treated as unusable for EVERY series here: positions are None, CoM is None, and the
reason is reported as excluded_by_list. The keypoint CSVs are never modified.

Per clip, under evaluation/2.2/<input-folder-name>/<clip>/:
    cov_overlay.mp4    video + skeleton + four trunk joints + CoM proxy (labelled),
                       frame number, banner and red border when CoM is None
    com_xy.png         CoM x and y (px) against time
    com_speed.png      CoM speed (px/s) against time
    gap_timeline.png   per joint: detector-dropped / low-confidence / excluded frames
    joint_metrics.csv  per joint (and the CoM proxy): mean/max velocity,
                       cumulative displacement, steps counted and skipped
and evaluation/2.2/<input-folder-name>/summary.csv with one row per clip.
Videos and PNGs are gitignored; the CSVs are small and may be committed.

Usage:
    python3 evaluation/2.2/report_metrics.py
    python3 evaluation/2.2/report_metrics.py --input-dir validation/output --no-video
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from metrics import (  # noqa: E402
    GAP_LOW_CONFIDENCE,
    GAP_MISSING_DETECTION,
    GAP_UNSPECIFIED,
    TRUNK_CENTROID_PROXY_WEIGHTS,
    JointSeries,
    compute_com,
    compute_displacement,
    compute_velocity,
    load_keypoints_csv,
)
from validation.analyze_occlusion_gaps import COCO17_NAMES, load_exclusions  # noqa: E402

EXCLUDED = "excluded_by_list"
# why a frame has no CoM, in priority order (a listed exclusion wins over a drop)
REASON_LABELS = {
    EXCLUDED: "excluded_by_list",
    GAP_MISSING_DETECTION: "detector_drop",
    GAP_LOW_CONFIDENCE: "low_confidence",
    GAP_UNSPECIFIED: "unspecified",
}
COM_SERIES_NAME = "com_trunk_centroid_proxy"

# Same 16 COCO-17 edges as validate_vitpose.py (overlay drawing only).
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]
TRUNK_LOOP = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]


@dataclass
class ClipResult:
    clip: str                       # video file name, e.g. IMG_2419.MOV
    fps: float
    joints: dict[str, JointSeries]  # exclusions applied
    com: JointSeries
    com_reasons: list[str | None]   # None where CoM exists, else a REASON_LABELS key
    excluded_frames: set[int]


def apply_exclusions(joints: dict[str, JointSeries], excluded: set[int]) -> dict[str, JointSeries]:
    """Copy of the series with excluded frames set to None (reason EXCLUDED)."""
    out = {}
    for name, s in joints.items():
        pos, rs = list(s.positions), list(s.gap_reasons)
        for f in excluded:
            if 0 <= f < len(pos):
                pos[f], rs[f] = None, EXCLUDED
        out[name] = JointSeries(name, pos, s.timestamps, rs)
    return out


def com_missing_reason(joints: dict[str, JointSeries], t: int) -> str:
    causes = {joints[n].gap_reasons[t] for n in TRUNK_CENTROID_PROXY_WEIGHTS
              if joints[n].positions[t] is None}
    for r in (EXCLUDED, GAP_MISSING_DETECTION, GAP_LOW_CONFIDENCE):
        if r in causes:
            return r
    return GAP_UNSPECIFIED


def analyse_clip(csv_path: Path, excluded: set[int] | None = None) -> ClipResult:
    with open(csv_path, newline="") as f:
        clip = next(csv.DictReader(f))["video"]
    joints = load_keypoints_csv(csv_path)
    n = len(next(iter(joints.values())).positions)
    exc = {f for f in (excluded or set()) if 0 <= f < n}
    joints = apply_exclusions(joints, exc)
    com = compute_com(joints)
    ts = com.timestamps
    fps = (n - 1) / (ts[-1] - ts[0]) if n > 1 and ts[-1] > ts[0] else 30.0
    reasons = [None if com.positions[t] is not None else com_missing_reason(joints, t)
               for t in range(n)]
    return ClipResult(clip, fps, joints, com, reasons, exc)


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.3f}"


def joint_metric_rows(res: ClipResult) -> list[list]:
    rows = []
    series = [(name, res.joints[name]) for name in COCO17_NAMES if name in res.joints]
    series.append((COM_SERIES_NAME, res.com))
    for name, s in series:
        v = [x for x in compute_velocity(s) if x is not None]
        d = compute_displacement(s)
        c = Counter(s.gap_reasons)
        rows.append([
            name, len(s.positions), c[GAP_MISSING_DETECTION], c[GAP_LOW_CONFIDENCE], c[EXCLUDED],
            len(v), _fmt(sum(v) / len(v)) if v else "", _fmt(max(v)) if v else "",
            _fmt(d.total), d.n_steps_counted, d.n_steps_skipped,
        ])
    return rows


JOINT_METRIC_COLUMNS = [
    "series", "n_frames", "frames_detector_drop", "frames_low_confidence", "frames_excluded_by_list",
    "n_velocity_values", "mean_velocity_px_s", "max_velocity_px_s",
    "cumulative_displacement_px", "displacement_steps_counted", "displacement_steps_skipped",
]
SUMMARY_COLUMNS = [
    "clip", "fps", "frames", "frames_with_com", "frames_without_com",
    "without_com_detector_drop", "without_com_excluded_by_list", "without_com_low_confidence",
    "without_com_unspecified", "n_com_speed_values", "mean_com_speed_px_s", "max_com_speed_px_s",
]


def summary_row(res: ClipResult) -> list:
    n = len(res.com.positions)
    with_com = sum(p is not None for p in res.com.positions)
    why = Counter(r for r in res.com_reasons if r is not None)
    v = [x for x in compute_velocity(res.com) if x is not None]
    return [
        res.clip, f"{res.fps:.3f}", n, with_com, n - with_com,
        why[GAP_MISSING_DETECTION], why[EXCLUDED], why[GAP_LOW_CONFIDENCE], why[GAP_UNSPECIFIED],
        len(v), _fmt(sum(v) / len(v)) if v else "", _fmt(max(v)) if v else "",
    ]


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    out, start = [], None
    for i, fl in enumerate(flags):
        if fl and start is None:
            start = i
        elif not fl and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


# --- plots ------------------------------------------------------------------------------

RAW_NOTE = "raw, unsmoothed, no gap-filling — CoM is a trunk-centroid PROXY"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _shade_no_com(ax, res: ClipResult) -> None:
    ts = res.com.timestamps
    for a, b in _runs([p is None for p in res.com.positions]):
        ax.axvspan(ts[a] - 0.5 / res.fps, ts[b] + 0.5 / res.fps, color="0.85", lw=0)


def plot_com_xy(res: ClipResult, path: Path) -> None:
    plt = _plt()
    ts = res.com.timestamps
    nan = float("nan")
    xs = [p[0] if p else nan for p in res.com.positions]
    ys = [p[1] if p else nan for p in res.com.positions]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, vals, label in ((ax1, xs, "CoM proxy x (px)"), (ax2, ys, "CoM proxy y (px, image coordinates, down)")):
        ax.plot(ts, vals, lw=1)
        _shade_no_com(ax, res)
        ax.set_ylabel(label)
    ax2.set_xlabel("time (s)")
    fig.suptitle(f"{res.clip} — CoM proxy position — {RAW_NOTE}\n(grey = no CoM)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_com_speed(res: ClipResult, path: Path) -> None:
    plt = _plt()
    ts = res.com.timestamps
    nan = float("nan")
    v = [x if x is not None else nan for x in compute_velocity(res.com)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, v, lw=1)
    _shade_no_com(ax, res)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("CoM proxy speed (px/s)")
    ax.set_title(f"{res.clip} — CoM proxy speed — {RAW_NOTE}\n(grey = no CoM)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


GAP_COLOURS = {GAP_MISSING_DETECTION: "#444444", GAP_LOW_CONFIDENCE: "#ff7f0e", EXCLUDED: "#9467bd"}
GAP_NAMES = {GAP_MISSING_DETECTION: "detector drop (no person box)",
             GAP_LOW_CONFIDENCE: "low confidence (< 0.5)", EXCLUDED: "excluded by list"}


def plot_gap_timeline(res: ClipResult, path: Path) -> None:
    plt = _plt()
    from matplotlib.patches import Patch
    ts = res.com.timestamps
    dt = 1.0 / res.fps
    names = [n for n in COCO17_NAMES if n in res.joints] + [COM_SERIES_NAME]
    series = {n: res.joints[n] for n in names if n in res.joints}
    series[COM_SERIES_NAME] = res.com
    fig, ax = plt.subplots(figsize=(10, 6))
    for row, name in enumerate(names):
        rs = series[name].gap_reasons
        for kind, colour in GAP_COLOURS.items():
            for a, b in _runs([r == kind for r in rs]):
                ax.broken_barh([(ts[a] - dt / 2, (b - a + 1) * dt)], (row - 0.4, 0.8),
                               facecolors=colour, lw=0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.set_title(f"{res.clip} — frames with no usable position per joint — {RAW_NOTE}", fontsize=10)
    ax.legend(handles=[Patch(color=GAP_COLOURS[k], label=GAP_NAMES[k]) for k in GAP_COLOURS],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- overlay video ------------------------------------------------------------------------

COL_SKELETON = (0, 200, 0)
COL_TRUNK = (0, 255, 255)
COL_COM = (255, 0, 255)
COL_ALERT = (0, 0, 220)
COL_EXCLUDED = (200, 60, 150)


def draw_frame(frame, res: ClipResult, t: int):
    import cv2
    h, w = frame.shape[:2]
    joints = res.joints
    scale = max(w, h) / 1920.0
    thick = max(2, int(3 * scale))
    font = 1.1 * scale + 0.3
    reason = res.com_reasons[t]
    banner = None

    if reason == EXCLUDED:
        banner = ("EXCLUDED (wrong-person lock) - keypoints not drawn", COL_EXCLUDED)
    else:
        pts = {n: joints[n].positions[t] for n in COCO17_NAMES if n in joints}
        for i, j in SKELETON_EDGES:
            a, b = pts.get(COCO17_NAMES[i]), pts.get(COCO17_NAMES[j])
            if a and b:
                cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), COL_SKELETON, thick)
        for name, p in pts.items():
            if p:
                cv2.circle(frame, (int(p[0]), int(p[1])), int(5 * scale) + 2, COL_SKELETON, -1)
        loop = [pts.get(n) for n in TRUNK_LOOP]
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if a and b:
                cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), COL_TRUNK, thick)
        for n in TRUNK_CENTROID_PROXY_WEIGHTS:
            p = pts.get(n)
            if p:
                cv2.circle(frame, (int(p[0]), int(p[1])), int(11 * scale) + 3, COL_TRUNK, thick)
        c = res.com.positions[t]
        if c:
            cx, cy, r = int(c[0]), int(c[1]), int(18 * scale) + 4
            cv2.circle(frame, (cx, cy), r, COL_COM, -1)
            cv2.line(frame, (cx - 2 * r, cy), (cx + 2 * r, cy), COL_COM, thick)
            cv2.line(frame, (cx, cy - 2 * r), (cx, cy + 2 * r), COL_COM, thick)
            cv2.putText(frame, "CoM proxy", (cx + r + 8, cy - r), cv2.FONT_HERSHEY_SIMPLEX,
                        font, COL_COM, thick)
        elif reason == GAP_MISSING_DETECTION:
            banner = ("CoM = None (detector drop: no person box)", COL_ALERT)
        else:
            banner = (f"CoM = None ({REASON_LABELS[reason]})", COL_ALERT)

    if banner:
        text, colour = banner
        cv2.rectangle(frame, (0, 0), (w, int(130 * scale) + 40), colour, -1)
        cv2.putText(frame, text, (int(20 * scale) + 10, int(100 * scale) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick)
        if reason != EXCLUDED:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, int(14 * scale) + 4)
    ts = res.com.timestamps[t]
    cv2.putText(frame, f"{res.clip}  frame {t}  t={ts:.2f}s", (int(20 * scale) + 10, h - int(110 * scale) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick + 2)
    cv2.putText(frame, f"CoM = trunk-centroid PROXY ({RAW_NOTE.split(' — ')[0]})",
                (int(20 * scale) + 10, h - int(40 * scale) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, font * 0.7, (255, 255, 255), max(2, thick - 1))
    return frame


def write_overlay(video_path: Path, out_path: Path, res: ClipResult) -> int:
    """Write cov_overlay.mp4 for the frames that have keypoint data; returns frames written."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"could not open {video_path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or res.fps
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    for t in range(len(res.com.positions)):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(draw_frame(frame, res, t))
        n += 1
    cap.release()
    writer.release()
    return n


# --- driver -------------------------------------------------------------------------------

def report_clip(csv_path: Path, out_root: Path, footage_dir: Path, excluded: set[int] | None,
                make_video: bool = True) -> tuple[ClipResult, list]:
    res = analyse_clip(csv_path, excluded)
    out = out_root / Path(res.clip).stem
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "joint_metrics.csv", JOINT_METRIC_COLUMNS, joint_metric_rows(res))
    plot_com_xy(res, out / "com_xy.png")
    plot_com_speed(res, out / "com_speed.png")
    plot_gap_timeline(res, out / "gap_timeline.png")
    if make_video:
        video = footage_dir / res.clip
        if video.exists():
            n = write_overlay(video, out / "cov_overlay.mp4", res)
            print(f"  overlay: {n} frame(s) written")
        else:
            print(f"  overlay skipped: {video} not found")
    return res, summary_row(res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/output_full"))
    parser.add_argument("--footage-dir", type=Path, default=Path("footage/test"))
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation/2.2"),
                        help="outputs go to <out-dir>/<input-folder-name>/")
    parser.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"))
    parser.add_argument("--no-video", action="store_true", help="skip cov_overlay.mp4")
    args = parser.parse_args()

    csv_paths = sorted(args.input_dir.glob("*_keypoints.csv"))
    if not csv_paths:
        print(f"No *_keypoints.csv files in {args.input_dir}.")
        return
    exclusions = load_exclusions(args.exclusions)
    out_root = args.out_dir / args.input_dir.resolve().name
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in csv_paths:
        with open(p, newline="") as f:
            clip = next(csv.DictReader(f))["video"]
        print(f"{clip}...")
        res, row = report_clip(p, out_root, args.footage_dir, exclusions.get(clip),
                               make_video=not args.no_video)
        rows.append(row)
        print(f"  frames={row[2]} with_com={row[3]} without_com={row[4]} "
              f"(detector_drop={row[5]} excluded={row[6]} low_conf={row[7]}) "
              f"mean_speed={row[10]} max_speed={row[11]} px/s (n={row[9]})")
    _write_csv(out_root / "summary.csv", SUMMARY_COLUMNS, rows)
    print(f"Written to {out_root}")


if __name__ == "__main__":
    main()
