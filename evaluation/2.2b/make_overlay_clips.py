"""
Short overlay clips for the ten largest speed differences between the trunk-centroid proxy
CoM and the segment-model CoM (rows of top_speed_differences.csv, written by
compare_com_models.py). Each clip covers the step (frame_from -> frame_to) padded by 15 frames
on each side (clamped at the clip edges), at the source video's full resolution.

Drawn on each frame: the skeleton of the 14 joints the segment model uses (colour by
confidence: green >= 0.7, orange 0.5-0.7; a joint below 0.5 is marked with a red X at its raw
position), the proxy CoM (magenta) and the segment-model CoM (cyan), a legend, and in the
corner the frame number, time and both speeds (px/s, step t-1 -> t; "n/a" where undefined).
The 0.7 colour break is drawing only, as in validate_vitpose.py; 0.5 is D-007's threshold
(metrics.py). Frames in the exclusion list show a banner and no skeleton; detector-dropped
frames show "NO DETECTION". Same exclusions as compare_com_models.py (it is reused for every
number); raw, unsmoothed, no gap-filling.

Videos go to evaluation/2.2b/overlays/<input-folder-name>/ (gitignored). A manifest (inputs,
HEAD, window list) goes to evaluation/2.2b/<input-folder-name>/overlay_clips_manifest.txt.

Usage:
    python3 evaluation/2.2b/make_overlay_clips.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import compare_com_models as cm  # noqa: E402
from metrics import (  # noqa: E402
    GAP_EXCLUDED,
    GAP_LOW_CONFIDENCE,
    GAP_MISSING_DETECTION,
    OCCLUSION_CONFIDENCE_THRESHOLD,
    SEX_TABLES,
    compute_velocity,
    segment_model_joints,
)
from validation.analyze_occlusion_gaps import _git, load_exclusions, sha256_file  # noqa: E402

PAD_FRAMES = 15
GOOD_CONF = 0.7  # drawing colour break only
COL_PROXY = (255, 0, 255)   # BGR magenta
COL_SEG = (255, 255, 0)     # BGR cyan
COL_GOOD, COL_MID, COL_BAD = (0, 200, 0), (0, 165, 255), (0, 0, 255)
COL_EXCLUDED = (200, 60, 150)
REASON_TEXT = {GAP_LOW_CONFIDENCE: "low confidence", GAP_MISSING_DETECTION: "detector drop",
               GAP_EXCLUDED: "excluded by list"}

# edges among the 14 joints the segment model uses (overlay drawing only)
EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ear", "right_ear"),
]


def window_for(frame_from: int, frame_to: int, pad: int, n_frames: int) -> tuple[int, int, bool, bool]:
    """(start, end, clamped_at_start, clamped_at_end): the step padded by `pad` frames each side,
    limited to the frames that exist (0 .. n_frames-1)."""
    start, end = frame_from - pad, frame_to + pad
    return max(0, start), min(n_frames - 1, end), start < 0, end > n_frames - 1


def load_raw(csv_path: Path) -> dict[int, dict[str, tuple[float, float, float]]]:
    """frame -> joint -> (x, y, confidence) for the 14 segment-model joints, as in the keypoint
    CSV (including joints below 0.5, which the series treat as gaps)."""
    needed = set(segment_model_joints())
    out: dict[int, dict[str, tuple[float, float, float]]] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["keypoint_name"] in needed:
                out.setdefault(int(r["frame_idx"]), {})[r["keypoint_name"]] = (
                    float(r["x"]), float(r["y"]), float(r["confidence"]))
    return out


def _conf_colour(conf: float) -> tuple[int, int, int]:
    if conf < OCCLUSION_CONFIDENCE_THRESHOLD:
        return COL_BAD
    return COL_GOOD if conf >= GOOD_CONF else COL_MID


def _fmt_speed(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f} px/s"


def draw_frame(frame, c: "cm.ClipComparison", raw, t: int, sex: str, speeds: tuple,
               info: dict | None = None):
    import cv2
    h, w = frame.shape[:2]
    scale = max(w, h) / 1920.0
    thick = max(2, int(3 * scale))
    font = 1.0 * scale + 0.3
    seg = c.seg[sex]
    reason = c.proxy.gap_reasons[t]
    banner = None

    if reason == GAP_EXCLUDED:
        banner = ("EXCLUDED (wrong-person lock): keypoints not drawn", COL_EXCLUDED)
    elif t not in raw:
        banner = ("NO DETECTION (no person box)", COL_BAD)
    else:
        pts = raw[t]
        for a, b in EDGES:
            if a in pts and b in pts:
                col = _conf_colour(min(pts[a][2], pts[b][2]))
                cv2.line(frame, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), col, thick)
        r = int(7 * scale) + 3
        for name, (x, y, conf) in pts.items():
            if conf < OCCLUSION_CONFIDENCE_THRESHOLD:
                R = r * 2
                cv2.line(frame, (int(x) - R, int(y) - R), (int(x) + R, int(y) + R), COL_BAD, thick + 1)
                cv2.line(frame, (int(x) - R, int(y) + R), (int(x) + R, int(y) - R), COL_BAD, thick + 1)
            else:
                cv2.circle(frame, (int(x), int(y)), r, _conf_colour(conf), -1)
        for pos, col, label, dy in ((c.proxy.positions[t], COL_PROXY, "proxy", -1),
                                    (seg.positions[t], COL_SEG, "segment", 1)):
            if pos is None:
                continue
            x, y, R = int(pos[0]), int(pos[1]), int(14 * scale) + 5
            cv2.circle(frame, (x, y), R, col, -1)
            cv2.circle(frame, (x, y), R + 3, (0, 0, 0), max(1, thick - 1))
            # proxy label goes left of its marker, segment label right, so close markers stay readable
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, thick)[0][0]
            tx = x - R - 8 - tw if dy < 0 else x + R + 8
            cv2.putText(frame, label, (tx, y + dy * R), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8, col, thick)

    if banner:
        text, colour = banner
        cv2.rectangle(frame, (0, 0), (w, int(120 * scale) + 30), colour, -1)
        cv2.putText(frame, text, (int(20 * scale) + 10, int(90 * scale) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick)

    # legend (top right)
    lines = [(COL_PROXY, "CoM proxy (trunk only)"), (COL_SEG, f"CoM segment model ({sex})"),
             (COL_GOOD, "joint conf >= 0.7"), (COL_MID, "joint conf 0.5 - 0.7"),
             (COL_BAD, "X = conf < 0.5 (gap)")]
    lh, lw = int(46 * scale) + 8, int(640 * scale) + 20
    x0, y0 = w - lw - 10, int(140 * scale) + 40
    cv2.rectangle(frame, (x0, y0), (w - 10, y0 + lh * len(lines) + 12), (30, 30, 30), -1)
    for i, (col, text) in enumerate(lines):
        yy = y0 + 10 + lh * i + lh // 2
        cv2.circle(frame, (x0 + 22, yy), int(11 * scale) + 3, col, -1)
        cv2.putText(frame, text, (x0 + 52, yy + int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    font * 0.65, (255, 255, 255), max(1, thick - 1))

    # corner text (bottom left)
    vp, vs = speeds
    ts = c.proxy.timestamps[t]
    out = [f"{c.clip}  frame {t}  t={ts:.2f}s",
           f"speed: proxy {_fmt_speed(vp)}  |  segment {_fmt_speed(vs)}"]
    if seg.positions[t] is None and reason not in (GAP_EXCLUDED, GAP_MISSING_DETECTION) and t in raw:
        out.append(f"segment model: None ({REASON_TEXT.get(seg.gap_reasons[t], 'unspecified')})")
    if info and t == info["frame_to"]:
        out.append(f"TARGET STEP rank {info['rank']}: frames {info['frame_from']}->{info['frame_to']}")
    elif info and t == info["frame_from"]:
        out.append(f"target step rank {info['rank']} starts: {info['frame_from']}->{info['frame_to']}")
    y = h - int(30 * scale) - 20 - int(60 * scale) * (len(out) - 1)
    for line in out:
        cv2.putText(frame, line, (int(20 * scale) + 10, y), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8,
                    (255, 255, 255), thick + 1)
        y += int(60 * scale)
    return frame


def read_top_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def make_clip(video: Path, out_path: Path, c: "cm.ClipComparison", raw, sex: str, row: dict,
              start: int, end: int) -> int:
    """Write frames start..end of `video` with the overlay; returns frames written."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"could not open {video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or c.fps
    for _ in range(start):          # decode sequentially: seeking is not frame-exact on all codecs
        if not cap.grab():
            break
    vp, vs = compute_velocity(c.proxy), compute_velocity(c.seg[sex])
    info = {"rank": int(row["rank"]), "frame_from": int(row["frame_from"]), "frame_to": int(row["frame_to"])}
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    for t in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(draw_frame(frame, c, raw, t, sex, (vp[t], vs[t]), info))
        n += 1
    cap.release()
    writer.release()
    return n


def build_manifest(windows: list[dict], input_dir: Path, top_csv: Path, exclusions_path: Path,
                   footage_dir: Path, sex: str, pad: int, today: str | None = None) -> str:
    L = ["Overlay clips manifest (evaluation/2.2b/make_overlay_clips.py)",
         f"date: {today or date.today().isoformat()}",
         f"input folder: {input_dir.as_posix()}",
         f"windows from: {top_csv.as_posix()} sha256 {sha256_file(top_csv) if top_csv.exists() else 'not found'}",
         f"segment-model sex table: {sex}",
         f"padding: {pad} frames each side (clamped at clip edges); full source resolution",
         "colours: proxy CoM magenta, segment-model CoM cyan; joint colour by confidence "
         "(green >= 0.7, orange 0.5-0.7, red X < 0.5)",
         "", "== Frame exclusion file"]
    if exclusions_path.exists():
        L += [f"path: {exclusions_path.as_posix()}", "contents (verbatim):",
              exclusions_path.read_text().rstrip("\n")]
    else:
        L.append("none")
    L += ["", "== Code (at run time)", f"git HEAD: {_git('rev-parse', 'HEAD')}"]
    dirty = _git("status", "--porcelain", "--", "metrics.py", "evaluation/2.2b/make_overlay_clips.py",
                 "evaluation/2.2b/compare_com_models.py", "validation/frame_exclusions.csv")
    L.append("metrics/scripts/exclusions modified since HEAD: "
             + ("unknown" if dirty == "unknown" else ("yes" if dirty else "no")))
    L += ["", "== Source videos"]
    for clip in sorted({w["clip"] for w in windows}):
        vp = footage_dir / clip
        L.append(f"{clip}: sha256 {sha256_file(vp) if vp.exists() else 'video not found'}")
    L += ["", "== Windows (rank, clip, step frames, window frames, clamped, n frames, file, bytes)"]
    for w in windows:
        clamp = ",".join(x for x, f in (("start", w["clamped_start"]), ("end", w["clamped_end"])) if f) or "no"
        L.append(f"{w['rank']}, {w['clip']}, {w['frame_from']}->{w['frame_to']}, "
                 f"{w['start']}-{w['end']}, clamped: {clamp}, {w['n_written']}, {w['file']}, {w['bytes']}")
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/output_full"))
    parser.add_argument("--footage-dir", type=Path, default=Path("footage/test"))
    parser.add_argument("--eval-dir", type=Path, default=Path("evaluation/2.2b"))
    parser.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"))
    parser.add_argument("--sex", choices=SEX_TABLES, default="male")
    parser.add_argument("--pad", type=int, default=PAD_FRAMES)
    args = parser.parse_args()

    folder = args.input_dir.resolve().name
    top_csv = args.eval_dir / folder / "top_speed_differences.csv"
    rows = read_top_rows(top_csv)
    exclusions = load_exclusions(args.exclusions)
    out_dir = args.eval_dir / "overlays" / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, tuple] = {}
    windows = []
    for row in rows:
        clip = row["clip"]
        if clip not in cache:
            csv_path = args.input_dir / f"{Path(clip).stem}_keypoints.csv"
            cache[clip] = (cm.compare_clip(csv_path, exclusions.get(clip)), load_raw(csv_path))
        c, raw = cache[clip]
        start, end, cs, ce = window_for(int(row["frame_from"]), int(row["frame_to"]), args.pad, c.n_frames)
        name = f"rank{int(row['rank']):02d}_{Path(clip).stem}_f{start}-{end}.mp4"
        video = args.footage_dir / clip
        print(f"rank {row['rank']}: {clip} frames {start}-{end} ...")
        n = make_clip(video, out_dir / name, c, raw, args.sex, row, start, end)
        windows.append({"rank": row["rank"], "clip": clip, "frame_from": row["frame_from"],
                        "frame_to": row["frame_to"], "start": start, "end": end,
                        "clamped_start": cs, "clamped_end": ce, "n_written": n,
                        "file": f"evaluation/2.2b/overlays/{folder}/{name}",
                        "bytes": (out_dir / name).stat().st_size})
    manifest = build_manifest(windows, args.input_dir, top_csv, args.exclusions, args.footage_dir,
                              args.sex, args.pad)
    (args.eval_dir / folder / "overlay_clips_manifest.txt").write_text(manifest)
    print(f"Videos in {out_dir}; manifest in {args.eval_dir / folder}")


if __name__ == "__main__":
    main()
