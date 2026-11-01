"""
Short overlay clips for the ten windows of evaluation/2.2b/output_full/top_speed_differences.csv
(the ten largest proxy-vs-F speed differences, D-025), drawing the nine candidate CoMs of
DECISIONS.md D-026 (A, B, C, D, E at 0.5/0.6/0.7/0.8, F) in distinct colours and marker shapes,
with a legend that also shows each candidate's current speed ("n/a" where undefined). The EdL
variants are not drawn (nine markers). Window rule, padding and the keypoint reading are the 2.2b
ones (make_overlay_clips.py is loaded from 2.2b, not copied). Raw, unsmoothed, no gap-filling.

The windows were chosen by proxy-vs-F disagreement, so they over-represent F's problems.

Skeleton lines are grey. Joint dots are shaded by confidence in the bands of the E sweep
(0.5-0.6, 0.6-0.7, 0.7-0.8, >= 0.8, darker = lower); a red X is a joint below 0.5 (a gap, D-007).
Videos go to evaluation/2.2c/overlays/<input-folder-name>/ (gitignored); the manifest to
evaluation/2.2c/<input-folder-name>/overlay_clips_manifest.txt.

Usage:
    python3 evaluation/2.2c/make_candidate_overlays.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

import compare_com_candidates as cc  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "overlay_2_2b", REPO_ROOT / "evaluation" / "2.2b" / "make_overlay_clips.py")
oc = importlib.util.module_from_spec(_spec)
sys.modules["overlay_2_2b"] = oc
_spec.loader.exec_module(oc)

from metrics import (  # noqa: E402
    GAP_EXCLUDED,
    GAP_MISSING_DETECTION,
    OCCLUSION_CONFIDENCE_THRESHOLD,
)
from validation.analyze_occlusion_gaps import _git, load_exclusions, sha256_file  # noqa: E402

# Okabe-Ito colours (colour-blind safe) as BGR; label -> (colour, marker, size factor)
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
STYLE = {
    cc.L_A: ((167, 121, 204), "circle", 1.0),            # reddish purple
    cc.L_B: ((0, 159, 230), "square", 1.0),               # orange
    cc.L_C: ((115, 158, 0), "diamond", 1.1),              # bluish green
    cc.L_D: ((233, 180, 86), "triangle_up", 1.1),         # sky blue
    "E_c0.5": ((66, 228, 240), "triangle_down", 1.1),     # yellow
    "E_c0.6": ((0, 94, 213), "cross", 1.1),               # vermillion
    "E_c0.7": ((178, 114, 0), "tilted_cross", 1.1),       # blue
    "E_c0.8": ((0, 0, 0), "star", 1.1),                   # black
    cc.L_F: ((255, 255, 255), "circle", 1.5),             # white, drawn first (largest)
}
DRAW_ORDER = [cc.L_F] + [l for l in cc.OVERLAY_LABELS if l != cc.L_F]
LEGEND_NAME = {
    cc.L_A: "A proxy (trunk)", cc.L_B: "B trunk point", cc.L_C: "C no head",
    cc.L_D: "D trunk+legs", "E_c0.5": "E cut-off 0.5", "E_c0.6": "E cut-off 0.6",
    "E_c0.7": "E cut-off 0.7", "E_c0.8": "E cut-off 0.8", cc.L_F: "F full model",
}
BANDS = [(0.5, 0.6, (90, 90, 90)), (0.6, 0.7, (125, 125, 125)), (0.7, 0.8, (170, 170, 170)),
         (0.8, float('inf'), (225, 225, 225))]
COL_BAD, COL_LINE, COL_EXCLUDED = (0, 0, 255), (150, 150, 150), (200, 60, 150)


def band_colour(conf: float):
    if conf < OCCLUSION_CONFIDENCE_THRESHOLD:
        return COL_BAD
    return next(col for lo, hi, col in BANDS if lo <= conf < hi)


def draw_marker(frame, pos, label: str, scale: float, size: float = 1.0) -> None:
    import cv2
    colour, kind, k = STYLE[label]
    x, y = int(pos[0]), int(pos[1])
    r = int((11 * scale + 4) * k * size)
    thick = max(2, int(3 * scale))
    if kind == "circle":
        cv2.circle(frame, (x, y), r + 3, BLACK, -1)
        cv2.circle(frame, (x, y), r, colour, -1)
        return
    types = {"square": cv2.MARKER_SQUARE, "diamond": cv2.MARKER_DIAMOND,
             "triangle_up": cv2.MARKER_TRIANGLE_UP, "triangle_down": cv2.MARKER_TRIANGLE_DOWN,
             "cross": cv2.MARKER_CROSS, "tilted_cross": cv2.MARKER_TILTED_CROSS,
             "star": cv2.MARKER_STAR}
    outline = WHITE if colour == BLACK else BLACK
    cv2.drawMarker(frame, (x, y), outline, types[kind], 2 * r, thick + 3)
    cv2.drawMarker(frame, (x, y), colour, types[kind], 2 * r, thick)


def _fmt_speed(v) -> str:
    return "n/a" if v is None else f"{v:.0f}"


def draw_frame(frame, c: "cc.ClipData", raw, t: int, info: dict | None = None):
    import cv2
    h, w = frame.shape[:2]
    scale = max(w, h) / 1920.0
    thick = max(2, int(3 * scale))
    font = 1.0 * scale + 0.3
    reason = c.series[cc.L_A].gap_reasons[t]
    banner = None
    if reason == GAP_EXCLUDED:
        banner = ("EXCLUDED (wrong-person lock): keypoints not drawn", COL_EXCLUDED)
    elif t not in raw:
        banner = ("NO DETECTION (no person box)", COL_BAD)
    else:
        pts = raw[t]
        for a, b in oc.EDGES:
            if a in pts and b in pts:
                cv2.line(frame, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])),
                         COL_LINE, thick)
        r = int(6 * scale) + 3
        for x, y, conf in pts.values():
            if conf < OCCLUSION_CONFIDENCE_THRESHOLD:
                R = r * 2
                cv2.line(frame, (int(x) - R, int(y) - R), (int(x) + R, int(y) + R), COL_BAD, thick + 1)
                cv2.line(frame, (int(x) - R, int(y) + R), (int(x) + R, int(y) - R), COL_BAD, thick + 1)
            else:
                cv2.circle(frame, (int(x), int(y)), r + 1, BLACK, -1)
                cv2.circle(frame, (int(x), int(y)), r, band_colour(conf), -1)
        for label in DRAW_ORDER:
            pos = c.series[label].positions[t]
            if pos is not None:
                draw_marker(frame, pos, label, scale)
    if banner:
        text, colour = banner
        cv2.rectangle(frame, (0, 0), (w, int(120 * scale) + 30), colour, -1)
        cv2.putText(frame, text, (int(20 * scale) + 10, int(90 * scale) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font, WHITE, thick)

    # legend (top right): candidates with current speed, then the joint-confidence bands
    rows = [(l, f"{LEGEND_NAME[l]}  {_fmt_speed(c.speeds[l][t])} px/s") for l in cc.OVERLAY_LABELS]
    bands = [(col, f"joint conf {lo:.1f}-{hi:.1f}" if hi != float("inf") else "joint conf >= 0.8")
             for lo, hi, col in BANDS] + [(COL_BAD, "X = conf < 0.5 (gap)")]
    lh, lw = int(40 * scale) + 8, int(720 * scale) + 20
    x0, y0 = w - lw - 10, int(140 * scale) + 40
    cv2.rectangle(frame, (x0, y0), (w - 10, y0 + lh * (len(rows) + len(bands)) + 12), (30, 30, 30), -1)
    for i, (label, text) in enumerate(rows):
        yy = y0 + 10 + lh * i + lh // 2
        draw_marker(frame, (x0 + 26, yy), label, scale, 0.6)
        cv2.putText(frame, text, (x0 + 62, yy + int(9 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    font * 0.55, WHITE, max(1, thick - 1))
    for j, (col, text) in enumerate(bands):
        yy = y0 + 10 + lh * (len(rows) + j) + lh // 2
        cv2.circle(frame, (x0 + 26, yy), int(8 * scale) + 3, col, -1)
        cv2.putText(frame, text, (x0 + 62, yy + int(9 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    font * 0.55, WHITE, max(1, thick - 1))

    out = [f"{c.clip}  frame {t}  t={c.series[cc.L_A].timestamps[t]:.2f}s"]
    if info and t == info["frame_to"]:
        out.append(f"TARGET STEP rank {info['rank']}: frames {info['frame_from']}->{info['frame_to']}")
    elif info and t == info["frame_from"]:
        out.append(f"target step rank {info['rank']} starts: {info['frame_from']}->{info['frame_to']}")
    y = h - int(30 * scale) - 20 - int(60 * scale) * (len(out) - 1)
    for line in out:
        cv2.putText(frame, line, (int(20 * scale) + 10, y), cv2.FONT_HERSHEY_SIMPLEX, font * 0.8,
                    WHITE, thick + 1)
        y += int(60 * scale)
    return frame


def make_clip(video: Path, out_path: Path, c, raw, row: dict, start: int, end: int) -> int:
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"could not open {video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or c.fps
    for _ in range(start):          # decode sequentially: seeking is not frame-exact on all codecs
        if not cap.grab():
            break
    info = {"rank": int(row["rank"]), "frame_from": int(row["frame_from"]), "frame_to": int(row["frame_to"])}
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    for t in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(draw_frame(frame, c, raw, t, info))
        n += 1
    cap.release()
    writer.release()
    return n


def build_manifest(windows, input_dir, top_csv, exclusions_path, footage_dir, folds, pad,
                   today=None) -> str:
    L = ["Candidate overlay clips manifest (evaluation/2.2c/make_candidate_overlays.py)",
         f"date: {today or date.today().isoformat()}",
         f"input folder: {input_dir.as_posix()}",
         f"windows from: {top_csv.as_posix()} sha256 {sha256_file(top_csv) if top_csv.exists() else 'not found'}",
         "windows chosen by proxy-vs-F speed difference (D-025): not representative of the clips",
         f"padding: {pad} frames each side (clamped at clip edges); full source resolution",
         "candidates drawn (nine): " + ", ".join(cc.OVERLAY_LABELS) + "; EdL variants not drawn",
         "markers: " + "; ".join(f"{LEGEND_NAME[l]} = {STYLE[l][1]} BGR{STYLE[l][0]}" for l in cc.OVERLAY_LABELS),
         "joint dots: grey shades by confidence band 0.5-0.6 / 0.6-0.7 / 0.7-0.8 / >= 0.8, red X < 0.5",
         "B and E use the w of the fold in which the drawn clip was held out: "
         + ", ".join(f"{r['held_out_clip']} {r['w']:.6f}" for r in folds),
         "", "== Frame exclusion file"]
    if exclusions_path.exists():
        L += [f"path: {exclusions_path.as_posix()}", "contents (verbatim):",
              exclusions_path.read_text().rstrip("\n")]
    else:
        L.append("none")
    L += ["", "== Code (at run time)", f"git HEAD: {_git('rev-parse', 'HEAD')}"]
    dirty = _git("status", "--porcelain", "--", "metrics.py", "evaluation/2.2c/compare_com_candidates.py",
                 "evaluation/2.2c/make_candidate_overlays.py",
                 "validation/frame_exclusions.csv")
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
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation/2.2c"))
    parser.add_argument("--windows-from", type=Path,
                        default=Path("evaluation/2.2b/output_full/top_speed_differences.csv"))
    parser.add_argument("--exclusions", type=Path, default=Path("validation/frame_exclusions.csv"))
    parser.add_argument("--sex", default="male")
    parser.add_argument("--pad", type=int, default=oc.PAD_FRAMES)
    args = parser.parse_args()

    folder = args.input_dir.resolve().name
    rows = oc.read_top_rows(args.windows_from)
    comps, folds = cc.run(args.input_dir, args.exclusions, args.sex)
    by_clip = {c.clip: c for c in comps}
    out_dir = args.out_dir / "overlays" / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    raws, windows = {}, []
    for row in rows:
        clip = row["clip"]
        c = by_clip[clip]
        if clip not in raws:
            raws[clip] = oc.load_raw(args.input_dir / f"{Path(clip).stem}_keypoints.csv")
        start, end, cs, ce = oc.window_for(int(row["frame_from"]), int(row["frame_to"]), args.pad, c.n_frames)
        name = f"rank{int(row['rank']):02d}_{Path(clip).stem}_f{start}-{end}.mp4"
        print(f"rank {row['rank']}: {clip} frames {start}-{end} ...")
        n = make_clip(args.footage_dir / clip, out_dir / name, c, raws[clip], row, start, end)
        windows.append({"rank": row["rank"], "clip": clip, "frame_from": row["frame_from"],
                        "frame_to": row["frame_to"], "start": start, "end": end,
                        "clamped_start": cs, "clamped_end": ce, "n_written": n,
                        "file": f"evaluation/2.2c/overlays/{folder}/{name}",
                        "bytes": (out_dir / name).stat().st_size})
    (args.out_dir / folder).mkdir(parents=True, exist_ok=True)
    (args.out_dir / folder / "overlay_clips_manifest.txt").write_text(build_manifest(
        windows, args.input_dir, args.windows_from, args.exclusions, args.footage_dir, folds, args.pad))
    print(f"Videos in {out_dir}; manifest in {args.out_dir / folder}")


if __name__ == "__main__":
    main()
