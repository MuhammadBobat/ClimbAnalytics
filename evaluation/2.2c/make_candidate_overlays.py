"""
Synchronised panel-grid overlay videos for the ten windows of
evaluation/2.2b/output_full/top_speed_differences.csv (the ten largest proxy-vs-F speed
differences, D-025), one video per window (DECISIONS.md D-026). Raw, unsmoothed, no gap-filling.

One panel per candidate, the same frame in every panel. Default panels: the survivors of the
triage (evaluation/2.2c/output_full/triage.csv) plus A and F, which are always kept; any
candidate, including a dropped one, can be rendered with --panels. Each panel is cropped and
zoomed to the climber with ONE fixed square crop per window: the box around every candidate's
position (all thirteen, whatever the panel list) over the window's frames, plus a margin of
MARGIN_TORSOS median torso lengths each side, made square and kept inside the frame. Per panel:
grey skeleton; joint dots shaded by confidence band (grey steps at 0.5/0.6/0.7/0.8, red X < 0.5,
a gap, D-007); that panel's candidate as one large coloured marker with a fading trail of its
last TRAIL frames; in every panel except A's, the proxy A as a small hollow grey ring; "undefined"
in red when the candidate is undefined in that frame; title = candidate name and current-frame
speed. The excluded-frame and no-detection banners are kept. Windows are the 2.2b ones (rule and
padding loaded from 2.2b make_overlay_clips.py, not copied); they were chosen by proxy-vs-F
disagreement, so they over-represent F's problems.

Videos go to evaluation/2.2c/overlays/<input-folder-name>/ (gitignored); the manifest to
evaluation/2.2c/<input-folder-name>/overlay_clips_manifest.txt.

Usage:
    python3 evaluation/2.2c/make_candidate_overlays.py
    python3 evaluation/2.2c/make_candidate_overlays.py --panels A_proxy,B_trunk_point,F_full --tag _B
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import statistics
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

from metrics import GAP_EXCLUDED, OCCLUSION_CONFIDENCE_THRESHOLD  # noqa: E402
from validation.analyze_occlusion_gaps import _git, load_exclusions, sha256_file  # noqa: E402

PANEL = 480            # panel image side, px
TITLE_H, HEADER_H, FOOTER_H = 44, 52, 44
TRAIL = 10             # frames in the fading trail (drawing choice, user-specified)
MARGIN_TORSOS = 1.5    # crop margin, in median torso lengths each side (drawing choice, D-026)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
COL_BAD, COL_EXCLUDED, COL_LINE, COL_RING = (0, 0, 255), (200, 60, 150), (150, 150, 150), (190, 190, 190)

# One colour per candidate (BGR); the first seven are Okabe-Ito. Panels are separate, so colours
# only need to be consistent per candidate.
COLOURS = {
    cc.L_A: (167, 121, 204), cc.L_B: (0, 159, 230), cc.L_C: (115, 158, 0), cc.L_D: (233, 180, 86),
    "E_c0.5": (66, 228, 240), "E_c0.6": (0, 94, 213), "E_c0.7": (178, 114, 0), "E_c0.8": (0, 0, 0),
    "EdL_c0.5": (75, 180, 60), "EdL_c0.6": (203, 192, 255), "EdL_c0.7": (128, 128, 0),
    "EdL_c0.8": (42, 42, 165), cc.L_F: (255, 255, 255),
}
BANDS = [(0.5, 0.6, (90, 90, 90)), (0.6, 0.7, (125, 125, 125)), (0.7, 0.8, (170, 170, 170)),
         (0.8, float("inf"), (225, 225, 225))]


def band_colour(conf: float):
    if conf < OCCLUSION_CONFIDENCE_THRESHOLD:
        return COL_BAD
    return next(col for lo, hi, col in BANDS if lo <= conf < hi)


def outline_for(colour) -> tuple:
    """White outline for dark colours, black for light ones."""
    return WHITE if sum(colour) < 330 else BLACK


# --- panel list ---------------------------------------------------------------------------------

def default_panels(triage_path: Path) -> list:
    """Survivors of the triage plus A and F, in cc.LABELS order."""
    with open(triage_path, newline="") as f:
        keep = {r["candidate"] for r in csv.DictReader(f) if r["dropped"] == "no"}
    keep |= {cc.L_A, cc.L_F}
    return [l for l in cc.LABELS if l in keep]


def parse_panels(arg: str) -> list:
    """Comma-separated candidate labels, validated, order kept, duplicates removed."""
    names = list(dict.fromkeys(x.strip() for x in arg.split(",") if x.strip()))
    bad = [n for n in names if n not in cc.LABELS]
    if bad or not names:
        raise ValueError(f"unknown panel(s) {bad}; choose from {', '.join(cc.LABELS)}")
    return names


def grid_shape(n: int) -> tuple[int, int]:
    """(columns, rows) for n panels."""
    cols = max(1, math.ceil(math.sqrt(n)))
    return cols, math.ceil(n / cols)


# --- crop ---------------------------------------------------------------------------------------

def window_crop(c, start: int, end: int, width: int, height: int,
                margin_torsos: float = MARGIN_TORSOS) -> tuple[int, int, int]:
    """(x0, y0, side): one fixed square crop for the window from the union of all candidates'
    positions over frames start..end plus a margin, inside the frame."""
    pts = [p for l in cc.LABELS for t in range(start, end + 1)
           if (p := c.series[l].positions[t]) is not None]
    full = min(width, height)
    if not pts:
        return 0, 0, full
    torsos = [L for L in c.torso[start:end + 1] if L]
    margin = margin_torsos * (statistics.median(torsos) if torsos else 200.0)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    side = int(min(max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * margin, full))
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    x0 = int(min(max(cx - side / 2, 0), width - side))
    y0 = int(min(max(cy - side / 2, 0), height - side))
    return x0, y0, side


# --- drawing ------------------------------------------------------------------------------------

def _fmt_speed(v) -> str:
    return "n/a" if v is None else f"{v:.0f} px/s"


def draw_panel(frame, c, raw, t: int, label: str, crop: tuple) -> "np.ndarray":
    """One panel (PANEL x PANEL image plus a title bar) for candidate `label` at frame t."""
    import cv2
    import numpy as np
    x0, y0, side = crop
    s = PANEL / side
    img = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (PANEL, PANEL), interpolation=cv2.INTER_AREA)

    def tp(p):
        return (int(round((p[0] - x0) * s)), int(round((p[1] - y0) * s)))

    reason = c.series[cc.L_A].gap_reasons[t]
    banner = None
    if reason == GAP_EXCLUDED:
        banner = ("EXCLUDED (wrong-person lock)", COL_EXCLUDED)
    elif t not in raw:
        banner = ("NO DETECTION (no person box)", COL_BAD)
    else:
        pts = raw[t]
        for a, b in oc.EDGES:
            if a in pts and b in pts:
                cv2.line(img, tp(pts[a]), tp(pts[b]), COL_LINE, 2)
        for x, y, conf in pts.values():
            q = tp((x, y))
            if conf < OCCLUSION_CONFIDENCE_THRESHOLD:
                cv2.line(img, (q[0] - 7, q[1] - 7), (q[0] + 7, q[1] + 7), COL_BAD, 2)
                cv2.line(img, (q[0] - 7, q[1] + 7), (q[0] + 7, q[1] - 7), COL_BAD, 2)
            else:
                cv2.circle(img, q, 5, BLACK, -1)
                cv2.circle(img, q, 4, band_colour(conf), -1)

    pos = c.series[label].positions
    col = COLOURS[label]
    if label != cc.L_A and c.series[cc.L_A].positions[t] is not None:
        cv2.circle(img, tp(c.series[cc.L_A].positions[t]), 7, COL_RING, 2)
    if pos[t] is not None:
        q = tp(pos[t])
        cv2.circle(img, q, 15, outline_for(col), -1)
        cv2.circle(img, q, 12, col, -1)
    else:
        cv2.putText(img, "undefined", (PANEL // 2 - 95, PANEL - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, COL_BAD, 3)
    # fading trail of the last TRAIL frames, drawn over the marker so small motion stays visible;
    # newest strongest; it breaks at undefined frames
    pts_k = [tp(pos[t - k]) if t - k >= 0 and pos[t - k] is not None else None
             for k in range(TRAIL + 1)]
    for k in range(1, TRAIL + 1):
        alpha = 0.9 * (1 - (k - 1) / TRAIL)
        layer = img.copy()
        if pts_k[k] is not None:
            if pts_k[k - 1] is not None:
                cv2.line(layer, pts_k[k - 1], pts_k[k], outline_for(col), 4)
                cv2.line(layer, pts_k[k - 1], pts_k[k], col, 2)
            cv2.circle(layer, pts_k[k], 5, outline_for(col), -1)
            cv2.circle(layer, pts_k[k], 3, col, -1)
        img = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
    if banner:
        cv2.rectangle(img, (0, 0), (PANEL, 34), banner[1], -1)
        cv2.putText(img, banner[0], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    title = np.full((TITLE_H, PANEL, 3), 30, dtype=np.uint8)
    cv2.circle(title, (22, TITLE_H // 2), 9, outline_for(col), -1)
    cv2.circle(title, (22, TITLE_H // 2), 7, col, -1)
    cv2.putText(title, f"{label}   {_fmt_speed(c.speeds[label][t])}", (42, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    return np.vstack([title, img])


def draw_grid(frame, c, raw, t: int, panels: list, crop: tuple, info: dict | None = None):
    """The whole grid image for frame t: header, panels row-major, footer."""
    import cv2
    import numpy as np
    cols, rows = grid_shape(len(panels))
    ph = PANEL + TITLE_H
    W, H = cols * PANEL, HEADER_H + rows * ph + FOOTER_H
    grid = np.full((H, W, 3), 20, dtype=np.uint8)
    for i, label in enumerate(panels):
        r, k = divmod(i, cols)
        grid[HEADER_H + r * ph:HEADER_H + (r + 1) * ph, k * PANEL:(k + 1) * PANEL] = \
            draw_panel(frame, c, raw, t, label, crop)
    head = f"{c.clip}  frame {t}  t={c.series[cc.L_A].timestamps[t]:.2f}s"
    if info and t == info["frame_to"]:
        head += f"   TARGET STEP rank {info['rank']}: frames {info['frame_from']}->{info['frame_to']}"
    elif info and t == info["frame_from"]:
        head += f"   target step rank {info['rank']} starts: {info['frame_from']}->{info['frame_to']}"
    cv2.putText(grid, head, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    y = H - FOOTER_H // 2
    x = 10
    for lo, hi, colr in BANDS:
        cv2.circle(grid, (x + 6, y), 6, colr, -1)
        text = f"{lo:.1f}-{hi:.1f}" if hi != float("inf") else ">= 0.8"
        cv2.putText(grid, text, (x + 18, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        x += 105
    cv2.putText(grid, "joint conf  |  red X < 0.5  |  grey ring = A proxy  |  trail = last "
                f"{TRAIL} frames", (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return grid


def make_clip(video: Path, out_path: Path, c, raw, row: dict, start: int, end: int,
              panels: list, crop: tuple) -> int:
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or c.fps
    for _ in range(start):          # decode sequentially: seeking is not frame-exact on all codecs
        if not cap.grab():
            break
    info = {"rank": int(row["rank"]), "frame_from": int(row["frame_from"]), "frame_to": int(row["frame_to"])}
    cols, rows = grid_shape(len(panels))
    size = (cols * PANEL, HEADER_H + rows * (PANEL + TITLE_H) + FOOTER_H)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    n = 0
    for t in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(draw_grid(frame, c, raw, t, panels, crop, info))
        n += 1
    cap.release()
    writer.release()
    return n


def build_manifest(windows, input_dir, top_csv, exclusions_path, footage_dir, folds, pad, panels,
                   triage_path, default, today=None) -> str:
    L = ["Candidate overlay clips manifest (evaluation/2.2c/make_candidate_overlays.py)",
         f"date: {today or date.today().isoformat()}",
         f"input folder: {input_dir.as_posix()}",
         f"windows from: {top_csv.as_posix()} sha256 {sha256_file(top_csv) if top_csv.exists() else 'not found'}",
         "windows chosen by proxy-vs-F speed difference (D-025): not representative of the clips",
         f"padding: {pad} frames each side (clamped at clip edges); full source resolution",
         "layout: synchronised grid, one panel per candidate, same frame in each, one video per window",
         "panels: " + ", ".join(panels)
         + ("  (default: triage survivors plus A and F)" if default else "  (set with --panels)"),
         f"triage file: {triage_path.as_posix()} sha256 "
         f"{sha256_file(triage_path) if triage_path.exists() else 'not found'}",
         f"panel image {PANEL} x {PANEL} px plus a {TITLE_H} px title bar; crop: one fixed square per "
         f"window from the union of all thirteen candidates' positions over the window's frames plus "
         f"a margin of {MARGIN_TORSOS} median torso lengths each side (drawing choice)",
         f"marker: one large coloured disc per panel, fading trail of the last {TRAIL} frames; grey "
         "hollow ring = A (every panel but A's); 'undefined' in red when the candidate is undefined",
         "colours (BGR): " + "; ".join(f"{l} {COLOURS[l]}" for l in panels),
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
                 "evaluation/2.2c/make_candidate_overlays.py", "validation/frame_exclusions.csv")
    L.append("metrics/scripts/exclusions modified since HEAD: "
             + ("unknown" if dirty == "unknown" else ("yes" if dirty else "no")))
    L += ["", "== Source videos"]
    for clip in sorted({w["clip"] for w in windows}):
        vp = footage_dir / clip
        L.append(f"{clip}: sha256 {sha256_file(vp) if vp.exists() else 'video not found'}")
    L += ["", "== Windows (rank, clip, step frames, window frames, clamped, crop x0 y0 side, "
          "n frames, file, bytes)"]
    for w in windows:
        clamp = ",".join(x for x, f in (("start", w["clamped_start"]), ("end", w["clamped_end"])) if f) or "no"
        L.append(f"{w['rank']}, {w['clip']}, {w['frame_from']}->{w['frame_to']}, "
                 f"{w['start']}-{w['end']}, clamped: {clamp}, crop {w['crop'][0]} {w['crop'][1]} "
                 f"{w['crop'][2]}, {w['n_written']}, {w['file']}, {w['bytes']}")
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
    parser.add_argument("--panels", default=None,
                        help="comma-separated candidate labels (default: triage survivors plus A and F)")
    parser.add_argument("--tag", default="", help="appended to the video names, e.g. _B")
    args = parser.parse_args()

    folder = args.input_dir.resolve().name
    triage_path = args.out_dir / folder / "triage.csv"
    default = args.panels is None
    panels = default_panels(triage_path) if default else parse_panels(args.panels)
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
        import cv2
        cap = cv2.VideoCapture(str(args.footage_dir / clip))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        crop = window_crop(c, start, end, width, height)
        name = f"grid_rank{int(row['rank']):02d}_{Path(clip).stem}_f{start}-{end}{args.tag}.mp4"
        print(f"rank {row['rank']}: {clip} frames {start}-{end} crop {crop} ...")
        n = make_clip(args.footage_dir / clip, out_dir / name, c, raws[clip], row, start, end, panels, crop)
        windows.append({"rank": row["rank"], "clip": clip, "frame_from": row["frame_from"],
                        "frame_to": row["frame_to"], "start": start, "end": end,
                        "clamped_start": cs, "clamped_end": ce, "n_written": n, "crop": crop,
                        "file": f"evaluation/2.2c/overlays/{folder}/{name}",
                        "bytes": (out_dir / name).stat().st_size})
    (args.out_dir / folder).mkdir(parents=True, exist_ok=True)
    mpath = args.out_dir / folder / f"overlay_clips_manifest{args.tag}.txt"
    mpath.write_text(build_manifest(windows, args.input_dir, args.windows_from, args.exclusions,
                                    args.footage_dir, folds, args.pad, panels, triage_path, default))
    print(f"Videos in {out_dir}; manifest {mpath}")


if __name__ == "__main__":
    main()
