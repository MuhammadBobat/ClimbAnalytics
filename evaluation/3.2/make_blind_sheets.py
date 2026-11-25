"""
Blind contact sheets for DISSERTATION_NOTES gaps row 38 (DECISIONS.md D-030 addendum): on the real frames that the
D-028-era run filled with the spline (flag tier1) and the D-030 run fills with the Kalman, which fill looks closer to where
the joint is in the image? A VISUAL JUDGEMENT BY ONE PERSON, NOT AN ERROR MEASUREMENT: there is no ground truth.

Fixed before the sheets were made (the author specified the sets, the markers, the blinding and the rule on 2026-11-25; the
choices marked [assistant] are the assistant's and are open to the author before judging):
  candidates   every joint-frame flagged tier1 in the previous run (evaluation/3.2/output_full_lock1) and tier2 in the new one
               (evaluation/3.2/output_full_lock1_d030): 224 frames over the five clips.
  top set      the 10 largest old-vs-new disagreements (Euclidean px between the two filled positions); ties by clip, joint, frame.
               Note: two of them can be adjacent frames of one gap event; each frame is judged on its own sheet.
  random set   10 frames at random from the candidates NOT in the top set [assistant: so the two sets are disjoint], 2 per clip
               [assistant: to spread across the five clips], python random.Random(SEED_SAMPLE) = 20261125; sampling uses the
               candidates sorted by (clip, joint, frame).
  sheet order  both sets are shuffled (random.Random(SEED_ORDER) = 20261127) so a sheet number reveals neither the size of
               the disagreement nor the clip.
  markers      on the middle panel (the gap frame): A and B = the two fills in a RANDOM order per frame (random.Random(SEED_LABELS)
               = 20261126; A is the old spline on about half the sheets); R = the raw low-confidence ViTPose point (the same
               neutral label on every sheet; absent where the frame has no detection, which the sheet says). Side panels: the
               joint's confident observed position (small white ring) or "gap frame", two frames either side.
  key          the A/B assignment is written ONLY to KEY_do_not_open_before_judging.csv (with the disagreement and the rank). The
               sheets show no tier, method or distance.
  rule         D-030 stands if the new Kalman fill is judged better or equal on at least 5 of the 10 random-set sheets; otherwise it
               is reopened. Judgements per sheet: A, B, equal (both about equally close), cannot_tell. [assistant] "equal" counts as
               better-or-equal; "cannot_tell" does not (the denominator stays 10). The top set is reported separately and does not
               enter the rule.

Usage:
    python3 evaluation/3.2/make_blind_sheets.py            # makes the sheets and the key
    python3 evaluation/3.2/score_blind_sheets.py            # after judgements.csv exists
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_SAMPLE, SEED_LABELS, SEED_ORDER = 20261125, 20261126, 20261127
N_TOP, PER_CLIP_RANDOM, SIDE_FRAMES = 10, 2, 2
PANEL_PX, MIN_CROP = 480, 240
ORANGE, CYAN, WHITE, GREY = (0, 140, 255), (255, 200, 0), (255, 255, 255), (170, 170, 170)   # BGR


def load_filled(folder: Path):
    """{(clip, joint, frame): row} for all body-joint rows of the *_filled.csv files in a 3.2 output folder."""
    out = {}
    for fn in sorted(folder.glob("*_filled.csv")):
        for r in csv.DictReader(open(fn, newline="")):
            out[(r["video"], r["keypoint_name"], int(r["frame_idx"]))] = r
    return out


def candidates(old, new):
    rows = []
    for k, ro in old.items():
        if ro["flag"] != "tier1":
            continue
        rn = new[k]
        if rn["flag"] != "tier2" or not rn["x_filled"] or not ro["x_filled"]:
            continue
        po = (float(ro["x_filled"]), float(ro["y_filled"]))
        pn = (float(rn["x_filled"]), float(rn["y_filled"]))
        raw = (float(ro["x_raw"]), float(ro["y_raw"]), float(ro["confidence"])) if ro["x_raw"] else None
        rows.append({"clip": k[0], "joint": k[1], "frame": k[2], "old": po, "new": pn, "raw": raw,
                     "dist": float(np.hypot(po[0] - pn[0], po[1] - pn[1]))})
    return rows


def choose(cands):
    ordered = sorted(cands, key=lambda c: (-c["dist"], c["clip"], c["joint"], c["frame"]))
    top = ordered[:N_TOP]
    for i, c in enumerate(top, 1):
        c["rank"] = i
    rest = sorted((c for c in cands if c not in top), key=lambda c: (c["clip"], c["joint"], c["frame"]))
    rng = random.Random(SEED_SAMPLE)
    by_clip = defaultdict(list)
    for c in rest:
        by_clip[c["clip"]].append(c)
    rnd = []
    for clip in sorted(by_clip):
        rnd += rng.sample(by_clip[clip], min(PER_CLIP_RANDOM, len(by_clip[clip])))
    return top, rnd


def read_frames(video: Path, wanted: set[int]):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    out, i, last = {}, 0, max(wanted)
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = fr
        i += 1
    cap.release()
    missing = wanted - set(out)
    if missing:
        raise SystemExit(f"{video}: frames not read: {sorted(missing)}")
    return out


def crop(img, cx, cy, side):
    """Square crop of `side` px centred on (cx, cy), black where outside the image; returns (panel, x0, y0)."""
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    canvas = np.zeros((side, side, 3), np.uint8)
    h, w = img.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return canvas, x0, y0


def put(panel, pt, x0, y0, scale, shape, label, colour):
    x, y = int(round((pt[0] - x0) * scale)), int(round((pt[1] - y0) * scale))
    if shape == "circle":                   # A: large ring, label up-left; B: small concentric ring, label down-right
        big = label == "A"
        cv2.circle(panel, (x, y), 14 if big else 7, colour, 2, cv2.LINE_AA)
        off = (-34, -14) if big else (12, 26)
        cv2.putText(panel, label, (x + off[0], y + off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(panel, label, (x + off[0], y + off[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    elif shape == "cross":
        cv2.drawMarker(panel, (x, y), colour, cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        cv2.putText(panel, label, (x + 12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    else:                                   # small ring: confident observed position on a side panel
        cv2.circle(panel, (x, y), 6, colour, 2, cv2.LINE_AA)


def title(panel, text):
    """Returns the panel with a black title strip above it."""
    strip = np.zeros((30, panel.shape[1], 3), np.uint8)
    cv2.putText(strip, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return np.vstack([strip, panel])


def make_sheet(c, sheet_id, a_is_old, frames, new, out_png):
    clip, joint, f = c["clip"], c["joint"], c["frame"]
    pts = [c["old"], c["new"]] + ([c["raw"][:2]] if c["raw"] else [])
    cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
    spread = max(np.hypot(p[0] - q[0], p[1] - q[1]) for p in pts for q in pts)
    side = int(max(MIN_CROP, 1.8 * spread + 100))
    scale = PANEL_PX / side
    panels = []
    for g in range(f - SIDE_FRAMES, f + SIDE_FRAMES + 1):
        if g not in frames:
            panels.append(title(np.zeros((PANEL_PX, PANEL_PX, 3), np.uint8), f"frame {g}: outside the clip"))
            continue
        crop_img, x0, y0 = crop(frames[g], cx, cy, side)
        panel = cv2.resize(crop_img, (PANEL_PX, PANEL_PX), interpolation=cv2.INTER_CUBIC)
        r = new.get((clip, joint, g))
        if g == f:
            fills = {"old": c["old"], "new": c["new"]}
            first, second = ("old", "new") if a_is_old else ("new", "old")
            put(panel, fills[first], x0, y0, scale, "circle", "A", ORANGE)
            put(panel, fills[second], x0, y0, scale, "circle", "B", CYAN)
            if c["raw"]:
                put(panel, c["raw"][:2], x0, y0, scale, "cross", "R", GREY)
            panel = title(panel, f"frame {g}: joint hidden" + ("" if c["raw"] else " (no R: no detection)"))
        else:
            obs = r is not None and r["flag"] == "observed" and r["x_filled"]
            if obs:
                put(panel, (float(r["x_filled"]), float(r["y_filled"])), x0, y0, scale, "ring", "", WHITE)
            panel = title(panel, f"frame {g}: " + ("observed (white ring)" if obs else "gap frame"))
        panels.append(panel)
    sheet = np.hstack(panels)
    head = np.zeros((70, sheet.shape[1], 3), np.uint8)
    cv2.putText(head, f"Sheet {sheet_id}   {clip}   {joint}   frame {f}   (crop {side} px square, same on every panel)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    cv2.putText(head, "A and B: the two candidate fills (which method is which is hidden).  R: raw low-confidence ViTPose point.  "
                "Which of A or B is closer to where the joint really is?",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, default=Path("evaluation/3.2/output_full_lock1"))
    ap.add_argument("--new", type=Path, default=Path("evaluation/3.2/output_full_lock1_d030"))
    ap.add_argument("--footage", type=Path, default=Path("footage/test"))
    ap.add_argument("--out", type=Path, default=Path("evaluation/3.2/blind_sheets_row38"))
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        print(f"Refusing to write into the non-empty folder {args.out}", file=sys.stderr)
        sys.exit(1)
    args.out.mkdir(parents=True, exist_ok=True)
    old, new = load_filled(args.old), load_filled(args.new)
    cands = candidates(old, new)
    top, rnd = choose(cands)
    order = random.Random(SEED_ORDER)
    order.shuffle(top)
    order.shuffle(rnd)
    labels = random.Random(SEED_LABELS)
    sheets = [("T", i, c) for i, c in enumerate(top, 1)] + [("R", i, c) for i, c in enumerate(rnd, 1)]
    need = defaultdict(set)
    for _, _, c in sheets:
        need[c["clip"]].update(range(max(c["frame"] - SIDE_FRAMES, 0), c["frame"] + SIDE_FRAMES + 1))
    video_frames = {clip: read_frames(args.footage / clip, w) for clip, w in need.items()}
    key = []
    for kind, i, c in sheets:
        sid = f"{kind}{i:02d}"
        a_is_old = labels.random() < 0.5
        make_sheet(c, sid, a_is_old, video_frames[c["clip"]], new, args.out / f"sheet_{sid}.jpg")
        key.append({"sheet": sid, "set": "top10_disagreement" if kind == "T" else "random10", "rank_by_disagreement": c.get("rank", ""),
                    "clip": c["clip"], "joint": c["joint"], "frame": c["frame"], "A_is": "old_spline" if a_is_old else "new_kalman",
                    "B_is": "new_kalman" if a_is_old else "old_spline", "disagreement_px": f"{c['dist']:.2f}",
                    "raw_confidence": f"{c['raw'][2]:.3f}" if c["raw"] else ""})
    with open(args.out / "KEY_do_not_open_before_judging.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(key)
    with open(args.out / "judgements_TEMPLATE.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["sheet", "judgement", "note"])
        for s in sheets:
            w.writerow([f"{s[0]}{s[1]:02d}", "", ""])
    per_clip = defaultdict(int)
    for c in rnd:
        per_clip[c["clip"]] += 1
    lines = ["Blind sheets for gaps row 38 (evaluation/3.2/make_blind_sheets.py)",
             f"candidates: {len(cands)} joint-frames flagged tier1 in {args.old.as_posix()} and tier2 in {args.new.as_posix()}",
             f"top set: the {N_TOP} largest old-vs-new disagreements (sheets T01-T10, shuffled); random set: {len(rnd)} frames from the other candidates, "
             f"{PER_CLIP_RANDOM} per clip, seed {SEED_SAMPLE} (sheets R01-R10, shuffled with seed {SEED_ORDER}); A/B order per frame, seed {SEED_LABELS}",
             f"random set per clip: {dict(sorted(per_clip.items()))}",
             "rule (fixed before looking): D-030 stands if the new Kalman fill is judged better or equal on at least 5 of the 10 random sheets, otherwise it is reopened; "
             "judgements A / B / equal / cannot_tell; equal counts as better-or-equal, cannot_tell does not; the top set is reported separately and does not enter the rule",
             "A visual judgement by one person, not an error measurement; no ground truth.",
             "The A/B key is in KEY_do_not_open_before_judging.csv; fill judgements_TEMPLATE.csv (copy it to judgements.csv) and run score_blind_sheets.py."]
    (args.out / "selection.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(sheets)} sheets written to {args.out}; random set per clip {dict(sorted(per_clip.items()))}")


if __name__ == "__main__":
    main()
