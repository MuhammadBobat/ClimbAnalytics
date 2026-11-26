"""
Blind panel review of the working CoM against A, F and the filled-limb variant (DECISIONS.md D-026 addendum, 2026-11-26).

Stage `select` (needs only the frame counts; run and COMMITTED before any panel is rendered): fixes 20 random windows
(4 per clip, 32 frames each, non-overlapping within a clip, drawn with the fixed seeds below) plus the 10 windows of the earlier
stress set (evaluation/2.2c/review.csv, the ten largest proxy-vs-F disagreements of the first review), and for every window a
random order of the four candidates. Writes
    evaluation/2.2c/random_review/windows.csv   set, window_id, clip, start, end (the stress set is rendered in the same blind
                                                format so it can be judged the same way; the earlier review notes in review.csv
                                                did not compare E-dL with A window by window)
    evaluation/2.2c/random_review/key.csv       window_id, P1..P4 = candidate shown in each panel  (THE KEY: do not open it
                                                before the judgements are written)
    evaluation/2.2c/random_review/judgements_template.csv   blank sheet for the author

Stage `render`: one grid video per window, evaluation/2.2c/overlays/random_review/<window_id>.mp4 (gitignored), four panels
labelled only P1..P4 (no candidate name, no speed, no candidate colour, no A ring), drawn from the 3.2 gap-filled run:
A_proxy, EdL_c0.5 (working CoM, raw-confidence limb gate), F_full, EdLfill_c0.5 (limb gate = position present, observed or
Kalman-filled, never Tier 3). Panels show the raw keypoint skeleton (red X below 0.5), one disc per panel with a fading trail,
"undefined" in red where the candidate has no position, one fixed square crop per window from the union of the four candidates
over the window plus 1.5 median torso lengths. Same marker colour in every panel.

Everything chosen here (seeds, 32-frame windows, 4 per clip, panel order) is an assistant choice, no literature source.
This is one person's visual judgement of plausibility, not an accuracy measure; no ground-truth CoM exists.

Usage:
    python3 evaluation/2.2c/random_review.py select
    python3 evaluation/2.2c/random_review.py render
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

# --- fixed in this file BEFORE any panel is rendered ------------------------------------------------------------------------
WINDOW_SEED = 20261127      # window starts: random.Random(WINDOW_SEED + clip index)
ORDER_SEED = 20261128       # shuffle of the 20 random windows' ids and of the panel order of every window
WINDOW_FRAMES = 32          # same length as the earlier stress windows (e.g. 64-95)
PER_CLIP = 4
CLIPS = ["IMG_2412", "IMG_2416", "IMG_2417", "IMG_2419", "IMG_2420"]
CANDIDATES = ["A_proxy", "EdL_c0.5", "F_full", "EdLfill_c0.5"]
FILLED_DIR = Path("evaluation/3.2/output_full_lock1_d030")
RAW_DIR = Path("validation/output_full_lock1")
OUT = Path("evaluation/2.2c/random_review")


def frame_counts(filled_dir: Path = FILLED_DIR) -> dict:
    out = {}
    for clip in CLIPS:
        with open(filled_dir / f"{clip}_filled.csv", newline="") as f:
            out[clip] = max(int(r["frame_idx"]) for r in csv.DictReader(f)) + 1
    return out


def draw_random_windows(counts: dict) -> list:
    """[(clip, start, end)] 4 per clip, in clip order then by start; starts uniform on [0, n - 32], rejection of any start within
    32 frames of an accepted one in the same clip (non-overlapping)."""
    out = []
    for i, clip in enumerate(CLIPS):
        rng = random.Random(WINDOW_SEED + i)
        starts = []
        while len(starts) < PER_CLIP:
            s = rng.randint(0, counts[clip] - WINDOW_FRAMES)
            if all(abs(s - o) >= WINDOW_FRAMES for o in starts):
                starts.append(s)
        out += [(clip, s, s + WINDOW_FRAMES - 1) for s in sorted(starts)]
    return out


def stress_windows(review_csv: Path = HERE / "review.csv") -> list:
    out = []
    with open(review_csv, newline="") as f:
        for r in csv.DictReader(f):
            a, b = r["window_frames"].split("-")
            out.append((int(r["rank"]), r["clip"].split(".")[0], int(a), int(b)))
    return out


def build_selection(counts: dict, review_csv: Path = HERE / "review.csv"):
    """(windows rows, key rows): random windows get ids R01..R20 in a shuffled order, stress windows S01..S10 in review rank order."""
    rnd = draw_random_windows(counts)
    rng = random.Random(ORDER_SEED)
    order = list(range(len(rnd)))
    rng.shuffle(order)
    windows = [["random", f"R{k + 1:02d}", *rnd[i], ""] for k, i in enumerate(order)]
    for rank, clip, a, b in stress_windows(review_csv):
        windows.append(["stress", f"S{rank:02d}", clip, a, b, rank])
    key = []
    for w in windows:                      # panel order drawn in window-list order from the same stream
        perm = CANDIDATES[:]
        rng.shuffle(perm)
        key.append([w[1], *perm])
    return windows, key


def select() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    windows, key = build_selection(frame_counts())
    for name, header, rows in (("windows.csv", ["set", "window_id", "clip", "start", "end", "stress_rank"], windows),
                               ("key.csv", ["window_id", "P1", "P2", "P3", "P4"], key),
                               ("judgements_template.csv",
                                ["window_id", "most_plausible_panel", "tie_or_none", "P1_jump", "P1_dropout", "P1_wrong_place",
                                 "P2_jump", "P2_dropout", "P2_wrong_place", "P3_jump", "P3_dropout", "P3_wrong_place",
                                 "P4_jump", "P4_dropout", "P4_wrong_place", "note"],
                                [[w[1]] + [""] * 15 for w in windows])):
        with open(OUT / name, "w", newline="") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(header)
            wr.writerows(rows)
    print(f"{len(windows)} windows written to {OUT}")


# --- rendering ------------------------------------------------------------------------------------------------------------

def render() -> None:
    import cv2
    import numpy as np
    import compare_com_candidates as cc
    import make_candidate_overlays as mo
    oc = mo.oc

    cc.enable_filled_candidate()
    comps, _ = cc.run(FILLED_DIR, Path("validation/frame_exclusions.csv"), "male", "*_filled.csv")
    by_clip = {c.clip.split(".")[0]: c for c in comps}
    with open(OUT / "windows.csv", newline="") as f:
        windows = list(csv.DictReader(f))
    with open(OUT / "key.csv", newline="") as f:
        key = {r["window_id"]: [r["P1"], r["P2"], r["P3"], r["P4"]] for r in csv.DictReader(f)}
    video_dir = Path("evaluation/2.2c/overlays/random_review")
    video_dir.mkdir(parents=True, exist_ok=True)
    COL = (0, 215, 255)   # one marker colour (BGR) for every panel
    P, TH = mo.PANEL, mo.TITLE_H

    def panel(frame, c, raw, t, label, name, crop):
        x0, y0, side = crop
        s = P / side
        img = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (P, P), interpolation=cv2.INTER_AREA)
        tp = lambda p: (int(round((p[0] - x0) * s)), int(round((p[1] - y0) * s)))
        banner = None
        if t not in raw:
            banner = ("NO DETECTION (no person box)", mo.COL_BAD)
        else:
            pts = raw[t]
            for a, b in oc.EDGES:
                if a in pts and b in pts:
                    cv2.line(img, tp(pts[a]), tp(pts[b]), mo.COL_LINE, 2)
            for x, y, conf in pts.values():
                q = tp((x, y))
                if conf < mo.OCCLUSION_CONFIDENCE_THRESHOLD:
                    cv2.line(img, (q[0] - 7, q[1] - 7), (q[0] + 7, q[1] + 7), mo.COL_BAD, 2)
                    cv2.line(img, (q[0] - 7, q[1] + 7), (q[0] + 7, q[1] - 7), mo.COL_BAD, 2)
                else:
                    cv2.circle(img, q, 5, mo.BLACK, -1)
                    cv2.circle(img, q, 4, mo.band_colour(conf), -1)
        pos = c.series[label].positions
        if pos[t] is not None:
            q = tp(pos[t])
            cv2.circle(img, q, 15, mo.BLACK, -1)
            cv2.circle(img, q, 12, COL, -1)
        else:
            cv2.putText(img, "undefined", (P // 2 - 95, P - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, mo.COL_BAD, 3)
        pts_k = [tp(pos[t - k]) if t - k >= 0 and pos[t - k] is not None else None for k in range(mo.TRAIL + 1)]
        for k in range(1, mo.TRAIL + 1):
            alpha = 0.9 * (1 - (k - 1) / mo.TRAIL)
            layer = img.copy()
            if pts_k[k] is not None:
                if pts_k[k - 1] is not None:
                    cv2.line(layer, pts_k[k - 1], pts_k[k], mo.BLACK, 4)
                    cv2.line(layer, pts_k[k - 1], pts_k[k], COL, 2)
                cv2.circle(layer, pts_k[k], 5, mo.BLACK, -1)
                cv2.circle(layer, pts_k[k], 3, COL, -1)
            img = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
        if banner:
            cv2.rectangle(img, (0, 0), (P, 34), banner[1], -1)
            cv2.putText(img, banner[0], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mo.WHITE, 2)
        title = np.full((TH, P, 3), 30, dtype=np.uint8)
        cv2.putText(title, name, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mo.WHITE, 2)
        return np.vstack([title, img])

    for w in windows:
        clip, start, end, wid = w["clip"], int(w["start"]), int(w["end"]), w["window_id"]
        c = by_clip[clip]
        raw = oc.load_raw(RAW_DIR / f"{clip}_keypoints.csv")
        cap = cv2.VideoCapture(str(Path("footage/test") / f"{clip}.MOV"))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pts = [p for l in CANDIDATES for t in range(start, end + 1) if (p := c.series[l].positions[t]) is not None]
        torsos = [L for L in c.torso[start:end + 1] if L]
        margin = mo.MARGIN_TORSOS * (statistics.median(torsos) if torsos else 200.0)
        full = min(W, H)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        side = int(min(max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * margin, full))
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        crop = (int(min(max(cx - side / 2, 0), W - side)), int(min(max(cy - side / 2, 0), H - side)), side)
        order = key[wid]
        for _ in range(start):
            cap.grab()
        size = (2 * P, mo.HEADER_H + 2 * (P + TH))
        writer = cv2.VideoWriter(str(video_dir / f"{wid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 30.0, size)
        for t in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            grid = np.full((size[1], size[0], 3), 20, dtype=np.uint8)
            for i, label in enumerate(order):
                r, k = divmod(i, 2)
                grid[mo.HEADER_H + r * (P + TH):mo.HEADER_H + (r + 1) * (P + TH), k * P:(k + 1) * P] = \
                    panel(frame, c, raw, t, label, f"P{i + 1}", crop)
            cv2.putText(grid, f"{wid}  {clip}  frame {t}  ({t - start + 1}/{end - start + 1})", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, mo.WHITE, 2)
            writer.write(grid)
        cap.release()
        writer.release()
        print(wid, clip, start, end, "->", video_dir / f"{wid}.mp4", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("select", "render"))
    a = ap.parse_args()
    select() if a.stage == "select" else render()
