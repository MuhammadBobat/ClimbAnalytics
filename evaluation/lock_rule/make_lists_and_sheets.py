"""
Case lists and contact sheets for the author's review (plan step 3; DECISIONS.md D-024, D-027).

Lists (CSV, evaluation/lock_rule/output_full/):
  multibox_frames.csv         every frame with >= 2 cached boxes at conf >= 0.5 (the legacy cut-off), one row
                              per alternative box: its position and size relative to the legacy choice and
                              whether it is "close" (see below)
  multibox_frames_summary.csv one row per such frame
  case_2420_2380_2400.csv     IMG_2420 frames 2380-2400 (named case, frame 2390): every cached box, the legacy
                              choice, the SELECTED setting, and the choice under EVERY feasible setting
  case_2419_585_630.csv       IMG_2419 frames 585-630: every cached box with its confidence, the provisional
                              reference box, the legacy choice, the selected setting
  changed_runs.csv            runs of frames where the selected setting differs from legacy
  lists_manifest.txt          counts, the legitimate ranges used and where they come from

"Close" alternative (those are the only multi-box frames that could hide a wrong lock showing no jump):
centre distance to the chosen box <= MAX_STEP_JUMP * h of the chosen box AND height ratio within
[1 / MAX_HRATIO, MAX_HRATIO]. MAX_STEP_JUMP and MAX_HRATIO are the largest values observed on the legacy
tracks of the five clips (legit_ranges.txt: max centre jump / h between consecutive frames, max height
ratio to the median of the last 10 accepted boxes), recomputed here, not typed in.

The setting drawn is the SELECTED one (selected_setting.json, written by select_setting.py from the author's
selection order). Sheets (PNG, evaluation/lock_rule/sheets/, gitignored; only these are built): IMG_2419 585-630,
the frames the selected setting accepted through the rescue tier in IMG_2412, IMG_2416 and IMG_2420, the changed
runs in IMG_2419, and the close-alternative frames. All cached boxes are thin yellow with their confidence,
the legacy choice red, the selected setting green (inset so equal boxes stay visible).

Usage:
    python3 evaluation/lock_rule/make_lists_and_sheets.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lock_common import (  # noqa: E402
    CLIPS, LockParams, fmt1, iou, legitimate_ranges, load_cache, reference_track_2419, run_rule, runs_of,
    same_box)
from simulate_gaps import params_from_row  # noqa: E402

OUT = Path("evaluation/lock_rule/output_full")
SHEETS = Path("evaluation/lock_rule/sheets")
FOOTAGE = Path("footage/test")
COLOURS = {"legacy": (0, 0, 255), "selected": (0, 200, 0)}
TILE_SCALE = 0.25


def bstr(b) -> str:
    return "" if b is None else f"{b.x:.1f},{b.y:.1f},{b.w:.1f},{b.h:.1f},{b.conf:.3f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)
    frames = {c: load_cache(c) for c in CLIPS}
    legacy = {c: run_rule(frames[c], LockParams.legacy())[0] for c in CLIPS}
    import json
    from simulate_gaps import params_from_row
    label = json.loads((OUT / "selected_setting.json").read_text())["label"]
    with open(OUT / "sweep_settings.csv", newline="") as f:
        SEL_PARAMS = params_from_row(next(r for r in csv.DictReader(f) if r["setting"] == label))
    res = {c: run_rule(frames[c], SEL_PARAMS) for c in CLIPS}
    SEL = {c: res[c][0] for c in CLIPS}
    TIERS = {c: [i.tier for i in res[c][1]] for c in CLIPS}
    legit = legitimate_ranges(legacy)
    max_jump, max_hr = legit["max_step_jump"], legit["max_hratio_vs_median10"]

    # --- multi-box frames
    alt_rows, sum_rows, close_frames = [], [], {c: [] for c in CLIPS}
    for c in CLIPS:
        for t, boxes in enumerate(frames[c]):
            main = [b for b in boxes if b.conf >= 0.5]
            if len(main) < 2:
                continue
            ch = legacy[c][t]
            ci = next(i for i, b in enumerate(boxes) if b is ch or b == ch)
            any_close = False
            for i, b in enumerate(boxes):
                if b.conf < 0.5 or i == ci:
                    continue
                d = math.hypot(b.center[0] - ch.center[0], b.center[1] - ch.center[1]) / ch.h
                hr = b.h / ch.h
                close = d <= max_jump and 1 / max_hr <= hr <= max_hr
                any_close |= close
                alt_rows.append([c, t, len(main), ci, bstr(ch), i, bstr(b), f"{d:.3f}", f"{hr:.3f}",
                                 f"{iou(b, ch):.3f}", close])
            if any_close:
                close_frames[c].append(t)
            sum_rows.append([c, t, len(main), any_close, not same_box(SEL[c][t], ch)])
    with open(OUT / "multibox_frames.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["clip", "frame", "n_boxes_ge_0_5", "legacy_idx", "legacy_box_x_y_w_h_conf", "alt_idx",
                    "alt_box_x_y_w_h_conf", "alt_centre_dist_over_h_chosen", "alt_h_over_h_chosen",
                    "alt_iou_with_chosen", "alt_close"])
        w.writerows(alt_rows)
    with open(OUT / "multibox_frames_summary.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["clip", "frame", "n_boxes_ge_0_5", "has_close_alternative", "selected_differs_from_legacy"])
        w.writerows(sum_rows)

    # --- IMG_2419 585-630
    ref = reference_track_2419(frames["IMG_2419"], legacy["IMG_2419"])
    with open(OUT / "case_2419_585_630.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame", "n_cached_boxes", "boxes_x_y_w_h_conf (conf >= 0.05, detector order)",
                    "provisional_reference_box", "legacy", "selected"])
        for t in range(585, 631):
            r = ref.get(t, (None, 0))[0]
            w.writerow([t, len(frames["IMG_2419"][t]), " | ".join(bstr(b) for b in frames["IMG_2419"][t]),
                        bstr(r), bstr(legacy["IMG_2419"][t]), bstr(SEL["IMG_2419"][t]), bstr(SEL["IMG_2419"][t])])

    # --- IMG_2420 2380-2400, choice under every feasible setting
    with open(OUT / "sweep_settings.csv", newline="") as f:
        feas = [params_from_row(r) for r in csv.DictReader(f) if r["feasible"] == "True"]
    f2420 = frames["IMG_2420"]
    tallies = {t: Counter() for t in range(2380, 2401)}
    for p in feas:
        ch, _ = run_rule(f2420, p, last=2400)
        for t in range(2380, 2401):
            tallies[t][None if ch[t] is None else fmt1(ch[t])] += 1
    with open(OUT / "case_2420_2380_2400.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["frame", "n_cached_boxes", "boxes_x_y_w_h_conf (detector order)", "legacy", "selected",
                    "n_feasible_settings", "n_same_as_legacy", "n_differ", "choices_under_feasible_settings"])
        changed_any = []
        for t in range(2380, 2401):
            leg = legacy["IMG_2420"][t]
            key = None if leg is None else fmt1(leg)
            same = tallies[t][key]
            differ = len(feas) - same
            if differ:
                changed_any.append(t)
            w.writerow([t, len(f2420[t]), " | ".join(bstr(b) for b in f2420[t]), bstr(leg),
                        bstr(SEL["IMG_2420"][t]), len(feas), same, differ,
                        "; ".join(f"{'none' if k is None else ','.join(k)} x{n}" for k, n in tallies[t].most_common())])

    # --- changed runs and rescued frames of the selected setting
    runs = {}
    with open(OUT / "changed_runs.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["setting", "clip", "first_frame", "last_frame", "n_frames", "gained", "lost", "replaced", "rescue_tier_frames"])
        for c in CLIPS:
            ch = [not same_box(a, b) for a, b in zip(SEL[c], legacy[c])]
            runs[c] = runs_of(ch)
            for s_, e_ in runs[c]:
                seg = range(s_, e_ + 1)
                g = sum(SEL[c][t] is not None and legacy[c][t] is None for t in seg)
                l_ = sum(SEL[c][t] is None and legacy[c][t] is not None for t in seg)
                w.writerow([label, c, s_, e_, e_ - s_ + 1, g, l_, (e_ - s_ + 1) - g - l_,
                            sum(TIERS[c][t] == "rescue" for t in seg)])
    rescued = {c: [t for t in range(len(SEL[c])) if TIERS[c][t] == "rescue"] for c in CLIPS}

    n_multi = Counter(r[0] for r in sum_rows)
    n_close = {c: len(v) for c, v in close_frames.items()}
    lines = ["Lists and sheets manifest (evaluation/lock_rule/make_lists_and_sheets.py)",
             f"legitimate ranges used for 'close alternative' (from the legacy tracks of the five clips, "
             f"IMG_2419 597-614 excluded; see legit_ranges.txt): centre distance <= {max_jump:.3f} h of the chosen "
             f"box (max centre jump / h between consecutive frames), height ratio within [{1 / max_hr:.3f}, {max_hr:.2f}] "
             f"(max height ratio to the median of the last 10 accepted boxes)",
             "multi-box frames (>= 2 cached boxes at conf >= 0.5), per clip: "
             + ", ".join(f"{c} {n_multi.get(c, 0)}" for c in CLIPS) + f"; total {sum(n_multi.values())}",
             "of those, frames with a close alternative (could hide a wrong lock that shows no jump), per clip: "
             + ", ".join(f"{c} {n_close[c]}" for c in CLIPS) + f"; total {sum(n_close.values())}",
             f"feasible settings used for the 2420 case: {len(feas)}",
             "2420 frames 2380-2400 where any feasible setting differs from legacy: "
             + (", ".join(map(str, changed_any)) or "none"),
             f"selected setting: {label}",
             "frames accepted through the rescue tier, per clip: " + ", ".join(f"{c} {len(rescued[c])}" for c in CLIPS),
             "changed runs vs legacy: " + "; ".join(f"{c}: {len(runs[c])} run(s)" for c in CLIPS if runs[c])]
    (OUT / "lists_manifest.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    make_sheets(frames, legacy, SEL, close_frames, runs, rescued)


def read_frames(video: Path, wanted: list[int]):
    import cv2
    cap = cv2.VideoCapture(str(video))
    want, out, t = set(wanted), {}, 0
    last = max(wanted)
    while t <= last:
        if t in want:
            ok, img = cap.read()
            if not ok:
                break
            out[t] = img
        elif not cap.grab():
            break
        t += 1
    cap.release()
    return out


def tile(img, boxes, chosen: dict, t: int, clip: str):
    import cv2
    s = TILE_SCALE
    im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for b in boxes:
        p1, p2 = (int(b.x * s), int(b.y * s)), (int((b.x + b.w) * s), int((b.y + b.h) * s))
        cv2.rectangle(im, p1, p2, (0, 220, 220), 1)
        cv2.putText(im, f"{b.conf:.2f}", (p1[0] + 2, p1[1] + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)
    for k, (name, b) in enumerate(chosen.items()):
        if b is None:
            continue
        inset = 3 * k
        p1 = (int(b.x * s) + inset, int(b.y * s) + inset)
        p2 = (int((b.x + b.w) * s) - inset, int((b.y + b.h) * s) - inset)
        cv2.rectangle(im, p1, p2, COLOURS[name], 2)
    cv2.rectangle(im, (0, 0), (im.shape[1], 16), (30, 30, 30), -1)
    none = [n for n, b in chosen.items() if b is None]
    cv2.putText(im, f"{clip[-4:]} f{t}  none: {','.join(none) if none else '-'}", (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return im


def sheet(path: Path, clip: str, idxs: list[int], frames, legacy, SEL, cols: int, title: str) -> None:
    import cv2
    import numpy as np
    imgs = read_frames(FOOTAGE / f"{clip}.MOV", idxs)
    tiles = [tile(imgs[t], frames[clip][t], {"legacy": legacy[clip][t], "selected": SEL[clip][t]}, t, clip)
             for t in idxs if t in imgs]
    if not tiles:
        return
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    head = 26
    canvas = np.full((head + rows * h, cols * w, 3), 20, dtype=np.uint8)
    cv2.putText(canvas, f"{title} | red legacy, green selected setting, yellow = every cached box with conf",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, t in enumerate(tiles):
        r, k = divmod(i, cols)
        canvas[head + r * h:head + (r + 1) * h, k * w:(k + 1) * w] = t
    cv2.imwrite(str(path), canvas)
    print(f"  {path} ({len(tiles)} tiles)")


def make_sheets(frames, legacy, SEL, close_frames, runs, rescued) -> None:
    print("sheets (selected setting only):")
    for old in SHEETS.glob("*.png"):
        old.unlink()                                   # earlier A / B sheets are superseded
    r = list(range(585, 631))
    sheet(SHEETS / "IMG_2419_585-607.png", "IMG_2419", r[:23], frames, legacy, SEL, 8, "IMG_2419 585-607")
    sheet(SHEETS / "IMG_2419_608-630.png", "IMG_2419", r[23:], frames, legacy, SEL, 8, "IMG_2419 608-630")
    for c in ("IMG_2412", "IMG_2416", "IMG_2420"):
        for part, i0 in enumerate(range(0, len(rescued[c]), 64)):
            sheet(SHEETS / f"rescued_{c}_{part + 1}.png", c, rescued[c][i0:i0 + 64], frames, legacy, SEL, 8,
                  f"{c}: frames accepted through the rescue tier")
    idxs = []
    for s_, e_ in runs["IMG_2419"]:
        n = e_ - s_ + 1
        idxs += list(range(s_, e_ + 1)) if n <= 24 else sorted({s_, e_, *[s_ + round(i * (n - 1) / 23) for i in range(24)]})
    sheet(SHEETS / "changed_IMG_2419.png", "IMG_2419", sorted(set(idxs)), frames, legacy, SEL, 8,
          "IMG_2419: frames where the selected setting differs from legacy")
    for c in CLIPS:
        fr = close_frames[c]
        if not fr:
            continue
        idxs = []
        for s_, e_ in runs_of_list(fr):
            n = e_ - s_ + 1
            idxs += list(range(s_, e_ + 1)) if n <= 3 else [s_, (s_ + e_) // 2, e_]
        for part, i0 in enumerate(range(0, len(idxs), 64)):
            sheet(SHEETS / f"closealt_{c}_{part + 1}.png", c, idxs[i0:i0 + 64], frames, legacy, SEL, 8,
                  f"{c}: multi-box frames with a close alternative (runs of > 3 sampled: first, middle, last)")


def runs_of_list(frs: list[int]) -> list[tuple[int, int]]:
    out, s, p = [], frs[0], frs[0]
    for f in frs[1:]:
        if f != p + 1:
            out.append((s, p))
            s = f
        p = f
    out.append((s, p))
    return out


if __name__ == "__main__":
    main()
