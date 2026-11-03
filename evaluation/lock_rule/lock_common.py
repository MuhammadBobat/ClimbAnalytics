"""Shared loaders and helpers for the lock-rule tools (DECISIONS.md D-024, D-027)."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from person_lock import Box, LockParams, LockState, LockInfo, iou, select_climber_box  # noqa: E402

CLIPS = ["IMG_2412", "IMG_2416", "IMG_2417", "IMG_2419", "IMG_2420"]
BASELINE_DIR = Path("validation/output_full")
CACHE_DIR = Path("validation/box_cache")
EXCLUSIONS_2419 = range(597, 615)       # validation/frame_exclusions.csv (D-024)
KNOWN_FAILURE = ("IMG_2419", 597, 614)  # the only confirmed wrong lock


def load_cache(clip: str, cache_dir: Path = CACHE_DIR) -> list[list[Box]]:
    """Per frame, every cached box (conf >= the cache floor) in the detector's order."""
    frames: dict[int, list[Box]] = {}
    with open(cache_dir / f"{clip}_boxes.csv", newline="") as f:
        for r in csv.DictReader(f):
            fi = int(r["frame_idx"])
            frames.setdefault(fi, [])
            if int(r["box_idx"]) >= 0:
                frames[fi].append(Box(float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"]),
                                      float(r["conf"])))
    return [frames[i] for i in range(len(frames))]


def load_baseline(clip: str, base_dir: Path = BASELINE_DIR) -> dict[int, tuple[tuple[str, ...], int]]:
    """frame -> ((x, y, w, h) as written, n_person_boxes) for frames that have a chosen box."""
    out: dict[int, tuple[tuple[str, ...], int]] = {}
    with open(base_dir / f"{clip}_keypoints.csv", newline="") as f:
        for r in csv.DictReader(f):
            fi = int(r["frame_idx"])
            if fi not in out:
                out[fi] = ((r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]), int(r["n_person_boxes"]))
    return out


def fmt1(box: Box) -> tuple[str, ...]:
    """The box as validate_vitpose.py writes it (one decimal)."""
    return (f"{box.x:.1f}", f"{box.y:.1f}", f"{box.w:.1f}", f"{box.h:.1f}")


def run_rule(frames: list[list[Box]], params: LockParams, start_state: LockState | None = None,
             first: int = 0, last: int | None = None, record_states: bool = False):
    """Replay the rule over frames[first:last+1]. Returns (chosen boxes, infos[, states]) where
    states[i] is the state AFTER frame first+i (so a trial can restart from states[t-1])."""
    st = start_state or LockState()
    last = len(frames) - 1 if last is None else last
    chosen, infos, states = [], [], []
    for t in range(first, last + 1):
        box, st, info = select_climber_box(frames[t], st, params)
        chosen.append(box)
        infos.append(info)
        if record_states:
            states.append(st)
    return (chosen, infos, states) if record_states else (chosen, infos)


def runs_of(flags: list[bool]) -> list[tuple[int, int]]:
    """(first, last) index of every maximal run of True."""
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def same_box(a: Box | None, b: Box | None) -> bool:
    return (a is None and b is None) or (a is not None and b is not None and fmt1(a) == fmt1(b))


def jump_stats(chosen: list[Box | None]) -> dict:
    """Centre jump / height of the previous box between consecutive accepted frames, and jumps
    across gaps (number of frames skipped and jump / h of the last box before the gap)."""
    j, gaps = [], []
    prev, prev_t = None, None
    for t, b in enumerate(chosen):
        if b is None:
            continue
        if prev is not None:
            d = math.hypot(b.center[0] - prev.center[0], b.center[1] - prev.center[1]) / prev.h
            (j if t == prev_t + 1 else gaps).append(d if t == prev_t + 1 else (t - prev_t - 1, d))
        prev, prev_t = b, t
    out = {"n_steps": len(j), "n_gap_jumps": len(gaps)}
    if j:
        s = sorted(j)
        out.update(median=statistics.median(j), p99=s[min(len(s) - 1, int(0.99 * len(s)))], max=s[-1],
                   n_over_0_3=sum(1 for x in j if x > 0.3))
    out["max_gap_jump"] = max((g[1] for g in gaps), default=None)
    return out


def height_ratio_flags(chosen: list[Box | None], limit: float = 2.0, k: int = 10) -> list[int]:
    """Frames whose chosen height differs from the median of the last k accepted heights by more
    than `limit` (rule-independent plausibility flag, not truth)."""
    flags, hist = [], []
    for t, b in enumerate(chosen):
        if b is None:
            continue
        if len(hist) >= 3:
            med = statistics.median(hist[-k:])
            if max(b.h / med, med / b.h) > limit:
                flags.append(t)
        hist.append(b.h)
    return flags


# --- reference track for the known failure (PROVISIONAL until the author confirms it) ----------

def reference_track_2419(frames: list[list[Box]], legacy: list[Box | None],
                         first: int = 592, last: int = 614) -> dict[int, tuple[Box | None, float]]:
    """For each frame in first..last: the cached box (any confidence) with the highest IoU to the
    linear interpolation of the legacy chosen boxes at first-1 and last+1, and that IoU; None if the
    best IoU is below 0.3. A proposal for the author to confirm on the contact sheets, not truth."""
    a, b = legacy[first - 1], legacy[last + 1]
    if a is None or b is None:
        raise ValueError("no legacy box on either side of the failure window")
    out = {}
    for t in range(first, last + 1):
        f = (t - (first - 1)) / (last + 1 - (first - 1))
        interp = Box(*(av + f * (bv - av) for av, bv in zip((a.x, a.y, a.w, a.h), (b.x, b.y, b.w, b.h))), 1.0)
        best = max(frames[t], key=lambda c: iou(c, interp), default=None)
        score = iou(best, interp) if best is not None else 0.0
        out[t] = (best if score >= 0.3 else None, score)
    return out


def legitimate_ranges(legacy_by_clip: dict[str, list[Box | None]]) -> dict:
    """Observed legitimate box motion on the legacy tracks of the five clips, excluding the known
    wrong-lock frames and any step touching them: max centre jump / h between consecutive frames,
    max height ratio between consecutive frames, max height ratio to the median of the last 10
    accepted boxes, and every (gap length, jump / h, height ratio) across a real detector gap."""
    jumps, steps, vs_med, gaps = [], [], [], []
    for clip, chosen in legacy_by_clip.items():
        ex = set(EXCLUSIONS_2419) if clip == "IMG_2419" else set()
        prev = prev_t = None
        hist = []
        for t, b in enumerate(chosen):
            if t in ex:
                prev = prev_t = None          # break the chain: no step touches an excluded frame
                continue
            if b is None:
                continue
            if prev is not None:
                d = math.hypot(b.center[0] - prev.center[0], b.center[1] - prev.center[1]) / prev.h
                r = max(b.h / prev.h, prev.h / b.h)
                if t == prev_t + 1:
                    jumps.append(d)
                    steps.append(r)
                else:
                    gaps.append((t - prev_t - 1, d, r))
            if len(hist) >= 10:
                med = statistics.median(hist[-10:])
                vs_med.append(max(b.h / med, med / b.h))
            hist.append(b.h)
            prev, prev_t = b, t
    return {"max_step_jump": max(jumps), "max_step_hratio": max(steps), "max_hratio_vs_median10": max(vs_med),
            "gaps": gaps, "n_steps": len(jumps)}
