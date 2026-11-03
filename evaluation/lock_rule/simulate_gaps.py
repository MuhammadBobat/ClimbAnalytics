"""
Gap simulation (plan item 3.5; DECISIONS.md D-024, D-027): hide the climber in REAL cached boxes and see
whether each rule re-acquires the climber when the hide ends, and whether it accepts another box while
the climber is hidden. This is a simulation on real boxes, not a real long gap: the five clips contain no
legitimate gap longer than 8 frames.

For each clip and hide length L in 5, 10, 20, 40, 80 frames, trials start at frame 20, 70, 120, ... when
the legacy rule tracked the climber on every frame from 10 before the start to 10 after the end and none
of those frames is in the exclusion list (deterministic, no random numbers). During the hide, every
cached box (any confidence) whose IoU with the legacy chosen box at that frame is >= 0.3 is removed;
other boxes (holds, other people) stay. The rule then runs from its state just before the hide.
Reported per trial set: re-acquired within 5 and within 10 frames after the hide (a chosen box with IoU >= 0.5 with
the legacy box of that frame), and trials in which any box was accepted during the hide.

LIMITATION (D-027): the climber is hidden IN PLACE and never reappears somewhere else, so the simulation cannot
show how a rule treats a climber who really moved away during the hide (false rejection of a climber who
returned elsewhere is untested), and the junk it measures is whatever real boxes the clips contain.

Settings simulated: EVERY row of sweep_settings.csv (legacy plus the full grid), so selection criteria and
neighbours can be evaluated on all of them. Output: sim_summary.csv (setting x L, clips pooled, with n
trials). Per-clip rows (setting x clip x L) are written by select_setting.py for the selected setting and
its neighbours only (sim_results.csv), to keep the committed files small.

Usage:
    python3 evaluation/lock_rule/simulate_gaps.py
"""

from __future__ import annotations

import csv
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lock_common import CLIPS, EXCLUSIONS_2419, LockParams, iou, load_cache, run_rule  # noqa: E402

OUT = Path("evaluation/lock_rule/output_full")
LENGTHS = (5, 10, 20, 40, 80)
STEP, FIRST, MARGIN, HIDE_IOU, REACQ_IOU = 50, 20, 10, 0.3, 0.5

_DATA = {}


def init() -> None:
    frames = {c: load_cache(c) for c in CLIPS}
    legacy = {c: run_rule(frames[c], LockParams.legacy())[0] for c in CLIPS}
    _DATA.update(frames=frames, legacy=legacy)


def trial_starts(clip: str, L: int) -> list[int]:
    leg, n = _DATA["legacy"][clip], len(_DATA["legacy"][clip])
    ex = set(EXCLUSIONS_2419) if clip == "IMG_2419" else set()
    out = []
    for s in range(FIRST, n, STEP):
        lo, hi = s - MARGIN, s + L + MARGIN
        if hi >= n:
            break
        if all(leg[t] is not None and t not in ex for t in range(lo, hi + 1)):
            out.append(s)
    return out


def params_from_row(r: dict) -> LockParams:
    f = lambda x: None if x in ("", "None") else float(x)  # noqa: E731
    return LockParams(conf_rescue=f(r["c"]), gate_g0=f(r["g0"]), gate_v=float(r["v"]),
                      size_s=f(r["s"]), size_k=int(r["K"]),
                      release_n=None if r["N"] in ("", "None") else int(r["N"]))


def simulate(params: LockParams) -> list[dict]:
    rows = []
    for clip in CLIPS:
        frames, leg = _DATA["frames"][clip], _DATA["legacy"][clip]
        _, _, states = run_rule(frames, params, record_states=True)
        for L in LENGTHS:
            starts = trial_starts(clip, L)
            n5 = n10 = n_fa = fa_frames = 0
            for s in starts:
                sub = []
                for t in range(s, s + L + MARGIN):
                    boxes = frames[t]
                    if t < s + L:
                        boxes = [b for b in boxes if iou(b, leg[t]) < HIDE_IOU]
                    sub.append(boxes)
                chosen, _ = run_rule(sub, params, start_state=states[s - 1])
                hidden = chosen[:L]
                after = chosen[L:]
                fa = sum(b is not None for b in hidden)
                n_fa += fa > 0
                fa_frames += fa
                ok = [b is not None and iou(b, leg[s + L + i]) >= REACQ_IOU for i, b in enumerate(after)]
                n5 += any(ok[:5])
                n10 += any(ok[:10])
            rows.append({"setting": params.label(), "clip": clip, "L": L, "n_trials": len(starts),
                         "reacq5": n5, "reacq10": n10, "trials_with_accept_during_hide": n_fa,
                         "accepted_frames_during_hide": fa_frames})
    return rows


def main() -> None:
    with open(OUT / "sweep_settings.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    grid = list({p.label(): p for p in (params_from_row(r) for r in rows)}.values())   # legacy appears twice
    with Pool(initializer=init) as pool:
        results = pool.map(simulate, grid, chunksize=8)
    flat = [r for res in results for r in res]
    pooled = {}
    for r in flat:
        k = (r["setting"], r["L"])
        d = pooled.setdefault(k, dict(n_trials=0, reacq5=0, reacq10=0, accept=0))
        d["n_trials"] += r["n_trials"]
        d["reacq5"] += r["reacq5"]
        d["reacq10"] += r["reacq10"]
        d["accept"] += r["trials_with_accept_during_hide"]
    with open(OUT / "sim_summary.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["setting", "L", "n_trials", "reacq5", "reacq10", "trials_with_accept_during_hide"])
        for (s, L), d in pooled.items():
            w.writerow([s, L, d["n_trials"], d["reacq5"], d["reacq10"], d["accept"]])
    print(f"{len(grid)} settings simulated; trials per L (pooled): "
          + ", ".join(f"L={L}: {pooled[(LockParams.legacy().label(), L)]['n_trials']}" for L in LENGTHS))


if __name__ == "__main__":
    main()
