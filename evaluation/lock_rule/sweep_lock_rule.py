"""
Sweep of the lock-rule family (person_lock.py; DECISIONS.md D-024, D-027) by replay on the cached
boxes (validation/box_cache/). Offline: no detector, no pose model.

Per setting and clip: frames with a chosen box, dropped frames, longest untracked run, frames changed
against the legacy choice, rescued and released frames, centre-jump statistics, a rule-independent
height-ratio flag count, and for IMG_2419 the failure window 592-614 against a PROVISIONAL reference
track (reference_track_2419; to be confirmed by the author on the contact sheets).

Hard filters (plan, in order): (i) IMG_2419 wrong-lock frames in 592-614 = 0; (ii) IMG_2417 identical to
legacy on every frame; (iii) the first 100 frames of every clip unchanged on every frame the legacy rule
tracked (no replaced and no lost frame). The literal reading of the plan, "first 100 frames identical to
validation/output", also forbids frames the new rule NEWLY tracks (legacy dropped them): a rescue tier
cannot satisfy that by construction (IMG_2412 frames 41-42 are detector drops inside the first 100).
Both are written (ok_first100_identical = literal, ok_first100_legacy_frames_unchanged = used for
feasibility); newly tracked frames are counted separately. A setting is feasible if it passes (i)-(iii). Feasible settings are ranked by a margin score, then by fewest
changed frames in the other clips, then fewest active components; the ranking is a convenience for
choosing sheets, not a decision. Only ONE real failure event exists, so thresholds are weakly identified.

Margin score = min of: g0 / J (gate vs the largest legitimate per-frame jump J), (g0 + v*gap) / jump for
every observed legitimate gap, F / (g0 + v*5) (the failure's centre distance F = 0.925 h against the
allowance at its gap of 5 frames), s / R (size limit vs the largest legitimate height ratio R, vs the
median of the last 10 accepted boxes), and 3.9 / s (the failure's height ratio). J, R, the gap events
and F come from the data (legitimate_ranges, and the failure boxes) and are written to
legit_ranges.txt. A rule with no gate and no size test has margin 0.

Outputs (evaluation/lock_rule/output_full/): sweep_settings.csv (one row per setting: parameters,
per-clip metrics, filters, margin), feasible_summary.txt, legit_ranges.txt, reference_track_2419.csv.

Usage:
    python3 evaluation/lock_rule/sweep_lock_rule.py
"""

from __future__ import annotations

import csv
import itertools
import math
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lock_common import (  # noqa: E402
    CLIPS, KNOWN_FAILURE, LockParams, height_ratio_flags, iou, jump_stats, legitimate_ranges, load_cache,
    reference_track_2419, run_rule, runs_of, same_box)

OUT = Path("evaluation/lock_rule/output_full")
GRID = {
    "g0": [None, 0.25, 0.5, 0.75, 1.0],
    "v": [0.0, 0.02, 0.05, 0.10],
    "s": [None, 1.75, 2.5, 3.5],
    "k": [1, 10, 30],
    "c": [None, 0.3, 0.2, 0.1, 0.05],
    "n": [None, 10, 20, 40],
}
FAIL_DIST_OVER_H = 0.925   # 2419: centre distance of the bystander box (597) to the last climber box (591), / h
FAIL_HRATIO = 3.9          # same boxes, climber h / bystander h
FAIL_GAP = 5               # frames without an accepted box before frame 597 (592-596)

_DATA = {}


def settings() -> list[LockParams]:
    out = []
    for g0, v, s, k, c, n in itertools.product(*GRID.values()):
        if g0 is None and v != 0:
            continue
        if g0 is None and s is None and k != 10:
            continue          # K only matters through the gate or the size test
        out.append(LockParams(conf_rescue=c, gate_g0=g0, gate_v=v, size_s=s, size_k=k, release_n=n))
    return out


def init() -> None:
    frames = {c: load_cache(c) for c in CLIPS}
    legacy = {c: run_rule(frames[c], LockParams.legacy())[0] for c in CLIPS}
    ref = reference_track_2419(frames["IMG_2419"], legacy["IMG_2419"])
    _DATA.update(frames=frames, legacy=legacy, ref=ref)


def evaluate(params: LockParams) -> dict:
    row = {}
    for clip in CLIPS:
        chosen, infos = run_rule(_DATA["frames"][clip], params)
        leg = _DATA["legacy"][clip]
        drops = runs_of([b is None for b in chosen])
        changed = [not same_box(a, b) for a, b in zip(chosen, leg)]
        gained = [a is not None and b is None for a, b in zip(chosen, leg)]
        lost = [a is None and b is not None for a, b in zip(chosen, leg)]
        replaced = [a is not None and b is not None and not same_box(a, b) for a, b in zip(chosen, leg)]
        j = jump_stats(chosen)
        row[clip] = {
            "n_frames": len(chosen), "n_dropped": sum(b is None for b in chosen),
            "longest_drop": max((e - s + 1 for s, e in drops), default=0),
            "n_changed": sum(changed), "n_changed_first100": sum(changed[:100]),
            "n_gained": sum(gained), "n_lost": sum(lost), "n_replaced": sum(replaced),
            "n_gained_first100": sum(gained[:100]),
            "n_lost_or_replaced_first100": sum(lost[:100]) + sum(replaced[:100]),
            "n_rescued": sum(i.tier == "rescue" for i in infos),
            "n_released": sum(i.tier == "release" for i in infos),
            "jump_max": j.get("max"), "jump_p99": j.get("p99"), "n_jump_over_0_3": j.get("n_over_0_3", 0),
            "n_hratio_flags": len(height_ratio_flags(chosen)),
        }
        if clip == KNOWN_FAILURE[0]:
            ch = _DATA["ref"]
            correct = wrong = dropped = rescued = 0
            for t in range(592, 615):
                b, (r, _) = chosen[t], ch[t]
                if b is None:
                    dropped += 1
                elif r is not None and iou(b, r) >= 0.5:
                    correct += 1
                    rescued += infos[t].tier == "rescue"
                else:
                    wrong += 1
            row[clip].update(win_correct=correct, win_wrong=wrong, win_dropped=dropped, win_rescued=rescued)
    return row


def margin(params: LockParams, legit: dict) -> float:
    if params.gate_g0 is None and params.size_s is None:
        return 0.0
    vals = []
    if params.gate_g0 is not None:
        vals.append(params.gate_g0 / legit["max_step_jump"])
        for gap, jump, _ in legit["gaps"]:
            vals.append((params.gate_g0 + params.gate_v * gap) / max(jump, 1e-9))
        vals.append(FAIL_DIST_OVER_H / (params.gate_g0 + params.gate_v * FAIL_GAP))
    if params.size_s is not None:
        vals.append(params.size_s / legit["max_hratio_vs_median10"])
        vals.append(FAIL_HRATIO / params.size_s)
    return min(vals)


def components(p: LockParams) -> int:
    return sum([p.gate_g0 is not None, p.size_s is not None, p.conf_rescue is not None, p.release_n is not None])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    init()
    legit = legitimate_ranges(_DATA["legacy"])
    grid = [LockParams.legacy()] + settings()
    with Pool(initializer=init) as pool:
        results = pool.map(evaluate, grid, chunksize=16)

    cols = ["setting", "legacy", "g0", "v", "s", "K", "c", "N", "components", "margin", "feasible",
            "ok_2419_no_wrong", "ok_2417_identical", "ok_first100_identical",
            "ok_first100_legacy_frames_unchanged"]
    metric_names = ["n_dropped", "longest_drop", "n_changed", "n_gained", "n_lost", "n_replaced",
                    "n_gained_first100", "n_rescued", "n_released", "jump_max",
                    "jump_p99", "n_jump_over_0_3", "n_hratio_flags"]
    for c in CLIPS:
        cols += [f"{c}_{m}" for m in metric_names]
    cols += ["win2419_correct", "win2419_wrong", "win2419_dropped", "win2419_rescued"]
    rows, feas, literal = [], [], 0
    for p, r in zip(grid, results):
        w = r["IMG_2419"]
        ok1 = w["win_wrong"] == 0
        ok2 = r["IMG_2417"]["n_changed"] == 0
        ok3_literal = all(r[c]["n_changed_first100"] == 0 for c in CLIPS)
        ok3 = all(r[c]["n_lost_or_replaced_first100"] == 0 for c in CLIPS)
        m = margin(p, legit)
        rec = [p.label(), p.is_legacy, p.gate_g0, p.gate_v, p.size_s, p.size_k, p.conf_rescue, p.release_n,
               components(p), f"{m:.3f}", (ok1 and ok2 and ok3 and not p.is_legacy), ok1, ok2, ok3_literal, ok3]
        for c in CLIPS:
            rec += [r[c][k] if r[c][k] is None or not isinstance(r[c][k], float) else f"{r[c][k]:.4f}"
                    for k in metric_names]
        rec += [w["win_correct"], w["win_wrong"], w["win_dropped"], w["win_rescued"]]
        rows.append(rec)
        if ok1 and ok2 and ok3 and not p.is_legacy:
            feas.append((p, r, m))
        if ok1 and ok2 and ok3_literal and not p.is_legacy:
            literal += 1
    with open(OUT / "sweep_settings.csv", "w", newline="") as f:
        wtr = csv.writer(f, lineterminator="\n")
        wtr.writerow(cols)
        wtr.writerows(rows)

    # reference track file
    with open(OUT / "reference_track_2419.csv", "w", newline="") as f:
        wtr = csv.writer(f, lineterminator="\n")
        wtr.writerow(["frame", "ref_x", "ref_y", "ref_w", "ref_h", "ref_conf", "iou_with_interpolation"])
        for t, (b, s) in _DATA["ref"].items():
            wtr.writerow([t] + ([f"{b.x:.1f}", f"{b.y:.1f}", f"{b.w:.1f}", f"{b.h:.1f}", f"{b.conf:.4f}"]
                                if b else ["", "", "", "", ""]) + [f"{s:.3f}"])

    lines = [f"Sweep of the lock-rule family on cached boxes: {len(grid) - 1} settings + legacy; replay equals "
             "baseline was checked first (replay_check.txt).",
             "Reference track for IMG_2419 592-614 is PROVISIONAL (reference_track_2419.csv).",
             f"Feasible (hard filters i-iii, iii = legacy-tracked frames unchanged in the first 100): "
             f"{len(feas)} of {len(grid) - 1}. Feasible under the literal filter iii (also no newly "
             f"tracked frame in the first 100): {literal}.", ""]
    if not feas:
        lines.append("No feasible setting.")
    else:
        for name, key in (("g0", lambda p: p.gate_g0), ("v", lambda p: p.gate_v), ("s", lambda p: p.size_s),
                          ("K", lambda p: p.size_k), ("c", lambda p: p.conf_rescue), ("N", lambda p: p.release_n)):
            tot, ok = {}, {}
            for p in grid[1:]:
                tot[key(p)] = tot.get(key(p), 0) + 1
            for p, _, _ in feas:
                ok[key(p)] = ok.get(key(p), 0) + 1
            lines.append(f"{name}: " + ", ".join(f"{'off' if k is None else k}: {ok.get(k, 0)}/{tot[k]}"
                                                  for k in sorted(tot, key=lambda x: (x is None, x))))
        ranked = sorted(feas, key=lambda x: (-round(x[2], 1), sum(x[1][c]["n_changed"] for c in CLIPS),
                                            components(x[0]), x[1]["IMG_2419"]["longest_drop"]))
        lines += ["", "Top 15 feasible settings by (margin rounded to 0.1 desc, changed frames, components, "
                  "2419 longest drop); a convenience for choosing sheets, not a decision:"]
        for p, r, m in ranked[:15]:
            w = r["IMG_2419"]
            lines.append(f"  {p.label()} | margin {m:.2f} | changed {sum(r[c]['n_changed'] for c in CLIPS)} | "
                         f"2419 window correct/wrong/dropped/rescued {w['win_correct']}/{w['win_wrong']}/"
                         f"{w['win_dropped']}/{w['win_rescued']} | rescued total {sum(r[c]['n_rescued'] for c in CLIPS)}")
    (OUT / "feasible_summary.txt").write_text("\n".join(lines) + "\n")

    gl = ", ".join(f"({g},{j:.3f},{r:.2f})" for g, j, r in legit["gaps"])
    (OUT / "legit_ranges.txt").write_text(
        "Observed legitimate box motion on the legacy tracks of the five clips (IMG_2419 597-614 and steps\n"
        "touching them excluded); source: validation/box_cache/ replayed with the legacy rule, equal to\n"
        "validation/output_full (replay_check.txt).\n"
        f"consecutive steps: n {legit['n_steps']}\n"
        f"max centre jump / h of previous box: {legit['max_step_jump']:.3f}\n"
        f"max height ratio between consecutive frames: {legit['max_step_hratio']:.2f}\n"
        f"max height ratio to the median of the last 10 accepted boxes: {legit['max_hratio_vs_median10']:.2f}\n"
        f"jumps across real detector gaps (gap frames, jump / h, height ratio): {gl}\n"
        f"known failure (IMG_2419): centre distance {FAIL_DIST_OVER_H} h, height ratio {FAIL_HRATIO}, gap {FAIL_GAP}\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
