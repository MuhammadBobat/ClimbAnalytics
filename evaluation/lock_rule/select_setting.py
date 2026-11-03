"""
Apply the author's selection order (stated 2026-11-03 after seeing the stage-2 results, so NOT
pre-registered; to be logged in D-027) to sweep_settings.csv and sim_summary.csv:

  a) the hard filters (sweep feasibility: IMG_2419 window has 0 wrong frames, IMG_2417 unchanged, first 100
     frames unchanged on every frame the legacy rule tracked);
  b) in the gap simulation, zero accepted non-climber boxes at EVERY hide length (pooled over the clips), and
     zero wrong frames in the 2419 window;
  c) among those: most correct frames in the 2419 window, then highest re-acquisition at 80 frames
     (re-acquired within 5 frames, then within 10);
  d) then the largest margin to the legitimate and failure bounds (sweep_lock_rule.margin), then the fewest
     active components.
b) is not relaxed here: if nothing passes it, the nearest settings are shown instead.

Writes selection.txt, selected_setting.json, selection_neighbours.csv (the selected setting and each parameter
moved one grid step, with the same columns) and sim_results.csv (per-clip simulation rows for those settings).

Usage:
    python3 evaluation/lock_rule/select_setting.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import simulate_gaps as sg  # noqa: E402
from lock_common import CLIPS, LockParams  # noqa: E402
from sweep_lock_rule import GRID, components  # noqa: E402

OUT = Path("evaluation/lock_rule/output_full")
LENGTHS = sg.LENGTHS


def load():
    with open(OUT / "sweep_settings.csv", newline="") as f:
        sweep = {}
        for r in csv.DictReader(f):
            sweep.setdefault(r["setting"], r)
    sim = {}
    with open(OUT / "sim_summary.csv", newline="") as f:
        for r in csv.DictReader(f):
            sim.setdefault(r["setting"], {})[int(r["L"])] = r
    return sweep, sim


def metrics(label: str, sweep, sim) -> dict:
    r, s = sweep[label], sim[label]
    other_changed = sum(int(r[f"{c}_n_changed"]) for c in CLIPS if c != "IMG_2419")
    m = {
        "setting": label, "feasible": r["feasible"] == "True", "margin": float(r["margin"]),
        "components": int(r["components"]),
        "win_correct": int(r["win2419_correct"]), "win_wrong": int(r["win2419_wrong"]),
        "win_dropped": int(r["win2419_dropped"]), "win_rescued": int(r["win2419_rescued"]),
        "changed_other_clips": other_changed,
        "rescued_total": sum(int(r[f"{c}_n_rescued"]) for c in CLIPS),
        "dropped_total": sum(int(r[f"{c}_n_dropped"]) for c in CLIPS),
        "ok_2417": r["ok_2417_identical"] == "True", "ok_first100": r["ok_first100_legacy_frames_unchanged"] == "True",
        "ok_first100_literal": r["ok_first100_identical"] == "True",
    }
    for L in LENGTHS:
        m[f"accept_L{L}"] = int(s[L]["trials_with_accept_during_hide"])
        m[f"reacq5_L{L}"] = int(s[L]["reacq5"])
        m[f"reacq10_L{L}"] = int(s[L]["reacq10"])
        m[f"n_L{L}"] = int(s[L]["n_trials"])
    m["passes_b"] = m["feasible"] and m["win_wrong"] == 0 and all(m[f"accept_L{L}"] == 0 for L in LENGTHS)
    return m


def label_of(g0, v, s, k, c, n) -> str:
    return LockParams(conf_rescue=c, gate_g0=g0, gate_v=v, size_s=s, size_k=k, release_n=n).label()


def neighbours(label: str, sweep) -> list[tuple[str, str]]:
    """(description, label) for each parameter moved one grid step up and down, canonicalised."""
    r = sweep[label]
    f = lambda x: None if x in ("", "None") else float(x)  # noqa: E731
    cur = dict(g0=f(r["g0"]), v=float(r["v"]), s=f(r["s"]), k=int(r["K"]), c=f(r["c"]),
               n=None if r["N"] in ("", "None") else int(r["N"]))
    out = []
    for key, grid in GRID.items():
        i = grid.index(cur[key])
        for d, name in ((-1, "down"), (1, "up")):
            j = i + d
            if not 0 <= j < len(grid):
                continue
            p = dict(cur)
            p[key] = grid[j]
            note = ""
            if p["g0"] is None and p["v"] != 0:
                p["v"], note = 0.0, " (v reset to 0: no gate)"
            if p["g0"] is None and p["s"] is None and p["k"] != 10:
                p["k"], note = 10, " (K reset to 10: unused)"
            lab = label_of(p["g0"], p["v"], p["s"], p["k"], p["c"], p["n"])
            if lab in sweep and lab != label:
                out.append((f"{key} {name} -> {grid[j] if grid[j] is not None else 'off'}{note}", lab))
    return out


def main() -> None:
    sweep, sim = load()
    all_m = {lab: metrics(lab, sweep, sim) for lab in sweep if lab in sim}
    legacy_label = LockParams.legacy().label()
    cand = [m for m in all_m.values() if m["setting"] != legacy_label]
    feas = [m for m in cand if m["feasible"]]
    passing = [m for m in feas if m["passes_b"]]
    lines = [f"Selection order applied (author, 2026-11-03; chosen AFTER the stage-2 results, not pre-registered).",
             f"settings in the grid: {len(cand)}; pass a) hard filters: {len(feas)} (literal first-100 filter: "
             f"{sum(1 for m in cand if m['feasible'] or False) and sum(1 for m in cand if m['ok_first100_literal'] and m['ok_2417'] and m['win_wrong'] == 0)}); "
             f"pass b) as well (zero accepted non-climber boxes at every hide length, pooled over the clips, and zero 2419 wrong frames): {len(passing)}"]
    if not passing:
        near = sorted(feas, key=lambda m: (sum(m[f"accept_L{L}"] for L in LENGTHS), -m["win_correct"]))[:10]
        lines += ["No setting passes b). Nearest by total accepted-in-hide trials (b is NOT relaxed):"]
        lines += [f"  {m['setting']} accepts {[m[f'accept_L{L}'] for L in LENGTHS]} correct {m['win_correct']}" for m in near]
        (OUT / "selection.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    key = lambda m: (-m["win_correct"], -m[f"reacq5_L80"], -m["reacq10_L80"], -round(m["margin"], 6), m["components"])  # noqa: E731
    ranked = sorted(passing, key=key)
    best = key(ranked[0])
    tied = [m for m in ranked if key(m) == best]
    lines += ["", f"ordering key: most 2419 window correct, then reacq<=5 at L=80, then reacq<=10 at L=80, then margin, then fewest components",
              f"settings tied for first on the whole key: {len(tied)}"]
    lines += [f"  {m['setting']} | margin {m['margin']:.3f} | components {m['components']}" for m in tied]
    # do N and v matter?
    def best_c(sub):
        return max((m["win_correct"], m["reacq5_L80"], m["reacq10_L80"]) for m in sub) if sub else None
    plain = [m for m in passing if m["setting"].endswith("N=off") and " v=0 " in m["setting"]]
    lines += ["", "Do settings that pass b) need v or N?",
              f"  best (2419 correct, reacq5@80, reacq10@80) over all passing: {best_c(passing)}",
              f"  best over passing settings with v = 0 and N off: {best_c(plain)}",
              f"  passing settings with v > 0: {sum(1 for m in passing if ' v=0 ' not in m['setting'])}; with N set: "
              f"{sum(1 for m in passing if not m['setting'].endswith('N=off'))}"]
    for name, sub in (("g0", lambda m: m["setting"].split()[0]), ("s", lambda m: m["setting"].split()[2]),
                      ("K", lambda m: m["setting"].split()[3]), ("c", lambda m: m["setting"].split()[4])):
        cnt = {}
        for m in passing:
            cnt[sub(m)] = cnt.get(sub(m), 0) + 1
        lines.append(f"  passing b) by {name}: " + ", ".join(f"{k}: {v}" for k, v in sorted(cnt.items())))
    # TIE-BREAK (mine, NOT in the author's order; to be confirmed): the tied settings agree on every sweep column;
    # prefer the higher total re-acquisition (within 5 frames) over all hide lengths, then grid order.
    tb = lambda m: -sum(m[f"reacq5_L{L}"] for L in LENGTHS)  # noqa: E731
    tied = sorted(tied, key=tb)
    sel = tied[0]
    lines += ["", "tie-break used (not part of the author's order): higher total reacq<=5 over all hide lengths, then grid order; "
              + ", ".join(f"{m['setting']}: {sum(m[f'reacq5_L{L}'] for L in LENGTHS)}" for m in tied),
              f"SELECTED: {sel['setting']}"]
    # neighbours table
    rows = [("selected", sel["setting"])] + neighbours(sel["setting"], sweep)
    cols = ["role", "setting", "feasible", "passes_b", "ok_2417", "ok_first100", "ok_first100_literal", "win_correct",
            "win_wrong", "win_dropped", "win_rescued"] + [f"accept_L{L}" for L in LENGTHS] + \
           [f"reacq5_L{L}" for L in LENGTHS] + ["margin", "components", "changed_other_clips", "rescued_total",
                                               "dropped_total"] + [f"n_L{L}" for L in LENGTHS]
    with open(OUT / "selection_neighbours.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for role, lab in rows:
            m = all_m[lab]
            w.writerow([role] + [m[c] if c != "setting" else lab for c in cols[1:]])
    lines += ["", "selected setting and neighbours (each parameter one grid step): see selection_neighbours.csv"]
    for role, lab in rows:
        m = all_m[lab]
        lines.append(f"  {role[:46]:46s} | {lab} | feas {m['feasible']} b {m['passes_b']} | 2419 c/w/d/r {m['win_correct']}/"
                     f"{m['win_wrong']}/{m['win_dropped']}/{m['win_rescued']} | accept@L {[m[f'accept_L{L}'] for L in LENGTHS]} | "
                     f"reacq5@L {[m[f'reacq5_L{L}'] for L in LENGTHS]} | margin {m['margin']:.2f} | comp {m['components']}")
    lines.append(f"hide-trial counts n per L: {[sel[f'n_L{L}'] for L in LENGTHS]}")
    (OUT / "selection.txt").write_text("\n".join(lines) + "\n")
    p = LockParams
    with open(OUT / "selected_setting.json", "w") as f:
        json.dump({"label": sel["setting"]}, f)
    # per-clip simulation rows for the selected setting and its neighbours
    sg.init()
    pr = []
    with open(OUT / "sweep_settings.csv", newline="") as f:
        byl = {}
        for r in csv.DictReader(f):
            byl.setdefault(r["setting"], r)
    for role, lab in rows + [("legacy", legacy_label)]:
        pr += sg.simulate(sg.params_from_row(byl[lab]))
    with open(OUT / "sim_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pr[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(pr)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
