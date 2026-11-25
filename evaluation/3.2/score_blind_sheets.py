"""
Scores the blind sheets of gaps row 38 once judgements.csv exists (columns sheet, judgement, note; judgement one of
A, B, equal, cannot_tell). Joins the judgements with KEY_do_not_open_before_judging.csv and applies the rule fixed before the
sheets were made (make_blind_sheets.py docstring): D-030 stands if the new Kalman fill is judged better or equal on at least
5 of the 10 random-set sheets, otherwise it is reopened; "equal" counts as better-or-equal, "cannot_tell" does not (the
denominator stays 10). The top set is reported separately and does not enter the rule. A visual judgement by one person,
NOT an error measurement.

Usage: python3 evaluation/3.2/score_blind_sheets.py [--dir evaluation/3.2/blind_sheets_row38]
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

RULE_MIN = 5
ALLOWED = {"A", "B", "equal", "cannot_tell"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("evaluation/3.2/blind_sheets_row38"))
    a = ap.parse_args()
    key = {r["sheet"]: r for r in csv.DictReader(open(a.dir / "KEY_do_not_open_before_judging.csv", newline=""))}
    jud = {r["sheet"]: r["judgement"].strip() for r in csv.DictReader(open(a.dir / "judgements.csv", newline=""))}
    bad = [s for s in key if jud.get(s) not in ALLOWED]
    if bad:
        print(f"Missing or invalid judgements (allowed: {sorted(ALLOWED)}): {bad}", file=sys.stderr)
        sys.exit(1)
    rows, out = [], {}
    for sid, k in key.items():
        j = jud[sid]
        if j == "A":
            verdict = "new_kalman" if k["A_is"] == "new_kalman" else "old_spline"
        elif j == "B":
            verdict = "new_kalman" if k["B_is"] == "new_kalman" else "old_spline"
        else:
            verdict = j
        rows.append({**k, "judgement": j, "closer": verdict})
    for st in ("random10", "top10_disagreement"):
        c = Counter(r["closer"] for r in rows if r["set"] == st)
        out[st] = c
    n_ok = out["random10"]["new_kalman"] + out["random10"]["equal"]
    stands = n_ok >= RULE_MIN
    L = ["# Blind sheets, gaps row 38: result (a visual judgement by one person, not an error measurement)", "",
         "| set | new Kalman closer | old spline closer | equal | cannot tell |", "|---|---|---|---|---|"]
    for st in ("random10", "top10_disagreement"):
        c = out[st]
        L.append(f"| {st} | {c['new_kalman']} | {c['old_spline']} | {c['equal']} | {c['cannot_tell']} |")
    L += ["", f"Rule (fixed before the sheets were made): D-030 stands if the new Kalman fill is better or equal on at least {RULE_MIN} of the 10 random sheets. "
          f"Random set: {n_ok} of 10 -> **{'D-030 stands' if stands else 'D-030 is reopened'}** under that rule.", "",
          "| sheet | set | clip | joint | frame | judgement | closer | note |", "|---|---|---|---|---|---|---|---|"]
    notes = {r["sheet"]: r.get("note", "") for r in csv.DictReader(open(a.dir / "judgements.csv", newline=""))}
    for r in sorted(rows, key=lambda r: (r["set"], r["sheet"])):
        L.append(f"| {r['sheet']} | {r['set']} | {r['clip']} | {r['joint']} | {r['frame']} | {r['judgement']} | {r['closer']} | {notes.get(r['sheet'], '')} |")
    (a.dir / "result.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:9]))


if __name__ == "__main__":
    main()
