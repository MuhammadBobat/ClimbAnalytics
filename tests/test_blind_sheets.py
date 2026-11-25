"""
Tests for evaluation/3.2/make_blind_sheets.py and score_blind_sheets.py (gaps row 38): selection, blinding bookkeeping and the
scoring rule. Synthetic data; nothing here is a judgement of any fill.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_blind_sheets", REPO_ROOT / "evaluation" / "3.2" / "make_blind_sheets.py")
mb = importlib.util.module_from_spec(_spec)
sys.modules["make_blind_sheets"] = mb
_spec.loader.exec_module(mb)

CLIPS = ["c1", "c2", "c3", "c4", "c5"]


def fake(n_per_clip=30):
    out = []
    for ci, clip in enumerate(CLIPS):
        for k in range(n_per_clip):
            d = 1.0 + ci * 0.1 + k * 0.5
            out.append({"clip": clip, "joint": "left_wrist", "frame": 100 + k, "old": (0.0, 0.0), "new": (d, 0.0),
                        "raw": None, "dist": d})
    return out


class SelectionTests(unittest.TestCase):
    def test_top_set_is_the_ten_largest_and_sets_are_disjoint(self):
        c = fake()
        top, rnd = mb.choose(c)
        self.assertEqual(len(top), 10)
        self.assertEqual(sorted((t["dist"] for t in top), reverse=True), sorted((x["dist"] for x in c), reverse=True)[:10])
        keys = lambda rows: {(r["clip"], r["joint"], r["frame"]) for r in rows}
        self.assertFalse(keys(top) & keys(rnd))

    def test_random_set_has_two_per_clip_and_is_reproducible(self):
        top, rnd = mb.choose(fake())
        self.assertEqual(len(rnd), 10)
        self.assertEqual({cl: sum(1 for r in rnd if r["clip"] == cl) for cl in CLIPS}, {cl: 2 for cl in CLIPS})
        top2, rnd2 = mb.choose(fake())
        self.assertEqual([(r["clip"], r["frame"]) for r in rnd], [(r["clip"], r["frame"]) for r in rnd2])
        self.assertEqual((mb.SEED_SAMPLE, mb.SEED_LABELS, mb.SEED_ORDER), (20261125, 20261126, 20261127))

    def test_clip_with_too_few_candidates_gives_what_it_has(self):
        c = [r for r in fake() if r["clip"] != "c3" or r["frame"] < 112]     # c3 keeps 12 frames, 10 of them in the top set? no: it has small distances
        top, rnd = mb.choose(c)
        self.assertEqual(sum(1 for r in rnd if r["clip"] == "c3"), 2)


class ScoringTests(unittest.TestCase):
    def run_scoring(self, judgements):
        """judgements: {sheet: (A_is, judgement)} for sheets R01..R10 (random set) and T01 (top set)."""
        d = Path(tempfile.mkdtemp())
        with open(d / "KEY_do_not_open_before_judging.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["sheet", "set", "rank_by_disagreement", "clip", "joint", "frame", "A_is", "B_is", "disagreement_px", "raw_confidence"])
            for sid, (a_is, _) in judgements.items():
                b_is = "new_kalman" if a_is == "old_spline" else "old_spline"
                w.writerow([sid, "top10_disagreement" if sid.startswith("T") else "random10", "", "c1", "left_wrist", 1, a_is, b_is, "1.0", ""])
        with open(d / "judgements.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["sheet", "judgement", "note"])
            for sid, (_, j) in judgements.items():
                w.writerow([sid, j, ""])
        r = subprocess.run([sys.executable, str(REPO_ROOT / "evaluation" / "3.2" / "score_blind_sheets.py"), "--dir", str(d)],
                           capture_output=True, text=True)
        return r, (d / "result.md").read_text() if (d / "result.md").exists() else ""

    def sheets(self, kalman_better, equal=0, cannot=0):
        """Ten random sheets: the first `kalman_better` judge the new Kalman closer (alternating which letter it is)."""
        out, i = {}, 0
        for n in range(kalman_better):
            a_is = "new_kalman" if n % 2 == 0 else "old_spline"
            out[f"R{n + 1:02d}"] = (a_is, "A" if a_is == "new_kalman" else "B")
        n = kalman_better
        for _ in range(equal):
            n += 1
            out[f"R{n:02d}"] = ("new_kalman", "equal")
        for _ in range(cannot):
            n += 1
            out[f"R{n:02d}"] = ("new_kalman", "cannot_tell")
        while n < 10:                                   # the rest: the old spline judged closer
            n += 1
            out[f"R{n:02d}"] = ("old_spline", "A")
        out["T01"] = ("old_spline", "A")
        return out

    def test_five_of_ten_stands_and_four_reopens(self):
        r, md = self.run_scoring(self.sheets(5))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("D-030 stands", md)
        r, md = self.run_scoring(self.sheets(4))
        self.assertIn("D-030 is reopened", md)

    def test_equal_counts_cannot_tell_does_not(self):
        r, md = self.run_scoring(self.sheets(3, equal=2))
        self.assertIn("D-030 stands", md)
        r, md = self.run_scoring(self.sheets(3, equal=1, cannot=1))
        self.assertIn("D-030 is reopened", md)

    def test_letter_is_mapped_through_the_key_and_top_set_does_not_enter_the_rule(self):
        j = self.sheets(4)
        j["T01"] = ("new_kalman", "A")                  # a top-set sheet judged for the Kalman must not rescue the rule
        r, md = self.run_scoring(j)
        self.assertIn("D-030 is reopened", md)

    def test_missing_or_invalid_judgement_is_refused(self):
        j = self.sheets(5)
        j["R03"] = ("new_kalman", "maybe")
        r, _ = self.run_scoring(j)
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
