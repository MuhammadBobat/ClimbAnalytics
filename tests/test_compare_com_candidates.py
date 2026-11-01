"""
Tests for evaluation/2.2c/compare_com_candidates.py: the leave-one-clip-out trunk-point fit, the
jitter and continuity definitions, coverage and shared-frame counts, the regression check against
2.2b and the manifest. Synthetic input; nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from metrics import JointSeries
from tests.test_compare_com_models import write_clip

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compare_com_candidates", REPO_ROOT / "evaluation" / "2.2c" / "compare_com_candidates.py")
cc = importlib.util.module_from_spec(_spec)
sys.modules["compare_com_candidates"] = cc
_spec.loader.exec_module(cc)


def planted(w_star, n=30, shift=0.0):
    """Samples whose F lies exactly at S + w* (H - S) (+ shift in x)."""
    out = []
    for i in range(n):
        s = (100.0 + 7 * i, 200.0 - 3 * i)
        h = (s[0] + 15 + (i % 5), s[1] + 120 + 2 * (i % 7))
        f = (s[0] + w_star * (h[0] - s[0]) + shift, s[1] + w_star * (h[1] - s[1]))
        out.append((s, h, f, 100.0 + i))
    return out


class FitTests(unittest.TestCase):
    def test_recovers_planted_weight(self):
        for w in (0.0, 0.3, 0.745, 1.0):
            self.assertAlmostEqual(cc.fit_trunk_weight(planted(w)), w, places=6)

    def test_clamps_to_unit_interval(self):
        self.assertEqual(cc.fit_trunk_weight(planted(1.5)), 1.0)
        self.assertEqual(cc.fit_trunk_weight(planted(-0.5)), 0.0)

    def test_no_samples_rejected(self):
        with self.assertRaises(ValueError):
            cc.fit_trunk_weight([])

    def test_mean_distance_px_and_torso(self):
        s = [((0.0, 0.0), (0.0, 10.0), (3.0, 4.0), 5.0)]
        self.assertAlmostEqual(cc.mean_distance(s, 0.0), 5.0)
        self.assertAlmostEqual(cc.mean_distance(s, 0.0, torso_units=True), 1.0)

    def test_loco_folds_exclude_held_out_clip(self):
        by_clip = {"a": planted(0.4), "b": planted(0.6), "c": planted(0.8)}
        folds = {r["held_out_clip"]: r for r in cc.loco_weights(by_clip)}
        self.assertEqual(set(folds), {"a", "b", "c"})
        self.assertEqual(folds["a"]["n_train"], 60)
        # a fold's w does not depend on its own held-out clip
        by_clip2 = {"a": planted(0.1), "b": by_clip["b"], "c": by_clip["c"]}
        folds2 = {r["held_out_clip"]: r for r in cc.loco_weights(by_clip2)}
        self.assertEqual(folds["a"]["w"], folds2["a"]["w"])
        self.assertNotEqual(folds["b"]["w"], folds2["b"]["w"])
        self.assertGreater(folds["a"]["heldout_px"], 0)


class DefinitionTests(unittest.TestCase):
    def test_jitter_zero_for_constant_velocity_and_known_value(self):
        ts = [i / 30 for i in range(5)]
        line = JointSeries("c", [(10.0 * i, 5.0 * i) for i in range(5)], ts)
        self.assertEqual(cc.jitter_series(line, [100.0] * 5), [None, 0.0, 0.0, 0.0, None])
        zig = JointSeries("c", [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)], ts[:4])
        j = cc.jitter_series(zig, [2.0] * 4)
        self.assertEqual(j, [None, 1.0, 1.0, None])   # |0 - 2 + 0| / 2 and |1 - 0 + 1| / 2

    def test_jitter_needs_three_defined_frames_and_torso(self):
        ts = [i / 30 for i in range(4)]
        s = JointSeries("c", [(0.0, 0.0), None, (2.0, 0.0), (3.0, 0.0)], ts)
        self.assertEqual(cc.jitter_series(s, [1.0] * 4), [None] * 4)
        s = JointSeries("c", [(0.0, 0.0), (1.0, 0.0), (5.0, 0.0), (6.0, 0.0)], ts)
        self.assertEqual(cc.jitter_series(s, [1.0, None, 2.0, 1.0]), [None, None, 1.5, None])   # |6 - 10 + 1| / 2

    def test_continuity(self):
        p = [(0, 0), (0, 0), None, (0, 0), None, None, (0, 0)]
        self.assertEqual(cc.continuity(p), (2, 3))
        self.assertEqual(cc.continuity([None, None]), (0, 0))

    def test_percentile_linear(self):
        self.assertAlmostEqual(cc.percentile([5, 1, 3, 2, 4], 0.95), 4.8)
        self.assertEqual(cc.percentile([7], 0.95), 7)


class ClipTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.p = self.d / "s_keypoints.csv"
        # 10 frames: frame 6 dropped, 8 excluded by list, left_wrist low on frame 3
        write_clip(self.p, [f for f in range(10) if f != 6], low={(3, "left_wrist")})
        self.c = cc.build_clip(self.p, {8}, "male", 0.7)

    def row(self, label, clip="s.MOV"):
        return dict(zip(cc.SUMMARY_COLUMNS, cc.summary_row(clip, label, [self.c])))

    def test_all_labels_built(self):
        self.assertEqual(set(self.c.series), set(cc.LABELS))
        self.assertEqual(len(cc.OVERLAY_LABELS), 9)

    def test_coverage_and_counts(self):
        a, f, e = self.row(cc.L_A), self.row(cc.L_F), self.row("E_c0.5")
        self.assertEqual((a["n_defined"], f["n_defined"], e["n_defined"]), (8, 7, 8))
        self.assertEqual(a["coverage"], "0.8000")
        self.assertEqual(f["coverage"], "0.7000")
        self.assertEqual(a["n_candidate_defined_f_not"], 1)   # frame 3
        self.assertEqual(f["n_f_defined_candidate_not"], 0)
        self.assertEqual(a["n_shared_with_f"], 7)
        self.assertEqual(self.row(cc.L_C)["n_defined"], 7)    # wrist gap, ears not needed

    def test_continuity_columns(self):
        a = self.row(cc.L_A)
        self.assertEqual(a["defined_to_none_per_1000_frames"], "200.0000")   # 5->6 and 7->8
        self.assertEqual(a["n_defined_runs"], 3)
        f = self.row(cc.L_F)
        self.assertEqual(f["n_defined_runs"], 4)                  # frame 3 splits the first run

    def test_limb_usage_diagnostics(self):
        e = self.row("E_c0.5")
        self.assertNotEqual(e["limb_segments_used_mean"], "")
        self.assertEqual(self.row(cc.L_A)["limb_segments_used_mean"], "")
        # frame 3 uses 10 of 12 segments (left forearm and hand dropped), others 12
        self.assertEqual(self.c.usage["E_c0.5"][3].__len__(), 10)
        self.assertEqual(len(self.c.usage["E_c0.5"][0]), 12)
        self.assertEqual(len(self.c.usage["E_c0.8"][0]), 12)

    def test_static_pose_has_zero_jitter_and_speed(self):
        r = self.row(cc.L_F)
        self.assertEqual(r["jitter_own_mean"], "0.0000")
        self.assertEqual(r["speed_own_max"], "0.0000")

    def test_frames_columns(self):
        header, rows = cc.frames_rows(self.c)
        self.assertEqual(len(header), 3 + 4 * len(cc.LABELS))
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[6][3], "")   # dropped frame: A x blank

    def test_regression_check_pass_and_fail(self):
        ref = self.d / "ref" / "s"
        ref.mkdir(parents=True)
        c = self.c
        c.clip = "s.MOV"
        header, rows = ["frame_idx", "proxy_x", "proxy_y", "seg_male_x", "seg_male_y"], []
        for t in range(c.n_frames):
            a, f = c.series[cc.L_A].positions[t], c.series[cc.L_F].positions[t]
            rows.append([t, cc._f(a[0]) if a else "", cc._f(a[1]) if a else "",
                         cc._f(f[0]) if f else "", cc._f(f[1]) if f else ""])
        cc.write_csv(ref / "frames.csv", header, rows)
        ok, msgs = cc.regression_check([c], self.d / "ref", {"frames": 10, cc.L_A: 8, cc.L_F: 7})
        self.assertTrue(ok, msgs)
        rows[2][1] = "123.0000"
        cc.write_csv(ref / "frames.csv", header, rows)
        ok, _ = cc.regression_check([c], self.d / "ref")
        self.assertFalse(ok)
        ok, _ = cc.regression_check([c], self.d / "ref", {"frames": 11})
        self.assertFalse(ok)

    def test_manifest_content(self):
        samples = cc.fit_samples(self.c.joints, self.c.series[cc.L_F], self.c.torso)
        folds = cc.loco_weights({"s.MOV": samples, "t.MOV": samples})
        text = cc.build_manifest([self.c], folds, Path("in"), Path("footage"), Path("nope.csv"),
                                 Path("nope.pdf"), "male", cc.FIT_TOL, (True, ["x"]))
        for needle in ("No candidate is chosen", "fold held out s.MOV", "removed head",
                       "scale 1/0.9306", "scale 1/0.8318", "jitter J(t)", "PASS"):
            self.assertIn(needle, text)


def table(spec):
    """spec: {candidate: {scope: (coverage, to_none, jitter)}} -> summary-style row dicts."""
    return [{"clip": sc, "candidate": cand, "coverage": v[0], "defined_to_none_per_1000_frames": v[1],
             "jitter_common_mean": v[2]} for cand, d in spec.items() for sc, v in d.items()]


class TriageTests(unittest.TestCase):
    SCOPES = ("ALL", "c1", "c2")

    def spec(self):
        good = ("0.9", "2.0", "0.010")
        s = {  # name: per-scope (coverage, transitions, jitter), as the strings of summary.csv
            "A_proxy": {sc: ("0.8", "5.0", "0.020") for sc in self.SCOPES},
            "F_full": {sc: ("0.7", "9.0", "0.030") for sc in self.SCOPES},
            "P": {sc: good for sc in self.SCOPES},                       # beats A and F everywhere
            "Q": {sc: ("0.85", "3.0", "0.015") for sc in self.SCOPES},   # dominated by P
            "R": {"ALL": good, "c1": good, "c2": ("0.9", "2.0", "0.025")},  # not better than A in c2
            "T": {sc: good for sc in self.SCOPES},                       # tie with P
            "M": {"ALL": ("", "2.0", "0.010"), "c1": good, "c2": good},  # missing value pooled
        }
        return s

    def result(self):
        tri = cc.triage(table(self.spec()))
        return {t["candidate"]: t for t in tri}, tri

    def test_dominated_in_every_scope(self):
        r, _ = self.result()
        self.assertTrue(r["Q"]["dominated"])
        self.assertEqual(r["Q"]["dominated_by"], ["P", "T"])   # P and T are equal and both beat Q
        self.assertTrue(r["Q"]["dropped"])

    def test_a_and_f_are_dominated_but_never_dropped(self):
        r, _ = self.result()
        for name in ("A_proxy", "F_full"):
            self.assertTrue(r[name]["dominated"])
            self.assertTrue(r[name]["protected"])
            self.assertFalse(r[name]["dropped"])

    def test_one_scope_is_not_enough(self):
        r, _ = self.result()
        # R beats A pooled and in c1 but not in c2 (jitter worse), so A is not dominated by R
        self.assertNotIn("R", r["A_proxy"]["dominated_by"])
        self.assertIn("R", r["A_proxy"]["per_scope"]["ALL"])
        self.assertNotIn("R", r["A_proxy"]["per_scope"]["c2"])

    def test_ties_do_not_dominate_each_other(self):
        r, _ = self.result()
        self.assertFalse(r["P"]["dominated"])
        self.assertFalse(r["T"]["dominated"])
        self.assertNotIn("T", r["P"]["per_scope"]["ALL"])

    def test_missing_value_never_dominates_or_is_dominated(self):
        r, _ = self.result()
        self.assertFalse(r["M"]["dominated"])
        self.assertNotIn("M", r["A_proxy"]["dominated_by"])

    def test_survivors_and_csv(self):
        r, tri = self.result()
        self.assertEqual(cc.survivors(tri), ["A_proxy", "F_full", "P", "R", "T", "M"])
        header, rows = cc.triage_csv_rows(tri)
        self.assertEqual(header[:5], ["candidate", "dominated", "dominated_by", "never_dropped", "dropped"])
        self.assertEqual(header[5:], [f"dominated_by_in_{sc}" for sc in self.SCOPES])
        q = next(row for row in rows if row[0] == "Q")
        self.assertEqual(q[1:5], ["yes", "P;T", "no", "yes"])

    def test_dominates_requires_a_strict_improvement(self):
        a = {"coverage": "0.9", "defined_to_none_per_1000_frames": "2.0", "jitter_common_mean": "0.01"}
        self.assertFalse(cc.dominates(a, dict(a)))
        b = dict(a, coverage="0.8")
        self.assertTrue(cc.dominates(a, b))
        self.assertFalse(cc.dominates(b, a))
        self.assertFalse(cc.dominates(dict(a, jitter_common_mean="0.02"), dict(a, coverage="0.8")))

    def test_manifest_has_rule_survivors_and_timing_statement(self):
        _, tri = self.result()
        text = cc.build_manifest([], [], Path("in"), Path("f"), Path("nope.csv"), Path("nope.pdf"),
                                 "male", cc.FIT_TOL, (True, []), tri=tri)
        for needle in ("== Triage", "chosen after the pooled first-run table", "No threshold",
                       "survivors", "A and F dominated but kept", "nothing is chosen or ranked"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
