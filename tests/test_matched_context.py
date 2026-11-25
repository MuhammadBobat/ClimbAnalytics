"""
Tests for evaluation/3.3/matched_context.py (spline against smoothed Kalman at matched context, D-029 addendum).
Synthetic skeletons; nothing here is a measurement and nothing validates a tier.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import occlusion_handling as oh
from tests.test_hide_and_fill import hf, trial, wiggly
from tests.test_occlusion_handling import to_joints

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("matched_context", REPO_ROOT / "evaluation" / "3.3" / "matched_context.py")
mc = importlib.util.module_from_spec(_spec)
sys.modules["matched_context"] = mc
_spec.loader.exec_module(mc)


def fills_for(joints, t):
    return mc.fill_batch(joints, [t])


class TestContextKnob(unittest.TestCase):
    """The context parameter really reaches the fill: a wired-but-ignored knob would make the whole comparison meaningless."""

    def setUp(self):
        self.joints = to_joints(wiggly())
        self.t = trial("left_wrist", 100, 3)

    def test_context_changes_the_spline(self):
        f = fills_for(self.joints, self.t)
        self.assertNotEqual(f[(0, "spline_c5")], f[(0, "spline_c10")])

    def test_context_changes_the_kalman(self):
        f = fills_for(self.joints, self.t)
        self.assertNotEqual(f[(0, "kalman_c5")], f[(0, "kalman_c10")])

    def test_shared_settings_equal_the_first_run(self):
        """spline_c5 and kalman_c10 are the first run's spline and kalman_smooth (defaults of OcclusionParams)."""
        f = fills_for(self.joints, self.t)
        old = hf.fill_batch(self.joints, [self.t], methods=["spline", "kalman_smooth"])
        self.assertEqual(f[(0, "spline_c5")], old[(0, "spline")])
        self.assertEqual(f[(0, "kalman_c10")], old[(0, "kalman_smooth")])

    def test_contexts_are_the_requested_ones(self):
        self.assertEqual(mc.CONFIGS["spline_c10"]["tier1_context"], 10)
        self.assertEqual(mc.CONFIGS["kalman_c5"]["tier2_context"], 5)
        self.assertEqual(oh.OcclusionParams().tier1_context, 5)     # the defaults are untouched
        self.assertEqual(oh.OcclusionParams().tier2_context, 10)

    def test_no_leakage_of_hidden_values(self):
        """Changing the TRUE values inside the hidden span must not change any fill."""
        other = to_joints(wiggly())
        s = other["left_wrist"]
        for f in self.t.span:
            x, y = s.positions[f]
            s.positions[f] = (x + 500.0, y - 500.0)
        a, b = fills_for(self.joints, self.t), fills_for(other, self.t)
        for cfg in ("spline_c5", "spline_c10", "kalman_c5", "kalman_c10", "hold"):
            self.assertEqual(a[(0, cfg)], b[(0, cfg)], cfg)


def fake_df(clip_effects):
    """Rows for 5 clips, one length, one group; Kalman minus spline error is clip_effects[i] on every trial of clip i."""
    rows = []
    for i, eff in enumerate(clip_effects):
        for k in range(6):
            base = 3.0 + 0.1 * k
            r = {"clip": f"c{i}", "length": 2, "joint": "left_wrist", "group": "limb", "start": 10 + 25 * k, "torso_px": 200.0,
                 "motion": 0.01, "motion_class": "smooth", "hold_mean_px": 5.0, "hold_mid_px": 5.0}
            for c in mc.CONTEXTS:
                r[f"spline_c{c}_mean_px"] = base
                r[f"kalman_c{c}_mean_px"] = base + eff
                r[f"spline_c{c}_mid_px"] = base
                r[f"kalman_c{c}_mid_px"] = base + eff
            rows.append(r)
    return pd.DataFrame(rows)


class TestAnalysis(unittest.TestCase):
    def run_analysis(self, effects):
        with tempfile.TemporaryDirectory() as d:
            summ, perclip, ctx = mc.analyse(fake_df(effects), Path(d), lengths=(2,))
        return {(r["group"], r["class"], r["context"]): r for r in summ}, perclip, ctx

    def test_paired_difference_and_survival(self):
        s, perclip, _ = self.run_analysis([-1.0] * 5)
        r = s[("limb", "all", 5)]
        self.assertAlmostEqual(r["diff_median"], -1.0)
        self.assertEqual(r["clips_kalman_lower"], 5)
        self.assertTrue(r["survives"])
        self.assertEqual(r["n_paired"], 30)
        self.assertEqual(len(perclip), 5 * 2 * 2)          # 5 clips x classes all and smooth (no sudden trials) x 2 contexts

    def test_three_of_five_clips_is_not_survival(self):
        s, _, _ = self.run_analysis([-1.0, -1.0, -1.0, 1.0, 1.0])
        r = s[("limb", "all", 10)]
        self.assertEqual(r["clips_kalman_lower"], 3)
        self.assertFalse(r["survives"])

    def test_four_of_five_with_negative_pooled_median_survives(self):
        s, _, _ = self.run_analysis([-1.0, -1.0, -1.0, -1.0, 1.0])
        self.assertTrue(s[("limb", "all", 10)]["survives"])

    def test_positive_pooled_median_does_not_survive(self):
        s, _, _ = self.run_analysis([1.0] * 5)
        self.assertFalse(s[("limb", "all", 5)]["survives"])

    def test_context_effect_sign(self):
        df = fake_df([0.0] * 5)
        df["kalman_c10_mean_px"] = df["kalman_c5_mean_px"] - 0.5      # more context helps the Kalman only
        with tempfile.TemporaryDirectory() as d:
            _, _, ctx = mc.analyse(df, Path(d), lengths=(2,))
        k = [r for r in ctx if r["method"] == "kalman" and r["class"] == "all"][0]
        sp = [r for r in ctx if r["method"] == "spline" and r["class"] == "all"][0]
        self.assertAlmostEqual(k["diff_median"], -0.5)
        self.assertAlmostEqual(sp["diff_median"], 0.0)

    def test_unavailable_fills_are_dropped_from_the_pair(self):
        df = fake_df([-1.0] * 5)
        df.loc[df.index[:3], "spline_c5_mean_px"] = np.nan
        with tempfile.TemporaryDirectory() as d:
            summ, _, _ = mc.analyse(df, Path(d), lengths=(2,))
        r = [x for x in summ if x["class"] == "all" and x["context"] == 5][0]
        self.assertEqual(r["n_trials"], 30)
        self.assertEqual(r["n_paired"], 27)


if __name__ == "__main__":
    unittest.main()
