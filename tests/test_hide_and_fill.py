"""
Tests for evaluation/3.3/hide_and_fill.py (hide-and-fill evaluation of the occlusion tiers, D-007, D-028, D-029).
Synthetic skeletons; nothing here is a measurement and nothing validates a tier.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import occlusion_handling as oh
from metrics import JointSeries
from tests.test_occlusion_handling import rigid_skeleton, to_joints

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("hide_and_fill", REPO_ROOT / "evaluation" / "3.3" / "hide_and_fill.py")
hf = importlib.util.module_from_spec(_spec)
sys.modules["hide_and_fill"] = hf
_spec.loader.exec_module(hf)

N = 300


def wiggly(n=N):
    """A rigid skeleton whose limb joints also wiggle, so the methods give different answers."""
    pos = rigid_skeleton(n)
    for j in oh.ARM_JOINTS + oh.LEG_JOINTS:
        k = oh.BODY_JOINTS.index(j)
        pos[j] = [(x + 6 * math.sin(0.21 * t + k), y + 5 * math.cos(0.17 * t + 2 * k)) for t, (x, y) in enumerate(pos[j])]
    return pos


def trial(joint, start, length, torso=100.0, motion=0.0):
    return hf.Trial("c", joint, length, start, torso, motion)


class TrialTests(unittest.TestCase):
    def test_starts_every_25_frames_from_10_and_context_required(self):
        joints = to_joints(wiggly())
        tr = hf.make_trials("c", joints, (5,))
        self.assertEqual(sorted({t.start for t in tr}), list(range(10, 286, 25)))
        self.assertEqual(len(tr), 12 * 12)                       # 12 starts x 12 joints
        for t in tr:
            self.assertGreaterEqual(t.start - hf.CONTEXT, 0)
            self.assertLessEqual(t.start + t.length - 1 + hf.CONTEXT, N - 1)

    def test_a_missing_joint_anywhere_in_the_window_removes_the_trial(self):
        joints = to_joints(wiggly(), {"right_knee": [(100, 100)]})
        starts = {t.start for t in hf.make_trials("c", joints, (5,))}
        self.assertNotIn(110, starts)           # window 100..124 contains the missing frame
        self.assertIn(85, starts)               # window 75..99 does not
        # a missing frame in a DIFFERENT joint removes the trial of every joint (all 12 must be observed)
        self.assertEqual(len([t for t in hf.make_trials("c", joints, (5,)) if t.start == 110]), 0)

    def test_torso_is_the_clip_median(self):
        joints = to_joints(wiggly())
        torso = hf.median_torso_px(joints)
        direct = []
        for t in range(N):
            ls, rs, lh, rh = (joints[n].positions[t] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
            direct.append(math.hypot((ls[0] + rs[0]) / 2 - (lh[0] + rh[0]) / 2, (ls[1] + rs[1]) / 2 - (lh[1] + rh[1]) / 2))
        self.assertAlmostEqual(torso, float(np.median(direct)), places=9)
        self.assertEqual({t.torso_px for t in hf.make_trials("c", joints, (5,))}, {torso})

    def test_motion_measure_is_the_peak_second_difference_of_the_true_positions_over_the_margin(self):
        pos = rigid_skeleton(N)
        pos["left_wrist"] = [(2.0 * t, -1.0 * t) for t in range(N)]
        pos["left_wrist"][52] = (104.0 + 30.0, -52.0)             # a 30 px kink: second differences 30, -60, 30 at 51..53
        joints = to_joints(pos)
        tr = {(t.joint, t.start): t for t in hf.make_trials("c", joints, (30,))}
        torso = hf.median_torso_px(joints)
        self.assertAlmostEqual(tr[("left_wrist", 35)].motion, 60.0 / torso, places=9)   # span 35..64: t = 34..65 contains 52
        self.assertLess(tr[("left_wrist", 10)].motion, 1e-9)                            # t = 9..40: a straight line
        self.assertLess(tr[("left_wrist", 60)].motion, 1e-9)                            # t = 59..90
        self.assertAlmostEqual(hf.peak_second_difference(pos["left_wrist"], 48, 54), 60.0, places=9)

    def test_terciles_are_within_each_length_and_joint_and_use_motion_only(self):
        rng = np.random.default_rng(1)
        trials = []
        for L in (3, 8):
            for j in ("left_wrist", "right_knee"):
                for i in range(30):
                    trials.append(hf.Trial("c", j, L, 10 + 25 * i, 100.0, float(rng.random() * (10 if (L, j) == (8, "right_knee") else 1))))
        cuts = hf.tercile_cuts(trials)
        self.assertEqual(set(cuts), {(3, "left_wrist"), (3, "right_knee"), (8, "left_wrist"), (8, "right_knee")})
        self.assertGreater(cuts[(8, "right_knee")][1], 3 * cuts[(8, "left_wrist")][1])     # cut points differ per cell
        cl = hf.classify(trials, cuts)
        for key in cuts:
            labels = [t.motion_class for t in cl if (t.length, t.joint) == key]
            self.assertEqual(labels.count("smooth"), 10)
            self.assertEqual(labels.count("sudden"), 10)
            self.assertEqual(labels.count("middle"), 10)
        for t in cl:
            lo, hi = cuts[(t.length, t.joint)]
            self.assertEqual(t.motion_class == "smooth", t.motion <= lo)

    def test_batches_one_gap_per_joint_and_spans_60_frames_apart(self):
        joints = to_joints(wiggly(1500))
        trials = hf.make_trials("c", joints, (10,))
        batches = hf.schedule_batches(trials)
        self.assertEqual(sorted(id(t) for b in batches for t in b), sorted(id(t) for t in trials))   # every trial once
        for b in batches:
            self.assertEqual(len({t.joint for t in b}), len(b))
            for i, a in enumerate(b):
                for c in b[i + 1:]:
                    first, second = (a, c) if a.start < c.start else (c, a)
                    self.assertGreaterEqual(second.start - (first.start + first.length - 1) - 1, hf.MIN_SEPARATION)


class FillAndScoreTests(unittest.TestCase):
    def setUp(self):
        self.joints = to_joints(wiggly())

    def test_the_real_fill_gaps_is_called_with_the_forced_method(self):
        seen = []
        real = hf.oh.fill_gaps

        def spy(joints, params=oh.OcclusionParams()):
            seen.append((params.force_method, {n: s.positions[100] for n, s in joints.items()}))
            return real(joints, params)
        hf.oh.fill_gaps = spy
        try:
            hf.fill_batch(self.joints, [trial("left_wrist", 100, 8)], ["spline", "kalman_smooth", "trunk_offset", "hold"])
        finally:
            hf.oh.fill_gaps = real
        self.assertEqual([s[0] for s in seen], ["spline", "kalman", "trunk_offset"])       # hold does not call it
        self.assertTrue(all(s[1]["left_wrist"] is None for s in seen))                      # the masked series was passed

    def test_hold_error_is_the_displacement_from_the_last_observed_frame(self):
        t = trial("left_wrist", 100, 4)
        fills = hf.fill_batch(self.joints, [t], ["hold"])
        mean, mid = hf.score(t, self.joints, fills[(0, "hold")])
        last = self.joints["left_wrist"].positions[99]
        errs = [math.hypot(self.joints["left_wrist"].positions[f][0] - last[0], self.joints["left_wrist"].positions[f][1] - last[1])
                for f in range(100, 104)]
        self.assertAlmostEqual(mean, float(np.mean(errs)), places=9)
        self.assertAlmostEqual(mid, float(np.mean([errs[1], errs[2]])), places=9)

    def test_a_spline_on_a_line_has_zero_error_at_any_length(self):
        pos = {j: [(2.0 * t + 3, -t + 7) for t in range(N)] for j in oh.BODY_JOINTS}
        joints = to_joints(pos)
        for L in (1, 10, 40):
            t = trial("left_wrist", 100, L)
            mean, _ = hf.score(t, joints, hf.fill_batch(joints, [t], ["spline"])[(0, "spline")])
            self.assertLess(mean, 1e-6, L)

    def test_unavailable_fill_scores_none(self):
        t = trial("left_hip", 100, 30)            # trunk joint: Tier 3 is unavailable by design
        fills = hf.fill_batch(self.joints, [t], ["trunk_offset"])
        self.assertIsNone(fills[(0, "trunk_offset")])
        self.assertEqual(hf.score(t, self.joints, None), (None, None))

    def test_clamp_has_no_effect_on_trunk_joints_and_mirror_none_on_limb_joints(self):
        for joint, a, b in (("left_hip", "kalman_smooth", "kalman_smooth_noclamp"),
                            ("left_wrist", "kalman_smooth", "kalman_smooth_mirror"),
                            ("left_hip", "kalman_causal", "kalman_causal_noclamp")):
            t = trial(joint, 100, 12)
            f = hf.fill_batch(self.joints, [t], [a, b])
            self.assertEqual(f[(0, a)], f[(0, b)], (joint, a, b))
        # and the switch that DOES act is visible: a mirror result differs from the Kalman result for a trunk joint
        t = trial("left_hip", 100, 12)
        f = hf.fill_batch(self.joints, [t], ["kalman_smooth", "kalman_smooth_mirror"])
        self.assertNotEqual(f[(0, "kalman_smooth")], f[(0, "kalman_smooth_mirror")])

    def test_the_input_joints_are_not_modified(self):
        before = {n: list(s.positions) for n, s in self.joints.items()}
        hf.fill_batch(self.joints, [trial("left_wrist", 100, 8), trial("right_knee", 200, 8)])
        self.assertEqual({n: s.positions for n, s in self.joints.items()}, before)


class LeakageTests(unittest.TestCase):
    """Changing the hidden true values must not change any fill (no method may see what it is asked to recover)."""

    def perturbed(self, joints, batch):
        out = {n: JointSeries(n, list(s.positions), s.timestamps, list(s.gap_reasons), list(s.confidences)) for n, s in joints.items()}
        for t in batch:
            s = out[t.joint]
            for f in t.span:
                s.positions[f] = (s.positions[f][0] + 500.0, s.positions[f][1] - 321.0)
                s.confidences[f] = 0.99
        return out

    def batch(self):
        return [trial("left_wrist", 60, 12), trial("right_knee", 150, 7), trial("left_hip", 230, 20)]

    def test_no_method_output_depends_on_the_hidden_true_values(self):
        joints = to_joints(wiggly())
        batch = self.batch()
        a = hf.fill_batch(joints, batch)
        b = hf.fill_batch(self.perturbed(joints, batch), batch)
        self.assertEqual(set(a), set(b))
        self.assertTrue(any(v is not None for v in a.values()))
        for key in a:
            self.assertEqual(a[key], b[key], key)
        # while the TRUE values did change, so the errors are not trivially equal
        t = batch[0]
        ea = hf.score(t, joints, a[(0, "spline")])[0]
        eb = hf.score(t, self.perturbed(joints, batch), b[(0, "spline")])[0]
        self.assertNotAlmostEqual(ea, eb, places=3)

    def test_the_leakage_test_can_fail(self):
        """Canary: with a mask that hides nothing, the same comparison detects the leak."""
        joints = to_joints(wiggly())
        batch = self.batch()
        leaky = lambda j, b: j      # noqa: E731
        a = hf.fill_batch(joints, batch, ["spline"], mask=leaky)
        b = hf.fill_batch(self.perturbed(joints, batch), batch, ["spline"], mask=leaky)
        self.assertNotEqual(a, b)

    def test_mask_removes_positions_reasons_and_confidences_of_the_hidden_span_only(self):
        joints = to_joints(wiggly())
        batch = self.batch()
        m = hf.mask_joints(joints, batch)
        for t in batch:
            for f in t.span:
                self.assertIsNone(m[t.joint].positions[f])
                self.assertIsNone(m[t.joint].confidences[f])
                self.assertEqual(m[t.joint].gap_reasons[f], hf.GAP_LOW_CONFIDENCE)
            self.assertIsNotNone(m[t.joint].positions[t.start - 1])
            self.assertIsNotNone(m[t.joint].positions[t.start + t.length])
        self.assertEqual(m["right_wrist"].positions, joints["right_wrist"].positions)


class AnalysisTests(unittest.TestCase):
    def test_crossover_is_the_first_length_with_median_paired_difference_at_least_zero(self):
        d = {1: np.array([-3.0, -1.0, -2.0]), 2: np.array([-1.0, -0.5, 4.0]), 3: np.array([0.0, 1.0, 2.0]), 4: np.array([-5.0])}
        self.assertEqual(hf.crossover((1, 2, 3, 4), d), 3)
        self.assertTrue(hf.recovers_after((1, 2, 3, 4), d, 3))
        self.assertIsNone(hf.crossover((1, 2), d))
        self.assertFalse(hf.recovers_after((1, 2), d, None))

    def test_pstats(self):
        s = hf.pstats(np.array([1.0, 2.0, 3.0, 4.0, np.nan]))
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["median"], 2.5)
        self.assertEqual(hf.pstats(np.array([np.nan]))["n"], 0)

    def test_analyse_on_a_planted_table(self):
        import pandas as pd
        rng = np.random.default_rng(0)
        rows = []
        for clip in ("a", "b", "c"):
            for L in (1, 2, 3):
                for joint in ("left_wrist", "left_hip"):
                    for i in range(12):
                        base = {"clip": clip, "length": L, "joint": joint, "group": hf.group_of(joint), "start": 10 + 25 * i,
                                "torso_px": 100.0, "motion": float(i), "motion_class": ["smooth", "middle", "sudden"][i % 3]}
                        hold = 10.0 * L + rng.random()
                        for m in hf.METHOD_ORDER:
                            if m == "hold":
                                v = hold
                            elif m == "spline":
                                v = hold + (L - 2) * 5.0 + rng.random()        # better than hold at L=1, worse from L=3 (planted)
                            else:
                                v = hold * 0.5
                            base[f"{m}_mean_px"] = v
                            base[f"{m}_mid_px"] = v
                        rows.append(base)
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            summ, perclip, paired, cross = hf.analyse(df, Path(tmp), (1, 2, 3))
            for name in ("summary_by_length.csv", "per_clip.csv", "paired_vs_hold.csv", "crossover.csv"):
                self.assertTrue((Path(tmp) / name).exists(), name)
            hf.write_report(Path(tmp), summ, cross, (1, 2, 3))
            self.assertTrue((Path(tmp) / "report.md").exists())
            c = {(r["set"], r["method"], r["class"]): r for r in cross}
            self.assertEqual(c[("limb", "spline", "all")]["crossover_length"], 2)   # diff = (L-2)*5 + noise: median >= 0 from L = 2
            self.assertEqual(c[("limb", "kalman_smooth", "all")]["crossover_length"], "none through 3")
            self.assertEqual(c[("trunk", "kalman_smooth_mirror", "all")]["crossover_length"], "none through 3")
            self.assertNotIn(("trunk", "trunk_offset", "all"), c)                    # Tier 3 is limb-only
            self.assertNotIn(("limb", "kalman_smooth_mirror", "all"), c)             # mirror is trunk-only
            ns = {(r["set"], r["method"], r["class"], r["length"]): r["n_trials"] for r in summ}
            self.assertEqual(ns[("limb", "spline", "all", 1)], 36)                    # 3 clips x 12
            self.assertEqual(ns[("limb", "spline", "smooth", 1)], 12)                 # 4 per clip x 3 clips


if __name__ == "__main__":
    unittest.main()
