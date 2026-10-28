"""
Tests for metrics.compute_com (ARCHITECTURE.md §4.3) — the trunk-centroid PROXY.

Correctness tests use SYNTHETIC joints with hand-computed answers; nothing here
is a measurement. The real-data test is a smoke test (invariants only) and
skips with a message if the gitignored validation output is absent.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import sys
import unittest
from pathlib import Path

from metrics import (
    GAP_LOW_CONFIDENCE,
    GAP_MISSING_DETECTION,
    TRUNK_CENTROID_PROXY_WEIGHTS,
    JointSeries,
    compute_com,
    compute_velocity,
    load_keypoints_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRUNK = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def make_joints(frames):
    """frames: list of dicts {joint: (x, y) | None}; timestamps at 30 fps."""
    ts = [i / 30.0 for i in range(len(frames))]
    return {
        name: JointSeries(name, [f[name] for f in frames], ts)
        for name in TRUNK
    }


def rect_frame(dx=0.0, dy=0.0, k=1.0):
    # shoulders (-1,0),(1,0); hips (-1,2),(1,2) -> CoM (0,1)
    return {
        "left_shoulder": (k * -1.0 + dx, k * 0.0 + dy),
        "right_shoulder": (k * 1.0 + dx, k * 0.0 + dy),
        "left_hip": (k * -1.0 + dx, k * 2.0 + dy),
        "right_hip": (k * 1.0 + dx, k * 2.0 + dy),
    }


class TrunkProxyTests(unittest.TestCase):
    def test_known_answer(self):
        com = compute_com(make_joints([rect_frame()]))
        self.assertEqual(com.positions[0], (0.0, 1.0))

    def test_equals_mean_of_hip_and_shoulder_midpoints(self):
        frame = {
            "left_shoulder": (10.0, 3.0), "right_shoulder": (14.0, 5.0),
            "left_hip": (11.0, 20.0), "right_hip": (17.0, 24.0),
        }
        sh_mid = (12.0, 4.0)
        hip_mid = (14.0, 22.0)
        expected = ((sh_mid[0] + hip_mid[0]) / 2, (sh_mid[1] + hip_mid[1]) / 2)
        got = compute_com(make_joints([frame])).positions[0]
        self.assertAlmostEqual(got[0], expected[0])
        self.assertAlmostEqual(got[1], expected[1])

    def test_translation(self):
        com = compute_com(make_joints([rect_frame(dx=100.0, dy=-40.0)]))
        self.assertEqual(com.positions[0], (100.0, -39.0))

    def test_left_right_swap_unchanged(self):
        frame = {
            "left_shoulder": (10.0, 3.0), "right_shoulder": (14.0, 5.0),
            "left_hip": (11.0, 20.0), "right_hip": (17.0, 24.0),
        }
        swapped = {
            "left_shoulder": frame["right_shoulder"], "right_shoulder": frame["left_shoulder"],
            "left_hip": frame["right_hip"], "right_hip": frame["left_hip"],
        }
        a = compute_com(make_joints([frame])).positions[0]
        b = compute_com(make_joints([swapped])).positions[0]
        self.assertAlmostEqual(a[0], b[0])
        self.assertAlmostEqual(a[1], b[1])

    def test_scale(self):
        com = compute_com(make_joints([rect_frame(k=3.0)]))
        self.assertEqual(com.positions[0], (0.0, 3.0))

    def test_missing_hip_gives_none_not_renormalised(self):
        f = rect_frame()
        f["left_hip"] = None
        com = compute_com(make_joints([rect_frame(), f, rect_frame()]))
        self.assertEqual(com.positions[0], (0.0, 1.0))
        self.assertIsNone(com.positions[1])
        self.assertEqual(com.positions[2], (0.0, 1.0))

    def test_gap_reason_propagates(self):
        joints = make_joints([rect_frame(), rect_frame(), rect_frame()])
        joints["left_hip"].positions[1] = None
        joints["left_hip"].gap_reasons[1] = GAP_LOW_CONFIDENCE
        joints["right_hip"].positions[2] = None
        joints["right_hip"].gap_reasons[2] = GAP_LOW_CONFIDENCE
        joints["left_shoulder"].positions[2] = None
        joints["left_shoulder"].gap_reasons[2] = GAP_MISSING_DETECTION
        com = compute_com(joints)
        self.assertEqual(com.gap_reasons, [None, GAP_LOW_CONFIDENCE, GAP_MISSING_DETECTION])

    def test_moving_com_feeds_velocity(self):
        # whole trunk moves (3,4) px/frame -> CoM speed 150 px/s at 30 fps
        frames = [rect_frame(dx=3.0 * i, dy=4.0 * i) for i in range(4)]
        v = compute_velocity(compute_com(make_joints(frames)))
        self.assertIsNone(v[0])
        for x in v[1:]:
            self.assertAlmostEqual(x, 150.0)

    def test_default_table_is_equal_weights(self):
        self.assertEqual(set(TRUNK_CENTROID_PROXY_WEIGHTS), set(TRUNK))
        self.assertEqual(len(set(TRUNK_CENTROID_PROXY_WEIGHTS.values())), 1)


class SwappableWeightsTests(unittest.TestCase):
    def test_custom_table_known_answer(self):
        # synthetic weights (NOT anthropometric): left_hip x3, right_hip x1
        joints = make_joints([{
            "left_shoulder": (0.0, 0.0), "right_shoulder": (0.0, 0.0),
            "left_hip": (0.0, 0.0), "right_hip": (4.0, 8.0),
        }])
        com = compute_com(joints, {"left_hip": 3.0, "right_hip": 1.0})
        self.assertEqual(com.positions[0], (1.0, 2.0))

    def test_scaling_all_weights_changes_nothing(self):
        joints = make_joints([rect_frame(dx=5.0)])
        a = compute_com(joints, {n: 1.0 for n in TRUNK}).positions[0]
        b = compute_com(joints, {n: 7.5 for n in TRUNK}).positions[0]
        self.assertAlmostEqual(a[0], b[0])
        self.assertAlmostEqual(a[1], b[1])

    def test_bad_weights_rejected(self):
        joints = make_joints([rect_frame()])
        with self.assertRaises(ValueError):
            compute_com(joints, {})
        with self.assertRaises(ValueError):
            compute_com(joints, {"left_hip": -1.0, "right_hip": 2.0})
        with self.assertRaises(KeyError):
            compute_com(joints, {"nose": 1.0})


class RealDataSmokeTest(unittest.TestCase):
    """Invariants only — no numeric expectations."""

    CSV = REPO_ROOT / "validation" / "output" / "IMG_2412_keypoints.csv"

    def test_com_inside_bounding_box_of_inputs(self):
        if not self.CSV.exists():
            msg = (f"SKIPPED real-data smoke test: {self.CSV.relative_to(REPO_ROOT)} "
                   "not found (validation/output/ is gitignored; run "
                   "validation/validate_vitpose.py to generate it)")
            print(msg, file=sys.stderr)
            self.skipTest(msg)
        joints = load_keypoints_csv(self.CSV)
        com = compute_com(joints)
        n_defined = 0
        for t, c in enumerate(com.positions):
            pts = [joints[name].positions[t] for name in TRUNK_CENTROID_PROXY_WEIGHTS]
            if any(p is None for p in pts):
                self.assertIsNone(c)
                continue
            n_defined += 1
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            self.assertGreaterEqual(c[0], min(xs) - 1e-9)
            self.assertLessEqual(c[0], max(xs) + 1e-9)
            self.assertGreaterEqual(c[1], min(ys) - 1e-9)
            self.assertLessEqual(c[1], max(ys) + 1e-9)
        self.assertGreater(n_defined, 0)


if __name__ == "__main__":
    unittest.main()
