"""
Tests for the three analyses in evaluation/2.2b/compare_com_models.py: constant-offset versus
time-varying split, speed correlation, and the largest speed differences. Synthetic poses with
hand-computed answers; nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_com_segment import COCO, stick_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compare_com_models", REPO_ROOT / "evaluation" / "2.2b" / "compare_com_models.py")
cm = sys.modules.get("compare_com_models") or importlib.util.module_from_spec(_spec)
if "compare_com_models" not in sys.modules:
    sys.modules["compare_com_models"] = cm
    _spec.loader.exec_module(cm)

DIFF_PX = 9.244557 - 5.0           # male segment-model CoM minus proxy, static stick figure
# weight of the left wrist keypoint in the male segment-model CoM: forearm CM (r = 0.4574 of
# mass 0.0162) plus the hand point mass 0.0061; masses sum to 1
WRIST_WEIGHT = 0.0162 * 0.4574 + 0.0061


def write_clip(path, n_frames, dx=lambda t: 0.0, jumps=None, conf=None):
    """Stick figure translated by dx(t) per frame; jumps {frame: (joint, dx_px)} move one joint;
    conf {(frame, joint): value} overrides the confidence (default 0.9)."""
    jumps, conf = jumps or {}, conf or {}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "timestamp_s", "keypoint_id", "keypoint_name",
                    "x", "y", "confidence"])
        for fr in range(n_frames):
            pose = stick_figure(dx=dx(fr))
            if fr in jumps:
                j, jd = jumps[fr]
                pose[j] = (pose[j][0] + jd, pose[j][1])
            for i, n in enumerate(COCO):
                w.writerow(["s.MOV", fr, f"{fr / 30:.9f}", i, n, pose[n][0], pose[n][1],
                            conf.get((fr, n), 0.9)])


class DecomposeTests(unittest.TestCase):
    def test_constant_offset_along_axis_has_no_timevarying_part(self):
        off, resid = cm.decompose_offset([(0.0, 2.0)] * 4, [(0.0, 10.0)] * 4)
        self.assertAlmostEqual(off, 0.2)
        self.assertTrue(all(abs(r) < 1e-12 for r in resid))

    def test_across_axis_wobble_is_timevarying(self):
        diffs = [(1.0, 2.0), (-1.0, 2.0)] * 2
        off, resid = cm.decompose_offset(diffs, [(0.0, 10.0)] * 4)
        self.assertAlmostEqual(off, 0.2)
        for r in resid:
            self.assertAlmostEqual(r, 0.1)

    def test_along_axis_variation_is_timevarying(self):
        off, resid = cm.decompose_offset([(0.0, 1.0), (0.0, 3.0)], [(0.0, 10.0)] * 2)
        self.assertAlmostEqual(off, 0.2)
        self.assertAlmostEqual(resid[0], 0.1)
        self.assertAlmostEqual(resid[1], 0.1)

    def test_rotated_axis(self):
        # axis (10, 0), d = (3, 4): along = 0.3 torso lengths; residual after removing it is (0, 4)/10
        off, resid = cm.decompose_offset([(3.0, 4.0)], [(10.0, 0.0)])
        self.assertAlmostEqual(off, 0.3)
        self.assertAlmostEqual(resid[0], 0.4)

    def test_static_clip_offset_is_all_constant(self):
        d = Path(tempfile.mkdtemp())
        write_clip(d / "s_keypoints.csv", 8)
        c = cm.compare_clip(d / "s_keypoints.csv", None)
        off, resid, n = cm.offset_for_clip(c, "male")
        self.assertEqual(n, 8)
        self.assertAlmostEqual(off, DIFF_PX / 10.0, places=5)  # toward the hips (+y), torso = 10 px
        self.assertTrue(all(abs(r) < 1e-9 for r in resid))
        row = dict(zip(cm.SUMMARY_COLUMNS, cm.summary_row("s.MOV", "male", "male", [c])))
        self.assertAlmostEqual(float(row["offset_along_torso_axis_torso"]), DIFF_PX / 10.0, places=4)
        self.assertAlmostEqual(float(row["timevarying_max_torso"]), 0.0, places=4)


class CorrelationTests(unittest.TestCase):
    def test_pearson_known_values(self):
        self.assertAlmostEqual(cm.pearson([(1, 2), (2, 4), (3, 6), (4, 8)]), 1.0)
        self.assertAlmostEqual(cm.pearson([(1, 4), (2, 3), (3, 2), (4, 1)]), -1.0)

    def test_pearson_undefined_cases(self):
        self.assertIsNone(cm.pearson([(1, 1), (2, 2)]))                  # fewer than 3
        self.assertIsNone(cm.pearson([(1, 5), (2, 5), (3, 5)]))          # no variance

    def test_identical_translation_gives_r_one_in_summary(self):
        d = Path(tempfile.mkdtemp())
        write_clip(d / "s_keypoints.csv", 12, dx=lambda t: 0.5 * t * t)  # speeds vary frame to frame
        c = cm.compare_clip(d / "s_keypoints.csv", None)
        row = dict(zip(cm.SUMMARY_COLUMNS, cm.summary_row("s.MOV", "male", "male", [c])))
        self.assertAlmostEqual(float(row["speed_corr_pearson"]), 1.0, places=4)
        all_row = dict(zip(cm.SUMMARY_COLUMNS, cm.summary_row("ALL", "male", "male", [c])))
        self.assertAlmostEqual(float(all_row["speed_corr_pearson"]), 1.0, places=4)

    def test_static_clip_has_empty_correlation(self):
        d = Path(tempfile.mkdtemp())
        write_clip(d / "s_keypoints.csv", 6)
        c = cm.compare_clip(d / "s_keypoints.csv", None)
        row = dict(zip(cm.SUMMARY_COLUMNS, cm.summary_row("s.MOV", "male", "male", [c])))
        self.assertEqual(row["speed_corr_pearson"], "")


class TopDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        # left wrist jumps 30 px in x at frame 4 only (confidence 0.55 there); the proxy does not
        # use the wrist, so only the segment-model CoM moves
        write_clip(self.d / "s_keypoints.csv", 10, jumps={4: ("left_wrist", 30.0)},
                   conf={(4, "left_wrist"): 0.55})
        self.c = cm.compare_clip(self.d / "s_keypoints.csv", None)
        self.expected = WRIST_WEIGHT * 30.0 * 30.0  # px/s: CoM shift in x times 30 fps

    def test_two_largest_are_the_steps_around_the_jump(self):
        rows = cm.top_speed_differences([self.c], "male", n=2)
        self.assertEqual(len(rows), 2)
        steps = sorted((r[2], r[3]) for r in rows)
        self.assertEqual(steps, [(3, 4), (4, 5)])
        for r in rows:
            self.assertAlmostEqual(float(r[5]), 0.0, places=3)                  # proxy speed
            self.assertAlmostEqual(float(r[6]), self.expected, places=2)        # segment speed
            self.assertAlmostEqual(float(r[7]), self.expected, places=2)

    def test_lowest_confidence_joints_listed_per_frame(self):
        rows = {(r[2], r[3]): r for r in cm.top_speed_differences([self.c], "male", n=2)}
        enter = rows[(3, 4)]
        self.assertTrue(enter[9].startswith("left_wrist:0.5500"))   # frame_to = 4 has the 0.55 joint
        self.assertEqual(len(enter[9].split("; ")), cm.LOWEST_CONF_N)
        self.assertNotIn("0.5500", enter[8])                        # frame_from = 3: all 0.9
        leave = rows[(4, 5)]
        self.assertTrue(leave[8].startswith("left_wrist:0.5500"))

    def test_default_cap_and_order(self):
        rows = cm.top_speed_differences([self.c], "male")
        self.assertEqual(len(rows), 9)  # 10 frames -> 9 steps, fewer than the cap of 10
        diffs = [float(r[7]) for r in rows]
        self.assertEqual(diffs, sorted(diffs, reverse=True))
        self.assertEqual([r[0] for r in rows], list(range(1, 10)))
        self.assertEqual(cm.TOP_N, 10)

    def test_pooled_over_clips(self):
        write_clip(self.d / "t_keypoints.csv", 10, jumps={6: ("right_wrist", 60.0)})
        c2 = cm.compare_clip(self.d / "t_keypoints.csv", None)
        c2.clip = "t.MOV"
        rows = cm.top_speed_differences([self.c, c2], "male", n=4)
        self.assertEqual({r[1] for r in rows}, {"s.MOV", "t.MOV"})
        self.assertEqual(rows[0][1], "t.MOV")  # the 60 px jump is the largest


class ManifestAndColumnsTests(unittest.TestCase):
    def test_new_columns_present_and_ldlj_last(self):
        for col in ("offset_along_torso_axis_torso", "offset_abs_torso", "timevarying_mean_torso",
                    "timevarying_median_torso", "timevarying_max_torso", "timevarying_rms_torso",
                    "speed_corr_pearson"):
            self.assertIn(col, cm.SUMMARY_COLUMNS)
        self.assertEqual(cm.SUMMARY_COLUMNS[-1], "ldlj_difference")

    def test_manifest_defines_analyses(self):
        d = Path(tempfile.mkdtemp())
        write_clip(d / "s_keypoints.csv", 6)
        c = cm.compare_clip(d / "s_keypoints.csv", None)
        text = cm.build_manifest([c], "male", d, d / "nofootage", d / "none.csv", d / "none.pdf",
                                 today="2026-10-28")
        self.assertIn("positive = toward the hips", text)
        self.assertIn("Pearson r", text)
        self.assertIn("top_speed_differences.csv", text)


if __name__ == "__main__":
    unittest.main()
