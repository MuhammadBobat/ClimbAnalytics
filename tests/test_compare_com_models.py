"""
Tests for evaluation/2.2b/compare_com_models.py on a synthetic static stick-figure clip.
Hand-computed: the proxy CoM is (0, 5); the male segment-model CoM is (0, 9.244557) (see
tests/test_com_segment.py); torso length is 10 px, so the difference is 4.244557 px =
0.4244557 torso lengths. Nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compare_com_models", REPO_ROOT / "evaluation" / "2.2b" / "compare_com_models.py")
cm = importlib.util.module_from_spec(_spec)
sys.modules["compare_com_models"] = cm
_spec.loader.exec_module(cm)

from tests.test_com_segment import COCO, stick_figure  # noqa: E402

DIFF_PX = 9.244557 - 5.0


def write_clip(path, detected, low=()):
    pose = stick_figure()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "timestamp_s", "keypoint_id", "keypoint_name",
                    "x", "y", "confidence"])
        for fr in detected:
            for i, n in enumerate(COCO):
                w.writerow(["s.MOV", fr, f"{fr / 30:.6f}", i, n, pose[n][0], pose[n][1],
                            0.2 if (fr, n) in low else 0.9])


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        p = self.d / "s_keypoints.csv"
        # 10 frames: frame 6 dropped by the detector, 8 excluded by list, left_wrist low on 3
        write_clip(p, [f for f in range(10) if f != 6], low={(3, "left_wrist")})
        self.c = cm.compare_clip(p, excluded={8})

    def test_counts_and_scope(self):
        row = dict(zip(cm.SUMMARY_COLUMNS, cm.summary_row("s.MOV", "male", "male", [self.c])))
        self.assertEqual(row["n_frames"], 10)
        self.assertEqual(row["n_excluded_by_list"], 1)
        self.assertEqual(row["n_proxy_defined"], 8)     # all but 6 (dropped) and 8 (excluded)
        self.assertEqual(row["n_segment_defined"], 7)   # also not 3 (left wrist low confidence)
        self.assertEqual(row["n_both_defined"], 7)
        self.assertEqual(row["n_only_proxy_defined"], 1)
        self.assertEqual(row["n_only_segment_defined"], 0)
        self.assertEqual(row["role"], "primary")
        self.assertEqual(row["ldlj_difference"], "n/a")

    def test_difference_in_px_and_torso_lengths(self):
        row = dict(zip(cm.SUMMARY_COLUMNS, cm.summary_row("s.MOV", "male", "male", [self.c])))
        self.assertAlmostEqual(float(row["pos_diff_px_mean"]), DIFF_PX, places=3)
        self.assertAlmostEqual(float(row["pos_diff_px_max"]), DIFF_PX, places=3)
        self.assertAlmostEqual(float(row["pos_diff_torso_median"]), DIFF_PX / 10.0, places=4)

    def test_comparison_only_on_frames_where_both_defined(self):
        diffs = cm.frame_differences(self.c, "male")
        defined = [t for t, d in enumerate(diffs) if d is not None]
        self.assertEqual(defined, [0, 1, 2, 4, 5, 7, 9])
        self.assertIsNone(diffs[3])  # proxy defined, segment model not -> not compared

    def test_speed_pairs_only_on_consecutive_defined_steps(self):
        pairs = cm.speed_pairs(self.c, "male")
        self.assertEqual(len(pairs), 3)  # steps 1, 2 and 5
        self.assertTrue(all(a == 0.0 and b == 0.0 for a, b in pairs))

    def test_sensitivity_rows_and_roles(self):
        row = dict(zip(cm.SUMMARY_COLUMNS, cm.summary_row("s.MOV", "female", "male", [self.c])))
        self.assertEqual(row["role"], "sensitivity")
        self.assertEqual(row["n_both_defined"], 7)  # same joints needed for every sex table

    def test_frames_csv_columns_and_blanks(self):
        header, rows = cm.frames_rows(self.c)
        self.assertEqual(header[:5], ["frame_idx", "timestamp_s", "proxy_x", "proxy_y", "torso_length_px"])
        by_frame = {r[0]: dict(zip(header, r)) for r in rows}
        self.assertEqual(by_frame[3]["seg_male_x"], "")      # segment model undefined
        self.assertNotEqual(by_frame[3]["proxy_y"], "")      # proxy defined
        self.assertEqual(by_frame[6]["proxy_y"], "")         # detector drop
        self.assertEqual(by_frame[8]["proxy_y"], "")         # excluded by list
        self.assertAlmostEqual(float(by_frame[0]["diff_torso_male"]), DIFF_PX / 10.0, places=4)

    def test_manifest_states_scope_and_rerun_note(self):
        paper = self.d / "paper.pdf"
        paper.write_bytes(b"abc")
        text = cm.build_manifest([self.c], "male", self.d, self.d / "nofootage", self.d / "none.csv",
                                 paper, today="2026-10-28")
        self.assertIn("ONLY on frames where both models are defined: 7 of 10", text)
        self.assertIn("only the proxy is defined (not compared): 1", text)
        self.assertIn("Re-run after the occlusion handling (3.2)", text)
        self.assertIn("LDLJ difference: not available until 5.1", text)
        self.assertIn("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", text)
        self.assertIn("primary = male", text)
        self.assertIn("Table 4 (p. 1228)", text)
        self.assertIn("video not found", text)

    def test_plot_is_written(self):
        out = self.d / "p.png"
        cm.plot_clip(self.c, "male", out)
        self.assertTrue(out.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
