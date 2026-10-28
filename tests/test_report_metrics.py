"""
Tests for evaluation/2.2/report_metrics.py. Synthetic input, hand-computed answers;
the overlay test writes a tiny synthetic video. Nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "report_metrics", REPO_ROOT / "evaluation" / "2.2" / "report_metrics.py")
rm = importlib.util.module_from_spec(_spec)
sys.modules["report_metrics"] = rm  # dataclasses looks the module up by name
_spec.loader.exec_module(rm)

NAMES = rm.COCO17_NAMES


def write_clip(path, frames, low=(), video="s.MOV", step=(3.0, 4.0)):
    """Every joint moves by `step` px/frame from (100, 100); confidence 0.2 where (frame, joint) in low."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "timestamp_s", "keypoint_id", "keypoint_name",
                    "x", "y", "confidence"])
        for fr in frames:
            for i, n in enumerate(NAMES):
                w.writerow([video, fr, f"{fr / 30:.6f}", i, n, 100 + step[0] * fr, 100 + step[1] * fr,
                            0.2 if (fr, n) in low else 0.9])


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_reasons_and_counts(self):
        # 10 frames 0..9; frame 3 dropped by detector; 5,6 excluded; left_hip low on 8
        p = self.d / "s_keypoints.csv"
        write_clip(p, [f for f in range(10) if f != 3], low={(8, "left_hip")})
        res = rm.analyse_clip(p, excluded={5, 6})
        row = dict(zip(rm.SUMMARY_COLUMNS, rm.summary_row(res)))
        self.assertEqual(row["frames"], 10)
        self.assertEqual(row["frames_with_com"], 6)       # 0,1,2,4,7,9
        self.assertEqual(row["frames_without_com"], 4)    # 3,5,6,8
        self.assertEqual(row["without_com_detector_drop"], 1)
        self.assertEqual(row["without_com_excluded_by_list"], 2)
        self.assertEqual(row["without_com_low_confidence"], 1)
        self.assertEqual(res.com_reasons[3], metrics.GAP_MISSING_DETECTION)
        self.assertEqual(res.com_reasons[5], rm.EXCLUDED)
        self.assertEqual(res.com_reasons[8], metrics.GAP_LOW_CONFIDENCE)
        self.assertIsNone(res.com_reasons[0])

    def test_excluded_frames_removed_from_every_joint_and_com(self):
        p = self.d / "s_keypoints.csv"
        write_clip(p, range(8))
        res = rm.analyse_clip(p, excluded={4})
        for s in list(res.joints.values()) + [res.com]:
            self.assertIsNone(s.positions[4])
        self.assertEqual(res.joints["left_wrist"].gap_reasons[4], rm.EXCLUDED)

    def test_priority_excluded_over_drop_over_low_conf(self):
        p = self.d / "s_keypoints.csv"
        write_clip(p, range(6), low={(2, "left_hip"), (3, "left_hip")})
        res = rm.analyse_clip(p, excluded={3})
        self.assertEqual(res.com_reasons[2], metrics.GAP_LOW_CONFIDENCE)
        self.assertEqual(res.com_reasons[3], rm.EXCLUDED)

    def test_speed_and_displacement_use_metrics(self):
        # (3,4) px/frame at 30 fps -> 150 px/s, 5 px per step; 6 frames -> 25 px
        p = self.d / "s_keypoints.csv"
        write_clip(p, range(6))
        res = rm.analyse_clip(p)
        row = dict(zip(rm.SUMMARY_COLUMNS, rm.summary_row(res)))
        # the synthetic CSV rounds timestamps to 6 decimals, so allow a tiny tolerance
        self.assertAlmostEqual(float(row["mean_com_speed_px_s"]), 150.0, places=1)
        self.assertAlmostEqual(float(row["max_com_speed_px_s"]), 150.0, places=1)
        self.assertEqual(row["n_com_speed_values"], 5)
        rows = {r[0]: dict(zip(rm.JOINT_METRIC_COLUMNS, r)) for r in rm.joint_metric_rows(res)}
        self.assertEqual(len(rows), 18)  # 17 joints + CoM proxy
        w = rows["left_wrist"]
        self.assertEqual((w["cumulative_displacement_px"], w["displacement_steps_counted"],
                          w["displacement_steps_skipped"]), ("25.000", 5, 0))
        self.assertAlmostEqual(float(w["mean_velocity_px_s"]), 150.0, places=1)
        self.assertEqual(rows[rm.COM_SERIES_NAME]["cumulative_displacement_px"], "25.000")

    def test_gap_counts_in_joint_metrics(self):
        p = self.d / "s_keypoints.csv"
        write_clip(p, [f for f in range(8) if f != 2], low={(5, "nose")})
        res = rm.analyse_clip(p, excluded={6})
        rows = {r[0]: dict(zip(rm.JOINT_METRIC_COLUMNS, r)) for r in rm.joint_metric_rows(res)}
        n = rows["nose"]
        self.assertEqual((n["n_frames"], n["frames_detector_drop"], n["frames_low_confidence"],
                          n["frames_excluded_by_list"]), (8, 1, 1, 1))
        # steps: 7 total; gaps at 2, 5, 6 kill steps 2,3,5,6,7 -> counted 2 (1 and 4), skipped 5
        self.assertEqual((n["displacement_steps_counted"], n["displacement_steps_skipped"]), (2, 5))

    def test_csv_and_plots_written_without_video(self):
        p = self.d / "s_keypoints.csv"
        write_clip(p, [f for f in range(10) if f != 3], low={(8, "left_hip")})
        out = self.d / "out"
        res, row = rm.report_clip(p, out, self.d / "nofootage", {5}, make_video=False)
        folder = out / "s"
        for name in ("joint_metrics.csv", "com_xy.png", "com_speed.png", "gap_timeline.png"):
            self.assertTrue((folder / name).stat().st_size > 0, name)
        self.assertFalse((folder / "cov_overlay.mp4").exists())


class OverlayTest(unittest.TestCase):
    def test_overlay_has_one_frame_per_data_frame(self):
        import cv2
        import numpy as np
        d = Path(tempfile.mkdtemp())
        w, h, n = 320, 480, 12
        vid = d / "s.MOV.mp4"
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for _ in range(n):
            vw.write(np.zeros((h, w, 3), dtype=np.uint8))
        vw.release()
        p = d / "s_keypoints.csv"
        # keypoints inside the 320x480 frame
        write_clip(p, [f for f in range(10) if f != 3], step=(1.0, 1.0))
        res = rm.analyse_clip(p, excluded={5})
        out = d / "o.mp4"
        written = rm.write_overlay(vid, out, res)
        self.assertEqual(written, 10)  # frames with data, not the video's 12
        cap = cv2.VideoCapture(str(out))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        self.assertEqual((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (w, h))


if __name__ == "__main__":
    unittest.main()
