"""
Tests for evaluation/2.2b/make_overlay_clips.py: window padding at clip edges, raw keypoint
loading, drawing (marker colours, banner), the clip writer and the manifest. Synthetic input;
nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from metrics import GAP_MISSING_DETECTION
from tests.test_com_segment import COCO, stick_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "make_overlay_clips", REPO_ROOT / "evaluation" / "2.2b" / "make_overlay_clips.py")
oc = importlib.util.module_from_spec(_spec)
sys.modules["make_overlay_clips"] = oc
_spec.loader.exec_module(oc)
cm = oc.cm

W, H = 500, 600


def write_clip(path, n_frames, low=None):
    """Static stick figure scaled x10 and placed at (100, 100): proxy CoM (100, 150), male
    segment-model CoM (100, 192.4). low {(frame, joint): confidence} overrides 0.9."""
    low = low or {}
    pose = stick_figure(dx=100.0, dy=100.0, k=10.0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "timestamp_s", "keypoint_id", "keypoint_name",
                    "x", "y", "confidence"])
        for fr in range(n_frames):
            for i, n in enumerate(COCO):
                w.writerow(["s.MOV", fr, f"{fr / 30:.9f}", i, n, pose[n][0], pose[n][1],
                            low.get((fr, n), 0.9)])


class WindowTests(unittest.TestCase):
    def test_padding_in_the_middle(self):
        self.assertEqual(oc.window_for(79, 80, 15, 2421), (64, 95, False, False))

    def test_clamped_at_start(self):
        self.assertEqual(oc.window_for(3, 4, 15, 100), (0, 19, True, False))

    def test_clamped_at_end(self):
        self.assertEqual(oc.window_for(95, 96, 15, 100), (80, 99, False, True))

    def test_exactly_at_the_edge_is_not_clamped(self):
        self.assertEqual(oc.window_for(15, 16, 15, 100), (0, 31, False, False))
        self.assertEqual(oc.window_for(83, 84, 15, 100), (68, 99, False, False))

    def test_zero_padding(self):
        self.assertEqual(oc.window_for(10, 11, 0, 50), (10, 11, False, False))

    def test_default_padding_is_fifteen(self):
        self.assertEqual(oc.PAD_FRAMES, 15)


class DrawTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        p = self.d / "s_keypoints.csv"
        write_clip(p, 4, low={(1, "left_wrist"): 0.3})
        self.c = cm.compare_clip(p, excluded={2})
        self.raw = oc.load_raw(p)

    def blank(self):
        import numpy as np
        return np.zeros((H, W, 3), dtype=np.uint8)

    def test_load_raw_keeps_joints_below_threshold(self):
        self.assertEqual(len(self.raw[0]), 14)
        self.assertAlmostEqual(self.raw[1]["left_wrist"][2], 0.3)
        self.assertEqual(self.raw[1]["left_wrist"][:2], (85.0, 200.0))

    def test_both_com_markers_have_their_own_colours(self):
        img = oc.draw_frame(self.blank(), self.c, self.raw, 0, "male", (None, None))
        self.assertEqual(tuple(img[150, 100]), oc.COL_PROXY)   # proxy (100, 150)
        self.assertEqual(tuple(img[192, 100]), oc.COL_SEG)     # segment model (100, 192.4)

    def test_joint_below_threshold_is_marked_red(self):
        img = oc.draw_frame(self.blank(), self.c, self.raw, 1, "male", (0.0, None))
        self.assertEqual(tuple(img[200, 85]), oc.COL_BAD)      # left wrist, X centre
        self.assertEqual(tuple(img[100, 90]), oc.COL_GOOD)     # left shoulder (90, 100), conf 0.9

    def test_segment_model_marker_absent_when_a_joint_is_a_gap(self):
        img = oc.draw_frame(self.blank(), self.c, self.raw, 1, "male", (0.0, None))
        self.assertEqual(tuple(img[150, 100]), oc.COL_PROXY)   # proxy still defined
        self.assertNotEqual(tuple(img[192, 100]), oc.COL_SEG)

    def test_excluded_frame_shows_banner_and_no_skeleton(self):
        img = oc.draw_frame(self.blank(), self.c, self.raw, 2, "male", (None, None))
        self.assertEqual(tuple(img[5, 5]), oc.COL_EXCLUDED)
        self.assertEqual(tuple(img[150, 100]), (0, 0, 0))      # no markers

    def test_frame_not_detected_shows_banner(self):
        c, raw = self.c, dict(self.raw)
        del raw[3]
        c.proxy.gap_reasons[3] = GAP_MISSING_DETECTION
        img = oc.draw_frame(self.blank(), c, raw, 3, "male", (None, None))
        self.assertEqual(tuple(img[5, 5]), oc.COL_BAD)


class ClipAndManifestTests(unittest.TestCase):
    def test_clip_has_one_frame_per_window_frame(self):
        import cv2
        import numpy as np
        d = Path(tempfile.mkdtemp())
        vid = d / "s.MOV.mp4"
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for _ in range(40):
            vw.write(np.zeros((H, W, 3), dtype=np.uint8))
        vw.release()
        p = d / "s_keypoints.csv"
        write_clip(p, 40)
        c, raw = cm.compare_clip(p, None), oc.load_raw(p)
        start, end, cs, ce = oc.window_for(19, 20, 15, c.n_frames)
        self.assertEqual((start, end), (4, 35))
        out = d / "o.mp4"
        n = oc.make_clip(vid, out, c, raw, "male", {"rank": "1", "frame_from": "19", "frame_to": "20"}, start, end)
        self.assertEqual(n, 32)
        cap = cv2.VideoCapture(str(out))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 32)
        self.assertEqual((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (W, H))

    def test_manifest_lists_inputs_head_and_windows(self):
        d = Path(tempfile.mkdtemp())
        (d / "s.MOV").write_bytes(b"abc")
        top = d / "top.csv"
        top.write_text("rank,clip\n1,s.MOV\n")
        ex = d / "ex.csv"
        ex_text = "# note\nvideo,start_frame,end_frame,reason\ns.MOV,8,9,test\n"
        ex.write_text(ex_text)
        windows = [{"rank": "1", "clip": "s.MOV", "frame_from": "3", "frame_to": "4", "start": 0, "end": 19,
                    "clamped_start": True, "clamped_end": False, "n_written": 20,
                    "file": "evaluation/2.2b/overlays/x/rank01_s_f0-19.mp4", "bytes": 1234}]
        text = oc.build_manifest(windows, d, top, ex, d, "male", 15, today="2026-10-28")
        self.assertIn("date: 2026-10-28", text)
        self.assertIn("padding: 15 frames each side", text)
        self.assertIn("git HEAD:", text)
        self.assertIn(ex_text.rstrip("\n"), text)
        self.assertIn("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", text)
        self.assertIn("1, s.MOV, 3->4, 0-19, clamped: start, 20,", text)
        self.assertIn("1234", text)


if __name__ == "__main__":
    unittest.main()
