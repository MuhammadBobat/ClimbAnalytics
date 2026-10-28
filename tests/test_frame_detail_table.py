"""
Tests for evaluation/2.2b/frame_detail_table.py: the table rows and the manifest. Synthetic input
(the static stick figure of tests/test_overlay_clips.py); nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_overlay_clips import write_clip  # static stick figure, 14 joints, x10 scale

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "frame_detail_table", REPO_ROOT / "evaluation" / "2.2b" / "frame_detail_table.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["frame_detail_table"] = fd
_spec.loader.exec_module(fd)


class FrameDetailTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.p = self.d / "s_keypoints.csv"
        # left_wrist low confidence on frame 2 -> segment model undefined there, proxy defined
        write_clip(self.p, 5, low={(2, "left_wrist"): 0.3})
        self.c = fd.cm.compare_clip(self.p, None)
        self.raw = fd.oc.load_raw(self.p)

    def test_rows(self):
        header, rows = fd.build_rows(self.c, self.raw, "male", 1, 3)
        self.assertEqual(header[:7], ["frame", "proxy_x", "proxy_y", "seg_male_x", "seg_male_y",
                                      "speed_proxy_px_s", "speed_segment_px_s"])
        self.assertEqual(len(header), 7 + 4 * 3)
        by = {r[0]: dict(zip(header, r)) for r in rows}
        self.assertEqual(by[1]["proxy_x"], "100.0000")
        self.assertNotEqual(by[1]["seg_male_y"], "")
        self.assertEqual(by[2]["seg_male_y"], "")             # undefined (gap)
        self.assertNotEqual(by[2]["proxy_y"], "")             # proxy still defined
        self.assertEqual(by[2]["left_shoulder_conf"], "0.9000")
        self.assertEqual(by[1]["speed_proxy_px_s"], "0.0000") # static figure

    def test_manifest(self):
        ex = self.d / "ex.csv"
        ex_text = "# note\nvideo,start_frame,end_frame,reason\ns.MOV,3,4,test\n"
        ex.write_text(ex_text)
        text = fd.build_manifest(self.p, ex, "s", 1, 3, "male", today="2026-10-28")
        self.assertIn("date: 2026-10-28", text)
        self.assertIn("clip: s, frames 1-3", text)
        self.assertIn(ex_text.rstrip("\n"), text)
        self.assertIn("git HEAD:", text)
        self.assertIn("modified since HEAD", text)
        self.assertIn("sha256 ", text)
        self.assertIn("blob hash", text)


if __name__ == "__main__":
    unittest.main()
