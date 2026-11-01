"""
Tests for evaluation/2.2c/make_candidate_overlays.py: confidence bands, marker styles, drawing and
the manifest. Synthetic input; nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_compare_com_candidates import cc
from tests.test_compare_com_models import write_clip

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "make_candidate_overlays", REPO_ROOT / "evaluation" / "2.2c" / "make_candidate_overlays.py")
mo = importlib.util.module_from_spec(_spec)
sys.modules["make_candidate_overlays"] = mo
_spec.loader.exec_module(mo)


class OverlayTests(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        p = d / "s_keypoints.csv"
        write_clip(p, range(6))
        self.c = cc.build_clip(p, None, "male", 0.7)
        self.raw = mo.oc.load_raw(p)

    def test_nine_distinct_styles(self):
        self.assertEqual(len(mo.STYLE), 9)
        self.assertEqual(set(mo.STYLE), set(cc.OVERLAY_LABELS))
        self.assertEqual(len({v[0] for v in mo.STYLE.values()}), 9)
        self.assertFalse(any(l in mo.STYLE for l in cc.EDL_LABELS))

    def test_band_colour_covers_scores_above_one(self):
        self.assertEqual(mo.band_colour(0.3), mo.COL_BAD)
        self.assertEqual(mo.band_colour(0.5), mo.BANDS[0][2])
        self.assertEqual(mo.band_colour(0.8), mo.BANDS[3][2])
        self.assertEqual(mo.band_colour(1.3), mo.BANDS[3][2])

    def test_draw_frame_marks_the_frame(self):
        frame = np.zeros((600, 500, 3), dtype=np.uint8)
        out = mo.draw_frame(frame.copy(), self.c, self.raw, 2, {"rank": 1, "frame_from": 1, "frame_to": 2})
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)

    def test_no_detection_banner(self):
        frame = np.zeros((600, 500, 3), dtype=np.uint8)
        out = mo.draw_frame(frame.copy(), self.c, {}, 2)
        self.assertTrue((out[10, 10] == np.array(mo.COL_BAD)).all())

    def test_manifest_lists_windows_and_folds(self):
        w = [{"rank": "1", "clip": "s.MOV", "frame_from": "1", "frame_to": "2", "start": 0, "end": 5,
              "clamped_start": True, "clamped_end": False, "n_written": 6, "file": "f.mp4", "bytes": 10}]
        text = mo.build_manifest(w, Path("in"), Path("top.csv"), Path("nope.csv"), Path("footage"),
                                 [{"held_out_clip": "s.MOV", "w": 0.7}], 15)
        for needle in ("nine", "EdL variants not drawn", "s.MOV 0.700000", "clamped: start", "not representative"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
