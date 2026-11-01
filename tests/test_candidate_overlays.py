"""
Tests for evaluation/2.2c/make_candidate_overlays.py: panel list (default from triage.csv and the
--panels option), grid shape, the fixed crop, panel drawing (undefined text, banners, A ring,
trail) and the manifest. Synthetic input; nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_compare_com_candidates import cc
import csv
from tests.test_com_segment import COCO, stick_figure

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "make_candidate_overlays", REPO_ROOT / "evaluation" / "2.2c" / "make_candidate_overlays.py")
mo = importlib.util.module_from_spec(_spec)
sys.modules["make_candidate_overlays"] = mo
_spec.loader.exec_module(mo)

W, H = 500, 600


def write_clip(path, detected, low=()):
    """Stick figure scaled x10 and placed at (200, 200), inside the synthetic frame."""
    pose = stick_figure(dx=200.0, dy=200.0, k=10.0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "timestamp_s", "keypoint_id", "keypoint_name",
                    "x", "y", "confidence"])
        for fr in detected:
            for i, n in enumerate(COCO):
                w.writerow(["s.MOV", fr, f"{fr / 30:.6f}", i, n, pose[n][0], pose[n][1],
                            0.2 if (fr, n) in low else 0.9])


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        p = self.d / "s_keypoints.csv"
        write_clip(p, [f for f in range(8) if f != 5], low={(3, "left_wrist")})
        self.c = cc.build_clip(p, {6}, "male", 0.7)
        self.raw = mo.oc.load_raw(p)
        self.frame = np.full((H, W, 3), 60, dtype=np.uint8)
        self.crop = mo.window_crop(self.c, 0, 7, W, H)

    def test_colours_cover_every_candidate_and_are_distinct(self):
        self.assertEqual(set(mo.COLOURS), set(cc.LABELS))
        self.assertEqual(len(set(mo.COLOURS.values())), len(cc.LABELS))

    def test_band_colour_covers_scores_above_one(self):
        self.assertEqual(mo.band_colour(0.3), mo.COL_BAD)
        self.assertEqual(mo.band_colour(0.5), mo.BANDS[0][2])
        self.assertEqual(mo.band_colour(1.3), mo.BANDS[3][2])

    def test_default_panels_are_survivors_plus_a_and_f(self):
        t = self.d / "triage.csv"
        t.write_text("candidate,dominated,dominated_by,never_dropped,dropped\n"
                     "A_proxy,yes,E_c0.5,yes,no\nB_trunk_point,yes,A_proxy,no,yes\n"
                     "E_c0.5,no,,no,no\nF_full,no,,yes,no\nC_no_head,yes,E_c0.5,no,yes\n")
        self.assertEqual(mo.default_panels(t), ["A_proxy", "E_c0.5", "F_full"])
        t.write_text("candidate,dominated,dominated_by,never_dropped,dropped\n"
                     "B_trunk_point,no,,no,no\n")   # A and F added even if absent
        self.assertEqual(mo.default_panels(t), ["A_proxy", "B_trunk_point", "F_full"])

    def test_parse_panels(self):
        self.assertEqual(mo.parse_panels("F_full, A_proxy,F_full,EdL_c0.8"),
                         ["F_full", "A_proxy", "EdL_c0.8"])
        for bad in ("A_proxy,nope", ""):
            with self.assertRaises(ValueError):
                mo.parse_panels(bad)

    def test_grid_shape(self):
        self.assertEqual([mo.grid_shape(n) for n in (1, 2, 3, 4, 5, 13)],
                         [(1, 1), (2, 1), (2, 2), (2, 2), (3, 2), (4, 4)])

    def test_crop_is_square_inside_frame_and_contains_all_positions(self):
        x0, y0, side = self.crop
        self.assertTrue(0 <= x0 and x0 + side <= W and 0 <= y0 and y0 + side <= H)
        for l in cc.LABELS:
            for p in self.c.series[l].positions:
                if p is not None:
                    self.assertTrue(x0 <= p[0] <= x0 + side and y0 <= p[1] <= y0 + side, l)
        # independent of the panel list: same crop whatever is drawn
        self.assertEqual(self.crop, mo.window_crop(self.c, 0, 7, W, H))

    def test_crop_without_positions_is_the_full_square(self):
        c = self.c
        saved = {l: list(c.series[l].positions) for l in cc.LABELS}
        for l in cc.LABELS:
            c.series[l].positions[:] = [None] * c.n_frames
        self.assertEqual(mo.window_crop(c, 0, 7, W, H), (0, 0, W))
        for l in cc.LABELS:
            c.series[l].positions[:] = saved[l]

    def test_panel_shape_and_undefined_text(self):
        red = np.array(mo.COL_BAD)
        # frame 3: left wrist low -> F undefined, A defined
        p_f = mo.draw_panel(self.frame, self.c, self.raw, 3, cc.L_F, self.crop)
        p_a = mo.draw_panel(self.frame, self.c, self.raw, 3, cc.L_A, self.crop)
        self.assertEqual(p_f.shape, (mo.PANEL + mo.TITLE_H, mo.PANEL, 3))
        body = lambda im: im[mo.TITLE_H + mo.PANEL - 70:, :]
        self.assertTrue((body(p_f) == red).all(axis=2).any())       # "undefined" drawn
        self.assertFalse((body(p_a) == red).all(axis=2).any())

    def test_banners_for_excluded_and_no_detection(self):
        ex = mo.draw_panel(self.frame, self.c, self.raw, 6, cc.L_A, self.crop)
        nd = mo.draw_panel(self.frame, self.c, self.raw, 5, cc.L_A, self.crop)
        self.assertTrue((ex[mo.TITLE_H + 5, 5] == np.array(mo.COL_EXCLUDED)).all())
        self.assertTrue((nd[mo.TITLE_H + 5, 5] == np.array(mo.COL_BAD)).all())

    def test_a_ring_only_in_other_panels(self):
        a = mo.draw_panel(self.frame, self.c, self.raw, 1, cc.L_A, self.crop)
        f = mo.draw_panel(self.frame, self.c, self.raw, 1, cc.L_F, self.crop)
        ring = np.array(mo.COL_RING)
        self.assertFalse((a[mo.TITLE_H:] == ring).all(axis=2).any())
        self.assertTrue((f[mo.TITLE_H:] == ring).all(axis=2).any())

    def test_grid_dimensions(self):
        g = mo.draw_grid(self.frame, self.c, self.raw, 1, [cc.L_A, "E_c0.5", cc.L_F], self.crop,
                         {"rank": 1, "frame_from": 0, "frame_to": 1})
        self.assertEqual(g.shape, (mo.HEADER_H + 2 * (mo.PANEL + mo.TITLE_H) + mo.FOOTER_H,
                                   2 * mo.PANEL, 3))

    def test_manifest_records_panels_crop_and_triage(self):
        w = [{"rank": "1", "clip": "s.MOV", "frame_from": "1", "frame_to": "2", "start": 0, "end": 5,
              "clamped_start": True, "clamped_end": False, "n_written": 6, "crop": (1, 2, 300),
              "file": "f.mp4", "bytes": 10}]
        text = mo.build_manifest(w, Path("in"), Path("top.csv"), Path("nope.csv"), Path("footage"),
                                 [{"held_out_clip": "s.MOV", "w": 0.7}], 15,
                                 [cc.L_A, cc.L_F], Path("triage.csv"), True)
        for needle in ("panels: A_proxy, F_full", "default: triage survivors plus A and F",
                       "crop 1 2 300", "s.MOV 0.700000", "clamped: start", "not representative",
                       "triage file"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
