"""
Synthetic tests for validation/analyze_occlusion_gaps.py (frame exclusions, trailing
drops, truncation flags). Hand-built CSVs; nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import tempfile
import unittest
from pathlib import Path

from validation import analyze_occlusion_gaps as a


def write(path, frames, low=()):
    """frames: detected frame_idx list; low: {(frame, joint)} with confidence 0.2."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "frame_idx", "timestamp_s", "keypoint_id", "keypoint_name",
                    "x", "y", "confidence"])
        for fr in frames:
            for i, n in enumerate(a.COCO17_NAMES):
                w.writerow(["s.MOV", fr, f"{fr / 30:.6f}", i, n, 0, 0,
                            0.2 if (fr, n) in low else 0.9])


class GapAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_excluded_frames_are_not_observations_or_gaps(self):
        p = self.d / "k.csv"
        low = {(f, "nose") for f in range(4, 10)}  # nose low on 4..9
        write(p, range(12), low)
        base = a.analyze_clip(p, 12)
        self.assertEqual([(e.start_frame, e.end_frame) for e in base.joint_gap_events["nose"]], [(4, 9)])
        ex = a.analyze_clip(p, 12, excluded={6, 7})
        # 4-5 and 8-9 are separate runs: the run does not continue across 6-7
        self.assertEqual([(e.start_frame, e.end_frame) for e in ex.joint_gap_events["nose"]],
                         [(4, 5), (8, 9)])
        self.assertTrue(all(e.adjacent_to_unobserved for e in ex.joint_gap_events["nose"]))
        self.assertEqual(ex.n_detected_frames, 10)
        self.assertEqual(ex.excluded_frames, [6, 7])
        # detector drops are unaffected by the exclusion list
        self.assertEqual(ex.missing_detection_events, [])

    def test_excluded_frames_do_not_become_detector_drops(self):
        p = self.d / "k.csv"
        write(p, [0, 1, 2, 4, 5])  # frame 3 dropped by the detector
        s = a.analyze_clip(p, 6, excluded={4})
        self.assertEqual([(e.start_frame, e.end_frame) for e in s.missing_detection_events], [(3, 3)])
        self.assertEqual(s.excluded_frames, [4])

    def test_trailing_drop_counted_with_frame_count(self):
        p = self.d / "k.csv"
        write(p, range(5))
        s = a.analyze_clip(p, 8)
        self.assertEqual([(e.start_frame, e.end_frame, e.touches_boundary)
                          for e in s.missing_detection_events], [(5, 7, True)])

    def test_boundary_flag_and_no_adjacency_inside_clip(self):
        p = self.d / "k.csv"
        write(p, range(10), {(f, "nose") for f in (0, 1, 5)})
        ev = a.analyze_clip(p, 10).joint_gap_events["nose"]
        self.assertEqual([(e.start_frame, e.end_frame, e.touches_boundary, e.adjacent_to_unobserved)
                          for e in ev], [(0, 1, True, False), (5, 5, False, False)])

    def test_load_exclusions_skips_comments(self):
        p = self.d / "ex.csv"
        p.write_text("# note\nvideo,start_frame,end_frame,reason\nA.MOV,3,5,because\n")
        self.assertEqual(a.load_exclusions(p), {"A.MOV": {3, 4, 5}})
        self.assertEqual(a.load_exclusions(self.d / "missing.csv"), {})


if __name__ == "__main__":
    unittest.main()
