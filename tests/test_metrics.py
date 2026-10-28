"""
Tests for metrics.py velocity / displacement (ARCHITECTURE.md §4.1, §4.2).

Correctness tests use SYNTHETIC series with hand-computed answers — nothing
here is a measurement from real footage. The one real-data test is a smoke
test (shape/invariant checks only, no numeric expectations) and is skipped if
the gitignored validation output is absent.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import metrics
from metrics import (
    GAP_LOW_CONFIDENCE,
    GAP_MISSING_DETECTION,
    JointSeries,
    compute_cumulative_displacement,
    compute_displacement,
    compute_velocity,
    count_gap_frames,
    load_keypoints_csv,
)

FPS = 30.0
REPO_ROOT = Path(__file__).resolve().parents[1]


def uniform_series(positions, fps=FPS):
    return JointSeries("j", list(positions), [i / fps for i in range(len(positions))])


def write_keypoints_csv(path, rows):
    """rows: (frame_idx, joint, x, y, confidence); timestamp_s = frame_idx/30."""
    fields = ["video", "frame_idx", "timestamp_s", "keypoint_id", "keypoint_name",
              "x", "y", "confidence", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for frame_idx, joint, x, y, conf in rows:
            w.writerow({
                "video": "synthetic.mov", "frame_idx": frame_idx,
                "timestamp_s": f"{frame_idx / FPS:.6f}", "keypoint_id": 0,
                "keypoint_name": joint, "x": x, "y": y, "confidence": conf,
                "bbox_x": 0, "bbox_y": 0, "bbox_w": 1, "bbox_h": 1,
            })


class VelocityTests(unittest.TestCase):
    def test_3_4_5_motion_at_30fps(self):
        # (3,4) px per frame = 5 px/frame = 150 px/s at 30 fps.
        series = uniform_series([(3.0 * i, 4.0 * i) for i in range(6)])
        v = compute_velocity(series)
        self.assertIsNone(v[0])
        for x in v[1:]:
            self.assertAlmostEqual(x, 150.0)

    def test_stationary_joint(self):
        series = uniform_series([(10.0, 20.0)] * 5)
        self.assertEqual(compute_velocity(series)[1:], [0.0] * 4)
        self.assertEqual(compute_cumulative_displacement(series), 0.0)

    def test_gap_makes_both_adjacent_steps_none(self):
        pos = [(3.0 * i, 4.0 * i) for i in range(6)]
        pos[3] = None
        v = compute_velocity(uniform_series(pos))
        self.assertIsNone(v[3])  # into the gap
        self.assertIsNone(v[4])  # out of the gap — not bridged
        for t in (1, 2, 5):
            self.assertAlmostEqual(v[t], 150.0)

    def test_uneven_timestamps_use_each_steps_own_dt(self):
        series = JointSeries(
            "j", [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0)], [0.0, 0.5, 0.75]
        )
        v = compute_velocity(series)
        self.assertAlmostEqual(v[1], 10.0 / 0.5)
        self.assertAlmostEqual(v[2], 20.0 / 0.25)


class DisplacementTests(unittest.TestCase):
    def setUp(self):
        self.n = 8
        self.series = uniform_series([(3.0 * i, 4.0 * i) for i in range(self.n)])

    def test_full_clip_is_5_times_steps(self):
        self.assertAlmostEqual(compute_cumulative_displacement(self.series), 5.0 * (self.n - 1))

    def test_window_3_to_6_counts_only_inside_steps(self):
        # steps t=4,5,6 -> 3 steps * 5 px. The old off-by-one gave 20.
        self.assertAlmostEqual(compute_cumulative_displacement(self.series, 3, 6), 15.0)

    def test_window_starting_at_zero_equals_full_clip(self):
        self.assertAlmostEqual(
            compute_cumulative_displacement(self.series, 0, self.n - 1),
            compute_cumulative_displacement(self.series),
        )

    def test_single_frame_window_is_zero(self):
        self.assertEqual(compute_cumulative_displacement(self.series, 4, 4), 0.0)

    def test_gap_step_skipped_and_counted(self):
        pos = [(3.0 * i, 4.0 * i) for i in range(self.n)]
        pos[3] = None  # kills steps t=3 and t=4
        res = compute_displacement(uniform_series(pos))
        self.assertAlmostEqual(res.total, 5.0 * (self.n - 1 - 2))
        self.assertEqual(res.n_steps_skipped, 2)
        self.assertEqual(res.n_steps_counted, self.n - 1 - 2)

    def test_skipped_count_respects_window(self):
        pos = [(3.0 * i, 4.0 * i) for i in range(self.n)]
        pos[3] = None
        res = compute_displacement(uniform_series(pos), 5, 7)  # steps 6,7 only
        self.assertEqual(res.n_steps_skipped, 0)
        self.assertAlmostEqual(res.total, 10.0)


class LoaderTests(unittest.TestCase):
    def test_missing_frame_is_reindexed_not_bridged(self):
        # frame 2 has no rows at all (a missing_detection frame).
        rows = [(f, "nose", 3.0 * f, 4.0 * f, 0.9) for f in (0, 1, 3)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.csv"
            write_keypoints_csv(p, rows)
            s = load_keypoints_csv(p)["nose"]
        self.assertEqual(len(s.positions), 4)
        self.assertIsNone(s.positions[2])
        self.assertEqual(s.gap_reasons[2], GAP_MISSING_DETECTION)
        v = compute_velocity(s)
        self.assertIsNone(v[2])
        self.assertIsNone(v[3])

    def test_confidence_threshold_boundary(self):
        # 0.499 is a gap, exactly 0.5 is not (D-007: occluded means C < 0.5).
        rows = [
            (0, "nose", 0.0, 0.0, 0.9),
            (1, "nose", 1.0, 0.0, 0.499),
            (2, "nose", 2.0, 0.0, 0.5),
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.csv"
            write_keypoints_csv(p, rows)
            s = load_keypoints_csv(p)["nose"]
        self.assertIsNone(s.positions[1])
        self.assertEqual(s.gap_reasons[1], GAP_LOW_CONFIDENCE)
        self.assertEqual(s.positions[2], (2.0, 0.0))
        self.assertIsNone(s.gap_reasons[2])

    def test_gap_causes_reported_separately(self):
        rows = [
            (0, "nose", 0.0, 0.0, 0.9),
            (1, "nose", 1.0, 0.0, 0.2),   # low confidence
            (3, "nose", 3.0, 0.0, 0.9),   # frame 2 dropped
            (4, "nose", 4.0, 0.0, 0.1),   # low confidence
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.csv"
            write_keypoints_csv(p, rows)
            s = load_keypoints_csv(p)["nose"]
        gaps = count_gap_frames(s)
        self.assertEqual(gaps.missing_detection, 1)
        self.assertEqual(gaps.low_confidence, 2)
        self.assertEqual(gaps.total, 3)

    def test_threshold_defined_once(self):
        self.assertEqual(metrics.OCCLUSION_CONFIDENCE_THRESHOLD, 0.5)


class RealDataSmokeTest(unittest.TestCase):
    """Shape / invariant checks only — no numeric expectations."""

    CSV = REPO_ROOT / "validation" / "output" / "IMG_2412_keypoints.csv"

    def test_real_keypoints_invariants(self):
        if not self.CSV.exists():
            msg = (f"SKIPPED real-data smoke test: {self.CSV.relative_to(REPO_ROOT)} "
                   "not found (validation/output/ is gitignored; run "
                   "validation/validate_vitpose.py to generate it)")
            print(msg, file=sys.stderr)
            self.skipTest(msg)
        all_series = load_keypoints_csv(self.CSV)
        self.assertEqual(len(all_series), 17)
        lengths = {len(s.positions) for s in all_series.values()}
        self.assertEqual(len(lengths), 1)
        for s in all_series.values():
            gaps = count_gap_frames(s)
            self.assertEqual(gaps.total, sum(1 for p in s.positions if p is None))
            self.assertEqual(gaps.unspecified, 0)
            for v in compute_velocity(s):
                self.assertTrue(v is None or (v >= 0 and math.isfinite(v)))
            d = compute_displacement(s)
            self.assertGreaterEqual(d.total, 0.0)
            self.assertEqual(d.n_steps_counted + d.n_steps_skipped, len(s.positions) - 1)


if __name__ == "__main__":
    unittest.main()
