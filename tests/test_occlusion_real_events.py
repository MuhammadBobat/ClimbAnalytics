"""
Real-event tests for occlusion_handling.py on a small committed fixture: the first 100 frames of IMG_2412 from the
legacy-lock run validation/output/ (tests/fixtures/IMG_2412_first100_keypoints.csv, our own keypoints, no video;
sha256 0c3da946fb2966883298fd07e9f034ccd5df11b235a1f69efcb68e03a30c0337, 1,666 rows; frames 41 and 42 have no person
box in that run). These tests record what the code does on real occlusion events. They do NOT validate any tier or
method: there is no ground truth for the hidden positions, and the tier boundaries are unvalidated heuristics
(hide-and-fill test, step 3.3). DECISIONS.md D-007, D-028 (proposed), D-030 (adopted 2026-11-25: Kalman for 1 to 20
frames, no position for a limb gap over 20).

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import csv
import math
import unittest
from pathlib import Path

import metrics
import occlusion_handling as oh
from occlusion_handling import Flag, fill_gaps

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "IMG_2412_first100_keypoints.csv"


def raw_row(joint, frame):
    with open(FIXTURE, newline="") as f:
        for r in csv.DictReader(f):
            if r["keypoint_name"] == joint and int(r["frame_idx"]) == frame:
                return float(r["x"]), float(r["y"]), float(r["confidence"])


class RealEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.joints = metrics.load_keypoints_csv(FIXTURE)
        cls.before = {n: (list(s.positions), list(s.gap_reasons), list(s.confidences)) for n, s in cls.joints.items()}
        cls.filled = fill_gaps(cls.joints)

    def events(self, joint):
        return [(e.first, e.last, e.tier, e.filled) for e in self.filled[joint].events]

    def test_gap_inventory_matches_the_3_1_events_plus_the_two_dropped_frames(self):
        # 3.1 body-joint events in these 100 frames: right wrist 64, right ankle 2, 8 and 10-38; frames 41-42 are a
        # whole-frame detector drop in this run, so every body joint has that 2-frame gap
        for j in oh.BODY_JOINTS:
            want = [(41, 42, Flag.TIER2, True)]
            if j == "right_wrist":
                want = [(41, 42, Flag.TIER2, True), (64, 64, Flag.TIER2, True)]
            if j == "right_ankle":                      # D-030: 10-38 is flag-only, so not filled
                want = [(2, 2, Flag.TIER2, True), (8, 8, Flag.TIER2, True), (10, 38, Flag.TIER3, False),
                        (41, 42, Flag.TIER2, True)]
            self.assertEqual(self.events(j), want, j)

    def test_right_wrist_frame_64_confidence_0_257(self):
        x, y, conf = raw_row("right_wrist", 64)
        self.assertAlmostEqual(conf, 0.2568, places=3)
        f = self.filled["right_wrist"]
        self.assertEqual(f.flags[64], Flag.TIER2)                           # D-030: Kalman (was Tier 1, spline)
        self.assertEqual(f.methods[64], "kalman")
        a, b, c = f.positions[63], f.positions[64], f.positions[65]
        self.assertEqual((a, c), (self.joints["right_wrist"].positions[63], self.joints["right_wrist"].positions[65]))
        # plausibility only: the filled point lies near the interval spanned by the observed neighbours
        for k in (0, 1):
            lo, hi = min(a[k], c[k]), max(a[k], c[k])
            self.assertTrue(lo - 5.0 <= b[k] <= hi + 5.0, (k, lo, b[k], hi))
        # the raw low-confidence value is a 36 px outlier in y; the fill does not follow it
        self.assertGreater(math.hypot(b[0] - x, b[1] - y), 30.0)
        self.assertIsNone(self.joints["right_wrist"].positions[64])        # raw stays a gap

    def test_whole_frame_drop_is_filled_for_every_body_joint(self):
        for j in oh.BODY_JOINTS:
            f = self.filled[j]
            self.assertEqual(f.flags[41:43], [Flag.TIER2, Flag.TIER2], j)      # D-030: Kalman (was Tier 1)
            self.assertEqual(f.raw.gap_reasons[41], metrics.GAP_MISSING_DETECTION)
        for t in (41, 42):
            ls, rs = self.filled["left_shoulder"].positions[t], self.filled["right_shoulder"].positions[t]
            self.assertIsNotNone(ls)
            self.assertGreater(math.hypot(ls[0] - rs[0], ls[1] - rs[1]), 50.0)     # a plausible shoulder width

    def test_right_ankle_10_to_38_is_flagged_without_a_position(self):
        """D-030: the 29-frame gap gets the flag tier3 and no position (it was a stale offset in D-028)."""
        f = self.filled["right_ankle"]
        self.assertEqual(f.flags[10:39], [Flag.TIER3] * 29)
        self.assertEqual(f.positions[10:39], [None] * 29)
        self.assertEqual(f.events[2].length, 29)
        self.assertFalse(f.events[2].filled)
        self.assertIsNone(f.events[2].method)
        js = oh.as_joint_series(self.filled)
        self.assertTrue(all(js["right_ankle"].positions[t] is None for t in range(10, 39)))

    def test_right_ankle_10_to_38_forced_stale_offset_is_unchanged_for_evaluation(self):
        """The old behaviour, kept behind force_method for the evaluation scripts (D-028 item 8): constant offset in the
        trunk-aligned frame, reference frame 9."""
        f = fill_gaps(self.joints, oh.OcclusionParams(force_method="trunk_offset"))["right_ankle"]
        self.assertEqual(f.flags[10:39], [Flag.TIER3] * 29)
        work = {n: self.filled[n].positions for n in oh.TRUNK}
        offsets = []
        for t in range(10, 39):
            o, u, w, s = oh._trunk_frame(work, t, "leg")
            d = (f.positions[t][0] - o[0], f.positions[t][1] - o[1])
            offsets.append(((d[0] * u[0] + d[1] * u[1]) / s, (d[0] * w[0] + d[1] * w[1]) / s))
        self.assertLess(max(abs(a - offsets[0][0]) for a, _ in offsets), 1e-9)
        self.assertLess(max(abs(b - offsets[0][1]) for _, b in offsets), 1e-9)
        self.assertEqual(f.events[2].note, "offset from frame 9")

    def test_raw_data_is_unchanged(self):
        for n, s in self.joints.items():
            self.assertEqual((s.positions, s.gap_reasons, s.confidences), self.before[n])
        fresh = metrics.load_keypoints_csv(FIXTURE)
        for n, s in fresh.items():
            self.assertEqual(s.positions, self.joints[n].positions)

    def test_no_face_points_in_the_result(self):
        self.assertEqual(set(self.filled), set(oh.BODY_JOINTS))


if __name__ == "__main__":
    unittest.main()
