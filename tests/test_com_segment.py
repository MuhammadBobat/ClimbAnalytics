"""
Tests for the de Leva (1996) segment-model CoM in metrics.py. Synthetic poses with
hand-computed answers; nothing here is a measurement.

TABLE_PERCENT below is an independent copy of the Table 4 numbers (percent) that the model
uses, read from the PDF (de Leva 1996, p. 1228; trunk = MIDS-MIDH row, shank = KJC-AJC
row) — it catches accidental edits of the stored table, not transcription errors in the
source copy.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import unittest

import metrics
from metrics import (
    GAP_EXCLUDED,
    GAP_LOW_CONFIDENCE,
    GAP_MISSING_DETECTION,
    JointSeries,
    SegmentSpec,
    compute_com_from_segments,
    compute_com_segment_model,
    segment_model_joints,
    segment_model_specs,
)

COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]

# percent: mass, and CM position from the proximal end where the segment has one
TABLE_PERCENT = {
    "male": {"head": (6.94,), "trunk": (43.46, 43.10), "upper_arm": (2.71, 57.72),
             "forearm": (1.62, 45.74), "hand": (0.61,), "thigh": (14.16, 40.95),
             "shank": (4.33, 43.95), "foot": (1.37,)},
    "female": {"head": (6.68,), "trunk": (42.57, 37.82), "upper_arm": (2.55, 57.54),
               "forearm": (1.38, 45.59), "hand": (0.56,), "thigh": (14.78, 36.12),
               "shank": (4.81, 43.52), "foot": (1.29,)},
}


def joints_from(frames):
    """frames: list of {joint: (x, y) | None}; missing keys are None. 30 fps timestamps."""
    ts = [i / 30.0 for i in range(len(frames))]
    return {n: JointSeries(n, [f.get(n) for f in frames], ts) for n in COCO}


def stick_figure(dx=0.0, dy=0.0, k=1.0):
    """Symmetric standing pose (x right, y down); head mass at ear midpoint (0, -1)."""
    base = {
        "left_ear": (-0.3, -1.0), "right_ear": (0.3, -1.0), "nose": (0.0, -0.8),
        "left_eye": (-0.1, -1.1), "right_eye": (0.1, -1.1),
        "left_shoulder": (-1.0, 0.0), "right_shoulder": (1.0, 0.0),
        "left_elbow": (-1.5, 5.0), "right_elbow": (1.5, 5.0),
        "left_wrist": (-1.5, 10.0), "right_wrist": (1.5, 10.0),
        "left_hip": (-0.5, 10.0), "right_hip": (0.5, 10.0),
        "left_knee": (-0.5, 20.0), "right_knee": (0.5, 20.0),
        "left_ankle": (-0.5, 30.0), "right_ankle": (0.5, 30.0),
    }
    return {n: (k * x + dx, k * y + dy) for n, (x, y) in base.items()}


class TableTests(unittest.TestCase):
    def test_mass_fractions_sum_to_one(self):
        # male 100.00%, female 99.99% in the source (rounded); mean 99.995%
        for sex in ("male", "female", "mean"):
            total = sum(s.mass_fraction for s in segment_model_specs(sex))
            self.assertAlmostEqual(total, 1.0, delta=2e-4, msg=sex)

    def test_fourteen_segments_and_fourteen_joints(self):
        specs = segment_model_specs("male")
        self.assertEqual(len(specs), 14)
        self.assertEqual(len(segment_model_joints()), 14)
        self.assertEqual(set(COCO) - set(segment_model_joints()), {"nose", "left_eye", "right_eye"})

    def test_stored_table_matches_independent_copy(self):
        for sex in ("male", "female"):
            by_name = {s.name: s for s in segment_model_specs(sex)}
            for seg, vals in TABLE_PERCENT[sex].items():
                names = [seg] if seg in ("head", "trunk") else [f"left_{seg}", f"right_{seg}"]
                for n in names:
                    spec = by_name[n]
                    self.assertAlmostEqual(spec.mass_fraction * 100, vals[0], places=9, msg=f"{sex} {n}")
                    if len(vals) == 2:
                        self.assertAlmostEqual(spec.cm_fraction * 100, vals[1], places=9, msg=f"{sex} {n}")
                    else:
                        self.assertIsNone(spec.cm_fraction)

    def test_mean_is_unweighted_average_of_tables(self):
        m = {s.name: s for s in segment_model_specs("male")}
        f = {s.name: s for s in segment_model_specs("female")}
        for s in segment_model_specs("mean"):
            self.assertAlmostEqual(s.mass_fraction, (m[s.name].mass_fraction + f[s.name].mass_fraction) / 2)
            if s.cm_fraction is not None:
                self.assertAlmostEqual(s.cm_fraction, (m[s.name].cm_fraction + f[s.name].cm_fraction) / 2)

    def test_sex_is_required_and_validated(self):
        joints = joints_from([stick_figure()])
        with self.assertRaises(TypeError):
            compute_com_segment_model(joints)
        with self.assertRaises(ValueError):
            compute_com_segment_model(joints, "other")
        with self.assertRaises(ValueError):
            segment_model_specs("Male")


class ToyCaseTests(unittest.TestCase):
    """Two-segment toy models; masses and ratios are fixtures, not anthropometric values."""

    def toy(self, m1=0.5, m2=0.5):
        a, b, c = (0.0, 0.0), (0.0, 2.0), (2.0, 2.0)
        joints = {n: JointSeries(n, [p], [0.0]) for n, p in (("a", a), ("b", b), ("c", c))}
        specs = [SegmentSpec("s1", m1, ("a",), ("b",), 0.5), SegmentSpec("s2", m2, ("b",), ("c",), 0.5)]
        return joints, specs

    def test_equal_masses(self):
        # segment CMs (0,1) and (1,2) -> CoM (0.5, 1.5)
        joints, specs = self.toy()
        self.assertEqual(compute_com_from_segments(joints, specs).positions[0], (0.5, 1.5))

    def test_unequal_masses(self):
        # 0.75*(0,1) + 0.25*(1,2) = (0.25, 1.25)
        joints, specs = self.toy(0.75, 0.25)
        self.assertEqual(compute_com_from_segments(joints, specs).positions[0], (0.25, 1.25))

    def test_weights_are_normalised_by_total_mass(self):
        joints, specs = self.toy(3.0, 1.0)  # same ratio as 0.75 / 0.25
        self.assertEqual(compute_com_from_segments(joints, specs).positions[0], (0.25, 1.25))

    def test_midpoint_endpoints_and_point_mass(self):
        pts = {"l": (0.0, 0.0), "r": (4.0, 0.0), "h": (2.0, 10.0), "f": (6.0, 6.0)}
        joints = {n: JointSeries(n, [p], [0.0]) for n, p in pts.items()}
        specs = [
            SegmentSpec("trunk", 0.5, ("l", "r"), ("h",), 0.4),  # (2,0) -> (2,10): CM (2, 4)
            SegmentSpec("foot", 0.5, ("f",)),                    # point mass at (6, 6)
        ]
        self.assertEqual(compute_com_from_segments(joints, specs).positions[0], (4.0, 5.0))

    def test_spec_validation(self):
        with self.assertRaises(ValueError):
            SegmentSpec("x", 0.1, ("a",), ("b",), None)
        with self.assertRaises(ValueError):
            SegmentSpec("x", 0.1, ("a",), None, 0.5)


class SegmentModelTests(unittest.TestCase):
    def test_stick_figure_known_answer_male(self):
        # hand-computed: sum(m*y) = -0.0694 + 0.4346*4.310 + 0.0542*2.886 + 0.0324*7.287
        #   + 0.0122*10 + 0.2832*14.095 + 0.0866*24.395 + 0.0274*30 = 9.244557 (masses sum to 1)
        # x = 0 by left/right symmetry
        com = compute_com_segment_model(joints_from([stick_figure()]), "male").positions[0]
        self.assertAlmostEqual(com[0], 0.0, places=9)
        self.assertAlmostEqual(com[1], 9.244557, places=5)

    def test_translation(self):
        a = compute_com_segment_model(joints_from([stick_figure()]), "male").positions[0]
        b = compute_com_segment_model(joints_from([stick_figure(dx=100.0, dy=-40.0)]), "male").positions[0]
        self.assertAlmostEqual(b[0], a[0] + 100.0, places=9)
        self.assertAlmostEqual(b[1], a[1] - 40.0, places=9)

    def test_scale(self):
        a = compute_com_segment_model(joints_from([stick_figure()]), "female").positions[0]
        b = compute_com_segment_model(joints_from([stick_figure(k=3.0)]), "female").positions[0]
        self.assertAlmostEqual(b[0], 3.0 * a[0], places=9)
        self.assertAlmostEqual(b[1], 3.0 * a[1], places=9)

    def test_left_right_swap(self):
        pose = stick_figure()
        pose["left_wrist"] = (-4.0, 3.0)   # make the pose asymmetric
        pose["right_knee"] = (2.0, 17.0)
        swapped = {}
        for n, p in pose.items():
            other = n.replace("left_", "R_").replace("right_", "left_").replace("R_", "right_")
            swapped[other] = p
        for sex in ("male", "female", "mean"):
            a = compute_com_segment_model(joints_from([pose]), sex).positions[0]
            b = compute_com_segment_model(joints_from([swapped]), sex).positions[0]
            self.assertAlmostEqual(a[0], b[0], places=9, msg=sex)
            self.assertAlmostEqual(a[1], b[1], places=9, msg=sex)

    def test_missing_joint_gives_none_for_every_required_joint(self):
        for j in segment_model_joints():
            pose = stick_figure()
            pose[j] = None
            com = compute_com_segment_model(joints_from([stick_figure(), pose, stick_figure()]), "male")
            self.assertIsNotNone(com.positions[0], j)
            self.assertIsNone(com.positions[1], j)
            self.assertIsNotNone(com.positions[2], j)

    def test_nose_and_eyes_are_not_used(self):
        a = compute_com_segment_model(joints_from([stick_figure()]), "male").positions[0]
        pose = stick_figure()
        pose["nose"], pose["left_eye"], pose["right_eye"] = (50.0, 50.0), (-9.0, 9.0), None
        b = compute_com_segment_model(joints_from([pose]), "male").positions[0]
        self.assertEqual(a, b)

    def test_gap_reason_propagates_with_priority(self):
        joints = joints_from([stick_figure()] * 4)
        joints["left_elbow"].positions[1] = None
        joints["left_elbow"].gap_reasons[1] = GAP_LOW_CONFIDENCE
        joints["left_elbow"].positions[2] = None
        joints["left_elbow"].gap_reasons[2] = GAP_LOW_CONFIDENCE
        joints["right_ankle"].positions[2] = None
        joints["right_ankle"].gap_reasons[2] = GAP_MISSING_DETECTION
        joints["left_hip"].positions[3] = None
        joints["left_hip"].gap_reasons[3] = GAP_EXCLUDED
        joints["right_ankle"].positions[3] = None
        joints["right_ankle"].gap_reasons[3] = GAP_MISSING_DETECTION
        com = compute_com_segment_model(joints, "male")
        self.assertEqual(com.gap_reasons, [None, GAP_LOW_CONFIDENCE, GAP_MISSING_DETECTION, GAP_EXCLUDED])

    def test_series_name_and_proxy_unchanged(self):
        joints = joints_from([stick_figure()])
        self.assertEqual(compute_com_segment_model(joints, "male").joint_name,
                         "com_segment_model_de_leva_1996_male")
        # the trunk-centroid proxy is the baseline and keeps its own name and result
        proxy = metrics.compute_com(joints)
        self.assertEqual(proxy.joint_name, "com_trunk_centroid_proxy")
        self.assertAlmostEqual(proxy.positions[0][1], 5.0)  # (shoulders y=0, hips y=10) -> 5


class ExclusionHelperTests(unittest.TestCase):
    def test_apply_frame_exclusions_copies_and_marks(self):
        joints = joints_from([stick_figure()] * 5)
        out = metrics.apply_frame_exclusions(joints, {1, 3, 99})
        self.assertIsNone(out["left_wrist"].positions[1])
        self.assertEqual(out["left_wrist"].gap_reasons[3], GAP_EXCLUDED)
        self.assertIsNotNone(joints["left_wrist"].positions[1])  # input untouched
        self.assertEqual(len(out["left_wrist"].positions), 5)


if __name__ == "__main__":
    unittest.main()
