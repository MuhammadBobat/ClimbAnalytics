"""
Tests for the candidate CoM estimators B-E in metrics.py (step 2.2c, DECISIONS.md D-026).
Synthetic poses with answers computed independently in the test from TABLE_PERCENT (an
independent copy of de Leva 1996 Table 4); nothing here is a measurement.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import unittest

from metrics import (
    GAP_LOW_CONFIDENCE,
    JointSeries,
    apply_frame_exclusions,
    blended_limb_usage,
    compute_com,
    compute_com_blended,
    compute_com_segment_model,
    compute_com_segment_model_no_head,
    compute_com_trunk_legs,
    compute_com_trunk_point,
    compute_com_working,
    compute_com_working_edl_raw_gate,
    position_gated,
    WORKING_COM_CUTOFF,
    renormalised_specs,
    segment_model_specs,
    trunk_cm_fraction,
)
from tests.test_com_segment import COCO, TABLE_PERCENT

# deliberately asymmetric, non-degenerate pose (x right, y down)
POSE = {
    "nose": (0.2, -0.7), "left_eye": (-0.1, -1.0), "right_eye": (0.3, -1.1),
    "left_ear": (-0.4, -0.9), "right_ear": (0.5, -1.2),
    "left_shoulder": (-1.1, 0.1), "right_shoulder": (0.9, -0.2),
    "left_elbow": (-2.0, 3.0), "right_elbow": (2.4, -4.0),
    "left_wrist": (-1.0, 6.5), "right_wrist": (3.5, -8.0),
    "left_hip": (-0.6, 9.0), "right_hip": (0.7, 10.2),
    "left_knee": (-1.5, 17.0), "right_knee": (2.0, 14.0),
    "left_ankle": (-0.8, 25.0), "right_ankle": (3.0, 21.0),
}
SEXES = ("male", "female", "mean")
LIMB_NAMES = [f"{s}_{n}" for s in ("left", "right")
              for n in ("upper_arm", "forearm", "hand", "thigh", "shank", "foot")]


def joints_of(pose, conf=0.9, confs=None, frames=1):
    """One JointSeries per joint, `frames` identical frames; confs {joint: c} overrides conf."""
    confs = confs or {}
    ts = [i / 30.0 for i in range(frames)]
    return {n: JointSeries(n, [pose.get(n)] * frames, ts, confidences=[confs.get(n, conf)] * frames)
            for n in COCO}


def swapped(pose):
    def sw(n):
        return n.replace("left_", "R_").replace("right_", "left_").replace("R_", "right_")
    return {sw(n): p for n, p in pose.items()}


def transformed(pose, k=1.0, dx=0.0, dy=0.0):
    return {n: (k * p[0] + dx, k * p[1] + dy) for n, p in pose.items()}


def at(series, t=0):
    return series.positions[t]


def candidates(joints):
    """Name -> series for A-F (E at the four cut-offs, fitted-w stand-in 0.7)."""
    out = {
        "A": compute_com(joints),
        "B": compute_com_trunk_point(joints, 0.7),
        "C": compute_com_segment_model_no_head(joints, "male"),
        "D": compute_com_trunk_legs(joints, "male"),
        "F": compute_com_segment_model(joints, "male"),
    }
    for c in (0.5, 0.6, 0.7, 0.8):
        out[f"E{c}"] = compute_com_blended(joints, "male", 0.7, c)
    return out


def reference_com(pose, sex, keep):
    """Independent segment-model CoM from TABLE_PERCENT for the kept segment kinds, mass
    fractions divided by their sum. keep: set of kinds among head, trunk, upper_arm, forearm,
    hand, thigh, shank, foot."""
    tab = TABLE_PERCENT["male" if sex == "male" else sex]
    def mid(a, b):
        return ((pose[a][0] + pose[b][0]) / 2, (pose[a][1] + pose[b][1]) / 2)
    items = []
    if "head" in keep:
        items.append((tab["head"][0], mid("left_ear", "right_ear")))
    if "trunk" in keep:
        m, r = tab["trunk"]
        p, d = mid("left_shoulder", "right_shoulder"), mid("left_hip", "right_hip")
        items.append((m, (p[0] + r / 100 * (d[0] - p[0]), p[1] + r / 100 * (d[1] - p[1]))))
    for side in ("left", "right"):
        for kind, pj, dj in (("upper_arm", "shoulder", "elbow"), ("forearm", "elbow", "wrist"),
                             ("thigh", "hip", "knee"), ("shank", "knee", "ankle")):
            if kind in keep:
                m, r = tab[kind]
                p, d = pose[f"{side}_{pj}"], pose[f"{side}_{dj}"]
                items.append((m, (p[0] + r / 100 * (d[0] - p[0]), p[1] + r / 100 * (d[1] - p[1]))))
        for kind, j in (("hand", "wrist"), ("foot", "ankle")):
            if kind in keep:
                items.append((tab[kind][0], pose[f"{side}_{j}"]))
    tot = sum(m for m, _ in items)
    return (sum(m * c[0] for m, c in items) / tot, sum(m * c[1] for m, c in items) / tot)


class RenormalisationTests(unittest.TestCase):
    def test_fractions_sum_to_one_c_d_e(self):
        for sex in SEXES:
            specs = segment_model_specs(sex)
            c, rc = renormalised_specs(specs, {"head"})
            self.assertAlmostEqual(sum(s.mass_fraction for s in c), 1.0, places=12)
            d_drop = {"head"} | {f"{s}_{n}" for s in ("left", "right")
                                 for n in ("upper_arm", "forearm", "hand")}
            d, rd = renormalised_specs(specs, d_drop)
            self.assertAlmostEqual(sum(s.mass_fraction for s in d), 1.0, places=12)
            limbs = [s for s in c if s.name != "trunk"]
            trunk = next(s for s in c if s.name == "trunk")
            self.assertAlmostEqual(trunk.mass_fraction + sum(s.mass_fraction for s in limbs), 1.0,
                                   places=12)
            self.assertEqual(rc.removed, ("head",))
            self.assertAlmostEqual(rc.scale * rc.kept_mass_before, 1.0, places=12)
            self.assertAlmostEqual(rc.removed_mass + rc.kept_mass_before,
                                   sum(s.mass_fraction for s in specs), places=12)
            self.assertEqual(len(rd.removed), 7)

    def test_male_kept_mass(self):
        _, r = renormalised_specs(segment_model_specs("male"), {"head"})
        self.assertAlmostEqual(r.removed_mass, 0.0694, places=12)
        self.assertAlmostEqual(r.kept_mass_before, 0.9306, places=12)

    def test_unknown_segment_rejected(self):
        with self.assertRaises(KeyError):
            renormalised_specs(segment_model_specs("male"), {"tail"})


class TrunkPointTests(unittest.TestCase):
    def test_ends_and_proxy(self):
        j = joints_of(POSE)
        sm = ((POSE["left_shoulder"][0] + POSE["right_shoulder"][0]) / 2,
              (POSE["left_shoulder"][1] + POSE["right_shoulder"][1]) / 2)
        hm = ((POSE["left_hip"][0] + POSE["right_hip"][0]) / 2,
              (POSE["left_hip"][1] + POSE["right_hip"][1]) / 2)
        for w, want in ((0.0, sm), (1.0, hm)):
            got = at(compute_com_trunk_point(j, w))
            self.assertAlmostEqual(got[0], want[0]); self.assertAlmostEqual(got[1], want[1])
        a, b = at(compute_com(j)), at(compute_com_trunk_point(j, 0.5))
        self.assertAlmostEqual(a[0], b[0]); self.assertAlmostEqual(a[1], b[1])

    def test_w_range(self):
        j = joints_of(POSE)
        for w in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                compute_com_trunk_point(j, w)

    def test_gap_gives_none_with_reason(self):
        pose = dict(POSE); pose["left_hip"] = None
        s = compute_com_trunk_point(joints_of(pose), 0.5)
        self.assertIsNone(at(s))


class ReducedModelTests(unittest.TestCase):
    def test_known_answer_c_and_d(self):
        j = joints_of(POSE)
        everything = {"head", "trunk", "upper_arm", "forearm", "hand", "thigh", "shank", "foot"}
        got = at(compute_com_segment_model_no_head(j, "male"))
        want = reference_com(POSE, "male", everything - {"head"})
        self.assertAlmostEqual(got[0], want[0]); self.assertAlmostEqual(got[1], want[1])
        got = at(compute_com_trunk_legs(j, "male"))
        want = reference_com(POSE, "male", {"trunk", "thigh", "shank", "foot"})
        self.assertAlmostEqual(got[0], want[0]); self.assertAlmostEqual(got[1], want[1])

    def test_c_ignores_ears_d_ignores_arms(self):
        base = joints_of(POSE)
        moved = dict(POSE)
        moved["left_ear"], moved["right_ear"] = (50.0, 50.0), (-60.0, 7.0)
        self.assertEqual(at(compute_com_segment_model_no_head(base, "male")),
                         at(compute_com_segment_model_no_head(joints_of(moved), "male")))
        moved = dict(POSE)
        for n in ("left_elbow", "right_elbow", "left_wrist", "right_wrist"):
            moved[n] = (99.0, -99.0)
        self.assertEqual(at(compute_com_trunk_legs(base, "male")),
                         at(compute_com_trunk_legs(joints_of(moved), "male")))

    def test_gap_in_kept_joint_gives_none_but_not_in_dropped(self):
        pose = dict(POSE); pose["left_ear"] = None
        self.assertIsNotNone(at(compute_com_segment_model_no_head(joints_of(pose), "male")))
        self.assertIsNone(at(compute_com_segment_model(joints_of(pose), "male")))
        pose = dict(POSE); pose["left_wrist"] = None
        self.assertIsNone(at(compute_com_segment_model_no_head(joints_of(pose), "male")))
        self.assertIsNotNone(at(compute_com_trunk_legs(joints_of(pose), "male")))
        pose = dict(POSE); pose["left_knee"] = None
        self.assertIsNone(at(compute_com_trunk_legs(joints_of(pose), "male")))


class BlendedTests(unittest.TestCase):
    def test_equals_b_when_every_limb_below_cutoff(self):
        j = joints_of(POSE, conf=0.9)
        b = at(compute_com_trunk_point(j, 0.7))
        e = at(compute_com_blended(j, "male", 0.7, 0.95))
        self.assertAlmostEqual(e[0], b[0]); self.assertAlmostEqual(e[1], b[1])
        self.assertEqual(blended_limb_usage(j, "male", 0.7, 0.95), [()])
        # limb positions missing altogether
        pose = {n: (None if n.split("_", 1)[-1] in ("elbow", "wrist", "knee", "ankle") else p)
                for n, p in POSE.items()}
        e = at(compute_com_blended(joints_of(pose), "male", 0.7, 0.5))
        self.assertAlmostEqual(e[0], b[0]); self.assertAlmostEqual(e[1], b[1])

    def test_equals_head_renormalised_full_model_at_trunk_cm(self):
        for sex in SEXES:
            j = joints_of(POSE, conf=0.95)
            e = at(compute_com_blended(j, sex, trunk_cm_fraction(sex), 0.8))
            c = at(compute_com_segment_model_no_head(j, sex))
            self.assertAlmostEqual(e[0], c[0]); self.assertAlmostEqual(e[1], c[1])
            self.assertEqual(len(blended_limb_usage(j, sex, 0.5, 0.8)[0]), 12)

    def test_defined_when_limb_joint_missing_and_drops_only_that_segment(self):
        pose = dict(POSE); pose["left_wrist"] = None
        j = joints_of(pose)
        e = at(compute_com_blended(j, "male", 0.6, 0.5))
        self.assertIsNotNone(e)
        used = blended_limb_usage(j, "male", 0.6, 0.5)[0]
        self.assertEqual(set(LIMB_NAMES) - set(used), {"left_forearm", "left_hand"})
        # same value as pushing that wrist below the cut-off with all else equal
        j2 = joints_of(POSE, confs={"left_wrist": 0.55})
        e2 = at(compute_com_blended(j2, "male", 0.6, 0.6))
        self.assertAlmostEqual(e[0], e2[0]); self.assertAlmostEqual(e[1], e2[1])

    def test_none_when_trunk_joint_missing(self):
        pose = dict(POSE); pose["right_shoulder"] = None
        s = compute_com_blended(joints_of(pose), "male", 0.6, 0.5)
        self.assertIsNone(at(s))

    def test_usage_non_increasing_with_cutoff(self):
        confs = {"left_wrist": 0.55, "right_knee": 0.65, "left_ankle": 0.75, "right_elbow": 0.85}
        j = joints_of(POSE, conf=0.9, confs=confs)
        counts = [len(blended_limb_usage(j, "male", 0.6, c)[0]) for c in (0.5, 0.6, 0.7, 0.8, 0.9)]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertLess(counts[-1], counts[0])

    def test_cutoff_is_inclusive(self):
        j = joints_of(POSE, conf=0.7)
        self.assertEqual(len(blended_limb_usage(j, "male", 0.6, 0.7)[0]), 12)

    def test_requires_confidences_and_valid_cutoff(self):
        bare = {n: JointSeries(n, [POSE[n]], [0.0]) for n in COCO}
        with self.assertRaises(ValueError):
            compute_com_blended(bare, "male", 0.6, 0.5)
        with self.assertRaises(ValueError):
            compute_com_blended(joints_of(POSE), "male", 0.6, 0.0)

    def test_excluded_frame_is_none_and_confidences_are_carried(self):
        j = joints_of(POSE, frames=3)
        ex = apply_frame_exclusions(j, {1})
        self.assertEqual(ex["left_wrist"].confidences, j["left_wrist"].confidences)
        s = compute_com_blended(ex, "male", 0.6, 0.5)
        self.assertIsNone(s.positions[1]); self.assertIsNotNone(s.positions[0])

    def test_confidence_length_validated(self):
        with self.assertRaises(ValueError):
            JointSeries("j", [(0, 0)], [0.0], confidences=[0.9, 0.9])

    def test_low_confidence_joint_in_blend_keeps_gap_reason_for_trunk(self):
        j = joints_of(POSE)
        j["left_hip"] = JointSeries("left_hip", [None], [0.0], [GAP_LOW_CONFIDENCE], [0.3])
        s = compute_com_blended(j, "male", 0.6, 0.5)
        self.assertEqual(s.gap_reasons[0], GAP_LOW_CONFIDENCE)


class WorkingComTests(unittest.TestCase):
    """The fallback: E-dL at 0.5 with the raw-confidence gate (the working CoM of 2026-11-01, D-026)."""
    def test_is_edl_at_the_d007_cutoff(self):
        self.assertEqual(WORKING_COM_CUTOFF, 0.5)
        confs = {"left_wrist": 0.55, "right_knee": 0.45}
        j = joints_of(POSE, conf=0.9, confs=confs)
        for sex in SEXES:
            got = compute_com_working_edl_raw_gate(j, sex)
            want = compute_com_blended(j, sex, trunk_cm_fraction(sex), 0.5)
            self.assertEqual(got.positions, want.positions)
        self.assertEqual(got.joint_name, "com_working_edl_mean")

    def test_equals_no_head_model_when_every_joint_is_present(self):
        j = joints_of(POSE, conf=0.9)
        a, c = at(compute_com_working_edl_raw_gate(j, "male")), at(compute_com_segment_model_no_head(j, "male"))
        self.assertAlmostEqual(a[0], c[0]); self.assertAlmostEqual(a[1], c[1])

    def test_defined_when_a_limb_is_lost_and_sex_is_required(self):
        pose = dict(POSE); pose["left_ankle"] = None
        self.assertIsNotNone(at(compute_com_working_edl_raw_gate(joints_of(pose), "male")))
        with self.assertRaises(TypeError):
            compute_com_working_edl_raw_gate(joints_of(POSE))


class NewWorkingComTests(unittest.TestCase):
    """The working CoM since 2026-11-26: EdLfill_c0.5 (position gate)."""

    def _filled(self):
        # left_wrist: position present (as if Kalman-filled) but raw confidence 0.2; right_knee: no position at all
        pose = dict(POSE); pose["right_knee"] = None
        return joints_of(pose, conf=0.9, confs={"left_wrist": 0.2, "right_knee": 0.2})

    def test_equals_edlfill_on_filled_data(self):
        j = self._filled()
        want = compute_com_blended(position_gated(j), "male", trunk_cm_fraction("male"), 0.5)
        self.assertEqual(compute_com_working(j, "male").positions, want.positions)
        # a filled limb counts, a missing one does not: differs from the raw-confidence gate only through left_wrist
        self.assertNotEqual(compute_com_working(j, "male").positions,
                            compute_com_working_edl_raw_gate(j, "male").positions)

    def test_equals_edl_on_unfilled_input(self):
        # no gap filling: a position exists only where confidence >= 0.5
        pose = dict(POSE); pose["left_wrist"] = None; pose["right_knee"] = None
        j = joints_of(pose, conf=0.9, confs={"left_wrist": 0.2, "right_knee": 0.2})
        for sex in SEXES:
            self.assertEqual(compute_com_working(j, sex).positions, compute_com_working_edl_raw_gate(j, sex).positions)

    def test_position_gated_does_not_modify_the_input_and_requires_nothing_else(self):
        j = self._filled()
        g = position_gated(j)
        self.assertEqual(j["left_wrist"].confidences, [0.2])
        self.assertEqual(g["left_wrist"].confidences, [1.0])
        self.assertEqual(g["right_knee"].confidences, [0.2])

    def test_sex_is_required(self):
        with self.assertRaises(TypeError):
            compute_com_working(joints_of(POSE))

    def test_2_2c_copy_of_position_gated_agrees_with_metrics(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "2.2c"))
        import compare_com_candidates as cc
        j = self._filled()
        a, b = cc.position_gated(j), position_gated(j)
        self.assertEqual(sorted(a), sorted(b))
        for n in a:
            self.assertEqual(a[n].positions, b[n].positions)
            self.assertEqual(a[n].confidences, b[n].confidences)
            self.assertEqual(a[n].gap_reasons, b[n].gap_reasons)
            self.assertEqual(a[n].joint_name, b[n].joint_name)


class InvarianceTests(unittest.TestCase):
    def _compare(self, a, b, tol=1e-9):
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            pa, pb = at(a[k]), at(b[k])
            self.assertAlmostEqual(pa[0], pb[0], delta=tol, msg=k)
            self.assertAlmostEqual(pa[1], pb[1], delta=tol, msg=k)

    def test_translation(self):
        base = candidates(joints_of(POSE))
        shifted = candidates(joints_of(transformed(POSE, dx=123.5, dy=-77.25)))
        want = {k: type(v)(v.joint_name, [(at(v)[0] + 123.5, at(v)[1] - 77.25)], v.timestamps)
                for k, v in base.items()}
        self._compare(want, shifted)

    def test_scale(self):
        k = 3.7
        base = candidates(joints_of(POSE))
        scaled = candidates(joints_of(transformed(POSE, k=k)))
        want = {n: type(v)(v.joint_name, [(k * at(v)[0], k * at(v)[1])], v.timestamps)
                for n, v in base.items()}
        self._compare(want, scaled, tol=1e-8)

    def test_left_right_swap(self):
        self._compare(candidates(joints_of(POSE)), candidates(joints_of(swapped(POSE))))

    def test_left_right_swap_with_per_joint_confidences(self):
        confs = {"left_wrist": 0.55, "right_ankle": 0.65}
        sw = {"right_wrist": 0.55, "left_ankle": 0.65}
        self._compare(candidates(joints_of(POSE, confs=confs)),
                      candidates(joints_of(swapped(POSE), confs=sw)))


if __name__ == "__main__":
    unittest.main()
