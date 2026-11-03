"""
Tests for person_lock.py (D-024 / D-027): the legacy rule is reproduced exactly, and each part of
the gated family (distance gate, size consistency, rescue tier, release) behaves as documented.
Synthetic boxes; nothing here is a measurement and no value is a project threshold.

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import random
import unittest

from person_lock import (Box, LockParams, LockState, explain_rejection, iou, legacy_candidate,
                         select_climber_box)


def legacy_reference(boxes, prev_center):
    """validation/validate_vitpose.py select_person_box, copied for comparison."""
    if not boxes:
        return None
    if prev_center is None:
        return max(boxes, key=lambda b: b.conf)
    px, py = prev_center
    return min(boxes, key=lambda b: (b.center[0] - px) ** 2 + (b.center[1] - py) ** 2)


def run(frames, params):
    st, out = LockState(), []
    for boxes in frames:
        box, st, info = select_climber_box(boxes, st, params)
        out.append((box, info))
    return out


CLIMBER = Box(580, 250, 240, 800, 0.9)          # centre (700, 650), h 800
BYSTANDER = Box(960, 1210, 100, 200, 0.55)       # centre (1010, 1310): 0.91 h away, h ratio 4


class LegacyTests(unittest.TestCase):
    def test_reproduces_select_person_box_on_random_sequences(self):
        rng = random.Random(1)
        for _ in range(50):
            frames = [[Box(rng.uniform(0, 900), rng.uniform(0, 900), rng.uniform(50, 300),
                           rng.uniform(100, 800), rng.uniform(0.5, 1.0))
                       for _ in range(rng.choice([0, 0, 1, 2, 3]))] for _ in range(40)]
            prev, st = None, LockState()
            for boxes in frames:
                want = legacy_reference(boxes, prev)
                got, st, _ = select_climber_box(boxes, st, LockParams.legacy())
                self.assertEqual(got, want)
                if want is not None:
                    prev = want.center

    def test_boxes_below_the_main_cutoff_are_ignored(self):
        low = Box(0, 0, 10, 10, 0.3)
        got, _, info = select_climber_box([low], LockState(), LockParams.legacy())
        self.assertIsNone(got)
        self.assertEqual(info.reason, "no_box")
        self.assertTrue(LockParams.legacy().is_legacy)

    def test_failure_mechanism_is_reproduced_by_legacy(self):
        # climber, five empty frames, then only the bystander: legacy accepts it and keeps it
        frames = [[CLIMBER]] * 3 + [[]] * 5 + [[BYSTANDER], [BYSTANDER]]
        out = run(frames, LockParams.legacy())
        self.assertEqual(out[8][0], BYSTANDER)
        self.assertEqual(out[9][0], BYSTANDER)


class GateTests(unittest.TestCase):
    P = LockParams(gate_g0=0.5)

    def test_far_box_is_rejected_near_box_accepted(self):
        out = run([[CLIMBER], [BYSTANDER]], self.P)
        self.assertIsNone(out[1][0])
        self.assertEqual(out[1][1].reason, "gated")
        near = Box(590, 255, 240, 790, 0.8)
        self.assertEqual(run([[CLIMBER], [near]], self.P)[1][0], near)

    def test_allowance_grows_with_the_gap(self):
        far = Box(580 + 600, 250, 240, 800, 0.8)            # 0.75 h to the right
        p0, p1 = LockParams(gate_g0=0.5), LockParams(gate_g0=0.5, gate_v=0.1)
        frames = [[CLIMBER]] + [[]] * 5 + [[far]]
        self.assertIsNone(run(frames, p0)[-1][0])
        self.assertEqual(run(frames, p1)[-1][0], far)        # 0.5 + 0.1 * 5 = 1.0 >= 0.75

    def test_nearest_among_passing_boxes(self):
        a, b = Box(600, 250, 240, 800, 0.8), Box(640, 250, 240, 800, 0.9)
        self.assertEqual(run([[CLIMBER], [b, a]], self.P)[1][0], a)

    def test_reference_height_is_the_median_of_the_last_k(self):
        p = LockParams(gate_g0=0.5, size_k=3)
        tall, short = Box(580, 250, 240, 1000, 0.9), Box(580, 250, 240, 400, 0.9)
        out = run([[tall], [tall], [short], [Box(580, 250, 240, 700, 0.9)]], p)
        self.assertIsNotNone(out[3][0])


class SizeTests(unittest.TestCase):
    def test_size_ratio_limits(self):
        p = LockParams(size_s=2.0, size_k=1)
        ok, too_small, too_big = Box(580, 250, 240, 500, 0.9), Box(580, 250, 100, 300, 0.9), Box(580, 250, 240, 1700, 0.9)
        self.assertIsNotNone(run([[CLIMBER], [ok]], p)[1][0])
        self.assertIsNone(run([[CLIMBER], [too_small]], p)[1][0])
        self.assertIsNone(run([[CLIMBER], [too_big]], p)[1][0])

    def test_long_reference_resists_a_persistent_wrong_lock(self):
        p = LockParams(size_s=2.0, size_k=30)
        frames = [[CLIMBER]] * 10 + [[Box(580, 250, 120, 300, 0.9)]] * 5
        out = run(frames, p)
        self.assertTrue(all(o[0] is None for o in out[10:]))


class RescueTests(unittest.TestCase):
    def test_rescue_used_only_when_no_main_box_passes(self):
        climber_low = Box(585, 255, 240, 790, 0.41)
        p = LockParams(gate_g0=0.5, conf_rescue=0.1)
        out = run([[CLIMBER], [BYSTANDER, climber_low]], p)
        self.assertEqual(out[1][0], climber_low)
        self.assertEqual(out[1][1].tier, "rescue")

    def test_main_box_that_passes_wins_even_if_a_rescue_box_is_nearer(self):
        main = Box(660, 250, 240, 800, 0.9)
        rescue_near = Box(581, 250, 240, 800, 0.2)
        p = LockParams(gate_g0=1.0, conf_rescue=0.1)
        out = run([[CLIMBER], [main, rescue_near]], p)
        self.assertEqual(out[1][0], main)
        self.assertEqual(out[1][1].tier, "main")

    def test_rescue_boxes_below_the_floor_are_ignored(self):
        p = LockParams(gate_g0=0.5, conf_rescue=0.3)
        self.assertIsNone(run([[CLIMBER], [Box(585, 255, 240, 790, 0.2)]], p)[1][0])

    def test_no_rescue_without_a_lock(self):
        p = LockParams(gate_g0=0.5, conf_rescue=0.1)
        self.assertIsNone(run([[Box(0, 0, 100, 100, 0.3)]], p)[0][0])


class ReleaseTests(unittest.TestCase):
    def test_release_after_n_frames_takes_highest_confidence_main_box(self):
        p = LockParams(gate_g0=0.5, release_n=5)
        frames = [[CLIMBER]] + [[]] * 5 + [[BYSTANDER]]
        out = run(frames, p)
        self.assertEqual(out[-1][0], BYSTANDER)
        self.assertEqual(out[-1][1].tier, "release")
        # one frame earlier (gap 4 < 5) it is still gated
        self.assertIsNone(run([[CLIMBER]] + [[]] * 4 + [[BYSTANDER]], p)[-1][0])

    def test_release_resets_the_history(self):
        p = LockParams(gate_g0=0.5, release_n=2, size_s=2.0)
        frames = [[CLIMBER], [], [], [BYSTANDER], [Box(965, 1212, 100, 210, 0.6)]]
        out = run(frames, p)
        self.assertEqual(out[3][0], BYSTANDER)
        self.assertIsNotNone(out[4][0])          # judged against the new (bystander) reference


class FailureScenarioTests(unittest.TestCase):
    """The IMG_2419 shape: climber lost, bystander the only box >= 0.5, climber weak but present."""

    def frames(self):
        climber_weak = Box(585, 255, 240, 790, 0.41)
        return ([[CLIMBER]] * 3 + [[]] * 5 + [[BYSTANDER, climber_weak]] * 6 + [[BYSTANDER, CLIMBER]])

    def test_legacy_locks_on_the_bystander(self):
        out = run(self.frames(), LockParams.legacy())
        self.assertEqual(out[8][0], BYSTANDER)
        self.assertEqual(out[-1][0], BYSTANDER)       # still the bystander when the climber is back

    def test_gate_alone_drops_instead_of_locking_wrongly(self):
        out = run(self.frames(), LockParams(gate_g0=0.5, gate_v=0.02))
        self.assertTrue(all(o[0] is None for o in out[8:14]))
        self.assertEqual(out[-1][0], CLIMBER)

    def test_gate_plus_rescue_keeps_the_climber(self):
        out = run(self.frames(), LockParams(gate_g0=0.5, gate_v=0.02, conf_rescue=0.1))
        self.assertTrue(all(o[0] is not None and o[0].h == 790 for o in out[8:14]))

    def test_release_alone_does_not_fix_it(self):
        out = run(self.frames(), LockParams(release_n=5))
        self.assertEqual(out[8][0], BYSTANDER)


class RejectionTests(unittest.TestCase):
    P = LockParams(gate_g0=0.5, size_s=2.0, conf_rescue=0.1)

    def state_after(self, *frames, params=None):
        st = LockState()
        for boxes in frames:
            _, st, _ = select_climber_box(boxes, st, params or self.P)
        return st

    def test_none_without_a_lock_or_checks(self):
        self.assertIsNone(explain_rejection([BYSTANDER], LockState(), self.P))
        st = self.state_after([CLIMBER], params=LockParams.legacy())
        self.assertIsNone(explain_rejection([BYSTANDER], st, LockParams.legacy()))

    def test_far_small_box_fails_both_checks(self):
        r = explain_rejection([BYSTANDER], self.state_after([CLIMBER]), self.P)
        self.assertEqual(r.box, BYSTANDER)
        self.assertEqual(r.check_failed, "gate+size")
        self.assertAlmostEqual(r.dist_over_h_ref, (310 ** 2 + 660 ** 2) ** 0.5 / 800, places=6)   # centres (700,650) and (1010,1310)
        self.assertEqual(r.allowance_over_h_ref, 0.5)
        self.assertAlmostEqual(r.h_ratio, 0.25)

    def test_gate_only_and_size_only(self):
        st = self.state_after([CLIMBER])
        far_same_size = Box(580 + 700, 250, 240, 800, 0.9)
        self.assertEqual(explain_rejection([far_same_size], st, self.P).check_failed, "gate")
        near_small = Box(600, 400, 240, 300, 0.9)
        self.assertEqual(explain_rejection([near_small], st, self.P).check_failed, "size")

    def test_passing_candidate_is_not_reported(self):
        st = self.state_after([CLIMBER])
        self.assertIsNone(explain_rejection([Box(590, 255, 240, 790, 0.9)], st, self.P))

    def test_reports_the_box_legacy_would_choose_not_every_box(self):
        st = self.state_after([CLIMBER])
        near = Box(590, 255, 240, 790, 0.9)
        self.assertIsNone(explain_rejection([BYSTANDER, near], st, self.P))     # legacy picks `near`, it passes
        self.assertEqual(legacy_candidate([BYSTANDER, near], st, self.P), near)

    def test_allowance_grows_with_gap(self):
        p = LockParams(gate_g0=0.5, gate_v=0.1)
        st = self.state_after([CLIMBER], [], [], params=p)
        r = explain_rejection([BYSTANDER], st, p)
        self.assertAlmostEqual(r.allowance_over_h_ref, 0.7)

    def test_low_confidence_boxes_are_not_legacy_candidates(self):
        st = self.state_after([CLIMBER])
        self.assertIsNone(explain_rejection([Box(900, 900, 50, 100, 0.3)], st, self.P))


class MiscTests(unittest.TestCase):
    def test_parameter_validation(self):
        for kw in ({"conf_rescue": 0.6}, {"conf_rescue": 0.0}, {"gate_g0": 0.0}, {"gate_v": 0.1},
                   {"size_s": 1.0}, {"size_k": 0}, {"release_n": 0}):
            with self.assertRaises(ValueError, msg=kw):
                LockParams(**kw)

    def test_state_is_not_modified(self):
        st = LockState()
        _, st2, _ = select_climber_box([CLIMBER], st, LockParams.legacy())
        self.assertIsNone(st.center)
        self.assertIsNotNone(st2.center)

    def test_iou(self):
        a = Box(0, 0, 10, 10, 1.0)
        self.assertAlmostEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, Box(20, 20, 5, 5, 1.0)), 0.0)
        self.assertAlmostEqual(iou(a, Box(5, 0, 10, 10, 1.0)), 50 / 150)

    def test_label(self):
        self.assertIn("g0=off", LockParams.legacy().label())


if __name__ == "__main__":
    unittest.main()
