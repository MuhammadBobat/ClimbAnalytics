"""
Tests for occlusion_handling.py (ARCHITECTURE.md §3; DECISIONS.md D-007, D-028 proposed). Synthetic skeletons with
known answers. Nothing here validates a tier: the tests record what the code does on constructed cases; the tier
boundaries and methods are unvalidated heuristics (hide-and-fill test, step 3.3).

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import math
import unittest

import metrics
from metrics import GAP_EXCLUDED, GAP_LOW_CONFIDENCE, GAP_MISSING_DETECTION, JointSeries
import occlusion_handling as oh
from occlusion_handling import (
    Flag, OcclusionParams, as_joint_series, classify_gap, clamp_to_segment, estimate_noise, fill_gaps, find_gaps,
    segment_bounds, summarise)

FPS = 30.0
N = 120
CU = {"left_elbow": (0.15, 0.6), "right_elbow": (0.15, -0.6), "left_wrist": (0.3, 0.9), "right_wrist": (0.3, -0.9),
      "left_knee": (-0.6, 0.3), "right_knee": (-0.6, -0.3), "left_ankle": (-1.2, 0.3), "right_ankle": (-1.2, -0.3)}
SWAP = {"left": "right", "right": "left"}


def swap_name(n):
    s, rest = n.split("_", 1)
    return f"{SWAP[s]}_{rest}"


def trunk_frame(t, k=1.0, dx=0.0, dy=0.0):
    """Moving, rotating, scaling trunk: hip midpoint H, unit u (hip -> shoulder midpoint), w, torso length L."""
    th = 0.3 * math.sin(0.05 * t)
    L = k * (100.0 + 10.0 * math.sin(0.07 * t))
    H = (k * (400.0 + 2.0 * t) + dx, k * (800.0 - 1.5 * t) + dy)
    u = (math.sin(th), -math.cos(th))          # y down: shoulders are above the hips
    w = (-u[1], u[0])
    return H, u, w, L


def rigid_skeleton(n=N, **kw):
    """Every body joint follows the trunk rigidly (fixed offset in torso lengths in the trunk-aligned frame)."""
    pos = {j: [] for j in oh.BODY_JOINTS}
    for t in range(n):
        H, u, w, L = trunk_frame(t, **kw)
        S = (H[0] + L * u[0], H[1] + L * u[1])
        pos["left_shoulder"].append((S[0] + 0.45 * L * w[0], S[1] + 0.45 * L * w[1]))
        pos["right_shoulder"].append((S[0] - 0.45 * L * w[0], S[1] - 0.45 * L * w[1]))
        pos["left_hip"].append((H[0] + 0.3 * L * w[0], H[1] + 0.3 * L * w[1]))
        pos["right_hip"].append((H[0] - 0.3 * L * w[0], H[1] - 0.3 * L * w[1]))
        for j, (cu, cw) in CU.items():
            O = S if j in oh.ARM_JOINTS else H
            pos[j].append((O[0] + L * (cu * u[0] + cw * w[0]), O[1] + L * (cu * u[1] + cw * w[1])))
    return pos


def to_joints(pos, gaps=None, reasons=None, confidences=True):
    """JointSeries per joint; gaps {joint: [(first, last), ...]} become None positions."""
    gaps = gaps or {}
    out = {}
    for j, p in pos.items():
        p = list(p)
        rs = [None] * len(p)
        for first, last in gaps.get(j, []):
            for t in range(first, last + 1):
                p[t] = None
                rs[t] = (reasons or {}).get(j, GAP_LOW_CONFIDENCE)
        out[j] = JointSeries(j, p, [i / FPS for i in range(len(p))], rs,
                             [0.9 if q is not None else 0.2 for q in p] if confidences else None)
    return out


def maxerr(a, b, frames):
    return max(math.hypot(a[t][0] - b[t][0], a[t][1] - b[t][1]) for t in frames)


class BasicsTests(unittest.TestCase):
    def test_boundaries_follow_d007(self):
        got = [classify_gap(n) for n in (1, 4, 5, 20, 21, 80)]
        self.assertEqual(got, [Flag.TIER1, Flag.TIER1, Flag.TIER2, Flag.TIER2, Flag.TIER3, Flag.TIER3])
        self.assertEqual((oh.TIER1_MAX_FRAMES, oh.TIER2_MAX_FRAMES), (4, 20))
        with self.assertRaises(ValueError):
            classify_gap(0)

    def test_find_gaps_causes_edges_and_excluded_walls(self):
        j = to_joints(rigid_skeleton(30), {"left_wrist": [(0, 1), (5, 5), (28, 29)]},
                      {"left_wrist": GAP_MISSING_DETECTION})["left_wrist"]
        self.assertEqual(find_gaps(j), [(0, 1), (5, 5), (28, 29)])
        j2 = metrics.apply_frame_exclusions({"left_wrist": j}, {10, 11})["left_wrist"]
        self.assertEqual(find_gaps(j2), [(0, 1), (5, 5), (28, 29)])        # excluded frames are not gaps
        j3 = metrics.apply_frame_exclusions({"left_wrist": to_joints(rigid_skeleton(30), {"left_wrist": [(5, 8)]})["left_wrist"]},
                                            {9, 10})["left_wrist"]
        self.assertEqual(find_gaps(j3), [(5, 8)])

    def test_input_is_never_modified_and_observed_frames_unchanged(self):
        joints = to_joints(rigid_skeleton(), {"left_wrist": [(10, 12)], "right_ankle": [(30, 40)]})
        before = {n: (list(s.positions), list(s.gap_reasons), list(s.confidences)) for n, s in joints.items()}
        filled = fill_gaps(joints)
        for n, s in joints.items():
            self.assertEqual((s.positions, s.gap_reasons, s.confidences), before[n])
            self.assertIs(filled[n].raw, s)
        f = filled["left_wrist"]
        for t, fl in enumerate(f.flags):
            if fl == Flag.OBSERVED:
                self.assertEqual(f.positions[t], joints["left_wrist"].positions[t])
        self.assertEqual(f.flags[10:13], [Flag.TIER1] * 3)

    def test_unequal_lengths_rejected(self):
        joints = to_joints(rigid_skeleton(30))
        joints["left_wrist"] = to_joints(rigid_skeleton(31))["left_wrist"]
        with self.assertRaises(ValueError):
            fill_gaps(joints)


class Tier1Tests(unittest.TestCase):
    def series_from(self, fx, fy, gap, n=40):
        pos = [(fx(i / FPS), fy(i / FPS)) for i in range(n)]
        pos_gap = list(pos)
        for t in range(gap[0], gap[1] + 1):
            pos_gap[t] = None
        s = JointSeries("left_wrist", pos_gap, [i / FPS for i in range(n)], confidences=[0.9] * n)
        return pos, s

    def test_cubic_polynomial_and_line_are_reproduced_exactly(self):
        pos, s = self.series_from(lambda t: 100 + 50 * t - 30 * t * t + 20 * t ** 3, lambda t: 5 - 7 * t, (15, 17))
        out = oh.fill_tier1(s, 15, 17, OcclusionParams())
        self.assertLess(maxerr(out, pos[15:18], range(3)), 1e-8)

    def test_arc_is_better_than_linear_interpolation(self):
        pos, s = self.series_from(lambda t: 200 * math.cos(2.0 * t), lambda t: 200 * math.sin(2.0 * t), (15, 18))
        spline = oh.fill_tier1(s, 15, 18, OcclusionParams())
        lin = [tuple(pos[14][k] + (pos[19][k] - pos[14][k]) * (i + 1) / 5 for k in (0, 1)) for i in range(4)]
        self.assertLess(maxerr(spline, pos[15:19], range(4)), 0.2 * maxerr(lin, pos[15:19], range(4)))

    def test_not_enough_observed_frames_is_unavailable(self):
        joints = to_joints(rigid_skeleton(40), {"left_wrist": [(0, 2)]})
        f = fill_gaps(joints)["left_wrist"]
        self.assertEqual(f.flags[:3], [Flag.UNAVAILABLE] * 3)
        self.assertTrue(f.events[0].touches_start)
        self.assertFalse(f.events[0].filled)
        f = fill_gaps(to_joints(rigid_skeleton(40), {"left_wrist": [(1, 2)]}))["left_wrist"]   # one frame on the left
        self.assertEqual(f.flags[1:3], [Flag.UNAVAILABLE] * 2)
        f = fill_gaps(to_joints(rigid_skeleton(40), {"left_wrist": [(37, 39)]}))["left_wrist"]  # clip end
        self.assertTrue(f.events[0].touches_end and not f.events[0].filled)

    def test_two_gaps_separated_by_one_observed_frame_are_filled_separately(self):
        pos = rigid_skeleton(60)
        joints = to_joints(pos, {"right_wrist": [(20, 22), (24, 25)]})
        f = fill_gaps(joints)["right_wrist"]
        self.assertEqual([e.length for e in f.events], [3, 2])
        self.assertEqual(f.flags[23], Flag.OBSERVED)
        self.assertTrue(all(fl == Flag.TIER1 for fl in f.flags[20:23] + f.flags[24:26]))

    def test_excluded_frame_is_a_wall(self):
        joints = to_joints(rigid_skeleton(60), {"right_wrist": [(20, 22)]})
        joints = metrics.apply_frame_exclusions(joints, {23, 24})
        f = fill_gaps(joints)["right_wrist"]
        self.assertEqual(f.flags[20:23], [Flag.UNAVAILABLE] * 3)       # no observed frame on the right before the wall
        self.assertTrue(f.events[0].adjacent_excluded)
        self.assertEqual(f.flags[23], Flag.UNAVAILABLE)                # excluded frames are never filled


class Tier2Tests(unittest.TestCase):
    def wave(self, n=80):
        return [(300 + 4.0 * t + 40 * math.sin(0.2 * t), 500 + 60 * math.cos(0.15 * t)) for t in range(n)]

    def test_line_is_reproduced_by_the_filter(self):
        pos = rigid_skeleton(60)
        pos["left_wrist"] = [(100 + 3.0 * t, 200 - 2.0 * t + (1.0 if t % 2 else 0.0) * 0.0) for t in range(60)]
        # add a deterministic wobble elsewhere on the clip so the noise estimate is not degenerate
        wob = [(100 + 3.0 * t + (0.3 if t % 2 else -0.3), 200 - 2.0 * t + (0.2 if t % 3 else -0.2)) for t in range(60)]
        pos["left_wrist"] = wob
        joints = to_joints(pos, {"left_wrist": [(25, 34)]})
        f = fill_gaps(joints)["left_wrist"]
        self.assertEqual(f.flags[25:35], [Flag.TIER2] * 10)
        self.assertEqual(set(f.methods[25:35]), {"kalman"})
        self.assertLess(maxerr(f.positions, [(100 + 3.0 * t, 200 - 2.0 * t) for t in range(60)], range(25, 35)), 2.5)

    def test_smooth_path_error_and_causal_option(self):
        w = self.wave()
        pos = rigid_skeleton(80)
        pos["right_wrist"] = w
        joints = to_joints(pos, {"right_wrist": [(30, 39)]})
        sm = fill_gaps(joints)["right_wrist"]
        ca = fill_gaps(joints, OcclusionParams(tier2_causal=True))["right_wrist"]
        e_sm, e_ca = maxerr(sm.positions, w, range(30, 40)), maxerr(ca.positions, w, range(30, 40))
        self.assertLess(e_sm, 12.0)
        self.assertLess(e_sm, e_ca)                                    # the backward pass uses the frames after the gap
        self.assertEqual(set(ca.methods[30:40]), {"kalman_causal"})

    def test_uncertainty_is_largest_in_the_middle(self):
        w = self.wave()
        pos = rigid_skeleton(80)
        pos["right_wrist"] = w
        f = fill_gaps(to_joints(pos, {"right_wrist": [(30, 41)]}))["right_wrist"]
        sig = f.sigma_px[30:42]
        self.assertTrue(all(s is not None for s in sig))
        self.assertGreater(max(sig), 2 * min(sig[0], sig[-1]))
        self.assertGreater(sig[len(sig) // 2], sig[0])
        self.assertIsNone(f.sigma_px[10])

    def test_clamp_keeps_the_segment_inside_the_observed_range(self):
        n = 60
        pos = rigid_skeleton(n)
        pos["right_elbow"] = [(300.0, 300.0)] * n
        # the forearm length grows linearly up to the gap and is 95 px after it: observed range [90, ~99.7]; a causal
        # constant-velocity fill keeps growing past the observed maximum
        d = [90.0 + t / 3.0 if t < 30 else 95.0 for t in range(n)]
        pos["right_wrist"] = [(300.0 + d[t], 300.0) for t in range(n)]
        joints = to_joints(pos, {"right_wrist": [(30, 39)]})
        lo, hi = segment_bounds(joints)["right_wrist"]
        for causal in (True, False):
            f = fill_gaps(joints, OcclusionParams(tier2_causal=causal))["right_wrist"]
            lens = [math.hypot(f.positions[t][0] - 300.0, f.positions[t][1] - 300.0) for t in range(30, 40)]
            self.assertTrue(all(lo - 1e-9 <= v <= hi + 1e-9 for v in lens), (causal, lens, lo, hi))
        raw = fill_gaps(joints, OcclusionParams(tier2_causal=True, tier2_clamp=False))["right_wrist"]
        lens = [math.hypot(raw.positions[t][0] - 300.0, raw.positions[t][1] - 300.0) for t in range(30, 40)]
        self.assertTrue(any(v > hi + 1.0 or v < lo - 1.0 for v in lens), "the unclamped causal fill should leave the range")

    def test_clamp_helper(self):
        self.assertEqual(clamp_to_segment((50.0, 0.0), (0.0, 0.0), (10.0, 20.0)), (20.0, 0.0))
        self.assertEqual(clamp_to_segment((5.0, 0.0), (0.0, 0.0), (10.0, 20.0)), (10.0, 0.0))
        self.assertEqual(clamp_to_segment((15.0, 0.0), (0.0, 0.0), (10.0, 20.0)), (15.0, 0.0))

    def test_noise_estimate(self):
        s = JointSeries("j", [(0.0, 0.0)] * 40, [i / FPS for i in range(40)])
        sd2, sa, sm = estimate_noise(s, OcclusionParams())
        self.assertLess(sd2, 1e-6)                                      # a constant series: floor only
        few = JointSeries("j", [(0.0, 0.0), None, (1.0, 1.0), None, (2.0, 2.0)], [i / FPS for i in range(5)])
        self.assertIsNone(estimate_noise(few, OcclusionParams()))

    def test_mirror_is_exact_for_an_affine_trunk_and_falls_back_to_kalman(self):
        pos = rigid_skeleton(80)
        joints = to_joints(pos, {"left_hip": [(30, 37)]})
        f = fill_gaps(joints)["left_hip"]
        self.assertEqual(f.flags[30:38], [Flag.TIER2] * 8)
        self.assertEqual(set(f.methods[30:38]), {"mirror"})
        self.assertLess(maxerr(f.positions, pos["left_hip"], range(30, 38)), 1e-8)
        f2 = fill_gaps(joints, OcclusionParams(tier2_mirror=False))["left_hip"]
        self.assertEqual(set(f2.methods[30:38]), {"kalman"})
        # another trunk joint lost inside the same frames: no mirror
        j3 = to_joints(pos, {"left_hip": [(30, 37)], "right_shoulder": [(34, 35)]})
        self.assertEqual(set(fill_gaps(j3)["left_hip"].methods[30:38]), {"kalman"})


class Tier3Tests(unittest.TestCase):
    def test_limb_follows_the_trunk_exactly_in_the_stale_model(self):
        pos = rigid_skeleton()
        joints = to_joints(pos, {"left_wrist": [(30, 60)], "right_ankle": [(40, 66)]})
        filled = fill_gaps(joints)
        for name, (a, b) in (("left_wrist", (30, 60)), ("right_ankle", (40, 66))):
            f = filled[name]
            self.assertEqual(f.flags[a:b + 1], [Flag.TIER3] * (b - a + 1))
            self.assertEqual(set(f.methods[a:b + 1]), {"trunk_offset"})
            self.assertLess(maxerr(f.positions, pos[name], range(a, b + 1)), 1e-8)

    def test_start_of_clip_uses_the_next_observed_offset(self):
        pos = rigid_skeleton()
        f = fill_gaps(to_joints(pos, {"left_knee": [(0, 24)]}))["left_knee"]
        self.assertEqual(f.flags[:25], [Flag.TIER3] * 25)
        self.assertLess(maxerr(f.positions, pos["left_knee"], range(25)), 1e-8)

    def test_never_observed_joint_is_unavailable(self):
        pos = rigid_skeleton(40)
        f = fill_gaps(to_joints(pos, {"left_wrist": [(0, 39)]}))["left_wrist"]
        self.assertEqual(set(f.flags), {Flag.UNAVAILABLE})
        self.assertIn("reference", f.events[0].note)

    def test_trunk_gap_over_20_frames_is_unavailable_and_dependants_follow(self):
        pos = rigid_skeleton()
        joints = to_joints(pos, {"left_hip": [(10, 40)], "left_wrist": [(15, 45)]})
        filled = fill_gaps(joints)
        hip = filled["left_hip"]
        self.assertEqual(hip.flags[10:41], [Flag.UNAVAILABLE] * 31)
        self.assertIn("root joint", hip.events[0].note)
        wrist = filled["left_wrist"]
        self.assertEqual(wrist.flags[15:41], [Flag.UNAVAILABLE] * 26)       # the trunk frame needs all four trunk joints
        self.assertEqual(wrist.flags[41:46], [Flag.TIER3] * 5)
        self.assertLess(maxerr(wrist.positions, pos["left_wrist"], range(41, 46)), 1e-8)

    def test_trunk_gap_up_to_20_frames_is_filled_first_so_the_limb_still_works(self):
        pos = rigid_skeleton()
        filled = fill_gaps(to_joints(pos, {"left_hip": [(30, 45)], "left_wrist": [(25, 60)]}))
        self.assertEqual(filled["left_hip"].flags[30:46], [Flag.TIER2] * 16)
        w = filled["left_wrist"]
        self.assertEqual(w.flags[25:61], [Flag.TIER3] * 36)
        self.assertLess(maxerr(w.positions, pos["left_wrist"], range(25, 61)), 1e-6)

    def test_face_points_are_not_handled(self):
        joints = to_joints(rigid_skeleton(40))
        joints["nose"] = JointSeries("nose", [None] * 40, [i / FPS for i in range(40)])
        self.assertNotIn("nose", fill_gaps(joints))


class AdapterAndInvarianceTests(unittest.TestCase):
    def test_as_joint_series_defaults_exclude_tier3_and_keep_confidences(self):
        pos = rigid_skeleton()
        joints = to_joints(pos, {"left_wrist": [(30, 60)], "right_wrist": [(10, 12)]})
        joints["nose"] = JointSeries("nose", [(1.0, 1.0)] * N, [i / FPS for i in range(N)])
        filled = fill_gaps(joints)
        js = as_joint_series(filled, raw_joints={"nose": joints["nose"]})
        self.assertIsNone(js["left_wrist"].positions[45])
        self.assertEqual(js["left_wrist"].gap_reasons[45], GAP_LOW_CONFIDENCE)
        self.assertIsNotNone(js["right_wrist"].positions[11])
        self.assertIsNone(js["right_wrist"].gap_reasons[11])
        self.assertEqual(js["right_wrist"].confidences[11], 0.2)             # raw confidence is carried, not invented
        self.assertIs(js["nose"], joints["nose"])
        t3 = as_joint_series(filled, include_tier3=True)
        self.assertIsNotNone(t3["left_wrist"].positions[45])
        # the adapter feeds metrics.py
        self.assertIsNotNone(metrics.compute_velocity(js["right_wrist"])[11])

    def test_summarise_counts_every_frame_once(self):
        filled = fill_gaps(to_joints(rigid_skeleton(), {"left_wrist": [(30, 60)], "right_wrist": [(10, 12)]}))
        s = summarise(filled)
        self.assertEqual(sum(s.values()), 12 * N)
        self.assertEqual((s["tier1"], s["tier3"]), (3, 31))

    def _scenario(self, **kw):
        pos = rigid_skeleton(**kw)
        w = [(300 + 4.0 * t + 40 * math.sin(0.2 * t), 500 + 60 * math.cos(0.15 * t)) for t in range(N)]
        pos["right_wrist"] = [(x * kw.get("k", 1.0) + kw.get("dx", 0.0), y * kw.get("k", 1.0) + kw.get("dy", 0.0)) for x, y in w]
        gaps = {"right_wrist": [(20, 22), (50, 58)], "left_ankle": [(70, 95)], "left_hip": [(30, 37)],
                "left_elbow": [(60, 66)]}
        return to_joints(pos, gaps)

    def test_translation_and_uniform_scale(self):
        base = fill_gaps(self._scenario())
        moved = fill_gaps(self._scenario(dx=57.0, dy=-31.0))
        scaled = fill_gaps(self._scenario(k=2.5))
        for name, f in base.items():
            self.assertEqual(f.flags, moved[name].flags)
            self.assertEqual(f.flags, scaled[name].flags)
            for t, p in enumerate(f.positions):
                if p is None:
                    continue
                self.assertAlmostEqual(moved[name].positions[t][0], p[0] + 57.0, delta=1e-5)
                self.assertAlmostEqual(moved[name].positions[t][1], p[1] - 31.0, delta=1e-5)
                self.assertAlmostEqual(scaled[name].positions[t][0], 2.5 * p[0], delta=1e-4)
                self.assertAlmostEqual(scaled[name].positions[t][1], 2.5 * p[1], delta=1e-4)

    def test_left_right_swap_and_determinism(self):
        joints = self._scenario()
        swapped = {swap_name(n): JointSeries(swap_name(n), s.positions, s.timestamps, s.gap_reasons, s.confidences)
                   for n, s in joints.items()}
        a, b = fill_gaps(joints), fill_gaps(swapped)
        for name, f in a.items():
            g = b[swap_name(name)]
            self.assertEqual(f.flags, g.flags, name)
            for p, q in zip(f.positions, g.positions):
                self.assertEqual(p is None, q is None)
                if p is not None:
                    self.assertAlmostEqual(p[0], q[0], delta=1e-6)
                    self.assertAlmostEqual(p[1], q[1], delta=1e-6)
        c = fill_gaps(joints)
        for name, f in a.items():
            self.assertEqual(f.positions, c[name].positions)


if __name__ == "__main__":
    unittest.main()
