"""
Tests for LDLJ and SPARC in metrics.py (DECISIONS.md D-006): synthetic speed profiles with
hand-derived answers; nothing here is a measurement.

The SPARC parameters used below (amp_threshold 0.05, omega_c_max 20 pi) are the values Balasubramanian
et al. (2015) used for their simulated example (Additional file 3, Fig. S2); here they are test
fixtures, NOT project values (the project values are an open question, D-006 addendum).

Run from the repo root:  python3 -m unittest discover -s tests -t .
"""

import math
import unittest

import metrics
from metrics import (
    GAP_MISSING_DETECTION,
    JointSeries,
    _ldlj,
    _sparc_from_spectrum,
    compute_ldlj,
    SPARC_REFERENCE_DEFAULTS,
    compute_sparc,
    sparc_n_fft,
    speed_profile,
)

SPARC_FIXTURE = dict(amp_threshold=0.05, omega_c_max=20 * math.pi, n_fft=4096)
REF = SPARC_REFERENCE_DEFAULTS


def bump(n=401, duration=1.0, amplitude=1.0, repeats=1):
    """Minimum-jerk speed bump(s): v = A * 16 s^2 (1-s)^2 per bump (peak A), n samples."""
    t = [duration * repeats * i / (n - 1) for i in range(n)]
    v = []
    for x in t:
        u = (x / duration) % 1.0
        if x > 0 and u == 0.0:
            u = 1.0
        v.append(amplitude * 16 * u * u * (1 - u) ** 2)
    return t, v


def scaled(t, v, k, m):
    return [m * x for x in t], [k * x for x in v]


class LdljTests(unittest.TestCase):
    def test_minimum_jerk_bump_known_answer(self):
        # v = A*16 s^2 (1-s)^2: d2v/dt2 = (16A/T^2)(2 - 12 s + 12 s^2), int_0^1 (..)^2 ds = 0.8,
        # so (T^3/A^2) * (16A)^2 * 0.8 / T^3 = 204.8 and LDLJ = -ln 204.8, whatever A and T are.
        for k, m in ((1.0, 1.0), (7.5, 0.4)):
            t, v = scaled(*bump(n=2001), k, m)
            self.assertAlmostEqual(compute_ldlj(t, v), -math.log(204.8), delta=0.01)

    def test_dimensionless_amplitude_k_and_duration_m(self):
        t, v = bump()
        base = compute_ldlj(t, v)
        for k, m in ((3.7, 1.0), (1.0, 2.3), (3.7, 2.3), (0.01, 0.2), (120.0, 9.0)):
            self.assertAlmostEqual(compute_ldlj(*scaled(t, v, k, m)), base, places=9, msg=(k, m))

    def test_exponent_5_is_not_dimensionless(self):
        # The same test with the (t2 - t1)^5 printed in the paper fails: it changes by -2 ln m
        # (and does not depend on k), so 5 is dimensionally inconsistent for a speed profile.
        t, v = bump()
        base5 = _ldlj(t, v, 5)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(_ldlj(*scaled(t, v, 3.7, 2.3), 5), base5, places=9)
        for k, m in ((3.7, 1.0), (1.0, 2.3), (3.7, 2.3)):
            self.assertAlmostEqual(_ldlj(*scaled(t, v, k, m), 5) - base5, -2 * math.log(m), places=9)
        # the two exponents agree when the duration is exactly 1 s (1^5 = 1^3), which hides the error
        self.assertAlmostEqual(_ldlj(t, v, 5), _ldlj(t, v, 3), places=12)

    def test_more_submovements_is_less_smooth(self):
        one, two = compute_ldlj(*bump(n=401)), compute_ldlj(*bump(n=801, repeats=2))
        self.assertLess(two, one)

    def test_undefined_cases(self):
        self.assertIsNone(compute_ldlj([0, 1], [1, 2]))                 # too few samples
        self.assertIsNone(compute_ldlj([0, 1, 2], [0, 0, 0]))           # no positive speed
        self.assertIsNone(compute_ldlj([0, 1, 2, 3], [1, 1, 1, 1]))     # zero integral
        with self.assertRaises(ValueError):
            compute_ldlj([0, 1, 1, 2], [0, 1, 1, 0])                    # not increasing
        with self.assertRaises(ValueError):
            compute_ldlj([0, 1, 2], [0, 1])


class SpeedProfileTests(unittest.TestCase):
    def series(self, positions):
        return JointSeries("c", positions, [i / 30.0 for i in range(len(positions))])

    def test_window_speeds_and_gaps(self):
        s = self.series([(0.0, 0.0), (3.0, 4.0), (6.0, 8.0), None, (9.0, 12.0), (12.0, 16.0)])
        t, v = speed_profile(s, 1, 2)
        self.assertEqual(t, [1 / 30, 2 / 30])
        self.assertAlmostEqual(v[0], 5.0 * 30)
        self.assertIsNone(speed_profile(s, 1, 4))     # a gap inside the window is never bridged
        self.assertIsNone(speed_profile(s, 0, 2))     # speed at frame 0 is undefined
        self.assertIsNone(speed_profile(s, 2, 99))
        self.assertIsNotNone(speed_profile(s, 5, 5))

    def test_gap_reason_irrelevant_and_input_not_modified(self):
        s = JointSeries("c", [(0.0, 0.0), None, (1.0, 0.0)], [0, 1, 2], [None, GAP_MISSING_DETECTION, None])
        self.assertIsNone(speed_profile(s, 1, 2))
        self.assertEqual(s.positions[1], None)


class SparcSpectrumTests(unittest.TestCase):
    """The arc-length and adaptive-cut-off rule on hand-built spectra."""

    OMEGA = [0, 1, 2, 3, 4, 5]
    VHAT = [1.0, 0.5, 0.2, 0.04, 0.06, 0.01]

    def test_cutoff_is_the_last_point_at_or_above_the_threshold(self):
        # threshold 0.05: last Vhat >= 0.05 is index 4 (0.06) -> wc = 4; x = w / 4
        x = [0, 0.25, 0.5, 0.75, 1.0]
        y = [1.0, 0.5, 0.2, 0.04, 0.06]
        want = -sum(math.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(4))
        self.assertAlmostEqual(_sparc_from_spectrum(self.OMEGA, self.VHAT, 0.05, 100.0), want)

    def test_omega_c_max_caps_the_cutoff(self):
        # wc = 2.5 -> points w <= 2.5 are 0, 1, 2; x = w / 2.5
        x, y = [0, 0.4, 0.8], [1.0, 0.5, 0.2]
        want = -sum(math.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(2))
        self.assertAlmostEqual(_sparc_from_spectrum(self.OMEGA, self.VHAT, 0.05, 2.5), want)

    def test_flat_spectrum_arc_length_is_minus_one(self):
        # Vhat = 1 up to wc: the curve is the straight line x in [0, 1]: arc length 1
        self.assertAlmostEqual(_sparc_from_spectrum([0, 1, 2, 3], [1, 1, 1, 1], 0.05, 10), -1.0)

    def test_undefined(self):
        self.assertIsNone(_sparc_from_spectrum(self.OMEGA, self.VHAT, 1.5, 10))   # nothing >= threshold
        self.assertIsNone(_sparc_from_spectrum(self.OMEGA, self.VHAT, 0.7, 10))   # wc = 0
        self.assertIsNone(_sparc_from_spectrum(self.OMEGA, [1, 0, 0, 0, 0, 0], 0.5, 10))  # one point


class SparcTests(unittest.TestCase):
    def test_parameters_are_required_keywords(self):
        t, v = bump()
        with self.assertRaises(TypeError):
            compute_sparc(t, v)
        with self.assertRaises(TypeError):
            compute_sparc(t, v, amp_threshold=0.05, omega_c_max=62.8)   # n_fft missing
        with self.assertRaises(TypeError):
            compute_sparc(t, v, 0.05, 62.8, 4096)                        # not positional

    def test_dimensionless_amplitude_k_and_duration_m(self):
        t, v = bump()
        base = compute_sparc(t, v, **SPARC_FIXTURE)
        self.assertIsNotNone(base)
        self.assertLess(base, 0)
        self.assertAlmostEqual(compute_sparc(*scaled(t, v, 7.3, 1.0), **SPARC_FIXTURE), base, places=9)
        # duration m (same samples, fs scales by 1/m, same n_fft): the whole frequency axis scales
        # and so does the adaptive cut-off, which stays below omega_c_max here, so SPARC is unchanged
        for m in (1.5, 2.0, 3.0):
            self.assertAlmostEqual(compute_sparc(*scaled(t, v, 2.0, m), **SPARC_FIXTURE), base, places=9)

    def test_more_submovements_is_less_smooth(self):
        one = compute_sparc(*bump(n=401), **SPARC_FIXTURE)
        two = compute_sparc(*bump(n=801, repeats=2), **SPARC_FIXTURE)
        self.assertLess(two, one)

    def test_undefined_and_invalid(self):
        t = [i / 30 for i in range(10)]
        self.assertIsNone(compute_sparc(t, [0.0] * 10, **SPARC_FIXTURE))
        self.assertIsNone(compute_sparc([0.0], [1.0], **SPARC_FIXTURE))
        with self.assertRaises(ValueError):
            compute_sparc(t, [1.0] * 10, amp_threshold=0.0, omega_c_max=1.0, n_fft=64)
        with self.assertRaises(ValueError):
            compute_sparc(t, [1.0] * 10, amp_threshold=0.05, omega_c_max=-1.0, n_fft=64)
        with self.assertRaises(ValueError):
            compute_sparc(t, [1.0] * 10, amp_threshold=0.05, omega_c_max=1.0, n_fft=4)


class SparcReferenceTests(unittest.TestCase):
    """Regression against the authors' reference code (github.com/siva82kb/SPARC, scripts/smoothness.py,
    commit 3650934, as stated by the author): its docstring example is a Gaussian speed profile
    exp(-5 t^2), t = -1 .. 1 in steps of 0.01 (np.arange(-1, 1, 0.01): 200 samples, last t = 0.99),
    fs = 100, giving -1.41403."""

    def test_constants_and_nfft(self):
        self.assertEqual(REF["amp_threshold"], 0.05)
        self.assertAlmostEqual(REF["omega_c_max"], 20 * math.pi)       # 10 Hz
        self.assertEqual(REF["padlevel"], 4)
        self.assertEqual(sparc_n_fft(200), 4096)                        # 2^(8 + 4)
        self.assertEqual(sparc_n_fft(201), 4096)
        self.assertEqual(sparc_n_fft(256), 2 ** 12)
        self.assertEqual(sparc_n_fft(257), 2 ** 13)

    def gaussian(self, n):
        t = [-1.0 + 0.01 * i for i in range(n)]
        return t, [math.exp(-5 * x * x) for x in t]

    def test_gaussian_docstring_example(self):
        t, v = self.gaussian(200)
        got = compute_sparc(t, v, amp_threshold=REF["amp_threshold"], omega_c_max=REF["omega_c_max"],
                            n_fft=sparc_n_fft(len(v), REF["padlevel"]))
        self.assertAlmostEqual(got, -1.41403, places=5)

    def test_endpoint_sample_changes_the_fifth_decimal(self):
        # With t = 1.00 included (201 samples) the same discretisation gives -1.41399, not -1.41403:
        # the docstring value corresponds to 200 samples. Recorded so the difference is not lost.
        t, v = self.gaussian(201)
        got = compute_sparc(t, v, amp_threshold=0.05, omega_c_max=20 * math.pi, n_fft=sparc_n_fft(201))
        self.assertAlmostEqual(got, -1.41399, places=5)
        self.assertNotAlmostEqual(got, -1.41403, places=5)


if __name__ == "__main__":
    unittest.main()
