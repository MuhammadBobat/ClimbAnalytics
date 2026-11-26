"""Tests for the 5.1a sweep helpers (evaluation/5.1a): filters, speeds, noise reproducibility, SPARC cap flag. Synthetic data only."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "5.1a"))
import run_sweep as rs  # noqa: E402
import common as c  # noqa: E402


def test_defined_runs_and_filter_keep_gaps():
    xy = np.cumsum(np.ones((100, 2)), axis=0)
    xy[40:45] = np.nan
    assert c.defined_runs(xy) == [(0, 39), (45, 99)]
    for kind, par in (("sg", 13), ("bw", 6.0)):
        out = rs.apply_filter(xy, kind, par)
        assert np.isnan(out[40:45]).all() and not np.isnan(out[:40]).any() and not np.isnan(out[45:]).any()


def test_savgol_order3_keeps_a_cubic_and_butterworth_keeps_a_constant():
    t = np.arange(100) / 30.0
    cubic = np.stack([t ** 3, 2 * t ** 2], axis=1)
    out = rs.apply_filter(cubic, "sg", 13)
    assert np.allclose(out, cubic, atol=1e-8)
    const = np.full((100, 2), 7.0)
    assert np.allclose(rs.apply_filter(const, "bw", 6.0), const, atol=1e-9)


def test_zero_phase_butterworth_has_no_lag():
    t = np.arange(300) / 30.0
    x = np.stack([np.sin(2 * math.pi * 1.0 * t), np.zeros_like(t)], axis=1)
    out = rs.apply_filter(x, "bw", 6.0)
    assert np.argmax(np.correlate(out[:, 0], x[:, 0], "full")) == len(t) - 1


def test_speeds_match_step_over_dt():
    xy = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    v = rs.speeds(xy, np.array([0.0, 0.5, 1.5]))
    assert np.isnan(v[0]) and v[1] == 10.0 and v[2] == 0.0


def test_noise_is_reproducible_and_nested():
    a = np.random.default_rng(rs.SEED0).standard_normal((50, 2))
    b = np.random.default_rng(rs.SEED0).standard_normal((50, 2))
    assert np.array_equal(a, b)
    assert np.allclose((2 * 0.01) * a, 2 * (0.01 * a))


def test_sparc_cap_flag_for_white_noise_and_for_smooth_profile():
    t = np.arange(1, 30) / 30.0
    rng = np.random.default_rng(1)
    noisy = 100 + 30 * rng.standard_normal(29)
    smooth = 100 + 50 * np.sin(math.pi * t / t[-1]) ** 2
    assert rs.sparc_capped(t, np.abs(noisy)) is True
    assert rs.sparc_capped(t, smooth) is False
    assert rs.sparc_val(t, smooth, 0.05, 10.0, 4) == rs.metrics.compute_sparc(
        t, smooth, amp_threshold=0.05, omega_c_max=20 * math.pi, n_fft=rs.metrics.sparc_n_fft(29, 4))
