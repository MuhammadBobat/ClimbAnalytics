"""Tests for the filled-input mode of evaluation/2.2c (D-026 addendum 2026-11-26) and the blind review selection. Synthetic data."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation" / "2.2c"))
import compare_com_candidates as cc  # noqa: E402
import random_review as rr  # noqa: E402

COLS = ["video", "frame_idx", "timestamp_s", "keypoint_name", "x_raw", "y_raw", "confidence", "x_filled", "y_filled", "flag", "method", "sigma_px"]


def write_filled(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLS)
        w.writerows(rows)


def test_load_filled_csv_uses_observed_and_tier2_never_tier3(tmp_path):
    rows = [
        ["c.MOV", 0, "0.000", "left_wrist", "1", "2", "0.9", "1.0000", "2.0000", "observed", "", ""],
        ["c.MOV", 1, "0.033", "left_wrist", "1", "2", "0.2", "1.5000", "2.5000", "tier2", "kalman", "0.5"],
        ["c.MOV", 2, "0.067", "left_wrist", "1", "2", "0.2", "", "", "tier3", "", ""],
        ["c.MOV", 3, "0.100", "left_wrist", "", "", "", "", "", "unavailable", "", ""],
    ]
    p = tmp_path / "c_filled.csv"
    write_filled(p, rows)
    j = cc.load_filled_csv(p)["left_wrist"]
    assert j.positions == [(1.0, 2.0), (1.5, 2.5), None, None]
    assert j.confidences == [0.9, 0.2, 0.2, None]          # RAW confidence kept
    assert j.gap_reasons[2] == cc.GAP_LOW_CONFIDENCE and j.gap_reasons[3] == cc.GAP_MISSING_DETECTION
    assert cc.FLAGS["c.MOV"]["left_wrist"] == ["observed", "tier2", "tier3", "unavailable"]


def test_position_gated_sets_confidence_only_where_a_position_exists():
    js = cc.JointSeries("x", [(0, 0), None, (1, 1)], [0, 1, 2], None, [0.3, 0.2, 0.9])
    g = cc.position_gated({"x": js})["x"]
    assert g.confidences == [1.0, 0.2, 1.0] and js.confidences == [0.3, 0.2, 0.9]


def test_extra_candidate_is_only_added_on_request():
    base = list(cc.LABELS)
    assert cc.L_EDLFILL not in base or cc.L_EDLFILL in cc.EXTRA_LABELS
    cc.enable_filled_candidate()
    cc.enable_filled_candidate()
    assert cc.LABELS.count(cc.L_EDLFILL) == 1
    cc.LABELS.remove(cc.L_EDLFILL)
    cc.EXTRA_LABELS.remove(cc.L_EDLFILL)


COUNTS = {"IMG_2412": 718, "IMG_2416": 917, "IMG_2417": 980, "IMG_2419": 850, "IMG_2420": 2421}


def test_random_windows_are_deterministic_in_bounds_and_non_overlapping():
    a, b = rr.draw_random_windows(COUNTS), rr.draw_random_windows(COUNTS)
    assert a == b and len(a) == 20
    for clip in rr.CLIPS:
        w = [x for x in a if x[0] == clip]
        assert len(w) == 4
        for _, s, e in w:
            assert 0 <= s and e == s + rr.WINDOW_FRAMES - 1 <= COUNTS[clip] - 1
        starts = sorted(s for _, s, _ in w)
        assert all(y - x >= rr.WINDOW_FRAMES for x, y in zip(starts, starts[1:]))


def test_key_is_a_permutation_per_window_and_selection_is_reproducible():
    w1, k1 = rr.build_selection(COUNTS)
    w2, k2 = rr.build_selection(COUNTS)
    assert (w1, k1) == (w2, k2) and len(w1) == 30 == len(k1)
    assert all(sorted(r[1:]) == sorted(rr.CANDIDATES) for r in k1)
    assert [w[1] for w in w1][:20] == [f"R{i:02d}" for i in range(1, 21)] and [w[1] for w in w1][20] == "S01"
