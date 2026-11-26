"""
Shared loading and windowing for step 5.1a (DECISIONS.md D-006 addendum, D-004 method).
Speed-profile check only: LDLJ and SPARC on windows of the CoM speed profile of real clips. NOT movement segments (those wait
for 4.1 and the hold-relative layer, D-018).

CoM input (author, 2026-11-26): working CoM E-dL at cut-off 0.5 (metrics.compute_com_working, male table) and the trunk-centroid
proxy A (metrics.compute_com), both on the D-030 gap-filled joints (occlusion_handling.fill_gaps re-run on
validation/output_full_lock1; checked against evaluation/3.2/output_full_lock1_d030/*_filled.csv by check_filled_equal).
E-dL gates limbs on RAW confidence (D-028 item 11), so filling does not change it; it only changes the trunk if a trunk joint
was filled (none was a gap in the lock1 run).
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import metrics  # noqa: E402
import occlusion_handling as oh  # noqa: E402

INPUT_DIR = REPO / "validation" / "output_full_lock1"
FILLED_DIR = REPO / "evaluation" / "3.2" / "output_full_lock1_d030"
EXCLUSIONS = REPO / "validation" / "frame_exclusions.csv"
CLIPS = ["IMG_2412", "IMG_2416", "IMG_2417", "IMG_2419", "IMG_2420"]
SEX = "male"
FPS = 30.0  # nominal; the real timestamps are used in every calculation (measured 29.95 to 30.00 per clip)


def load_exclusions() -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    with open(EXCLUSIONS, newline="") as f:
        rows = [r for r in f if not r.startswith("#")]
    for r in csv.DictReader(rows):
        clip = r["video"].split(".")[0]
        out.setdefault(clip, set()).update(range(int(r["start_frame"]), int(r["end_frame"]) + 1))
    return out


class Clip:
    """One clip: filled joints, both CoM series (positions as float arrays with NaN where None), torso scale, tier3 mask."""

    def __init__(self, name: str):
        self.name = name
        raw = metrics.load_keypoints_csv(INPUT_DIR / f"{name}_keypoints.csv")
        n = len(next(iter(raw.values())).positions)
        ex = {t for t in load_exclusions().get(name, set()) if 0 <= t < n}
        raw = metrics.apply_frame_exclusions(raw, ex)
        self.filled = oh.fill_gaps(raw, oh.OcclusionParams())
        self.joints = oh.as_joint_series(self.filled, raw)
        self.n = n
        self.times = np.asarray(next(iter(raw.values())).timestamps, dtype=float)
        self.com_edl = metrics.compute_com_working(self.joints, SEX)
        self.com_a = metrics.compute_com(self.joints)
        self.xy_edl = to_array(self.com_edl.positions)
        self.xy_a = to_array(self.com_a.positions)
        self.tier3 = np.zeros(n, dtype=bool)
        for fj in self.filled.values():
            for t, fl in enumerate(fj.flags):
                if fl == oh.Flag.TIER3:
                    self.tier3[t] = True
        ls, rs = self.joints["left_shoulder"].positions, self.joints["right_shoulder"].positions
        lh, rh = self.joints["left_hip"].positions, self.joints["right_hip"].positions
        tl = []
        for t in range(n):
            if None in (ls[t], rs[t], lh[t], rh[t]):
                continue
            sm = ((ls[t][0] + rs[t][0]) / 2, (ls[t][1] + rs[t][1]) / 2)
            hm = ((lh[t][0] + rh[t][0]) / 2, (lh[t][1] + rh[t][1]) / 2)
            tl.append(math.hypot(sm[0] - hm[0], sm[1] - hm[1]))
        self.torso = float(np.median(tl))  # clip median torso length, px

    def series(self, xy: np.ndarray) -> metrics.JointSeries:
        pos = [None if np.isnan(p[0]) else (float(p[0]), float(p[1])) for p in xy]
        return metrics.JointSeries("com", pos, list(self.times))

    def which(self, com: str) -> np.ndarray:
        return {"edl": self.xy_edl, "A": self.xy_a}[com]


def to_array(positions) -> np.ndarray:
    a = np.full((len(positions), 2), np.nan)
    for i, p in enumerate(positions):
        if p is not None:
            a[i] = p
    return a


def defined_runs(xy: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs [a, b] (inclusive) of frames with a defined position."""
    ok = ~np.isnan(xy[:, 0])
    runs, start = [], None
    for t, o in enumerate(ok):
        if o and start is None:
            start = t
        if not o and start is not None:
            runs.append((start, t - 1)); start = None
    if start is not None:
        runs.append((start, len(ok) - 1))
    return runs


def windows(clip: Clip, seconds: int) -> list[tuple[int, int]]:
    """Non-overlapping tiles of L = round(seconds * 30) frames from frame 0; a tile [s, s+L) is kept when the E-dL CoM is defined on
    every frame of it and no frame in it has any body joint flagged tier3. A rejected tile is dropped, never re-slid.
    The same tiles are used for E-dL, A, every filter and every noise level (A and E-dL share the trunk, so A is defined there too)."""
    L = int(round(seconds * FPS))
    out = []
    for s in range(0, clip.n - L + 1, L):
        sl = slice(s, s + L)
        if np.isnan(clip.xy_edl[sl, 0]).any() or np.isnan(clip.xy_a[sl, 0]).any() or clip.tier3[sl].any():
            continue
        out.append((s, s + L - 1))
    return out


def check_filled_equal(clip: Clip) -> tuple[int, int]:
    """Compare the re-run fill with the committed-run CSV (x_filled, y_filled, flag) -> (rows compared, rows different)."""
    path = FILLED_DIR / f"{clip.name}_filled.csv"
    if not path.exists():
        return (0, -1)
    n = bad = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            name, t = r["keypoint_name"], int(r["frame_idx"])
            if name not in clip.filled:
                continue
            fj = clip.filled[name]
            n += 1
            same_flag = fj.flags[t].value == r["flag"]
            p = fj.positions[t]
            if r["x_filled"] == "":
                same_pos = p is None
            else:
                same_pos = p is not None and abs(p[0] - float(r["x_filled"])) < 6e-5 and abs(p[1] - float(r["y_filled"])) < 6e-5
            if not (same_flag and same_pos):
                bad += 1
    return n, bad
