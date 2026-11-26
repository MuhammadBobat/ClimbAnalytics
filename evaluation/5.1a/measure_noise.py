"""
Step 5.1a, part 0 (run BEFORE the noise sweep): how noisy is each clip's own CoM, so that the synthetic noise levels can be checked
against it. Output: evaluation/5.1a/noise_estimates.csv (+ printed table).

Method (the assistant's, fixed before the sweep, no literature source): take the 1 s windows of common.windows whose peak CoM speed
(E-dL, filled, unfiltered, torso lengths per second) is in the LOWER TERCILE of its clip (slow stretches, where the true path is
nearly smooth). Per window and axis, fit a cubic polynomial in time and take the residuals. Two estimates per clip, in px and in
torso lengths (clip median torso length):
  resid_sd     sqrt(sum of squared residuals / (N - 4)) pooled over the clip's slow windows and both axes
  d2_sigma     robust second-difference estimate as in D-028 item 5: MAD-scaled (1.4826) sd of x(t+1) - 2x(t) + x(t-1) on the same
               windows, pooled over x and y, divided by sqrt(6) (white-noise assumption; the true path's acceleration inflates it)
Pose-model keypoint error is not white (neighbouring frames are correlated), so these are descriptive scales, not a noise model.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

OUT = Path(__file__).resolve().parent


def main():
    rows = []
    for name in c.CLIPS:
        clip = c.Clip(name)
        ws = c.windows(clip, 1)
        pk = []
        for a, b in ws:
            ser = clip.series(clip.xy_edl)
            sp = c.metrics.speed_profile(ser, a + 1, b)
            pk.append(max(sp[1]) / clip.torso)
        thr = np.percentile(pk, 100 / 3)
        slow = [w for w, p in zip(ws, pk) if p <= thr]
        ss, n_res, d2 = 0.0, 0, []
        for a, b in slow:
            t = clip.times[a:b + 1] - clip.times[a]
            for ax in (0, 1):
                y = clip.xy_edl[a:b + 1, ax]
                coef = np.polyfit(t, y, 3)
                r = y - np.polyval(coef, t)
                ss += float(np.sum(r * r)); n_res += len(y) - 4
                d2.extend(np.diff(y, 2).tolist())
        d2 = np.asarray(d2)
        sd_res = (ss / n_res) ** 0.5
        sig_d2 = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / 6 ** 0.5
        rows.append([name, len(ws), len(slow), f"{thr:.4f}", f"{clip.torso:.2f}", f"{sd_res:.3f}", f"{sd_res / clip.torso:.5f}",
                     f"{sig_d2:.3f}", f"{sig_d2 / clip.torso:.5f}"])
    hdr = ["clip", "n_1s_windows", "n_slow_windows", "slow_peak_speed_max_torso_per_s", "median_torso_px", "resid_sd_px",
           "resid_sd_torso", "d2_sigma_px", "d2_sigma_torso"]
    with open(OUT / "noise_estimates.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(hdr); w.writerows(rows)
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(map(str, r)))


if __name__ == "__main__":
    main()
