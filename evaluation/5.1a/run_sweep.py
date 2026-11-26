"""
Step 5.1a sweep (protocol: PROTOCOL.md, committed before this ran). Writes raw per-window results to
evaluation/5.1a/output/raw_results.npz (gitignored, regenerable); analyze.py turns them into the committed CSVs.
LDLJ and SPARC are the real metrics.compute_ldlj / metrics.compute_sparc. D-006, D-004.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

metrics = c.metrics
OUT = Path(__file__).resolve().parent / "output"
LEVELS = [0.0025, 0.005, 0.01, 0.02, 0.04]
N_SEEDS = 20
SEED0 = 20261126
DURATIONS = [1, 2, 3]
FILTERS = [("none", None, None), ("sg_7f", "sg", 7), ("sg_13f", "sg", 13), ("sg_25f", "sg", 25),
           ("bw_3Hz", "bw", 3.0), ("bw_6Hz", "bw", 6.0), ("bw_12Hz", "bw", 12.0)]
COMS = ["edl", "A"]
REF = {"amp": 0.05, "fc": 10.0, "pad": 4}
SPARC_SETTINGS = [("ref", 0.05, 10.0, 4), ("amp0.01", 0.01, 10.0, 4), ("amp0.02", 0.02, 10.0, 4), ("amp0.10", 0.10, 10.0, 4),
                  ("amp0.20", 0.20, 10.0, 4), ("fc3", 0.05, 3.0, 4), ("fc5", 0.05, 5.0, 4), ("fc15", 0.05, 15.0, 4),
                  ("pad0", 0.05, 10.0, 0), ("pad2", 0.05, 10.0, 2), ("pad6", 0.05, 10.0, 6), ("pad8", 0.05, 10.0, 8)]
CAP = 20 * math.pi


def apply_filter(xy, kind, par):
    if kind is None:
        return xy
    out = xy.copy()
    for a, b in c.defined_runs(xy):
        n = b - a + 1
        seg = xy[a:b + 1]
        if kind == "sg":
            if n < par:
                out[a:b + 1] = np.nan  # no window uses a run this short (windows need >= 30 frames)
                continue
            out[a:b + 1] = savgol_filter(seg, par, 3, axis=0, mode="interp")
        else:
            if n < 30:
                out[a:b + 1] = np.nan
                continue
            sos = butter(2, par, btype="low", fs=30.0, output="sos")
            out[a:b + 1] = sosfiltfilt(sos, seg, axis=0)
    return out


def speeds(xy, times):
    v = np.full(len(xy), np.nan)
    d = np.hypot(*(xy[1:] - xy[:-1]).T)
    v[1:] = d / np.diff(times)
    return v


def sparc_val(t, v, amp, fc, pad):
    r = metrics.compute_sparc(t, v, amp_threshold=amp, omega_c_max=2 * math.pi * fc, n_fft=metrics.sparc_n_fft(len(v), pad))
    return np.nan if r is None else r


def sparc_capped(t, v):
    """True when the adaptive cut-off of the reference setting is the 10 Hz cap (same rule as metrics._sparc_from_spectrum)."""
    n_fft = metrics.sparc_n_fft(len(v))
    sp = np.abs(np.fft.rfft(v, n=n_fft))
    if sp[0] <= 0:
        return False
    fs = (len(t) - 1) / (t[-1] - t[0])
    om = 2 * math.pi * np.arange(len(sp)) * fs / n_fft
    above = np.nonzero(sp / sp[0] >= REF["amp"])[0]
    if len(above) == 0:
        return False
    return float(om[above[-1]]) >= CAP


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    clips = [c.Clip(n) for n in c.CLIPS]
    W = []  # (clip index, duration, a, b)
    for ci, clip in enumerate(clips):
        for d in DURATIONS:
            for a, b in c.windows(clip, d):
                W.append((ci, d, a, b))
    nW = len(W)
    conds = [(0.0, -1)] + [(lv, k) for lv in LEVELS for k in range(N_SEEDS)]
    nC = len(conds)
    ldlj = {(com, f[0]): np.full((nC, nW), np.nan) for com in COMS for f in FILTERS}
    sparc = {(com, f[0]): np.full((nC, nW), np.nan) for com in COMS for f in FILTERS}
    capd = {(com, f[0]): np.zeros((nC, nW), dtype=bool) for com in COMS for f in FILTERS}
    sweep = {s[0]: np.full((nC, nW), np.nan) for s in SPARC_SETTINGS}
    peak = np.full(nW, np.nan)
    check_mismatch = 0
    check_n = 0
    for ci, clip in enumerate(clips):
        idx = [i for i, w in enumerate(W) if w[0] == ci]
        rng_z = [np.random.default_rng(SEED0 + k).standard_normal((clip.n, 2)) for k in range(N_SEEDS)]
        for cond_i, (lv, k) in enumerate(conds):
            for com in COMS:
                base = clip.which(com)
                xy = base if lv == 0 else base + lv * clip.torso * rng_z[k]
                for fname, kind, par in FILTERS:
                    xf = apply_filter(xy, kind, par)
                    v = speeds(xf, clip.times)
                    for i in idx:
                        _, d, a, b = W[i]
                        t, s = clip.times[a + 1:b + 1], v[a + 1:b + 1]
                        if np.isnan(s).any():
                            continue
                        ldlj[(com, fname)][cond_i, i] = np.nan if (r := metrics.compute_ldlj(t, s)) is None else r
                        sparc[(com, fname)][cond_i, i] = sparc_val(t, s, REF["amp"], REF["fc"], REF["pad"])
                        capd[(com, fname)][cond_i, i] = sparc_capped(t, s)
                        if cond_i == 0 and fname == "none" and com == "edl":
                            ser = clip.series(xy)
                            sp = metrics.speed_profile(ser, a + 1, b)
                            check_n += 1
                            if sp is None or not np.allclose(sp[1], s, rtol=0, atol=1e-9) or not np.allclose(sp[0], t, rtol=0, atol=1e-12):
                                check_mismatch += 1
                            peak[i] = s.max() / clip.torso
                        if fname == "none" and com == "edl":
                            for sname, amp, fc, pad in SPARC_SETTINGS:
                                sweep[sname][cond_i, i] = sparc_val(t, s, amp, fc, pad)
        print(f"{clip.name}: done ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(
        OUT / "raw_results.npz", W=np.array(W), conds=np.array(conds), peak=peak,
        **{f"ldlj|{k[0]}|{k[1]}": v for k, v in ldlj.items()}, **{f"sparc|{k[0]}|{k[1]}": v for k, v in sparc.items()},
        **{f"cap|{k[0]}|{k[1]}": v for k, v in capd.items()}, **{f"sweep|{k}": v for k, v in sweep.items()})
    info = {"n_windows": nW, "n_conditions": nC, "speed_profile_check_windows": check_n, "speed_profile_check_mismatch": check_mismatch,
            "clips": c.CLIPS, "levels": LEVELS, "n_seeds": N_SEEDS, "seed0": SEED0, "seconds": round(time.time() - t0)}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
