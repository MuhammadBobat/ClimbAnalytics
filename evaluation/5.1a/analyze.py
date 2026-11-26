"""
Step 5.1a analysis: raw per-window results -> committed CSVs (descriptive; no p-values, no thresholds). See PROTOCOL.md.
Sign convention: higher LDLJ and higher (less negative) SPARC = smoother. Delta < 0 = scored as LESS smooth than the reference.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c
import run_sweep as rs

OUT = Path(__file__).resolve().parent / "output"
Z = np.load(OUT / "raw_results.npz")
W = Z["W"]; conds = Z["conds"]; peak = Z["peak"]
CI, DUR = W[:, 0], W[:, 1]
nW = len(W)
LEVELS = rs.LEVELS; NS = rs.N_SEEDS
lvl_idx = {lv: [i for i, (l, k) in enumerate(conds) if l == lv] for lv in LEVELS}
NAMES = c.CLIPS
METRICS = ["ldlj", "sparc"]
FILT = [f[0] for f in rs.FILTERS]

# slow / middle / fast terciles of peak speed, per duration, pooled over clips
cls = np.empty(nW, dtype=object)
for d in rs.DURATIONS:
    m = DUR == d
    lo, hi = np.percentile(peak[m], [100 / 3, 200 / 3])
    cls[m & (peak <= lo)] = "slow"
    cls[m & (peak > hi)] = "fast"
    cls[m & (peak > lo) & (peak <= hi)] = "middle"


def get(metric, com, filt):
    return Z[f"{metric}|{com}|{filt}"]


def q(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    return tuple(np.percentile(x, [50, 25, 75]))


def f4(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(header); w.writerows(rows)


def clip_medians(arr, d):
    """(n_conditions, 5) per-clip median over the clip's windows of duration d."""
    out = np.full((arr.shape[0], 5), np.nan)
    for ci in range(5):
        m = (CI == ci) & (DUR == d)
        if m.any():
            with np.errstate(all="ignore"):
                out[:, ci] = np.nanmedian(arr[:, m], axis=1)
    return out


def ranks(row):
    return rankdata(-row, method="average")  # rank 1 = highest = smoothest


def rho(a, b):
    if np.isnan(a).any() or np.isnan(b).any():
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def seed_mean_delta(arr, lv, base):
    """per window: mean over seeds of (m - base); also per-window sd over seeds; windows with an undefined value in any seed or base -> nan"""
    sub = arr[lvl_idx[lv]] - base[None, :]
    return np.nanmean(sub, axis=0), np.nanstd(arr[lvl_idx[lv]], axis=0)


# 1. noise-free per clip -------------------------------------------------------------------------
rows = []
for com in rs.COMS:
    for filt in FILT:
        for d in rs.DURATIONS:
            med = {}
            for mt in METRICS:
                med[mt] = clip_medians(get(mt, com, filt)[:1], d)[0]
            rk = {mt: ranks(med[mt]) for mt in METRICS}
            for ci in range(5):
                m = (CI == ci) & (DUR == d)
                lj = get("ldlj", com, filt)[0, m]; sp = get("sparc", com, filt)[0, m]
                a, b, cc = q(lj); e, f_, g = q(sp)
                rows.append([com, filt, d, NAMES[ci], int(m.sum()), int((~np.isnan(lj)).sum()), f4(a), f4(b), f4(cc), f4(e), f4(f_), f4(g),
                             int(rk["ldlj"][ci]) if rk["ldlj"][ci] == int(rk["ldlj"][ci]) else rk["ldlj"][ci],
                             int(rk["sparc"][ci]) if rk["sparc"][ci] == int(rk["sparc"][ci]) else rk["sparc"][ci]])
write("noise_free_per_clip.csv", ["com", "filter", "duration_s", "clip", "n_windows", "n_defined_ldlj", "ldlj_median", "ldlj_q1", "ldlj_q3",
                                  "sparc_median", "sparc_q1", "sparc_q3", "rank_ldlj_1smoothest", "rank_sparc_1smoothest"], rows)

# 2. noise effect (change from the same window's noise-free value, same filter) --------------------
rows = []
for com in rs.COMS:
    for filt in FILT:
        for mt in METRICS:
            arr = get(mt, com, filt)
            base = arr[0]
            for lv in LEVELS:
                dl = np.nanmean(arr[lvl_idx[lv]] - base[None, :], axis=0)
                sd = np.nanstd(arr[lvl_idx[lv]], axis=0)
                n_undef = int(np.isnan(arr[lvl_idx[lv]]).sum())
                for d in rs.DURATIONS:
                    for scope in ["ALL"] + NAMES:
                        for klass in (["all", "slow", "middle", "fast"] if scope == "ALL" else ["all"]):
                            m = (DUR == d) & ~np.isnan(base)
                            if scope != "ALL":
                                m &= CI == NAMES.index(scope)
                            if klass != "all":
                                m &= cls == klass
                            a, b, cc = q(dl[m]); s = q(sd[m])[0]
                            rows.append([com, filt, d, scope, klass, lv, mt, int(m.sum()), len(set(CI[m])), f4(a), f4(b), f4(cc), f4(s),
                                         int(np.isnan(arr[lvl_idx[lv]][:, m]).sum())])
write("noise_effect.csv", ["com", "filter", "duration_s", "scope", "class", "noise_level_torso", "metric", "n_windows", "n_clips",
                           "median_delta", "delta_q1", "delta_q3", "median_seed_sd", "n_undefined_window_seeds"], rows)

# 3. filtering alone (noise-free): change from the unfiltered value of the same window ---------------
rows = []
for com in rs.COMS:
    for filt in FILT[1:]:
        for mt in METRICS:
            dl = get(mt, com, filt)[0] - get(mt, com, "none")[0]
            for d in rs.DURATIONS:
                for klass in ["all", "slow", "middle", "fast"]:
                    m = (DUR == d) & ~np.isnan(dl)
                    if klass != "all":
                        m &= cls == klass
                    a, b, cc = q(dl[m])
                    rows.append([com, filt, d, klass, mt, int(m.sum()), f4(a), f4(b), f4(cc)])
write("filter_effect.csv", ["com", "filter", "duration_s", "class", "metric", "n_windows", "median_delta_vs_unfiltered", "q1", "q3"], rows)

# 4. clip ranking stability --------------------------------------------------------------------------
rows = []
for com in rs.COMS:
    for mt in METRICS:
        for d in rs.DURATIONS:
            ref_none = ranks(clip_medians(get(mt, com, "none")[:1], d)[0])
            for filt in FILT:
                arr = get(mt, com, filt)
                cm = clip_medians(arr, d)
                ref_same = ranks(cm[0])
                for lv in [0.0] + LEVELS:
                    ids = [0] if lv == 0 else lvl_idx[lv]
                    rk = [ranks(cm[i]) for i in ids]
                    r_none = [rho(r, ref_none) for r in rk]
                    r_same = [rho(r, ref_same) for r in rk]
                    ident_none = np.mean([np.array_equal(r, ref_none) for r in rk])
                    ident_same = np.mean([np.array_equal(r, ref_same) for r in rk])
                    rows.append([com, mt, d, filt, lv, len(ids), " ".join(f"{x:g}" for x in ref_none), " ".join(f"{x:g}" for x in ref_same),
                                 f4(np.nanmean(r_none)), f4(np.nanmin(r_none)), f4(ident_none),
                                 f4(np.nanmean(r_same)), f4(np.nanmin(r_same)), f4(ident_same)])
write("ranking_stability.csv", ["com", "metric", "duration_s", "filter", "noise_level_torso", "n_seeds", "rank_noisefree_unfiltered_by_clip(2412 2416 2417 2419 2420)",
                                "rank_noisefree_same_filter", "mean_spearman_vs_unfiltered_noisefree", "min_spearman_vs_unfiltered_noisefree",
                                "share_identical_order_vs_unfiltered_noisefree", "mean_spearman_vs_same_filter_noisefree",
                                "min_spearman_vs_same_filter_noisefree", "share_identical_order_vs_same_filter_noisefree"], rows)

# 5. slow versus fast at the same noise --------------------------------------------------------------
rows = []
for com in rs.COMS:
    for filt in FILT:
        for mt in METRICS:
            arr = get(mt, com, filt); base = arr[0]
            for lv in LEVELS:
                dl = np.nanmean(arr[lvl_idx[lv]] - base[None, :], axis=0)
                for d in rs.DURATIONS:
                    out = {}
                    for kl in ["slow", "middle", "fast"]:
                        m = (DUR == d) & (cls == kl) & ~np.isnan(dl)
                        out[kl] = (q(dl[m])[0], int(m.sum()), q(base[m])[0], len(set(CI[m])))
                    rows.append([com, filt, d, lv, mt, out["slow"][1], out["fast"][1], f4(out["slow"][2]), f4(out["fast"][2]),
                                 f4(out["slow"][0]), f4(out["middle"][0]), f4(out["fast"][0]), f4(out["slow"][0] - out["fast"][0]),
                                 out["slow"][3], out["fast"][3]])
write("slow_vs_fast.csv", ["com", "filter", "duration_s", "noise_level_torso", "metric", "n_slow", "n_fast", "noisefree_median_slow", "noisefree_median_fast",
                           "median_delta_slow", "median_delta_middle", "median_delta_fast", "slow_minus_fast", "n_clips_slow", "n_clips_fast"], rows)

# 6. metric versus peak speed (descriptive Spearman) -----------------------------------------------------
rows = []
for com in rs.COMS:
    for filt in ["none", "sg_13f", "bw_6Hz"]:
        for mt in METRICS:
            arr = get(mt, com, filt)
            for lv in [0.0, 0.01]:
                vals = arr[0] if lv == 0 else np.nanmean(arr[lvl_idx[lv]], axis=0)
                for d in rs.DURATIONS:
                    for scope in ["ALL"] + NAMES:
                        m = (DUR == d) & ~np.isnan(vals)
                        if scope != "ALL":
                            m &= CI == NAMES.index(scope)
                        r = rho(peak[m], vals[m]) if m.sum() >= 4 else np.nan
                        rows.append([com, filt, d, scope, lv, mt, int(m.sum()), f4(r)])
write("metric_vs_peak_speed.csv", ["com", "filter", "duration_s", "scope", "noise_level_torso", "metric", "n_windows", "spearman_rho_with_peak_speed"], rows)

# 7. SPARC cut-off at the cap, per filter ---------------------------------------------------------------------
rows = []
for com in rs.COMS:
    for filt in FILT:
        cap = Z[f"cap|{com}|{filt}"]; sp = get("sparc", com, filt)
        for lv in [0.0] + LEVELS:
            ids = [0] if lv == 0 else lvl_idx[lv]
            for d in rs.DURATIONS + ["all"]:
                m = (DUR == d) if d != "all" else np.ones(nW, bool)
                ok = ~np.isnan(sp[ids][:, m])
                share = float(cap[ids][:, m][ok].mean()) if ok.any() else np.nan
                rows.append([com, filt, lv, d, int(m.sum()), f4(share)])
write("sparc_cutoff_at_cap.csv", ["com", "filter", "noise_level_torso", "duration_s", "n_windows", "share_of_defined_cases_with_adaptive_cutoff_at_10Hz_cap"], rows)

# 8. SPARC parameter sweep ----------------------------------------------------------------------------------------
rows = []
ref = Z["sweep|ref"]
for sname, amp, fc, pad in rs.SPARC_SETTINGS:
    arr = Z[f"sweep|{sname}"]
    for d in rs.DURATIONS:
        cm = clip_medians(arr, d); ref_cm = clip_medians(ref, d)
        rk0 = ranks(cm[0]); rkref = ranks(ref_cm[0])
        mm = (DUR == d)
        dref = arr[0, mm] - ref[0, mm]
        a, b, cc = q(dref)
        n_undef = int(np.isnan(arr[0, mm]).sum())
        row = [sname, amp, fc, pad, d, int(mm.sum()), n_undef, " ".join(f4(x) for x in cm[0]), " ".join(f"{x:g}" for x in rk0),
               f4(rho(cm[0], ref_cm[0])), f4(a), f4(b), f4(cc)]
        for lv in LEVELS:
            dl = np.nanmean(arr[lvl_idx[lv]] - arr[0][None, :], axis=0)
            rr = [rho(cm[i], ref_cm[0]) for i in lvl_idx[lv]]
            row += [f4(q(dl[mm])[0]), f4(np.nanmean(rr))]
        rows.append(row)
hdr = ["setting", "amp_threshold", "fc_hz", "padlevel", "duration_s", "n_windows", "n_undefined_noisefree", "noisefree_median_by_clip(2412 2416 2417 2419 2420)",
       "rank_by_clip", "spearman_clip_order_vs_reference_setting", "median_delta_vs_reference_setting_noisefree", "q1", "q3"]
for lv in LEVELS:
    hdr += [f"median_delta_vs_own_noisefree@{lv}", f"mean_spearman_vs_reference_noisefree_order@{lv}"]
write("sparc_parameter_sweep.csv", hdr, rows)
# 9. A versus E-dL (speed-profile check only; the comparison on movement segments waits for 4.1) and between-clip spread -----------
rows = []
for mt in METRICS:
    for filt in FILT:
        for d in rs.DURATIONS:
            ce = clip_medians(get(mt, "edl", filt)[:1], d)[0]; ca = clip_medians(get(mt, "A", filt)[:1], d)[0]
            m = DUR == d
            dd = get(mt, "A", filt)[0, m] - get(mt, "edl", filt)[0, m]
            rows.append([mt, filt, d, " ".join(f"{x:g}" for x in ranks(ce)), " ".join(f"{x:g}" for x in ranks(ca)), f4(rho(ce, ca)),
                         f4(np.nanmax(ce) - np.nanmin(ce)), f4(np.nanmax(ca) - np.nanmin(ca)), f4(q(dd)[0]), f4(q(dd)[1]), f4(q(dd)[2])])
write("com_agreement_and_clip_spread.csv", ["metric", "filter", "duration_s", "rank_edl", "rank_A", "spearman_rank_edl_vs_A", "range_of_clip_medians_edl",
                                            "range_of_clip_medians_A", "median_window_diff_A_minus_edl", "q1", "q3"], rows)
print("written:", sorted(p.name for p in OUT.glob("*.csv")))
print("class counts per duration:", {d: {k: int(((DUR == d) & (cls == k)).sum()) for k in ['slow', 'middle', 'fast']} for d in rs.DURATIONS})
print("windows per clip/duration:", {NAMES[ci]: {d: int(((CI == ci) & (DUR == d)).sum()) for d in rs.DURATIONS} for ci in range(5)})
