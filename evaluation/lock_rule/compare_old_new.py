"""
What changed when the analysis moved from the legacy-lock run (validation/output_full) to the gated-lock run
(validation/output_full_lock1), DECISIONS.md D-027. Compares the 3.1 gap statistics and the 2.2c candidate
comparison (and the keypoint rows underneath) between the old and the new folders. Nothing old is written.

Expectation stated by the author: only IMG_2419 and the 33 newly tracked frames (IMG_2412 2, IMG_2416 24, IMG_2420 7)
move. Checks made here: (1) keypoint rows that differ outside IMG_2419 and outside newly tracked frames (expected 0);
(2) candidate positions that differ at frames whose keypoint rows are unchanged (expected 0 for A, C, D, F; B, E and
E-dL move because their trunk weight w is refitted leave-one-clip-out on different data); (3) speed and jitter
values that differ farther than one or two frames from a changed frame (expected 0).

Writes evaluation/lock_rule/output_lock1/rerun_comparison.md, rerun_3_1.csv, rerun_2_2c.csv.

Usage:
    python3 evaluation/lock_rule/compare_old_new.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compare_gated_run import load_kp  # noqa: E402

OUT = Path("evaluation/lock_rule/output_lock1")
CLIPS = ["IMG_2412", "IMG_2416", "IMG_2417", "IMG_2419", "IMG_2420"]
OLD_KP, NEW_KP = Path("validation/output_full"), Path("validation/output_full_lock1")
O31, N31 = Path("evaluation/3.1/output_full"), Path("evaluation/3.1/output_full_lock1")
O22, N22 = Path("evaluation/2.2c/output_full"), Path("evaluation/2.2c/output_full_lock1")
LABELS = ["A_proxy", "B_trunk_point", "C_no_head", "D_trunk_legs", "E_c0.5", "E_c0.6", "E_c0.7", "E_c0.8",
          "EdL_c0.5", "EdL_c0.6", "EdL_c0.7", "EdL_c0.8", "F_full"]
FIXED = ["A_proxy", "C_no_head", "D_trunk_legs", "F_full"]            # no fitted weight
PIPELINE = {"left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_ankle",
            "right_ankle"}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    return None if x in ("", None) else float(x)


def main() -> None:
    md = ["# Old (legacy lock, validation/output_full) vs new (gated lock, validation/output_full_lock1)\n"]

    # --- 1. keypoint rows
    changed = {}
    md.append("## 1. Keypoint rows\n")
    md.append("| clip | frames with rows old / new | newly tracked | lost | rows differ (box changed) | outside IMG_2419 and not newly tracked |\n|---|---|---|---|---|---|")
    unexplained_total = 0
    for c in CLIPS:
        old, new = load_kp(OLD_KP / f"{c}_keypoints.csv"), load_kp(NEW_KP / f"{c}_keypoints.csv")
        gained = sorted(t for t in new if t not in old)
        lost = sorted(t for t in old if t not in new)
        diff = sorted(t for t in new if t in old and any(a != b for a, b in zip(new[t], old[t])))
        changed[c] = set(gained) | set(lost) | set(diff)
        unexpl = 0 if c == "IMG_2419" else len(lost) + len(diff)
        unexplained_total += unexpl
        md.append(f"| {c} | {len(old)} / {len(new)} | {len(gained)} | {len(lost)} | {len(diff)} | {unexpl} |")
    md.append(f"\nKeypoint rows that changed outside IMG_2419 other than newly tracked frames: **{unexplained_total}** (expected 0).\n")

    # --- 2. 3.1
    o_sum, n_sum = rows(O31 / "summary.csv"), rows(N31 / "summary.csv")
    o_ev, n_ev = rows(O31 / "events.csv"), rows(N31 / "events.csv")
    o_fl, n_fl = rows(O31 / "frame_losses.csv"), rows(N31 / "frame_losses.csv")
    out31 = []
    md.append("## 2. 3.1 gap statistics (pipeline joints = shoulders, wrists, hips, ankles; all 17 where stated)\n")
    md.append("| clip | detector-dropped frames old/new | excluded-by-list old/new | frames used old/new | detector-drop events old/new | longest drop old/new | joint events (17) old/new | pipeline-joint events old/new | pipeline events <5 / 5-20 / >20 old | same, new | low-confidence frames, pipeline joints old/new |\n|---|---|---|---|---|---|---|---|---|---|---|")

    def per_clip(summary, events, losses, clip):
        s = [r for r in summary if r["clip"] == clip + ".MOV" and r["unit"] == "frames"]   # one row per joint
        dropped = int(s[0]["frames_detector_dropped"])
        excl = int(s[0]["frames_excluded_by_list"])
        used = int(s[0]["frames_used"])
        dl = [int(r["length_frames"]) for r in losses if r["clip"] == clip + ".MOV" and r["kind"] == "detector_drop"]
        ev = [e for e in events if e["clip"] == clip + ".MOV"]
        pev = [e for e in ev if e["pipeline_joint"] == "True"]
        lens = [int(e["length_frames"]) for e in pev]
        bins = (sum(1 for x in lens if x < 5), sum(1 for x in lens if 5 <= x <= 20), sum(1 for x in lens if x > 20))
        low = sum(int(r["frames_low_conf"]) for r in s if r["joint"] in PIPELINE)
        return dict(dropped=dropped, excluded=excl, used=used, drop_events=len(dl), longest=max(dl, default=0),
                    events17=len(ev), pipeline_events=len(pev), bins=bins, low=low)
    tot_o = tot_n = None
    for c in CLIPS + ["ALL"]:
        if c == "ALL":
            o = {k: (tuple(sum(x) for x in zip(*[per_clip(o_sum, o_ev, o_fl, cc)[k] for cc in CLIPS])) if k == "bins"
                     else (max if k == "longest" else sum)(per_clip(o_sum, o_ev, o_fl, cc)[k] for cc in CLIPS))
                 for k in per_clip(o_sum, o_ev, o_fl, CLIPS[0])}
            n = {k: (tuple(sum(x) for x in zip(*[per_clip(n_sum, n_ev, n_fl, cc)[k] for cc in CLIPS])) if k == "bins"
                     else (max if k == "longest" else sum)(per_clip(n_sum, n_ev, n_fl, cc)[k] for cc in CLIPS))
                 for k in per_clip(n_sum, n_ev, n_fl, CLIPS[0])}
        else:
            o, n = per_clip(o_sum, o_ev, o_fl, c), per_clip(n_sum, n_ev, n_fl, c)
        md.append(f"| {c} | {o['dropped']} / {n['dropped']} | {o['excluded']} / {n['excluded']} | {o['used']} / {n['used']} | "
                  f"{o['drop_events']} / {n['drop_events']} | {o['longest']} / {n['longest']} | {o['events17']} / {n['events17']} | "
                  f"{o['pipeline_events']} / {n['pipeline_events']} | {o['bins']} | {n['bins']} | {o['low']} / {n['low']} |")
        for k in o:
            out31.append([c, k, o[k], n[k]])
    with open(OUT / "rerun_3_1.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["clip", "quantity", "old", "new"])
        w.writerows(out31)

    # --- 3. 2.2c summary
    o22, n22 = rows(O22 / "summary.csv"), rows(N22 / "summary.csv")
    key = lambda rs: {(r["clip"], r["candidate"]): r for r in rs}  # noqa: E731
    ko, kn = key(o22), key(n22)
    metrics = [("n_defined", "defined frames"), ("coverage", "coverage"), ("n_shared_with_f", "shared with F"),
               ("pos_px_mean", "pos diff to F, mean px"), ("pos_torso_mean", "pos diff to F, mean torso"),
               ("jitter_common_mean", "jitter (common support), mean"), ("jitter_common_p95", "jitter (common), p95"),
               ("defined_to_none_per_1000_frames", "defined-to-None per 1,000"), ("n_defined_runs", "defined runs")]
    long22 = []
    for (clip, cand), r in kn.items():
        for m, _ in metrics:
            a, b = fnum(ko[(clip, cand)][m]), fnum(r[m])
            long22.append([clip, cand, m, a, b, None if a is None or b is None else round(b - a, 6)])
    with open(OUT / "rerun_2_2c.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["clip", "candidate", "metric", "old", "new", "delta"])
        w.writerows(long22)
    md.append("\n## 3. 2.2c candidate comparison, pooled (ALL); old -> new (delta)\n")
    md.append("| candidate | defined | coverage | pos diff mean px | pos diff mean torso | jitter common mean | jitter common p95 | defined-to-None /1,000 | runs |\n|---|---|---|---|---|---|---|---|---|")
    for cand in LABELS:
        cells = []
        for m, _ in metrics:
            if m in ("n_shared_with_f",):
                continue
            a, b = fnum(ko[("ALL", cand)][m]), fnum(kn[("ALL", cand)][m])
            d = b - a
            cells.append(f"{a:g} -> {b:g} ({d:+.4g})")
        md.append(f"| {cand} | " + " | ".join(cells) + " |")
    md.append("\nShared frames with F (pooled): old %s -> new %s.\n" % (ko[("ALL", "F_full")]["n_shared_with_f"], kn[("ALL", "F_full")]["n_shared_with_f"]))
    md.append("### Per clip, survivors of the triage (A_proxy, EdL_c0.5, F_full): defined / pos diff px / jitter common mean, old -> new\n")
    md.append("| clip | candidate | defined | pos diff mean px | jitter common mean |\n|---|---|---|---|---|")
    for c in [x + ".MOV" for x in CLIPS]:
        for cand in ("A_proxy", "EdL_c0.5", "F_full"):
            a, b = ko[(c, cand)], kn[(c, cand)]
            md.append(f"| {c[:-4]} | {cand} | {a['n_defined']} -> {b['n_defined']} | {a['pos_px_mean']} -> {b['pos_px_mean']} | {a['jitter_common_mean']} -> {b['jitter_common_mean']} |")

    # --- 4. frame-level exactness
    md.append("\n## 4. Frame-level checks (2.2c frames.csv, old vs new)\n")
    md.append("| clip | changed keypoint frames | A/C/D/F positions differing at unchanged frames | B/E/E-dL position shift at unchanged frames (max px) | speed/jitter differing farther than 1/2 frames from a changed frame |\n|---|---|---|---|---|")
    far_total = fixed_total = 0
    for c in CLIPS:
        old_f = {int(r["frame_idx"]): r for r in rows(O22 / c / "frames.csv")}
        new_f = {int(r["frame_idx"]): r for r in rows(N22 / c / "frames.csv")}
        ch = changed[c]
        near1 = {t + d for t in ch for d in (-1, 0, 1)}
        near2 = {t + d for t in ch for d in (-2, -1, 0, 1, 2)}
        fixed_diff = far = 0
        shift = 0.0
        for t, r in new_f.items():
            o = old_f[t]
            for lab in FIXED:
                if t not in ch and (r[f"{lab}_x"], r[f"{lab}_y"]) != (o[f"{lab}_x"], o[f"{lab}_y"]):
                    fixed_diff += 1
            for lab in LABELS:
                if lab not in FIXED and t not in ch and r[f"{lab}_x"] and o[f"{lab}_x"]:
                    shift = max(shift, abs(float(r[f"{lab}_x"]) - float(o[f"{lab}_x"])), abs(float(r[f"{lab}_y"]) - float(o[f"{lab}_y"])))
            for lab in FIXED:
                if r[f"{lab}_speed_px_s"] != o[f"{lab}_speed_px_s"] and t not in near1:
                    far += 1
                if r[f"{lab}_jitter_torso"] != o[f"{lab}_jitter_torso"] and t not in near2:
                    far += 1
        far_total += far
        fixed_total += fixed_diff
        md.append(f"| {c} | {len(ch)} | {fixed_diff} | {shift:.3f} | {far} |")
    md.append(f"\nTotals: A/C/D/F positions differing at unchanged frames {fixed_total}; speed/jitter of A/C/D/F differing beyond the neighbours of a changed frame {far_total} (both expected 0). "
              "The trunk-point weight w of B, E and E-dL is refitted leave-one-clip-out on the new frames, so those candidates shift everywhere by the amount in the column above.\n")

    # --- 5. weights and triage
    ow, nw = rows(O22 / "loco_trunk_weight.csv"), rows(N22 / "loco_trunk_weight.csv")
    md.append("## 5. Fitted trunk weight per fold and triage\n")
    md.append("| held-out clip | w old | w new | delta |\n|---|---|---|---|")
    for a, b in zip(ow, nw):
        md.append(f"| {a['held_out_clip'][:-4]} | {a['w']} | {b['w']} | {float(b['w']) - float(a['w']):+.6f} |")
    ot, nt = rows(O22 / "triage.csv"), rows(N22 / "triage.csv")
    od = {r["candidate"]: (r["dominated"], r["dropped"]) for r in ot}
    nd = {r["candidate"]: (r["dominated"], r["dropped"]) for r in nt}
    md.append("\nTriage (dropped candidates) old %s; new %s; identical: **%s**.\n" % (
        [k for k, v in od.items() if v[1] == "yes"], [k for k, v in nd.items() if v[1] == "yes"], od == nd))
    (OUT / "rerun_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
