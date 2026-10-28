"""
Steps 2-5: agreement with the saved ViTPose-L keypoints, and z-signal
diagnostics on BlazePose's world landmarks.

Exploratory spike. Not part of the pipeline. Not a logged decision.
Reads already-saved ViTPose-L CSVs read-only (validation/output/); does not
regenerate them, does not import the main pipeline.

Usage:
    ./.venv/bin/python analyze.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = REPO_ROOT / "footage" / "test"
VITPOSE_DIR = REPO_ROOT / "validation" / "output"
OUTPUT_DIR = Path(__file__).parent / "output"

CLIP_NAMES = ["IMG_2412", "IMG_2416", "IMG_2417", "IMG_2419", "IMG_2420"]
COMPARE_JOINTS = ["left_wrist", "right_wrist", "left_hip", "right_hip", "left_shoulder", "right_shoulder", "left_ankle", "right_ankle"]

# Stationarity definition (engineering heuristic for this spike only, not a
# project threshold): a joint is "stationary" at frame t if its normalized
# image-space (x, y) displacement from frame t-1 is below STATIONARY_THRESH,
# sustained for at least STATIONARY_MIN_RUN consecutive frames. Normalized
# units: 0.01 = 1% of frame width/height per frame.
STATIONARY_THRESH = 0.01
STATIONARY_MIN_RUN = 5


def get_video_dims(clip: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(CLIPS_DIR / f"{clip}.MOV"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def load_vitpose(clip: str) -> pd.DataFrame:
    path = VITPOSE_DIR / f"{clip}_keypoints.csv"
    df = pd.read_csv(path)
    df["video"] = clip
    return df


def load_blazepose_image(clip: str) -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_DIR / "blazepose_image_landmarks.csv")
    return df[df["video"] == clip].copy()


def load_blazepose_world(clip: str) -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_DIR / "blazepose_world_landmarks.csv")
    return df[df["video"] == clip].copy()


def step2_detection_and_visibility(clip: str, bp_img: pd.DataFrame, vp: pd.DataFrame) -> dict:
    bp_frames = set(bp_img["frame_idx"].unique())
    vp_frames = set(vp["frame_idx"].unique())
    joint_vis = bp_img[bp_img["landmark_name"].isin(COMPARE_JOINTS)]
    return {
        "video": clip,
        "blazepose_frames_detected": len(bp_frames),
        "vitpose_frames_detected": len(vp_frames),
        "frames_lost_vs_vitpose": len(vp_frames - bp_frames),
        "mean_visibility_wrists_ankles_hips_shoulders": round(joint_vis["visibility"].mean(), 4),
    }


def step3_agreement(clip: str, bp_img: pd.DataFrame, vp: pd.DataFrame, w: int, h: int) -> tuple[dict, pd.DataFrame]:
    rows = []
    for joint in COMPARE_JOINTS:
        bp_j = bp_img[bp_img["landmark_name"] == joint][["frame_idx", "x", "y"]].rename(columns={"x": "bp_x", "y": "bp_y"})
        vp_j = vp[vp["keypoint_name"] == joint][["frame_idx", "x", "y"]].rename(columns={"x": "vp_x", "y": "vp_y"})
        merged = bp_j.merge(vp_j, on="frame_idx", how="inner")
        if merged.empty:
            continue
        merged["bp_x_px"] = merged["bp_x"] * w
        merged["bp_y_px"] = merged["bp_y"] * h
        merged["dist_px"] = np.sqrt((merged["bp_x_px"] - merged["vp_x"]) ** 2 + (merged["bp_y_px"] - merged["vp_y"]) ** 2)
        merged["joint"] = joint
        merged["video"] = clip
        rows.append(merged[["video", "joint", "frame_idx", "dist_px"]])
    if not rows:
        return {"video": clip, "median_dist_px": None, "p95_dist_px": None}, pd.DataFrame()
    all_dist = pd.concat(rows, ignore_index=True)
    return {
        "video": clip,
        "median_dist_px": round(all_dist["dist_px"].median(), 1),
        "p95_dist_px": round(all_dist["dist_px"].quantile(0.95), 1),
        "n_comparisons": len(all_dist),
    }, all_dist


def find_stationary_segments(xy: np.ndarray) -> list[tuple[int, int]]:
    """xy: (n_frames, 2) normalized image-space coords for one joint, one clip,
    indexed by frame order. Returns list of (start_idx, end_idx) inclusive
    runs where frame-to-frame displacement stayed below STATIONARY_THRESH for
    at least STATIONARY_MIN_RUN consecutive frames."""
    if len(xy) < 2:
        return []
    disp = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    is_still = disp < STATIONARY_THRESH  # is_still[i] = transition i->i+1 was small
    segments = []
    run_start = None
    for i, still in enumerate(is_still):
        if still:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= STATIONARY_MIN_RUN:
                segments.append((run_start, i))
            run_start = None
    if run_start is not None and (len(is_still) - run_start) >= STATIONARY_MIN_RUN:
        segments.append((run_start, len(is_still)))
    return segments


def step4_and_5(clip: str, bp_img: pd.DataFrame, bp_world: pd.DataFrame) -> dict:
    frames = sorted(bp_img["frame_idx"].unique())

    def get_xy_image(name: str) -> dict:
        d = bp_img[bp_img["landmark_name"] == name].set_index("frame_idx")
        return {f: (d.loc[f, "x"], d.loc[f, "y"]) for f in frames if f in d.index}

    def get_xyz_world(name: str) -> dict:
        d = bp_world[bp_world["landmark_name"] == name].set_index("frame_idx")
        return {f: (d.loc[f, "x"], d.loc[f, "y"], d.loc[f, "z"]) for f in frames if f in d.index}

    lhip_w, rhip_w = get_xyz_world("left_hip"), get_xyz_world("right_hip")
    lsh_w, rsh_w = get_xyz_world("left_shoulder"), get_xyz_world("right_shoulder")
    lank_w, rank_w = get_xyz_world("left_ankle"), get_xyz_world("right_ankle")
    lhip_img = get_xy_image("left_hip")
    rhip_img = get_xy_image("right_hip")
    lank_img = get_xy_image("left_ankle")
    rank_img = get_xy_image("right_ankle")

    common = sorted(set(lhip_w) & set(rhip_w) & set(lsh_w) & set(rsh_w) & set(lank_w) & set(rank_w))
    torso_lean = []
    ankle_z_rel = {"left": [], "right": []}
    hip_mid_img = []
    for f in common:
        hip_mid_z = (lhip_w[f][2] + rhip_w[f][2]) / 2
        sh_mid_z = (lsh_w[f][2] + rsh_w[f][2]) / 2
        torso_lean.append((f, sh_mid_z - hip_mid_z))
        ankle_z_rel["left"].append((f, lank_w[f][2] - hip_mid_z))
        ankle_z_rel["right"].append((f, rank_w[f][2] - hip_mid_z))
        if f in lhip_img and f in rhip_img:
            hip_mid_img.append((f, (lhip_img[f][0] + rhip_img[f][0]) / 2, (lhip_img[f][1] + rhip_img[f][1]) / 2))

    # 4a/4b: save time series
    ts_path = OUTPUT_DIR / f"{clip}_z_timeseries.csv"
    with open(ts_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["frame_idx", "torso_lean_z", "left_ankle_z_rel_hip", "right_ankle_z_rel_hip"])
        la = dict(ankle_z_rel["left"])
        ra = dict(ankle_z_rel["right"])
        for f, lean in torso_lean:
            writer.writerow([f, f"{lean:.5f}", f"{la[f]:.5f}", f"{ra[f]:.5f}"])

    # 4c: stationary segments on hip-midpoint (image space) -> jitter (std) of each world-z series within those segments
    hip_mid_xy = np.array([[x, y] for _, x, y in hip_mid_img])
    hip_frames_ordered = [f for f, _, _ in hip_mid_img]
    hip_segments = find_stationary_segments(hip_mid_xy)
    torso_lean_dict = dict(torso_lean)
    segment_stds = []
    for start, end in hip_segments:
        seg_frames = hip_frames_ordered[start:end]
        seg_vals = [torso_lean_dict[f] for f in seg_frames if f in torso_lean_dict]
        if len(seg_vals) >= 2:
            segment_stds.append(float(np.std(seg_vals)))

    # 4d: sign flips of torso lean
    lean_vals = [v for _, v in torso_lean]
    sign_flips = sum(1 for i in range(1, len(lean_vals)) if np.sign(lean_vals[i]) != np.sign(lean_vals[i - 1]) and lean_vals[i] != 0 and lean_vals[i-1] != 0)
    flip_frames = [common[i] for i in range(1, len(lean_vals)) if np.sign(lean_vals[i]) != np.sign(lean_vals[i - 1]) and lean_vals[i] != 0 and lean_vals[i-1] != 0]

    # 5: ankle-specific stationary segments (per ankle, image space) -> std of that ankle's world z within those segments
    ankle_stability = {}
    for side, img_dict, z_dict in [("left", lank_img, dict(ankle_z_rel["left"])), ("right", rank_img, dict(ankle_z_rel["right"]))]:
        frames_ordered = sorted(img_dict.keys())
        xy = np.array([img_dict[f] for f in frames_ordered])
        segs = find_stationary_segments(xy)
        stds = []
        for start, end in segs:
            seg_frames = frames_ordered[start:end]
            seg_vals = [z_dict[f] for f in seg_frames if f in z_dict]
            if len(seg_vals) >= 2:
                stds.append(float(np.std(seg_vals)))
        ankle_stability[side] = {
            "n_stationary_segments": len(segs),
            "mean_z_std_within_segments": round(float(np.mean(stds)), 5) if stds else None,
        }

    return {
        "video": clip,
        "torso_lean_range": (round(min(lean_vals), 4), round(max(lean_vals), 4)) if lean_vals else None,
        "torso_lean_sign_flips": sign_flips,
        "torso_lean_flip_frames": flip_frames,
        "n_hip_stationary_segments": len(hip_segments),
        "torso_lean_jitter_mean_std_within_stationary": round(float(np.mean(segment_stds)), 5) if segment_stds else None,
        "left_ankle_stability": ankle_stability["left"],
        "right_ankle_stability": ankle_stability["right"],
        "timeseries_csv": str(ts_path.name),
    }


def main():
    step2_rows, step3_rows, step45_rows = [], [], []
    all_agreement = []

    for clip in CLIP_NAMES:
        vp_path = VITPOSE_DIR / f"{clip}_keypoints.csv"
        if not vp_path.exists():
            print(f"WARNING: no saved ViTPose CSV for {clip}, skipping comparison steps")
            continue
        w, h = get_video_dims(clip)
        bp_img = load_blazepose_image(clip)
        bp_world = load_blazepose_world(clip)
        vp = load_vitpose(clip)

        step2_rows.append(step2_detection_and_visibility(clip, bp_img, vp))
        s3_summary, s3_detail = step3_agreement(clip, bp_img, vp, w, h)
        step3_rows.append(s3_summary)
        if not s3_detail.empty:
            all_agreement.append(s3_detail)

        step45_rows.append(step4_and_5(clip, bp_img, bp_world))
        print(f"{clip}: done")

    pd.DataFrame(step2_rows).to_csv(OUTPUT_DIR / "step2_detection_visibility.csv", index=False)
    pd.DataFrame(step3_rows).to_csv(OUTPUT_DIR / "step3_agreement_summary.csv", index=False)
    if all_agreement:
        pd.concat(all_agreement, ignore_index=True).to_csv(OUTPUT_DIR / "step3_agreement_detail.csv", index=False)

    with open(OUTPUT_DIR / "step4_5_z_diagnostics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "torso_lean_min", "torso_lean_max", "torso_lean_sign_flips",
            "n_hip_stationary_segments", "torso_lean_jitter_mean_std",
            "left_ankle_n_segments", "left_ankle_mean_z_std",
            "right_ankle_n_segments", "right_ankle_mean_z_std",
        ])
        for r in step45_rows:
            lo, hi = r["torso_lean_range"] if r["torso_lean_range"] else (None, None)
            writer.writerow([
                r["video"], lo, hi, r["torso_lean_sign_flips"],
                r["n_hip_stationary_segments"], r["torso_lean_jitter_mean_std_within_stationary"],
                r["left_ankle_stability"]["n_stationary_segments"], r["left_ankle_stability"]["mean_z_std_within_segments"],
                r["right_ankle_stability"]["n_stationary_segments"], r["right_ankle_stability"]["mean_z_std_within_segments"],
            ])

    print("\n=== Step 2: detection & visibility ===")
    print(pd.DataFrame(step2_rows).to_string(index=False))
    print("\n=== Step 3: agreement with ViTPose-L ===")
    print(pd.DataFrame(step3_rows).to_string(index=False))
    print("\n=== Step 4/5: z diagnostics ===")
    for r in step45_rows:
        print(f"\n{r['video']}:")
        print(f"  torso lean range: {r['torso_lean_range']}, sign flips: {r['torso_lean_sign_flips']} at frames {r['torso_lean_flip_frames']}")
        print(f"  hip-stationary segments: {r['n_hip_stationary_segments']}, mean torso-lean-z std within them: {r['torso_lean_jitter_mean_std_within_stationary']}")
        print(f"  left ankle: {r['left_ankle_stability']}")
        print(f"  right ankle: {r['right_ankle_stability']}")

    print(f"\nWrote CSVs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
