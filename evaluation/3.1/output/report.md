# Occlusion-gap statistics (real data, D-007 / ARCHITECTURE.md §3)

Threshold used: per-joint confidence < 0.5 (D-007's own threshold, not validate_vitpose.py's QA thresholds).

Whole-frame drops (missing_detection) and joint-level confidence gaps are reported separately below and never merged into one count.

## Per-clip fps (measured from real timestamps, not assumed)

- IMG_2412.MOV: 29.99 fps, 100 frames observed
- IMG_2416.MOV: 30.00 fps, 100 frames observed
- IMG_2417.MOV: 29.95 fps, 100 frames observed
- IMG_2419.MOV: 30.00 fps, 100 frames observed
- IMG_2420.MOV: 30.00 fps, 100 frames observed

## 1. Whole-frame drops (missing_detection) — per clip

These frames are excluded entirely from the joint-level analysis in section 2/3 below (no confidence value exists for any joint on a frame with no detection at all) — reported here on their own, never counted a second time per joint.

- IMG_2412.MOV: 1 event(s) total, 2 frame(s) excluded from joint-level counts
    <5 frames: 1 event(s) (0.067-0.067s), 0 touching a clip boundary, 0 next to a dropped/excluded frame
- IMG_2416.MOV: 0 event(s) total, 0 frame(s) excluded from joint-level counts
- IMG_2417.MOV: 0 event(s) total, 0 frame(s) excluded from joint-level counts
- IMG_2419.MOV: 0 event(s) total, 0 frame(s) excluded from joint-level counts
- IMG_2420.MOV: 1 event(s) total, 1 frame(s) excluded from joint-level counts
    <5 frames: 1 event(s) (0.033-0.033s), 0 touching a clip boundary, 0 next to a dropped/excluded frame

## 2. Joint-level gaps (confidence < 0.5) — all clips

Total joint-level gap events: 29

    <5 frames: 19 event(s) (0.033-0.133s), 0 touching a clip boundary, 0 next to a dropped/excluded frame
    5-20 frames: 9 event(s) (0.167-0.634s), 5 touching a clip boundary, 1 next to a dropped/excluded frame
    >20 frames: 1 event(s) (0.967-0.967s), 0 touching a clip boundary, 0 next to a dropped/excluded frame

Per-joint event counts:

- nose: 4 event(s)
- left_eye: 4 event(s)
- right_eye: 4 event(s)
- left_ear: 0 event(s)
- right_ear: 1 event(s)
- left_shoulder: 0 event(s)
- right_shoulder: 0 event(s)
- left_elbow: 0 event(s)
- right_elbow: 0 event(s)
- left_wrist: 0 event(s)
- right_wrist: 7 event(s)
- left_hip: 0 event(s)
- right_hip: 0 event(s)
- left_knee: 6 event(s)
- right_knee: 0 event(s)
- left_ankle: 0 event(s)
- right_ankle: 3 event(s)

**Provenance:** All clips are the author's own footage, filmed on a fixed camera in the same gym.

## 3. Per-clip joint-level breakdown

- IMG_2412.MOV: 5 joint-level event(s) total
    right_ear: <5 frames=1
    right_wrist: <5 frames=1
    right_ankle: <5 frames=2, >20 frames=1
- IMG_2416.MOV: 6 joint-level event(s) total
    right_wrist: <5 frames=4, 5-20 frames=2
- IMG_2417.MOV: 10 joint-level event(s) total
    nose: <5 frames=1, 5-20 frames=2
    left_eye: <5 frames=1, 5-20 frames=2
    right_eye: <5 frames=2, 5-20 frames=2
- IMG_2419.MOV: 1 joint-level event(s) total
    left_eye: <5 frames=1
- IMG_2420.MOV: 7 joint-level event(s) total
    nose: <5 frames=1
    left_knee: <5 frames=5, 5-20 frames=1

## 4. Pipeline joints (8): left_shoulder, right_shoulder, left_wrist, right_wrist, left_hip, right_hip, left_ankle, right_ankle

Same events as section 2, restricted to joints used by the CoM proxy (D-005 follow-up) and the four-state classifier (ARCHITECTURE.md §4.4). Denominator for '% of used frames' is 497 used frames per joint.

Per joint:

- left_shoulder: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 497 used frames
- right_shoulder: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 497 used frames
- left_wrist: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 497 used frames
- right_wrist: 7 event(s) (5 / 2 / 0 by <5 / 5-20 / >20 frames), 20 low-confidence frame(s) = 4.02% of 497 used frames
- left_hip: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 497 used frames
- right_hip: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 497 used frames
- left_ankle: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 497 used frames
- right_ankle: 3 event(s) (2 / 0 / 1 by <5 / 5-20 / >20 frames), 31 low-confidence frame(s) = 6.24% of 497 used frames

Per clip (pipeline joints combined):

- IMG_2412.MOV: 4 event(s), 32 low-confidence joint-frame(s) of 784 used joint-frames (8 joints x 98 used frames)
- IMG_2416.MOV: 6 event(s), 19 low-confidence joint-frame(s) of 800 used joint-frames (8 joints x 100 used frames)
- IMG_2417.MOV: 0 event(s), 0 low-confidence joint-frame(s) of 800 used joint-frames (8 joints x 100 used frames)
- IMG_2419.MOV: 0 event(s), 0 low-confidence joint-frame(s) of 800 used joint-frames (8 joints x 100 used frames)
- IMG_2420.MOV: 0 event(s), 0 low-confidence joint-frame(s) of 792 used joint-frames (8 joints x 99 used frames)

Total: 10 event(s) (7 / 2 / 1).
Longest pipeline-joint gap: IMG_2412.MOV right_ankle, frames 10-38 (29 frames, 0.967 s); touches clip boundary: False; next to a dropped/excluded frame: False

## Counts and n

- n = 5 clip(s), 500 frame(s) in range, 497 used for joint-level counts (detected, not excluded by list)
- Frames excluded by list: 0
- Whole-frame-drop events: 2
- Joint-level gap events (17 joints): 29
- Joint-level observations (used frames x 17 joints): 8449
- Longest joint-level gap: IMG_2412.MOV right_ankle, frames 10-38 (29 frames, 0.967 s); touches clip boundary: False; next to a dropped/excluded frame: False

## Known limitation

A keypoint with confidence >= 0.5 during a real occlusion (a "confidently wrong" hallucinated position) would not be caught by this threshold at all. This is a limit of the statistic itself, not something this script attempts to detect.
