# Occlusion-gap statistics (real data, D-007 / ARCHITECTURE.md §3)

Threshold used: per-joint confidence < 0.5 (D-007's own threshold, not validate_vitpose.py's QA thresholds).

Whole-frame drops (missing_detection) and joint-level confidence gaps are reported separately below and never merged into one count.

## Per-clip fps (measured from real timestamps, not assumed)

- IMG_2412.MOV: 29.99 fps, 718 frames observed
- IMG_2416.MOV: 30.00 fps, 917 frames observed
- IMG_2417.MOV: 29.95 fps, 980 frames observed
- IMG_2419.MOV: 30.00 fps, 850 frames observed
- IMG_2420.MOV: 30.00 fps, 2421 frames observed

## 1. Whole-frame drops (missing_detection) — per clip

These frames are excluded entirely from the joint-level analysis in section 2/3 below (no confidence value exists for any joint on a frame with no detection at all) — reported here on their own, never counted a second time per joint.

- IMG_2412.MOV: 1 event(s) total, 2 frame(s) excluded from joint-level counts
    <5 frames: 1 event(s) (0.067-0.067s), 0 touching a clip boundary, 0 next to a dropped/excluded frame
- IMG_2416.MOV: 8 event(s) total, 24 frame(s) excluded from joint-level counts
    <5 frames: 6 event(s) (0.033-0.067s), 0 touching a clip boundary, 0 next to a dropped/excluded frame
    5-20 frames: 2 event(s) (0.233-0.267s), 0 touching a clip boundary, 0 next to a dropped/excluded frame
- IMG_2417.MOV: 0 event(s) total, 0 frame(s) excluded from joint-level counts
- IMG_2419.MOV: 3 event(s) total, 7 frame(s) excluded from joint-level counts
    <5 frames: 2 event(s) (0.033-0.033s), 0 touching a clip boundary, 0 next to a dropped/excluded frame
    5-20 frames: 1 event(s) (0.167-0.167s), 0 touching a clip boundary, 0 next to a dropped/excluded frame
    plus 18 frame(s) excluded by list (not detector drops), frames 597-614
- IMG_2420.MOV: 5 event(s) total, 7 frame(s) excluded from joint-level counts
    <5 frames: 5 event(s) (0.033-0.100s), 0 touching a clip boundary, 0 next to a dropped/excluded frame

## 2. Joint-level gaps (confidence < 0.5) — all clips

Total joint-level gap events: 558

    <5 frames: 428 event(s) (0.033-0.134s), 2 touching a clip boundary, 13 next to a dropped/excluded frame
    5-20 frames: 113 event(s) (0.167-0.668s), 3 touching a clip boundary, 3 next to a dropped/excluded frame
    >20 frames: 17 event(s) (0.700-2.905s), 0 touching a clip boundary, 0 next to a dropped/excluded frame

Per-joint event counts:

- nose: 132 event(s)
- left_eye: 131 event(s)
- right_eye: 129 event(s)
- left_ear: 3 event(s)
- right_ear: 1 event(s)
- left_shoulder: 0 event(s)
- right_shoulder: 0 event(s)
- left_elbow: 33 event(s)
- right_elbow: 5 event(s)
- left_wrist: 79 event(s)
- right_wrist: 28 event(s)
- left_hip: 0 event(s)
- right_hip: 0 event(s)
- left_knee: 9 event(s)
- right_knee: 0 event(s)
- left_ankle: 2 event(s)
- right_ankle: 6 event(s)

**Provenance:** All clips are the author's own footage, filmed on a fixed camera in the same gym.

## 3. Per-clip joint-level breakdown

- IMG_2412.MOV: 134 joint-level event(s) total
    nose: <5 frames=18, 5-20 frames=11
    left_eye: <5 frames=28, 5-20 frames=13
    right_eye: <5 frames=14, 5-20 frames=5, >20 frames=1
    right_ear: <5 frames=1
    left_elbow: <5 frames=14, 5-20 frames=2
    left_wrist: <5 frames=19, 5-20 frames=3
    right_wrist: <5 frames=1
    right_ankle: <5 frames=3, >20 frames=1
- IMG_2416.MOV: 97 joint-level event(s) total
    nose: <5 frames=19, 5-20 frames=1
    left_eye: <5 frames=15, 5-20 frames=3
    right_eye: <5 frames=22, 5-20 frames=2
    left_elbow: <5 frames=7
    left_wrist: <5 frames=16, 5-20 frames=3
    right_wrist: <5 frames=7, 5-20 frames=2
- IMG_2417.MOV: 117 joint-level event(s) total
    nose: <5 frames=16, 5-20 frames=8, >20 frames=2
    left_eye: <5 frames=20, 5-20 frames=4, >20 frames=4
    right_eye: <5 frames=27, 5-20 frames=9, >20 frames=1
    left_elbow: <5 frames=2, 5-20 frames=2
    left_wrist: <5 frames=11, 5-20 frames=5, >20 frames=1
    right_wrist: <5 frames=1
    left_knee: <5 frames=1, >20 frames=1
    left_ankle: <5 frames=1, >20 frames=1
- IMG_2419.MOV: 93 joint-level event(s) total
    nose: <5 frames=23, 5-20 frames=2, >20 frames=1
    left_eye: <5 frames=20, 5-20 frames=5, >20 frames=1
    right_eye: <5 frames=17, 5-20 frames=2, >20 frames=1
    left_elbow: <5 frames=1, 5-20 frames=1
    left_wrist: <5 frames=12, 5-20 frames=2
    right_wrist: <5 frames=2
    left_knee: <5 frames=1
    right_ankle: <5 frames=2
- IMG_2420.MOV: 117 joint-level event(s) total
    nose: <5 frames=24, 5-20 frames=7
    left_eye: <5 frames=13, 5-20 frames=3, >20 frames=2
    right_eye: <5 frames=20, 5-20 frames=8
    left_ear: <5 frames=3
    left_elbow: <5 frames=3, 5-20 frames=1
    right_elbow: <5 frames=4, 5-20 frames=1
    left_wrist: <5 frames=7
    right_wrist: <5 frames=8, 5-20 frames=7
    left_knee: <5 frames=5, 5-20 frames=1

## 4. Pipeline joints (8): left_shoulder, right_shoulder, left_wrist, right_wrist, left_hip, right_hip, left_ankle, right_ankle

Same events as section 2, restricted to joints used by the CoM proxy (D-005 follow-up) and the four-state classifier (ARCHITECTURE.md §4.4). Denominator for '% of used frames' is 5828 used frames per joint.

Per joint:

- left_shoulder: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 5828 used frames
- right_shoulder: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 5828 used frames
- left_wrist: 79 event(s) (65 / 13 / 1 by <5 / 5-20 / >20 frames), 238 low-confidence frame(s) = 4.08% of 5828 used frames
- right_wrist: 28 event(s) (19 / 9 / 0 by <5 / 5-20 / >20 frames), 90 low-confidence frame(s) = 1.54% of 5828 used frames
- left_hip: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 5828 used frames
- right_hip: 0 event(s) (0 / 0 / 0 by <5 / 5-20 / >20 frames), 0 low-confidence frame(s) = 0.00% of 5828 used frames
- left_ankle: 2 event(s) (1 / 0 / 1 by <5 / 5-20 / >20 frames), 31 low-confidence frame(s) = 0.53% of 5828 used frames
- right_ankle: 6 event(s) (5 / 0 / 1 by <5 / 5-20 / >20 frames), 38 low-confidence frame(s) = 0.65% of 5828 used frames

Per clip (pipeline joints combined):

- IMG_2412.MOV: 27 event(s), 98 low-confidence joint-frame(s) of 5728 used joint-frames (8 joints x 716 used frames)
- IMG_2416.MOV: 28 event(s), 72 low-confidence joint-frame(s) of 7144 used joint-frames (8 joints x 893 used frames)
- IMG_2417.MOV: 20 event(s), 118 low-confidence joint-frame(s) of 7840 used joint-frames (8 joints x 980 used frames)
- IMG_2419.MOV: 18 event(s), 34 low-confidence joint-frame(s) of 6600 used joint-frames (8 joints x 825 used frames)
- IMG_2420.MOV: 22 event(s), 75 low-confidence joint-frame(s) of 19312 used joint-frames (8 joints x 2414 used frames)

Total: 115 event(s) (90 / 22 / 3).
Longest pipeline-joint gap: IMG_2417.MOV left_ankle, frames 167-196 (30 frames, 1.002 s); touches clip boundary: False; next to a dropped/excluded frame: False

## Counts and n

- n = 5 clip(s), 5886 frame(s) in range, 5828 used for joint-level counts (detected, not excluded by list)
- Frames excluded by list: 18
- Whole-frame-drop events: 17
- Joint-level gap events (17 joints): 558
- Joint-level observations (used frames x 17 joints): 99076
- Longest joint-level gap: IMG_2417.MOV left_eye, frames 111-197 (87 frames, 2.905 s); touches clip boundary: False; next to a dropped/excluded frame: False

## Known limitation

A keypoint with confidence >= 0.5 during a real occlusion (a "confidently wrong" hallucinated position) would not be caught by this threshold at all. This is a limit of the statistic itself, not something this script attempts to detect.
