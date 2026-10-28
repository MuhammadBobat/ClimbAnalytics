# BlazePose z-value feasibility spike — REPORT

**Exploratory spike. Not part of the pipeline. Not a logged decision.**

Every number below comes from code actually run in this folder (`run_spike.py`,
`analyze.py`, `make_overlays.py`), against this project's own footage. Where
something couldn't be measured, that's stated explicitly rather than guessed.

## Setup

- Model: MediaPipe Pose Landmarker, "heavy" variant (BlazePose GHUM), Tasks API.
- `mediapipe` 0.10.35 (pinned — 1.0.x crashes on macOS Apple Silicon, per
  `DECISIONS.md` D-016; confirmed still true for this spike's own fresh install).
- Licence: **Apache 2.0** (confirmed via `pip show mediapipe`, not memory).
- `PoseLandmarker` / `PoseLandmarkerOptions` API confirmed directly against
  the installed package (not recalled from memory) — see `README.md`.
- Clips: `footage/test/{IMG_2412,IMG_2416,IMG_2417,IMG_2419,IMG_2420}.MOV`,
  first 100 frames each — matched exactly to the frame range already covered
  by the saved ViTPose-L keypoints in `validation/output/`, so the comparison
  in Steps 2-3 is apples-to-apples. `Indoor_Bouldering_V3_Rock_Spot` has a
  saved ViTPose-L CSV but its source video is no longer in `footage/test/`,
  so it's excluded from this spike entirely — not silently dropped.

**One real bug found and fixed along the way:** reusing one `PoseLandmarker`
instance across multiple clips throws `ValueError: Input timestamp must be
monotonically increasing`, because the Tasks API's VIDEO running mode
enforces strictly increasing timestamps *per instance*, and each clip's
timestamps restart from 0. Fixed by creating a fresh landmarker per clip
(`run_spike.py`). Found by actually running it on a second clip, not
anticipated from the docs.

## Step 2 — Detection rate & visibility

| Clip | BlazePose frames detected | ViTPose-L frames detected | Frames lost vs. ViTPose-L | Mean visibility (wrists/ankles/hips/shoulders) |
|---|---|---|---|---|
| IMG_2412 | 100/100 | 98/100 | 0 | 0.858 |
| IMG_2416 | 100/100 | 100/100 | 0 | 0.729 |
| IMG_2417 | 100/100 | 100/100 | 0 | 0.752 |
| IMG_2419 | 100/100 | 100/100 | 0 | 0.779 |
| IMG_2420 | 100/100 | 99/100 | 0 | 0.693 |

BlazePose detected a pose in every frame across all 5 clips — it never lost
a frame that ViTPose-L caught (ViTPose-L itself missed 1-2 frames in two
clips, per the existing D-002a record). Mean visibility for the four limb
joints this spike cares about sits in the 0.69-0.86 range — lower than
ViTPose-L's own ~0.87-0.90 mean confidence from D-002a, though the two
scores aren't the same measurement (different models, different calibration
— don't average or directly compare them numerically, same caveat D-016
already noted).

## Step 3 — 2D agreement with ViTPose-L

| Clip | Median distance (px) | 95th pct distance (px) | n comparisons |
|---|---|---|---|
| IMG_2412 | 12.1 | 44.9 | 784 |
| IMG_2416 | 8.8 | 25.4 | 800 |
| IMG_2417 | 12.9 | 52.4 | 800 |
| IMG_2419 | 11.6 | 62.1 | 800 |
| **IMG_2420** | **21.5** | **65.8** | 792 |

On a ~1080×1920 frame, medians of 9-13px are a small fraction of the frame
and plausibly just normal model-to-model disagreement for the same real
joint. **IMG_2420 disagrees noticeably worse than the other four clips** —
both its median (21.5px) and its per-joint breakdown stand out:

| Joint | IMG_2412 | IMG_2416 | IMG_2417 | IMG_2419 | **IMG_2420** |
|---|---|---|---|---|---|
| left_ankle | 13.9 | 9.5 | 11.0 | 15.1 | **42.0** |
| left_hip | 8.1 | 18.0 | 18.6 | 19.0 | **42.5** |
| right_ankle | 24.2 | 12.9 | 14.3 | 11.9 | 17.5 |
| right_hip | 16.3 | 13.8 | 16.0 | 20.1 | 16.2 |

IMG_2420's left ankle and left hip are the specific outliers (42px each) —
not a whole-clip problem, a localised one. Flagging this rather than
explaining it: this spike didn't investigate why (candidates would include
occlusion, a body orientation BlazePose handles worse, or a genuine
ViTPose-L error on that side — not distinguished here).

## Step 4 — z-signal diagnostics (all root-relative, from world landmarks)

**Stationarity definition used throughout** (engineering heuristic for this
spike, not a project-wide threshold): a joint is "stationary" at frame *t*
if its normalized image-space displacement from frame *t-1* is below 0.01
(1% of frame width/height), sustained for ≥5 consecutive frames.

| Clip | Torso lean z range | Sign flips (100 frames) | Hip-stationary segments | Mean z-std within stationary segments |
|---|---|---|---|---|
| IMG_2412 | -0.213 to 0.024 | **13** | 1 | 0.079 |
| IMG_2416 | -0.025 to 0.070 | 5 | 1 | 0.022 |
| IMG_2417 | -0.036 to 0.063 | 3 | 1 | 0.027 |
| IMG_2419 | -0.030 to 0.143 | 2 | 1 | 0.044 |
| IMG_2420 | -0.138 to 0.004 | 2 | 3 | 0.012 |

**Sign/consistency check — done by actually looking, not inferred from the
numbers alone.** IMG_2412 flips sign 13 times in 100 frames — implausibly
often for a real postural reversal during continuous climbing (roughly once
every 7 frames). I pulled the actual video frames at one flip (frames 10→11)
and compared them side by side: **the climber's pose is visually static
between those two frames** — same position, same grip, no visible lean
change — while torso_lean_z reads `+0.00192 → -0.00053 → +0.00238` across
frames 9-11. That's sub-centimeter-scale noise crossing zero, not a real
lean reversal. **Most of the 13 flips in IMG_2412 are very likely jitter
near a near-zero value, not genuine postural changes** — reported as a
direct observation on one checked instance, not generalised to claim this
for every flip in every clip without checking each one.

## Step 5 — Untested hypothesis: ankle z stability during visually-stationary foot segments

Hypothesis as given: while an ankle is planted, hip distance from the wall
≈ -(ankle z - hip z). **Not claimed as measured here** — no contact
timeline exists in this spike, so only the narrower, explicitly-scoped
check was run: is ankle z stable during segments where that ankle's 2D
image position is visually stationary (same 0.01/5-frame definition, applied
per-ankle)?

| Clip | Left ankle: stationary segments | Left ankle: mean z-std | Right ankle: stationary segments | Right ankle: mean z-std |
|---|---|---|---|---|
| IMG_2412 | 2 | 0.027 | 4 | 0.033 |
| IMG_2416 | 2 | 0.052 | 3 | 0.023 |
| IMG_2417 | 4 | 0.044 | 5 | 0.032 |
| IMG_2419 | 2 | 0.035 | 1 | **0.104** |
| IMG_2420 | 6 | 0.020 | 2 | 0.012 |

Ankle z-std during visually-stationary segments is mostly in the 0.02-0.05
range (world-landmark units, roughly metres, hip-relative) — i.e. the ankle
position wobbles by a few centimetres even while the foot looks visually
planted in 2D. One clear outlier: IMG_2419's right ankle at 0.104 (only 1
qualifying segment, so this is a single short window, not a stable
estimate). **This is a jitter/noise characterisation only — it says nothing
about whether z actually tracks true distance to the wall**, which would
need the real contact timeline and/or ground truth this spike doesn't have.

One supporting (not conclusive) observation from the overlay videos: in
frames where the climber's hips are visibly pulled back from the wall while
the feet stay planted near it, both ankle-z-relative-to-hip values read
clearly positive (e.g. `+0.20`, `+0.21` in one `IMG_2412` frame) — the right
qualitative direction for "feet closer to the wall than hips" under
MediaPipe's z-sign convention. This is one visually-checked frame, not a
systematic validation.

## Step 6 — Overlay videos

`output/IMG_2412_blazepose_overlay.mp4`, `IMG_2416_blazepose_overlay.mp4`,
`IMG_2420_blazepose_overlay.mp4` — skeleton + live torso-lean-z / ankle-z
readout, 100 frames each. Spot-checked frame 50 of `IMG_2412` directly: skeleton
correctly tracks the climber's actual joints (confirmed by eye, same
verification habit as the earlier ViTPose-L overlay bug in this project).

## Step 7 — Ground truth

Not provided this session. Error-against-ground-truth was not computed —
nothing here should be read as an accuracy number against real-world
measurements, only as internal consistency/jitter/agreement-with-ViTPose-L.

## Recommendation

**Need ground truth before any adoption decision — not yet usable, not
ruled out either.**

What this spike does support:
- BlazePose runs reliably on this footage (100/100 frames detected, every
  clip) and its 2D joint positions agree with ViTPose-L to within ~9-13px
  median on 4 of 5 clips (IMG_2420 is a flagged outlier, localised to
  left hip/ankle).
- The z-signal has internal structure, not pure noise — it responds
  qualitatively in the expected direction in at least the one visually-checked
  frame, and most clips show ~1 stable hip-stationary segment with
  reasonably tight jitter (0.01-0.08 range).

What it does not support:
- No measurement against real-world distance exists, so "usable
  distance-to-wall signal" is still unverified. Sign flips in IMG_2412
  specifically show the raw z-signal needs smoothing/filtering before any
  downstream use — using it raw would read as 13 lean reversals in under
  4 seconds, which is not real.
- The ankle-stability numbers characterise jitter, not accuracy — a low std
  during a stationary segment doesn't mean the z-value is *correct*, only
  that it's *consistent* with itself.

**If this is pursued further:** get real ground truth (side-camera clips or
tape-measured static poses, per the original Step 7 ask) before drawing any
conclusion stronger than "the signal exists and isn't pure noise." A draft
decision-log addendum is in `DRAFT_D016_addendum.md` for review — not
applied to `DECISIONS.md`.
