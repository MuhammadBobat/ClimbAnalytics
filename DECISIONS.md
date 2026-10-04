# DECISIONS.md — Decision Log

**Purpose:** every design/methodology decision that isn't purely mechanical gets an entry here, with its source. This file *is* the working draft of your dissertation's methodology chapter — write entries as if a reader outside your head needs to understand why you did something, not just what.

**Why this file instead of scattered comments or chat history:** chat history disappears between Claude Code sessions and doesn't survive to your write-up; comments in code get lost/buried as files are refactored; this file is the single durable, searchable, chronological record. Code comments should reference an entry ID here (e.g. `# see DECISIONS.md D-004`) rather than duplicating the full reasoning inline.

**Maintenance rule (also stated in `CLAUDE.md`):** any time a threshold, model, algorithm, or methodological choice is made — by you or by Claude Code — add an entry here *in the same session*, before moving on. Retroactively reconstructing "why did I pick 0.5s" six months later, from memory, is exactly what this file prevents.

---

## Entry template (copy this for new entries)

```
## D-0XX: [short title]
**Date:** YYYY-MM-DD
**Decision:** [what was decided, precisely]
**Rationale:** [why]
**Source(s):** [citation(s), or "no literature source — engineering heuristic, validate empirically"]
**Status:** proposed | implemented | validated | superseded
**Superseded by:** [D-0YY, if applicable]
```

---

## D-001: Clarify manual-labelling requirement with supervisor
**Date:** 2026-07-27
**Decision:** Raise with supervisor directly which specific downstream task ("manual labelling") they were anticipating: quantitative pose accuracy evaluation, hold detection, a learned classifier replacing rule-based interpretation, or human-rated climb comparisons for validation.
**Rationale:** as written, the current proposal (pretrained models, rule-based feedback, qualitative evaluation) does not require manual annotation. Supervisor's comment likely anticipates scope adjacent to, but not currently inside, the proposal.
**Source(s):** none — internal project clarification.
**Status:** proposed (awaiting supervisor response)

## D-002: Pose estimation model — ViTPose-L (COCO-25)
**Date:** 2026-07-27
**Decision:** Use ViTPose-L as the primary 2D pose estimator, paired with a bounding-box selection heuristic (nearest-to-previous-frame / largest-confidence) for single-climber tracking.
**Rationale:** highest accuracy of three benchmarked models on climbing-specific footage (86.6% vs. MediaPipe 83.5%, YOLOv8-pose 75.3%); fewest missing-detection frames (best temporal robustness); no real-time constraint in this project so ViTPose's slower inference is not a real cost; top-down architecture gives an explicit, controllable point for single-person locking.
**Source(s):** Maschek, A. & Schedl, D.C. (2025). "The Way Up: A Dataset for Hold Usage Detection in Sport Climbing." arXiv:2505.12854. Tables 4–6 (per-model accuracy, sensitivity, precision, missing-detection counts).
**Status:** validated (qualitative) on own footage — see D-002a. Quantitative accuracy comparison against MediaPipe/YOLOv8-pose on *this project's* footage (the literal reading of this follow-up) remains undone and stays deferred under D-010, since it would need ground-truth joint annotation.
**Follow-up required:** ~~D-002a — replicate a small version of this comparison on own footage; log result here regardless of outcome.~~ Completed 2026-10-04 as a qualitative check instead — see D-002a.

## D-002a: ViTPose-L validation on own gym footage — results
**Date:** 2026-10-04
**Decision:** ViTPose-L (via the D-011 implementation route) is confirmed fit for purpose on this project's own footage, based on per-keypoint confidence/missing-detection statistics plus direct visual inspection of the overlay video, across 5 clips (first 100 frames each via `--max-frames 100`) and one full clip. D-002's status is updated accordingly (see above). This is **not** the quantitative accuracy-replication exercise literally described in `ARCHITECTURE.md` §2's "Action required" line (ViTPose vs. MediaPipe vs. YOLOv8-pose numbers on own footage) — that needs ground-truth joint annotation, which `D-010` defers. What's validated here is narrower and more practical: does ViTPose track a real climber on real gym footage well enough to build the rest of the pipeline on top of, including under occlusion.

**Results** (from `validation/validate_vitpose.py`, `validation/output/summary.csv`):

| Clip | Frames tested | Missing-detection | Flagged frames | Mean keypoint confidence |
|---|---|---|---|---|
| IMG_2412.MOV | 100 | 2 (2%) | 0 | 0.871 |
| IMG_2416.MOV | 100 | 0 | 0 | 0.897 |
| IMG_2417.MOV | 100 | 0 | 0 | 0.876 |
| IMG_2419.MOV | 100 | 0 | 0 | 0.883 |
| IMG_2420.MOV | 100 | 1 (1%) | 0 | 0.873 |

Zero frames across all five clips tripped the script's own low-confidence/many-low-keypoints flags (`validation/output/*_flags.csv` empty except for the missing-detection rows above).

**Occlusion-specific finding, directly relevant to D-007:** `IMG_2412.MOV`, frame 64 — right wrist occluded behind the climber's head. Per-keypoint confidence dropped to 0.257 (the single lowest wrist confidence recorded across all tested frames; correctly below the planned `C < 0.5` occlusion threshold in `ARCHITECTURE.md` §3), but the predicted `(x, y)` position stayed smooth and plausible: `(611.6, 868.0)` at frame 63 → `(610.8, 904.2)` at frame 64 → `(596.0, 863.1)` at frame 65 — a small, continuous wobble, not a discontinuous jump or garbage coordinate. Confirmed directly from `IMG_2412_keypoints.csv` and matched against the author's visual read of the overlay video. ViTPose's heatmap regression appears to degrade gracefully under brief occlusion rather than failing outright — a good sign for how Tier 1 interpolation (D-007, <5-frame gaps) will interact with it: the signal it has to interpolate around during occlusion is a plausible estimate, not noise.

**Limitations — explicitly not covered by this entry:**
- Only the first 100 frames of 5 of the 6 available clips were tested (a few seconds each), not full climbs. One full-clip run exists (`Indoor_Bouldering_V3_Rock_Spot.mp4`, from 2026-08-02) but hasn't been reviewed against this same confidence/missing-detection lens yet.
- No ground-truth comparison against MediaPipe or YOLOv8-pose was performed on own footage — see decision text above. If quantitative pose evaluation is ever promoted out of D-010's deferred status, that comparison should be redone then, not retrofitted from this entry.
- Single occlusion event reviewed in detail; no systematic sweep yet of occlusion types (hand behind torso vs. behind head vs. climber facing away) across clips.
**Source(s):** direct output of `validation/validate_vitpose.py` against 5 own-gym clips, 2026-10-04 (`validation/output/summary.csv`, `*_flags.csv`, `*_keypoints.csv`); visual inspection of `IMG_2412_overlay.mp4` by the author.
**Status:** implemented — ViTPose-L confirmed fit for purpose on own footage (qualitative, partial-clip sample). Full-clip review and a broader occlusion sweep remain open, lower-priority follow-ups rather than blockers.

## D-003: Four-state motion detection method — z-score thresholding, not cusum
**Date:** 2026-07-27
**Decision:** Detect per-joint moving/static state via z-score thresholding on the velocity signal (adapted from RGB video, not IMU), not Boulanger's original cusum-based statistical change-point method.
**Rationale:** Boulanger's cusum method requires a supervised learning phase with manually annotated climbs to construct per-sensor statistical models — this reintroduces the exact manual-annotation burden the project is scoped to avoid. The z-score approach is self-calibrating per clip (uses the clip's own mean/std) and requires no training set.
**Source(s):** Boulanger, J., Seifert, L., Hérault, R. & Coeurjolly, J-F. (2015). "Automatic Sensor-Based Detection and Classification of Climbing Activities." IEEE Sensors Journal 16(3), 742–749 — Section III-A (cusum method, learning-phase requirement) and Section IV-A (four-state definitions, reused verbatim in this project). Beltrán, R., Richter, J. & Heinkel, U. — method referenced in Beltrán, Richter, Köstermeyer & Heinkel (2023) "Climbing Technique Evaluation by Means of Skeleton Video Stream Analysis," Sensors 23, 8216 — describes standard-score (z-score) motion segmentation on RGB-derived joint velocity signals as the detection technique adopted here.
**Status:** proposed

## D-004: Threshold and duration values — empirical sweep methodology
**Date:** 2026-07-27
**Decision:** All threshold and minimum-duration values in this project (z-score threshold for motion detection, hysteresis duration for state-flicker suppression, occlusion tier boundaries) will be determined by sweeping a small candidate range on real test footage and selecting the value that best matches independently judged (by the author, as an experienced climber) ground truth — not asserted from theory.
**Rationale:** this mirrors the only concrete precedent found in the literature for this exact type of decision.
**Source(s):** Maschek & Schedl (2025) determined their 0.5-second hold-usage confirmation threshold by empirically testing durations between 0 and 2 seconds and selecting the best-performing value — same methodology, note this is a precedent for *how to choose a threshold*, not a value reused directly, since their threshold solved a different problem (hold-usage confirmation, which this project does not implement — hold detection is out of scope, see `PROPOSAL.md` §7).
**Status:** proposed — actual swept values to be logged as sub-entries once footage is available (e.g. D-004a for the four-state z-score threshold, D-004b for hysteresis duration).

## D-005: Whole-body CoM as primary tracked quantity
**Date:** 2026-07-27
**Decision:** Track whole-body centre of mass as the central performance-related quantity, computed as a weighted average of tracked joint positions (weighting scheme: TBD, see follow-up).
**Rationale:** both key sources treat pelvis/CoM motion as the structurally central variable in climbing movement analysis, not an incidental one — Boulanger's entire four-state framework is defined by pelvis motion crossed with limb motion; the JFMK biomechanics review lists "Center of Mass Shift" as one of five fundamental climbing principles and reports an empirical association between low CoM-to-wall distance and competition ranking.
**Source(s):** Dech, S. & Kittel, R. (2026). "Biomechanical Principles and Techniques — A Systematization for Sport Climbing." J. Funct. Morphol. Kinesiol. 11, 103 — Section 3.1.3 (Principle of Center of Mass Shift), citing Werner, Gebert & Kauer (1999) on CoM-distance vs. competition rank. Sibella, F., Frosio, I., Schena, F. & Borghese, N.A. (2007). "3D analysis of the body center of mass in rock climbing." Human Movement Science 26(6), 841–852 — cited by both Boulanger (2015) and Dech & Kittel (2026) as the direct precedent for CoM as a climbing-specific measurement.
**Status:** proposed
**Follow-up required:** decide and log exact joint-weighting scheme (full anthropometric segment-mass model vs. simplified proxy) once implementation begins — see `ARCHITECTURE.md` §4.3.

## D-006: Smoothness metric — Log Dimensionless Jerk (LDLJ)
**Date:** 2026-07-27 (formula verified 2026-07-27, superseding the earlier unverified version of this entry)
**Decision:** Use LDLJ as the primary smoothness metric, applied to CoM velocity as the default signal. Formula:
```
LDLJ = -ln( (t2 - t1)^3 / v_peak^2  ·  ∫_{t1}^{t2} |d²v/dt²|² dt )
```
**Rationale:** LDLJ is the most established/widely used smoothness metric in the movement-science literature; different smoothness metrics can disagree, even in direction, on the same movement (e.g. one comparison study found LDLJ, NARJ, and zero-crossing count judged backward movements smoother while SPARC found the opposite), so picking one defensible, well-documented metric and stating the choice explicitly is preferable to an ad hoc formula.
**Source(s):** Originally developed by Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A. & Burdet, E. — cited as the standard reference across the secondary sources below. Formula (including the previously error-prone cubed duration term) confirmed directly via: Gulde, P. & Hermsdörfer, J. (2023). "Corrigendum: Smoothness metrics in complex movement tasks." Frontiers in Neurology, DOI: 10.3389/fneur.2023.1279682 — explicitly corrects a missing `^3` term in the published LDLJ formula, giving the exact corrected form used above. Cross-checked against: a comparison study (Sensors 23(3), 1158, 2023) computing LDLJ via "the Python code provided by Balasubramanian et al." on upper-limb movement data. **Still recommended before final implementation:** locate and skim the original Balasubramanian et al. paper directly (likely "On the analysis of movement smoothness," J. NeuroEngineering Rehabil., 2015) to confirm this is the same paper being referenced across secondary sources — the above verification is solid on the formula itself but the primary paper's exact title/venue has not been independently fetched and read in this session.
**Status:** implemented (formula) — primary-source read still pending as a minor follow-up, not a blocker.

## D-007: Occlusion handling — three-tier framework
**Date:** 2026-07-27
**Decision:** Adopt the three-tier occlusion framework in `ARCHITECTURE.md` §3 (cubic spline < 5 frames; Kalman filter 5–20 frames; root-lock + CoM substitution > 20 frames).
**Rationale:** single fixed-camera 2D tracking will have frequent occlusion (hands behind torso, limb crossing) — this is a known, literature-confirmed failure mode for climbing pose estimation specifically, not a hypothetical edge case.
**Source(s):** the three-tier structure itself is an original design for this project — **no direct literature source**, validate empirically. The underlying occlusion problem is confirmed by: Maschek & Schedl (2025), Results section — "longer occlusions of the hands behind the climber's upper body frequently led to detection inaccuracies."
**Status:** proposed — not yet validated against real occlusion events.

## D-008: Camera assumption — perpendicular, fixed
**Date:** 2026-07-27
**Decision:** Assume camera mounted exactly perpendicular to the wall plane, fixed position, no pan/tilt/zoom during a clip.
**Rationale:** simplifies all 2D displacement-based metrics to a consistent frame of reference; matches the setup used in the two closest precedent projects.
**Source(s):** Maschek & Schedl (2025) — camera positioned at a 90° angle to the wall after preliminary testing of alternative angles (their Fig. 2 comparison). Reiff, D. (2024). "Using Computer Vision to Assess Bouldering Performance" [Roboflow blog, non-peer-reviewed, cited as implementation precedent only, not academic literature] — fixed front-facing camera setup.
**Status:** implemented (assumption baked into scope, see `PROPOSAL.md` §4)

## D-009: Hold detection — confirmed out of scope
**Date:** 2026-07-27
**Decision:** No hold detection component, in any form, as core or stretch scope.
**Rationale:** every reviewed precedent implementing hold detection required a custom-trained, gym/dataset-specific model needing hand-annotated data, with generalisation problems across different walls even after that investment.
**Source(s):** Reiff (2024) [Roboflow blog] — custom object detection + custom colour classification models, both hand-annotated. Ludford, G. (2024). "Development of a climbing performance analysis tool using computer vision." The Plymouth Student Scientist 17(2) — trained a custom YOLOv7 hold-detection model on a 1717-image, 63,897-annotation dataset, achieving 87.6% mAP but requiring substantial dataset work; noted persistent issues with colour classification accuracy (72–88%) despite this investment. Maschek & Schedl (2025) — coach-annotated hold usage across only 22 videos still represented significant specialist labour.
**Status:** implemented (see `PROPOSAL.md` §7)

## D-010: Evaluation — qualitative now, quantitative deferred
**Date:** 2026-07-27
**Decision:** Ship qualitative evaluation only for the current project phase; explicitly leave open (not reject) a future quantitative pose-accuracy evaluation component.
**Rationale:** quantitative pose evaluation requires ground-truth annotation, which is a real cost — deferring keeps the option open without committing scope/time now.
**Source(s):** Maschek & Schedl (2025) — provides a template methodology if this is pursued later (coach-annotated ground truth, accuracy/precision/sensitivity via confusion matrix, temporal IoU).
**Status:** proposed

## D-011: ViTPose-L implementation route — HuggingFace `transformers`, not `mmpose`, plus YOLOv8n as the upstream bounding-box source
**Date:** 2026-07-27
**Decision:** Implement ViTPose-L via the HuggingFace `transformers` library (`VitPoseForPoseEstimation` + `AutoImageProcessor`, checkpoint `usyd-community/vitpose-plus-large`) rather than OpenMMLab's `mmpose`/`mmcv`. Pair it with YOLOv8n (`ultralytics`, COCO-pretrained, person class only, boxes only — not YOLOv8-**pose**) purely as the upstream person-bounding-box source, since ViTPose is a top-down method requiring an external detector (D-002).
**Rationale:** `mmpose`'s dependency stack (`mmcv`, `mmengine`, `mmdet`) requires version-matched, sometimes platform-compiled wheels and has a known history of fragile installs, particularly on newer Python versions and non-Linux/non-CUDA platforms. This project's dev environment is Python 3.14 on Apple Silicon (arm64, CPU-only, no CUDA) — well outside `mmcv`'s well-tested combinations — so `mmpose` was not attempted. `transformers`' ViTPose implementation is pure PyTorch with no compiled extensions and was installed and validated directly (see Status). For the person-bbox detector, the model card's own example uses RTDetr (`PekingU/rtdetr_r50vd_coco_o365`, ~166MB), but YOLOv8n (~6MB) is far cheaper per-frame on CPU and sufficient for the single-box-per-frame selection heuristic in `ARCHITECTURE.md` §2 — this is a substitution of detector, not of the ViTPose model itself, and is **not** the documented YOLOv8-**pose** fallback (that fallback is for replacing ViTPose's keypoint estimation entirely, not invoked here).
**Source(s):** no literature source — engineering/environment decision. Confirmed directly in this project's dev environment on 2026-07-27: `transformers` 5.14.1, `torch` 2.11.0 (CPU-only), `ultralytics` 8.4.108; `usyd-community/vitpose-plus-large` downloaded and loaded successfully (434M parameters); `yolov8n.pt` downloaded and loaded successfully.
**Status:** implemented — dependency install and model load validated in-environment. This is *not* the same as D-002a (accuracy replication against own footage), which still requires test clips and remains outstanding.
**Addendum (2026-07-27, found via smoke test, not docs):** `vitpose-plus-large` is a mixture-of-experts checkpoint (6 expert heads: COCO validation=0, AIC=1, MPII=2, AP-10K=3, APT-36K=4, COCO-WholeBody=5) and the forward pass raises `ValueError` unless an explicit `dataset_index` tensor is passed. `dataset_index=0` (COCO) is used throughout `validation/validate_vitpose.py`. This was not obvious from the model card's basic usage example (which uses the non-MoE `vitpose-base-simple` checkpoint) — only surfaced by actually running inference end-to-end, confirming it's worth doing that rather than trusting a paraphrased doc summary.

## D-012: Keypoint format correction — ViTPose outputs COCO-17, not "COCO-25"
**Date:** 2026-07-27
**Decision:** Correct `ARCHITECTURE.md` §1 and §2: ViTPose-L (and every `usyd-community` ViTPose/ViTPose+ checkpoint, including `vitpose-plus-large`) outputs the standard **17-keypoint COCO format** (Nose, L/R Eye, L/R Ear, L/R Shoulder, L/R Elbow, L/R Wrist, L/R Hip, L/R Knee, L/R Ankle). No "COCO 25-keypoint" ViTPose checkpoint exists among the surveyed releases — that figure does not correspond to any real ViTPose variant and has been removed from `ARCHITECTURE.md`.
**Rationale:** discovered directly while validating model loading for D-011 — `usyd-community/vitpose-plus-large`'s `config.id2label` returns exactly the 17 standard COCO joints, nothing more. This does not block any metric in `ARCHITECTURE.md` §4: CoM (hip/shoulder midpoints), per-joint velocity, and the four-state classifier (pelvis = hip midpoint, limbs = wrists/ankles) all use joints present in COCO-17. No scope or metric-definition change follows from this correction — it is a factual fix to the model's I/O spec only.
**Source(s):** direct inspection of `usyd-community/vitpose-plus-large`'s loaded config in this project's environment, 2026-07-27.
**Status:** implemented (`ARCHITECTURE.md` corrected in the same session)


## D-013: Cross-session scale calibration — open question, needed before attempt comparison is built
**Date:** 2026-08-02
**Decision:** Not yet decided. Flagging that the whole-climb attempt-to-attempt comparison feature (`PROPOSAL.md` §6) likely needs some form of pixel-to-real-world scale calibration before its metrics are meaningful, since two filming sessions of the "same" perpendicular setup won't have the camera at exactly identical distance/position from the wall each time. Without calibration, raw pixel-space displacement/velocity numbers from two different sessions aren't directly comparable, even though both individually satisfy the D-008 perpendicular-camera assumption.
**Rationale:** Maschek & Schedl needed a homography transformation to get real-world distances even with a perpendicular, deliberately-positioned camera, specifically because camera position varied slightly session-to-session — the same condition this project's attempt-comparison feature will face. They found a lightweight approach (four manually-identified corner reference points) was nearly as accurate as a much denser calibration grid (1.51px vs. 1.20px mean error), suggesting a full calibration rig isn't necessary — a cheap per-session reference-point step may be sufficient.
**Source(s):** Maschek, A. & Schedl, D.C. (2025). "The Way Up: A Dataset for Hold Usage Detection in Sport Climbing." arXiv:2505.12854 — "Dataset" section, homography methodology and four-corner vs. 350-point calibration accuracy comparison.
**Status:** proposed — not required for core pipeline (which only needs internal, single-session consistency); required before implementing the attempt-comparison feature specifically. Revisit when that feature is picked up.
 
## D-014: tIoU as the planned metric if quantitative evaluation (D-010) is picked up
**Date:** 2026-08-02
**Decision:** If/when the deferred quantitative evaluation in D-010 is pursued, use temporal Intersection-over-Union (tIoU) as the primary metric for scoring how well the four-state classifier's timing matches ground truth — not just whether a state was detected at some point, but whether it was detected during the correct frame range.
**Rationale:** tIoU is a standard, well-precedented metric for exactly this kind of "did the detected time range match reality" question in activity/state recognition, and gives a graded temporal-alignment score rather than a coarse yes/no.
**Source(s):** Maschek & Schedl (2025) — tIoU formula (`|Tp ∩ Tg| / |Tp ∪ Tg|`) and its use alongside accuracy/precision/sensitivity, both at a 0.0 and 0.5 tIoU threshold, to separately capture "was it detected at all" vs. "was the timing right." Originally from Heilbron et al. (2015), "ActivityNet," cited by Maschek & Schedl as the source of the tIoU metric in activity recognition generally.
**Status:** proposed — dependent on D-010 being promoted from deferred to active; no action needed until then.

## D-016: Foot keypoints — ViTPose-WholeBody and MediaPipe both explored, staying on ViTPose-L body-only for now
**Date:** 2026-10-04
**Decision:** Confirmed ViTPose-L's current checkpoint cannot produce foot/toe keypoints, explored two routes to get them (ViTPose-WholeBody via `dataset_index`, and MediaPipe Pose as a second model), and decided to stick with ViTPose-L's standard 17-keypoint body output for now rather than adopt either. Foot placement/smoothness metrics are not currently in `PROPOSAL.md` scope (§3, §5) — this entry is a feasibility note for if/when that changes, not a scope change itself.

**Route 1 — `dataset_index=5` (the "COCO-WholeBody" MoE expert on `vitpose-plus-large`): does not work.** Verified directly that the model's decode head outputs exactly 17 heatmap channels (`[1, 17, 64, 48]`) regardless of which of the 6 expert indices (0–5) is passed — `dataset_index` only selects which backbone expert processes image features, it does not change the fixed output head, which this checkpoint always wires to standard COCO-17. Separately, and worse: confidence scores produced under `dataset_index=5` came out in the ~2.0–3.1 range instead of the expected [0,1] (e.g. `IMG_2412.MOV` mean_confidence 2.56 vs. 0.87 under index 0) — not merely lower-quality, but off-scale and unusable by every threshold in `validate_vitpose.py`, which assumes [0,1]. No checkpoint in the `usyd-community` HuggingFace `transformers` port exposes a true COCO-WholeBody (133-keypoint) decode head; that only exists in OpenMMLab's original `mmpose`-based ViTPose-WholeBody configs, and installing `mmpose`/`mmcv` to reach it reintroduces exactly the dependency fragility D-011 already ruled out on this Python 3.14 / Apple Silicon / no-CUDA environment (confirmed on 2026-10-04: `mmcv` 2.2.0, `mmpose` 1.3.2, `mmengine` 0.10.7 are all ~2 years stale on PyPI, built against PyTorch versions well behind this project's `torch` 2.11). `validate_vitpose.py`'s `VITPOSE_COCO_DATASET_INDEX` has been reverted to `0`.

**Route 2 — MediaPipe Pose (`mediapipe` package), separate model, exploratory only: works, but not adopted.** MediaPipe Pose has real heel/foot-index (toe) landmarks. A scratchpad-only script (not committed; not part of any module boundary) ran MediaPipe's "heavy" Pose Landmarker on `IMG_2412.MOV` (150 frames, CPU): mean visibility left_heel 0.780, right_heel 0.724, left_foot_index 0.667, right_foot_index 0.593 (min values as low as 0.192–0.339) — real foot tracking, but visibly weaker and noisier than ViTPose's body-joint confidence on the same footage (D-002a: ~0.87–0.90 mean, rarely below 0.5). Hit one unrelated but notable engineering snag along the way: `mediapipe` 1.0.x hard-crashes (native abort, not a Python exception) on macOS Apple Silicon specifically — confirmed upstream bug, [google-ai-edge/mediapipe#6356](https://github.com/google-ai-edge/mediapipe/issues/6356); pinning to `mediapipe==0.10.35` avoids it. Worth remembering if this is revisited: `mediapipe` must stay pinned to `0.10.35`, not left to float to latest.

**Why not adopt MediaPipe (even just for feet) right now:** two reasons, independent of the scope question. (1) D-002's own literature comparison already found MediaPipe meaningfully behind ViTPose on climbing footage generally (83.5% vs. 86.6% accuracy, and far more missing-detection frames — 150+ consecutive vs. 2–10), and today's foot-specific numbers don't contradict that pattern. (2) Running a second full model per frame alongside ViTPose-L is a real architecture and runtime-cost decision (every frame processed twice), not a drop-in addition — not something to fold in quietly alongside a pose-model validation session.
**Rationale:** both routes were evaluated with actual runs against real footage (not estimated), per CLAUDE.md rule 9. Decision to not adopt either is a scope/cost call, not a finding that either route is impossible — MediaPipe-for-feet specifically remains a live, validated-as-feasible option (modulo the version pin) if foot placement is later promoted into `PROPOSAL.md` scope, most likely as an augmentation (MediaPipe feet + ViTPose body) rather than a replacement, given ViTPose's clear edge everywhere else.
**Source(s):** direct experimentation in this project's environment, 2026-10-04 — `transformers`/`usyd-community/vitpose-plus-large` dataset_index sweep; `mediapipe` 0.10.35 Pose Landmarker ("heavy") on own footage; PyPI version check for `mmcv`/`mmpose`/`mmengine`; GitHub issue google-ai-edge/mediapipe#6356.
**Status:** implemented (exploration complete, decision made to stay on ViTPose-L body-only). Revisit if/when foot placement or smoothness is promoted into core scope.
 