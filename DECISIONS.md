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
**Status:** closed (2026-10-04) — author has made an executive decision not to pursue supervisor clarification on this point; scope is being decided by the author's own judgement going forward. Recorded as closed, not deleted, since it documents a real change in how scope gets decided on this project.

## D-002: Pose estimation model — ViTPose-L (17-keypoint COCO — see D-012; this heading originally said "COCO-25", corrected 2026-10-04)
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
**Addendum (2026-10-04):** Directly confirmed from Beltrán et al.'s own methodology text (Section 4.5, not a search snippet this time): "The technique used is based on a *standard score* on the joint velocity signal sampled at each frame," with the z-score given in their Eq. 9 as `z = (v − μ) / σ` — matches this entry's description exactly, closing the "grounded in a search snippet, not a full read" gap previously flagged. One refinement worth noting for D-004, not yet adopted: Beltrán et al. don't use a simple crossover of the z-score signal against `threshold_z`; they additionally compare the resulting n-σ graph against 50% of its own peak value, specifically to filter out detection-noise jitter before calling a phase transition. Flagging as a candidate refinement if D-004's empirical sweep finds the plain crossover too flicker-prone — not implemented now, just noted so it isn't lost.
**Source(s) confirmed via:** Beltrán, R., Richter, J., Köstermeyer, D. & Heinkel, U. (2023). "Climbing Technique Evaluation by Means of Skeleton Video Stream Analysis." Sensors 23(19), 8216. [PMC10574944](https://pmc.ncbi.nlm.nih.gov/articles/PMC10574944) — Section 4.5, Eq. 9.

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
**Source(s):** Originally developed by Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A. & Burdet, E. — cited as the standard reference across the secondary sources below. Formula (including the previously error-prone cubed duration term) confirmed directly via: Gulde, P. & Hermsdörfer, J. (2023). "Corrigendum: Smoothness metrics in complex movement tasks." Frontiers in Neurology, DOI: 10.3389/fneur.2023.1279682 — explicitly corrects a missing `^3` term in the published LDLJ formula, giving the exact corrected form used above. Cross-checked against: a comparison study (Sensors 23(3), 1158, 2023) computing LDLJ via "the Python code provided by Balasubramanian et al." on upper-limb movement data. **Primary-source identity — confirmed 2026-10-04:** Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A. & Burdet, E. (2015). "On the analysis of movement smoothness." *Journal of NeuroEngineering and Rehabilitation* 12:112. DOI: [10.1186/s12984-015-0090-9](https://doi.org/10.1186/s12984-015-0090-9). [Imperial Spiral record](https://spiral.imperial.ac.uk/entities/publication/0140ac66-2946-4c82-a1c3-fc546c8cb6f1) — title, journal, volume, year, and full author list confirmed directly, matching the "likely" venue guessed in the original version of this entry. This closes the identity question — confirmed to be the same paper referenced across the secondary sources below, not a different Balasubramanian smoothness paper.
**Honest limit of this check:** the exact formula *text* inside the primary PDF itself could not be independently re-read in this session — every mirror tried (PMC, HAL/Sorbonne, BioMedCentral) returned a reCAPTCHA wall, a 403, or a rate limit rather than article text. The formula above still rests on the two secondary-source cross-checks already in this entry (the Frontiers corrigendum, which exists specifically to correct an error in this paper's own published formula, and the Sensors comparison study's re-implementation of "the Python code provided by Balasubramanian et al."), not a fresh primary read. Narrower than "verified against primary text" — state that distinction if this is cited in the methodology chapter.
**Status:** implemented (formula, secondary-source verified) — primary-source identity confirmed; primary-source formula text unread due to access barriers, not treated as a blocker given the existing secondary verification.

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
**Status:** superseded by D-017 (2026-10-04) — original reasoning (annotation cost, generalisation) stands as a historical record of why hold detection was excluded at the time; D-017 reverses this on the grounds that assisted-annotation tooling has lowered the annotation-cost premise.
**Superseded by:** D-017

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
**Status:** implemented (`ARCHITECTURE.md` corrected in the same session). **Addendum (2026-10-04):** D-002's own heading in this file still said "(COCO-25)" despite this entry — a stale factual claim in a title, not just leftover prose, so fixed directly rather than left as historical colour. See D-002's heading.


## D-013: Cross-session scale calibration — open question, needed before attempt comparison is built
**Date:** 2026-08-02
**Decision:** Not yet decided. Flagging that the whole-climb attempt-to-attempt comparison feature (`PROPOSAL.md` §6) likely needs some form of pixel-to-real-world scale calibration before its metrics are meaningful, since two filming sessions of the "same" perpendicular setup won't have the camera at exactly identical distance/position from the wall each time. Without calibration, raw pixel-space displacement/velocity numbers from two different sessions aren't directly comparable, even though both individually satisfy the D-008 perpendicular-camera assumption.
**Rationale:** Maschek & Schedl needed a homography transformation to get real-world distances even with a perpendicular, deliberately-positioned camera, specifically because camera position varied slightly session-to-session — the same condition this project's attempt-comparison feature will face. They found a lightweight approach (four manually-identified corner reference points) was nearly as accurate as a much denser calibration grid (1.51px vs. 1.20px mean error), suggesting a full calibration rig isn't necessary — a cheap per-session reference-point step may be sufficient.
**Source(s):** Maschek, A. & Schedl, D.C. (2025). "The Way Up: A Dataset for Hold Usage Detection in Sport Climbing." arXiv:2505.12854 — "Dataset" section, homography methodology and four-corner vs. 350-point calibration accuracy comparison.
**Status:** proposed — not required for core pipeline (which only needs internal, single-session consistency); required before implementing the attempt-comparison feature specifically. Revisit when that feature is picked up. **Update (2026-10-07):** D-020 proposes layout matching of detected hold layouts as the way to resolve this, with the manual four-corner method as fallback; still not decided or implemented.
 
## D-014: tIoU as the planned metric if quantitative evaluation (D-010) is picked up
**Date:** 2026-08-02
**Decision:** If/when the deferred quantitative evaluation in D-010 is pursued, use temporal Intersection-over-Union (tIoU) as the primary metric for scoring how well the four-state classifier's timing matches ground truth — not just whether a state was detected at some point, but whether it was detected during the correct frame range.
**Rationale:** tIoU is a standard, well-precedented metric for exactly this kind of "did the detected time range match reality" question in activity/state recognition, and gives a graded temporal-alignment score rather than a coarse yes/no.
**Source(s):** Maschek & Schedl (2025) — tIoU formula (`|Tp ∩ Tg| / |Tp ∪ Tg|`) and its use alongside accuracy/precision/sensitivity, both at a 0.0 and 0.5 tIoU threshold, to separately capture "was it detected at all" vs. "was the timing right." Originally from Heilbron et al. (2015), "ActivityNet," cited by Maschek & Schedl as the source of the tIoU metric in activity recognition generally.
**Status:** proposed — dependent on D-010 being promoted from deferred to active; no action needed until then.

## D-015: Known-groups validation as a supplementary check for threshold selection
**Date:** 2026-10-04 (reconstructed — `ARCHITECTURE.md` §7 already referenced this entry by ID, but it was never actually written up here; writing it now from that section's own text rather than leaving the reference dangling)
**Decision:** Treat "known-groups validation" as a supplementary validity check for the four-state classifier's threshold selection (D-004), run alongside — not instead of — the qualitative evaluation already committed to in D-010. Method: group test clips by self-reported climber experience level, compute two per-clip summary stats from the four-state timeline (`% time pelvis-immobile` vs. `pelvis-moving`; `% time in Hold Interaction`, used as an approximate proxy for an exploratory/performatory split), and check whether the pattern across experience groups goes in the literature-predicted direction. The `threshold_z` value that produces the cleanest such separation is adopted — this doubles as D-004's actual threshold-selection methodology, not a separate pass. Full I/O spec (per-clip inputs/outputs, suggested standalone script location) lives in `ARCHITECTURE.md` §7, not duplicated here.
**Rationale:** without ground-truth annotation (deferred under D-010), there is no way to directly score the four-state classifier's accuracy. "Known-groups validity" — checking whether a measure can distinguish between groups that theory predicts should differ — is a standard technique used generally in instrument validation for exactly this situation (no gold-standard criterion available). Applying it here gives a second, more structured line of evidence for picking `threshold_z`, on top of the author's own subjective judgement (which D-004 already relies on alone).
**Explicit limitation — state this honestly, do not gloss over it:** `% time in Hold Interaction` is used only as an approximate, convenience proxy for the exploratory/performatory movement distinction from the climbing motor-control literature associated with Seifert (co-author on Boulanger, Seifert, Hérault & Coeurjolly (2015), already cited in D-003). This project has **no validated operationalisation** linking "Hold Interaction" (pelvis immobile + ≥1 limb moving, per the four-state model) to the formal exploratory/performatory construct — the mapping is this project's own approximation for a plausibility check, not a claim that the two are the same measurement. Do not present a known-groups result as validating the exploratory/performatory distinction itself in the write-up.
**CoM-to-wall exclusion:** CoM distance-to-wall (a depth/z-axis quantity) is excluded from this check because it is not measurable at all with a single perpendicular 2D camera (D-008) — a hard consequence of the camera setup every metric in this project already works within, not a limitation specific to this check.
**Source(s):** "known-groups validity" as a general technique — no single climbing-specific source; standard concept in measurement/instrument validation methodology generally, applied here rather than taken from a climbing-specific precedent. The exploratory/performatory distinction traces to the climbing motor-control literature associated with Seifert; the specific mapping onto this project's Hold Interaction state is this project's own approximation, not sourced from a paper that defines it that way — flagged above, not hidden.
**Status:** proposed — not required for the core pipeline (`ARCHITECTURE.md` §1–§6); required as a prerequisite check before `threshold_z` (D-004) is finalised.

## D-016: Foot keypoints — ViTPose-WholeBody and MediaPipe both explored, staying on ViTPose-L body-only for now
**Date:** 2026-10-04
**Decision:** Confirmed ViTPose-L's current checkpoint cannot produce foot/toe keypoints, explored two routes to get them (ViTPose-WholeBody via `dataset_index`, and MediaPipe Pose as a second model), and decided to stick with ViTPose-L's standard 17-keypoint body output for now rather than adopt either. Foot placement/smoothness metrics are not currently in `PROPOSAL.md` scope (§3, §5) — this entry is a feasibility note for if/when that changes, not a scope change itself.

**Route 1 — `dataset_index=5` (the "COCO-WholeBody" MoE expert on `vitpose-plus-large`): does not work.** Verified directly that the model's decode head outputs exactly 17 heatmap channels (`[1, 17, 64, 48]`) regardless of which of the 6 expert indices (0–5) is passed — `dataset_index` only selects which backbone expert processes image features, it does not change the fixed output head, which this checkpoint always wires to standard COCO-17. Separately, and worse: confidence scores produced under `dataset_index=5` came out in the ~2.0–3.1 range instead of the expected [0,1] (e.g. `IMG_2412.MOV` mean_confidence 2.56 vs. 0.87 under index 0) — not merely lower-quality, but off-scale and unusable by every threshold in `validate_vitpose.py`, which assumes [0,1]. No checkpoint in the `usyd-community` HuggingFace `transformers` port exposes a true COCO-WholeBody (133-keypoint) decode head; that only exists in OpenMMLab's original `mmpose`-based ViTPose-WholeBody configs, and installing `mmpose`/`mmcv` to reach it reintroduces exactly the dependency fragility D-011 already ruled out on this Python 3.14 / Apple Silicon / no-CUDA environment (confirmed on 2026-10-04: `mmcv` 2.2.0, `mmpose` 1.3.2, `mmengine` 0.10.7 are all ~2 years stale on PyPI, built against PyTorch versions well behind this project's `torch` 2.11). `validate_vitpose.py`'s `VITPOSE_COCO_DATASET_INDEX` has been reverted to `0`.

**Route 2 — MediaPipe Pose (`mediapipe` package), separate model, exploratory only: works, but not adopted.** MediaPipe Pose has real heel/foot-index (toe) landmarks. A scratchpad-only script (not committed; not part of any module boundary) ran MediaPipe's "heavy" Pose Landmarker on `IMG_2412.MOV` (150 frames, CPU): mean visibility left_heel 0.780, right_heel 0.724, left_foot_index 0.667, right_foot_index 0.593 (min values as low as 0.192–0.339) — real foot tracking, but visibly weaker and noisier than ViTPose's body-joint confidence on the same footage (D-002a: ~0.87–0.90 mean, rarely below 0.5). Hit one unrelated but notable engineering snag along the way: `mediapipe` 1.0.x hard-crashes (native abort, not a Python exception) on macOS Apple Silicon specifically — confirmed upstream bug, [google-ai-edge/mediapipe#6356](https://github.com/google-ai-edge/mediapipe/issues/6356); pinning to `mediapipe==0.10.35` avoids it. Worth remembering if this is revisited: `mediapipe` must stay pinned to `0.10.35`, not left to float to latest.

**Why not adopt MediaPipe (even just for feet) right now:** two reasons, independent of the scope question. (1) D-002's own literature comparison already found MediaPipe meaningfully behind ViTPose on climbing footage generally (83.5% vs. 86.6% accuracy, and far more missing-detection frames — 150+ consecutive vs. 2–10), and today's foot-specific numbers don't contradict that pattern. (2) Running a second full model per frame alongside ViTPose-L is a real architecture and runtime-cost decision (every frame processed twice), not a drop-in addition — not something to fold in quietly alongside a pose-model validation session.
**Rationale:** both routes were evaluated with actual runs against real footage (not estimated), per CLAUDE.md rule 9. Decision to not adopt either is a scope/cost call, not a finding that either route is impossible — MediaPipe-for-feet specifically remains a live, validated-as-feasible option (modulo the version pin) if foot placement is later promoted into `PROPOSAL.md` scope, most likely as an augmentation (MediaPipe feet + ViTPose body) rather than a replacement, given ViTPose's clear edge everywhere else.
**Source(s):** direct experimentation in this project's environment, 2026-10-04 — `transformers`/`usyd-community/vitpose-plus-large` dataset_index sweep; `mediapipe` 0.10.35 Pose Landmarker ("heavy") on own footage; PyPI version check for `mmcv`/`mmpose`/`mmengine`; GitHub issue google-ai-edge/mediapipe#6356.
**Status:** implemented (exploration complete, decision made to stay on ViTPose-L body-only). Revisit if/when foot placement or smoothness is promoted into core scope.

## D-017: Hold detection — hybrid gym-specific detector + colour-matching, superseding D-009
**Date:** 2026-10-04
**Decision:** Reverse D-009. Add a hold-detection component: (1) a single-class object detector ("hold" vs. background) fine-tuned from a pretrained **YOLOv8n** checkpoint (picked over YOLOv11n for consistency with the YOLOv8n already in use for person-box detection, D-011 — same library, same weights-management pattern, no new dependency) on hand-annotated frames from this project's own gym, to localise hold bounding boxes against lighting, climber occlusion, and background clutter; (2) classical HSV colour-matching, applied within each detected hold's bounding box against this gym's known route-colour palette, to assign each hold to a route/grade (this gym grades by hold colour). Detection (localisation) and colour-matching (classification) are kept as separate steps rather than training a multi-class detector, since colour-matching is cheap, interpretable, and needs no per-colour training examples.

**Why this reverses D-009's original reasoning:** D-009 ruled out hold detection specifically because of hand-annotation cost. That premise has changed: **Roboflow's model-assisted labelling** (auto-suggested boxes from a partially-trained model, corrected rather than drawn from scratch) meaningfully lowers the per-image labelling cost that was the actual blocker in D-009 — not the modelling difficulty, not the generalisation problem (still unresolved, see Limitation 1). Tool named explicitly here to close the follow-up this entry originally left open; no measured labelling-time figure is claimed — if/when annotation actually happens, log the real time taken as a D-017a sub-entry rather than relying on this generic claim.

**Limitation 1 — gym-specific, conflicts with `PROPOSAL.md` §3's generalisability goal for this component:** a detector fine-tuned on this gym's holds will not generalise to a different wall/gym without re-annotating and retraining. Scoped specifically to this one component — pose estimation, metrics, and feedback generation remain gym-agnostic. Do not present this component as satisfying the generalisability goal in the write-up.

**Limitation 2 — reintroduces model fine-tuning/training:** required a corresponding `CLAUDE.md` rule 4 update, narrowly scoped to this one detector (done in the same session as this entry — see `CLAUDE.md`). Not a blanket removal of the fine-tuning prohibition.

**Colour palette — explicit placeholder, not fabricated:** this project's gym's actual route-colour palette (which hex/HSV ranges correspond to which grades) is not yet recorded anywhere in this repo. Per `CLAUDE.md` rule 9, this must come from the author's own knowledge of their gym, not be guessed — left as `GYM_COLOUR_PALETTE = None  # TBD, see DECISIONS.md D-017a` in code until supplied.
**Rationale:** keeps the harder, label-expensive part of the problem (localisation under lighting/occlusion/clutter) to a single-class detector — an easier learning problem than the multi-class/custom-colour-classifier models reviewed in D-009 — while keeping route/grade assignment in cheap, interpretable, non-learned HSV matching. Avoids Ludford's (2024) reported colour-classification accuracy problems (72–88%), since this project's matching only needs one known gym's fixed colour set, not generalised conditions.
**Source(s):** D-009's original citations (Reiff 2024, Ludford 2024, Maschek & Schedl 2025) remain the evidentiary basis for the generalisation/annotation-cost concerns, reweighed rather than superseded. Roboflow model-assisted labelling — no specific measured labelling-time citation logged; see D-017a follow-up.
**Status:** proposed — author has authorized starting implementation (2026-10-04). **Update (2026-10-04):** this entry originally flagged supervisor sign-off (`PROPOSAL.md` §9) as a required gate before/while building. The author has since made an explicit executive decision to proceed without it. Recorded here as what actually happened, not as a retroactive claim that sign-off took place.
**Supersedes:** D-009

## D-017a: Gym colour→grade mapping — partially supplied (grade bands known, HSV calibration still pending)
**Date:** 2026-10-04
**Decision:** Author supplied this gym's real colour→grade-band mapping directly, now encoded in `hold_colours_config.py`'s `GYM_COLOUR_GRADE_BANDS`:
```
green=V0, white=V0-V1, blue=V1-V3, black=V2-V4, pink=V2-V5,
red=V3-V5, purple=V5-V7, yellow=V7-V8, orange=V8+
```
Several entries are grade *bands*, not single values — colour alone does not pin an exact grade at this gym. `hold_detection.py`'s output reports the band as given, never a fabricated single number narrower than this mapping actually supports.

**What this does *not* resolve:** the numeric HSV ranges for each colour name remain unmeasured — "green" is a name, not an HSV range, and the actual pixel values depend on this gym's specific holds and lighting. `GYM_COLOUR_HSV_REFERENCE` in `hold_colours_config.py` stays `None` per colour until measured via `training/calibrate_hold_colours.py` against real sample photos. Until then, `hold_detection.py`'s colour-matching returns `"unassigned"` for every detection — this is working as intended, not a bug, per `CLAUDE.md` rule 9.
**Rationale:** the grade-band mapping is a known fact the author could simply state; the HSV ranges are not a fact anyone can simply state without measuring real holds under this gym's actual lighting — conflating the two would mean fabricating the harder half of this entry.
**Source(s):** author's direct knowledge of their own gym, supplied 2026-10-04. HSV ranges: none yet — pending `training/calibrate_hold_colours.py` measurement against real sample photos (log that measurement as an addendum to this entry once it happens, including sample-photo count per colour).
**Status:** proposed — grade-band mapping implemented; HSV calibration outstanding, blocking real (non-"unassigned") colour-matching output.
 
## D-018: Promote hold-relative feedback — contact timeline, hold-to-hold segmentation, and derived timing/foot/balance metrics
**Date:** 2026-10-07
**Decision:** Supersede the exclusion of hold-relative feedback written into `PROPOSAL.md` §7 and `ARCHITECTURE.md` §8's intro (D-017 alone — localisation plus colour/route assignment — gave static hold positions, not a per-hold interaction timeline; this entry adds the missing piece). Add to core scope, in this order of dependency:
1. **Keypoint-to-hold contact timeline** (`hold_contact.py`): for each frame, which hold each wrist/ankle keypoint is in contact with, aggregated into touch events `{hold_id, limb, start_frame, end_frame, route_label}`.
2. **Hold-to-hold segmentation of existing metrics:** CoM velocity/displacement, four-state fractions and LDLJ computed per interval between successive contact events, in addition to per clip. LDLJ needs a **minimum segment duration** below which it is not reported (value TBD — determine by the D-004 sweep methodology, do not assert one; segments flagged as dynamic moves should be reported with that context, since LDLJ's smoothness reading is not equivalent across static and ballistic moves).
3. **Timing metrics derived from the timeline (no new model):** climb time (first to last contact), pace (contact events per second), pre-climb pause (first frame with a detected climber to first contact), dwell per hold (contact duration), flight time (gap between a limb's release and its next contact).
4. **Foot usage ratio and foot readjustment count:** share of contact events made by ankles; a readjustment is a same-ankle release and re-contact on the same hold within a window (window TBD, empirical). Uses the ankle keypoints ViTPose-L already outputs; **no toe/fingertip keypoints** (D-016 stays on body-only — a foot keypoint change would be its own decision).
5. **Balance proxy:** horizontal offset of the CoM from the centre of the supporting contact points in the image plane (normalisation TBD).

**Real dependencies — not buildable today:** needs the four-state classifier calibrated (Phase 4) and the hold detector trained and colour-calibrated (Phase 0.5.2–0.5.5).

**Contact provenance rule:** each keypoint carries whether it was observed, interpolated (D-007 tier 1/2) or stale (tier 3). Contacts decided from tier-3 keypoints are recorded as **unknown**, not guessed, and each clip reports the share of its contact timeline that rests on observed or short-gap keypoints. The tier boundaries are themselves unvalidated heuristics (D-007), so contact accuracy inherits that uncertainty. Contact accuracy must be measured on own footage before any feature below is trusted (task 0.5.7).

**Known risks, stated up front:**
- Hold IDs depend on the per-clip layout merge in `ARCHITECTURE.md` §8.1 (sampled frames, IoU clustering); if that merge is noisy, IDs may not be stable enough for a clean timeline. Validate empirically.
- 2D proximity cannot distinguish a hand near a hold from a hand gripping it, and wrist/ankle keypoints are not the actual contact points. A 2026 preprint reports markedly better contact detection with fingertip/toe keypoints from a different pose model (Sapiens) than with wrist/ankle keypoints from ViTPose-L and YOLOv8-pose. Read only as a fetched summary, preprint status, not independently verified here; licence and compute cost unchecked. Treat as an open spike, not a decision.
- Foot readjustment (item 4) and the balance proxy (item 5) have **no literature source — engineering heuristics, validate empirically**. Do not present them as established measures.
- Feedback wording stays comparative/descriptive (D-010, D-015 limitations): none of these metrics is ground-truth validated as an indicator of climbing quality.

**Distance-to-wall is not replaced and remains out of scope:** it is a depth quantity not measurable with the D-008 single perpendicular camera (D-015). The in-plane items above (balance proxy, foot usage) are partial proxies for the same intent, not a measurement of it. Testing BlazePose z as a 2.5D route is a separate optional spike, not part of this decision.

**Rationale:** with hold detection in scope the per-hold timeline is the natural next layer, and segmenting the existing metrics by hold-to-hold interval makes feedback specific ("the move between holds 3 and 4") rather than whole-clip. It also reduces the LDLJ dyno problem by isolating a ballistic move in its own segment.
**Source(s):** Frontiers in Psychology (2020) climbing fluency study, DOI 10.3389/fpsyg.2020.00249 — climbing time and immobility ratio as fluency indicators (page read via fetch, first 100,000 of ~102,000 characters; authors and exact title **not yet confirmed — confirm before citing**). "Training-Free Hold-Usage Detection in Sport Climbing with Foundation Pose Models", arXiv:2609.30026v1 (preprint) — climb time, pace, dwell, foot-usage ratio and flight time derived from a hold-usage event stream. Boulanger et al. (2015) per-hold touch framing already cited in D-003. Items 4–5: no literature source, per above.
**Status:** proposed — build blocked on Phase 4 and Phase 0.5.2–0.5.5.
**Supersedes:** the hold-relative-feedback exclusion in `PROPOSAL.md` §7 and `ARCHITECTURE.md` §8 (not a numbered decision).

## D-019: Route auto-definition and route progress
**Date:** 2026-10-07
**Decision:** Do not require manual start/finish annotation. Derive the route from the hold layout (`ARCHITECTURE.md` §8.1) and the contact timeline:
- **Route membership:** holds whose colour-matched route label (D-017a) equals the route the climber touches, spatially clustered, selecting the cluster the climber contacts. This rests on the author's knowledge that same-colour problems do not overlap on this wall — **an assumption to validate on real footage**, not a tested fact.
- **Start holds:** the first one or two holds the climber touches (one hold if both hands go on the same hold).
- **Finish hold:** the topmost hold of the route cluster, assuming the climber completes the problem.
- **Route progress:** number or fraction of route holds touched, and the furthest hold reached; for a failed attempt it is reported as progress short of the finish, not as a finish.
The gym's own start/finish markers (small silver round metal tags) are **not** relied on: they are unlikely to be reliably visible to the detector.
**Known risks:** a missed hold shortens the route and overstates progress (detector recall must be measured); "furthest hold" by vertical position fails on traverses (ordering rule TBD — decide when hold_contact.py exists, log as an addendum); colour→route membership depends on HSV calibration (D-017a, task 0.5.5), and Ludford (2024) reported 72–88% colour classification accuracy (cited in D-009); "first touched" start holds depend on the contact timeline being correct at the start of a clip.
**Rationale:** avoids per-problem manual input, consistent with the hybrid detector's purpose (D-017).
**Source(s):** author's knowledge of own gym (2026-10-07). No literature source for the route-definition heuristics — engineering heuristics, validate empirically.
**Status:** proposed — blocked on D-018's dependencies.

## D-020: Per-move attempt-to-attempt comparison by hold sequence — stretch goal; layout matching proposed as the D-013 resolution
**Date:** 2026-10-07
**Decision:** Replace the whole-climb comparison (`PROJECT_PLAN.md` 7.2, `PROPOSAL.md` §6) with a per-move comparison aligned by hold sequence as the stretch feature. Whole-climb aggregate deltas remain as the summary line of the same feature, not a separate build. **Stretch only — nothing in the core pipeline depends on it.**
**Alignment:** within one clip, hold identity comes from the per-clip layout (§8.1) and needs no extra step. Across clips, the detected hold layouts are matched (hold centres, colour as a constraint, robust fit such as RANSAC) to estimate a plane homography between sessions, which also gives hold identity across sessions. This proposes **layout matching as the resolution of D-013** instead of manual reference points; manual four-corner points (Maschek & Schedl) remain the fallback. The homography is exact only for points on the wall plane; keypoints on the climber (in front of the wall) are warped approximately — a limitation to state, not yet tested.
**What needs alignment:** only comparisons of pixel-valued quantities (displacement, velocity) and hold identity across sessions. Time-based measures (D-018 item 3), four-state fractions and dimensionless quantities (LDLJ, ratios such as path length over straight-line distance between holds) do not.
**Not a replacement for distance-to-wall:** per-move comparison is an in-plane comparison tool; it does not measure depth (see D-018).
**Wording:** comparative only ("longer than your previous attempt on this move"), never evaluative (D-010, D-015).
**Rationale:** hold sequence is a more natural alignment than DTW and makes advice for the next attempt move-specific.
**Source(s):** D-013 (Maschek & Schedl 2025, homography and four-corner calibration). Layout matching: no literature source — engineering proposal, validate empirically.
**Status:** proposed — stretch, after the core pipeline and D-018/D-019 features work.

## D-021: BlazePose (MediaPipe) z-values as a distance-to-wall signal — exploratory spike, inconclusive, not adopted
**Date:** 2026-10-07
**Decision:** Ran as an isolated feasibility spike (`experiments/blazepose_z_spike/`, branch `spike/blazepose-z`), outside the runtime pipeline — not a pipeline change. BlazePose z is **not adopted** and is not a project goal. Distance-to-wall stays out of scope: it is a depth quantity not measurable with the single perpendicular 2D camera this project assumes (D-008, D-015; `PROPOSAL.md` §7). The result is **inconclusive, not a rejection** — no ground truth was available in this session, so no accuracy figure exists either way.

**Setup:** `mediapipe` 0.10.35 (pinned — 1.0.x crashes on macOS Apple Silicon, per D-016, re-confirmed on this spike's own fresh install), `PoseLandmarker` Tasks API, "heavy" model, Apache 2.0 licence (confirmed via `pip show`, not memory). 5 clips × first 100 frames, matched exactly to the frame range already covered by the saved ViTPose-L keypoints in `validation/output/`, so the comparison is apples-to-apples. `Indoor_Bouldering_V3_Rock_Spot` was excluded — it has a saved ViTPose-L CSV, but its source video is no longer in `footage/test/`.

**Findings, taken directly from `experiments/blazepose_z_spike/REPORT.md`:**
- **Detection rate:** BlazePose detected a pose in 100/100 frames on all 5 clips — it never lost a frame ViTPose-L caught (ViTPose-L itself: 98/100 on IMG_2412, 100/100 on IMG_2416/2417/2419, 99/100 on IMG_2420, per D-002a). Mean visibility (wrists/ankles/hips/shoulders) per clip: IMG_2412 0.858, IMG_2416 0.729, IMG_2417 0.752, IMG_2419 0.779, IMG_2420 0.693 — lower than ViTPose-L's own ~0.87–0.90 mean confidence, though the two scores are not the same measurement (different models/calibration, not directly comparable).
- **2D agreement with ViTPose-L** (median / 95th-percentile pixel distance, n comparisons): IMG_2412 12.1 / 44.9px (n=784), IMG_2416 8.8 / 25.4px (n=800), IMG_2417 12.9 / 52.4px (n=800), IMG_2419 11.6 / 62.1px (n=800), **IMG_2420 21.5 / 65.8px (n=792) — flagged outlier**. IMG_2420's per-joint breakdown shows this is localised, not whole-clip: left_ankle 42.0px and left_hip 42.5px (vs. right_ankle 17.5px, right_hip 16.2px on the same clip). The spike did not investigate the cause (candidates: occlusion, a body orientation BlazePose handles worse, or a genuine ViTPose-L error on that side — not distinguished).
- **Torso-lean z sign flips (100 frames per clip):** IMG_2412 13 (range -0.213 to 0.024), IMG_2416 5, IMG_2417 3, IMG_2419 2, IMG_2420 2. For IMG_2412 specifically, frames 10→11 were pulled and visually compared: the climber's pose was static between them, while torso_lean_z read `+0.00192 → -0.00053 → +0.00238` across frames 9–11 — sub-centimetre-scale noise crossing zero, not a real lean reversal. Reported as a direct observation on this one checked instance, not generalised to every flip in every clip.
- **Ankle z jitter during visually-stationary segments** (mean std, world-landmark units ≈ metres, hip-relative): mostly 0.02–0.05 across clips/ankles; one outlier, IMG_2419's right ankle at 0.104 (only 1 qualifying segment — a single short window, not a stable estimate). This characterises jitter only, not accuracy.
- **One supporting, non-systematic observation:** in one visually-checked `IMG_2412` frame where the hips were visibly pulled back from the wall while the feet stayed planted near it, both ankle-z-relative-to-hip values read clearly positive (`+0.20`, `+0.21`) — the qualitatively expected direction under MediaPipe's z-sign convention, from one frame, not a systematic validation.

**Limits, stated honestly:**
- Only 100 frames per clip were tested, not full climbs.
- No ground truth exists, so everything above is internal consistency and agreement-with-ViTPose-L only — no real-world accuracy claim is supportable.
- The hypothesis as originally posed (while an ankle is planted, hip distance from the wall ≈ -(ankle z - hip z)) was **not validated** — no contact timeline existed in this spike to identify actual plant/release events, so only the narrower check (ankle z stability during *visually*-stationary segments) was run.
- "Stationary" was defined by this spike (normalized image-space displacement < 0.01, sustained ≥5 frames) — an engineering heuristic for this spike only, not a project-wide threshold (same status as D-004's not-yet-swept values).
- D-002's literature finding that MediaPipe loses more frames than ViTPose-L on climbing footage generally is **not overturned** by this spike's 100-frame, 5-clip sample.

**Implementation note worth keeping:** reusing one `PoseLandmarker` instance across multiple clips throws `ValueError: Input timestamp must be monotonically increasing` — the Tasks API's VIDEO running mode enforces strictly increasing timestamps per instance, and each clip's timestamps restart from 0. Fix: create a fresh landmarker per clip. Found by actually running it on a second clip, not anticipated from the docs.

**What would change this:** real ground truth (a side-view camera, or tape-measured static poses) plus smoothing/filtering of the raw z series — the sign-flip finding above shows raw z is not usable without it. Until ground truth exists, **do not use z in any metric.**
**Rationale:** per CLAUDE.md rule 9, every number above comes from code actually run against this project's own footage in `experiments/blazepose_z_spike/`, not estimated.
**Source(s):** `experiments/blazepose_z_spike/REPORT.md` (direct experimentation, own footage, 2026-10-07); `mediapipe` package (Apache 2.0, confirmed via `pip show`). No literature source for z-accuracy in climbing specifically — none found, consistent with the gap already noted when this spike was commissioned.
**Status:** proposed — inconclusive, not adopted. Cross-reference D-016 (foot keypoints / prior MediaPipe exploration), D-015 and D-018 (why depth/distance-to-wall matters but stays out of scope).
