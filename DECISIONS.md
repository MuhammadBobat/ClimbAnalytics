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
**Status:** proposed — must be validated on own gym footage before finalising (see D-002a below).
**Follow-up required:** D-002a — replicate a small version of this comparison on own footage; log result here regardless of outcome.

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
