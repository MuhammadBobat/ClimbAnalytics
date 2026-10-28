# Dissertation notes

Working notes for the dissertation write-up: one section per topic, one reference register, one gaps table. **This file records; it does not decide.** Decisions live in `DECISIONS.md`; this file points to them.

## How to read and maintain this file

- **Status** is copied from `DECISIONS.md` as it appears there (statuses as at the last commit touching that file, `87c8f0f`). It is never upgraded here. Where an entry carries several status lines (D-005, D-019, D-022), the entry's own first status line is shown.
- **Results are file paths**, not retyped numbers. A number appears here only where it is needed to say what a result cannot show, and then with its source path.
- **"not recorded in the repo"** means the repo (`DECISIONS.md`, the git log, `evaluation/`, `validation/`) does not say. Nothing is reconstructed or guessed; every such item is also listed in the gaps table at the end so it can be filled in from memory.
- **Failures and negative results are recorded**, not only successes. A result that did not come from a real run is never recorded. A citation that was not checked is marked unverified.
- Update the matching topic section at the end of every development step (`CLAUDE.md` rule 11).

## Topics

| Topic | Decisions |
|---|---|
| A. Scope and evaluation plan | D-001, D-010, D-014 |
| B. Pose model and keypoint set | D-002, D-002a, D-011, D-012, D-016, D-023 |
| C. Person detection and single-climber tracking | D-024 |
| D. Camera assumption, scale calibration and depth | D-008, D-013, D-021 |
| E. Occlusion and gap handling | D-007 |
| F. Centre of mass | D-005, D-025 |
| G. Smoothness metric | D-006 |
| H. Four-state classifier and thresholds | D-003, D-004, D-015 |
| I. Hold detection and colour calibration | D-009, D-017, D-017a, D-022 |
| J. Hold-relative analysis and attempt comparison | D-018, D-019, D-020 |

D-025 (the full 14-joint segment-model CoM is not adopted as it stands; the trunk-centroid proxy remains the working CoM) is in topic F. D-024 (tracking failure) is in topic C; the tracking fix will be an addendum or a later ID.

---

## A. Scope and evaluation plan

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-001 | Clarify manual-labelling requirement with supervisor | closed (2026-10-04) — author has made an executive decision not to pursue supervisor clarification on this point; scope is being decided by the author's own judgement going forward. Recorded as closed, not deleted, since it docume … |
| D-010 | Evaluation — qualitative now, quantitative deferred | proposed |
| D-014 | tIoU as the planned metric if quantitative evaluation (D-010) is picked up | proposed — dependent on D-010 being promoted from deferred to active; no action needed until then. |

**Question:** Who settles the project's scope, and how is the system to be judged? D-001 asked what "manual labelling" the supervisor anticipated; D-010 fixes the evaluation plan (qualitative now, quantitative deferred); D-014 names the metric (temporal IoU) to use if a quantitative evaluation is picked up.

**What was tried, in order (dates and commit hashes):** 
1. D-001 and D-010 are dated 2026-07-27 and first appear in git in `5181280` (2026-08-02, the commit that added `DECISIONS.md`). D-014 is dated 2026-08-02 and first appears in `dd343be` (2026-10-04).
2. D-001 was closed with the author's executive decision not to pursue supervisor clarification (entry dated 2026-10-04; first in git in `6f1eb27`, 2026-10-07).
3. D-010 is applied as qualitative evaluation only: visual inspection of overlays (D-002a, `dd343be`, 2026-10-04) and the author's review of overlay clips (D-025, `87c8f0f`, 2026-10-28). No quantitative evaluation was started.

**Data and sample sizes:** None for D-001, D-010 or D-014 themselves. There is no ground-truth annotation of any kind in the repo. The footage used for the qualitative checks is described under topic B.

**Results (file paths):** None for this topic. The qualitative evidence sits under topic B (`validation/output/summary.csv`), topic E (`evaluation/3.1/`) and topic F (`evaluation/2.2b/output_full/`).

**Decision and status:** D-001 closed; D-010 proposed; D-014 proposed (dependent on D-010 being promoted). Statuses as in the table above.

**What it cannot show:** Nothing quantitative about accuracy: no ground truth exists, so no accuracy, precision, sensitivity or tIoU figure exists for any component. What the supervisor's original comment meant, and whether it was ever answered, is not recorded in the repo (D-001 says only that it was closed).

**References:** register rows R1, R14.

## B. Pose model and keypoint set

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-002 | Pose estimation model — ViTPose-L (17-keypoint COCO — see D-012; this heading originally said "COCO-25", corrected 2026-10-04) | validated (qualitative) on own footage — see D-002a. Quantitative accuracy comparison against MediaPipe/YOLOv8-pose on *this project's* footage (the literal reading of this follow-up) remains undone and stays deferred under D-010, … |
| D-002a | ViTPose-L validation on own gym footage — results | implemented — ViTPose-L confirmed fit for purpose on own footage (qualitative, partial-clip sample). Full-clip review and a broader occlusion sweep remain open, lower-priority follow-ups rather than blockers. |
| D-011 | ViTPose-L implementation route — HuggingFace `transformers`, not `mmpose`, plus YOLOv8n as the upstream bounding-box source | implemented — dependency install and model load validated in-environment. This is *not* the same as D-002a (accuracy replication against own footage), which still requires test clips and remains outstanding. |
| D-012 | Keypoint format correction — usyd-community ViTPose checkpoints output COCO-17, not "COCO-25" (scoped 2026-10-28, see addendum) | implemented (`ARCHITECTURE.md` corrected in the same session). **Addendum (2026-10-04):** D-002's own heading in this file still said "(COCO-25)" despite this entry — a stale factual claim in a title, not just leftover prose, so f … |
| D-016 | Foot keypoints — ViTPose-WholeBody and MediaPipe both explored, staying on ViTPose-L body-only for now | implemented (exploration complete, decision made to stay on ViTPose-L body-only). Revisit if/when foot placement or smoothness is promoted into core scope. |
| D-023 | Evaluate the easy_ViTPose coco_25 checkpoint as a foot-keypoint candidate | proposed — evaluation not started. |

**Question:** Which 2D pose model, through which implementation route, outputting which keypoints, and does it track a real climber on this project's own footage well enough to build on? Can foot or toe keypoints be obtained?

**What was tried, in order (dates and commit hashes):** 
1. 2026-07-27, D-002: ViTPose-L chosen from the literature (Maschek & Schedl 2025, R1), paired with a nearest-box / largest-confidence person selection. D-011: implemented via Hugging Face `transformers` with `usyd-community/vitpose-plus-large` and YOLOv8n as the box source, not `mmpose` (not attempted: Python 3.14 on Apple Silicon, CPU-only). Smoke test found the mixture-of-experts checkpoint needs `dataset_index` (0 = COCO used throughout). D-012: the checkpoint's config gives 17 COCO keypoints, not 25. First in git in `5181280` (2026-08-02), together with `validation/validate_vitpose.py`.
2. 2026-10-04, `dd343be`: D-002a, ViTPose-L run on five own clips (first 100 frames each, `--max-frames 100`) and visually inspected; one occlusion event reviewed (IMG_2412 frame 64, right wrist confidence 0.257, position stayed smooth). D-016: `dataset_index=5` tried for foot keypoints and rejected (the decode head stays at 17 channels, and confidences came out off-scale, about 2.0 to 3.1); MediaPipe Pose "heavy" tried on IMG_2412 (150 frames) and not adopted (weaker visibility than ViTPose-L); `mmcv`/`mmpose`/`mmengine` PyPI check; `mediapipe` 1.0.x crash on Apple Silicon, pinned to 0.10.35.
3. 2026-10-04, `e7140a7` and `39d60aa`: stale "COCO-25" claims corrected in `ARCHITECTURE.md` and in D-002's heading.
4. 2026-10-28, `8870e73`: the author's re-check of Maschek & Schedl found they ran a COCO-plus-feet "coco_25" checkpoint (easy_ViTPose); D-002's accuracy numbers come from that checkpoint, not the one this project runs (D-002, D-012 and D-016 addenda). D-023 opened to evaluate that checkpoint as a foot-keypoint candidate; evaluation not started.
5. 2026-10-28: ViTPose-L re-run on the five full clips (`validation/output_full/`), with the output-folder guard and the `n_person_boxes` column added to `validate_vitpose.py` (`3fd56f0`). The first 100 frames of the full run reproduced the earlier 100-frame run exactly (recorded in the D-007 whole-clip addendum, `00ec2a0`).

**Data and sample sizes:** Five own clips (IMG_2412, 2416, 2417, 2419, 2420). D-002a: first 100 frames of each (500 frames). Whole-clip run: 5,886 frames (718, 917, 980, 850 and 2,421 frames; `validation/output_full/summary.csv`), of which 5,846 had a person box (detector drops: 40). MediaPipe feet exploration: IMG_2412, 150 frames. The sixth, downloaded clip (Rock_Spot) is not the author's footage and is excluded from all later analysis (D-002a note, 2026-10-28). Recording device, resolution, climber and route per clip: not recorded in the repo.

**Results (file paths):** `validation/output/summary.csv` and `*_flags.csv` (100-frame run; table in D-002a); `validation/output_full/summary.csv` (whole clips); D-016 for the foot-keypoint exploration (its numbers are in the entry text; the exploration script was scratchpad-only and is not in the repo); library versions in `evaluation/3.1/output_full/manifest.txt`. All of `validation/output*/` is gitignored.

**Decision and status:** D-002 validated (qualitative); D-002a implemented; D-011 implemented; D-012 implemented; D-016 implemented (stay on ViTPose-L body-only); D-023 proposed (not started). Statuses as in the table above.

**What it cannot show:** D-002a is not a quantitative accuracy comparison (no ground truth; it stays deferred under D-010), covers only partial clips for the 2026-10-04 run, and reviewed one occlusion event in detail. D-002's accuracy figures belong to a different checkpoint. Nothing here says the easy_ViTPose coco_25 weights load, what their 25 keypoints are, or how accurate they are on this footage (D-023, unverified). The MediaPipe comparison is one clip and compares confidence and visibility scores that are not the same measurement.

**References:** register rows R1, R21, R22, R23, R24, R25.

## C. Person detection and single-climber tracking

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-024 | Single-climber lock-on — observed failure and options | proposed |

**Question:** How is the single climber kept locked across frames, and where does the lock-on fail? D-002 and D-011 fix the box source (YOLOv8n, person class) and the selection rule; D-024 records an observed failure.

**What was tried, in order (dates and commit hashes):** 
1. 2026-07-27, D-002 and D-011 (`5181280`): box from YOLOv8n; take the highest-confidence box on the first frame, then the box nearest the previous centre; no distance limit.
2. 2026-10-28, `3fd56f0`: `validate_vitpose.py` refuses to overwrite an existing output folder and logs `n_person_boxes` (people detected per frame) at the end of its keypoint and flag CSVs; tracking logic unchanged. Whole-clip run (`validation/output_full/`).
3. 2026-10-28: boxes checked per clip. The only confirmed wrong-person lock is IMG_2419 frames 597-614 (a bystander on the mat; the climber's box was dropped for frames 592-596, then stayed weak). Extra boxes elsewhere were mostly holds, partial people or duplicate boxes. Those frames are listed in `validation/frame_exclusions.csv` and excluded from the gap statistics (list introduced in `e0a597e`).
4. 2026-10-28: D-024 logged with the failure mechanism and the fix options, none chosen (`6517c14`), then the detector-confidence observation for IMG_2419 (`599ef09`). The tracking fix is to be an addendum or a later ID; none is built.

**Data and sample sizes:** Five whole clips (5,886 frames). Frame-level counts of frames with more than one person box, per clip: not recorded in the repo (the `n_person_boxes` column exists in the gitignored `validation/output_full/*_keypoints.csv`). The frames inspected visually (IMG_2419 frame 605, IMG_2420 frames 1850 and 2390, and detector re-runs on IMG_2419 frames 590-615) are described in D-024; the images and the re-run script are not in the repo.

**Results (file paths):** `validation/frame_exclusions.csv` (the one exclusion, with its reason and mechanism); D-024 in `DECISIONS.md`; `validation/output_full/summary.csv` for detection rates; `evaluation/3.1/output_full/frame_losses.csv` for detector drops and the excluded span.

**Decision and status:** D-024 proposed (observation and options only; no decision on the fix). Statuses as in the table above.

**What it cannot show:** It is not a tracking error rate: it rests on one confirmed failure and a handful of spot checks, with no per-frame identity labels. No distance gate or other fix has been tried, so nothing is known about their effect. D-024 notes that a distance gate alone would stop the wrong lock but still leave the climber untracked for about the same frames.

**References:** register rows R25.

## D. Camera assumption, scale calibration and depth

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-008 | Camera assumption — perpendicular, fixed | implemented (assumption baked into scope, see `PROPOSAL.md` §4) |
| D-013 | Cross-session scale calibration — open question, needed before attempt comparison is built | proposed — not required for core pipeline (which only needs internal, single-session consistency); required before implementing the attempt-comparison feature specifically. Revisit when that feature is picked up. **Update (2026-10 … |
| D-021 | BlazePose (MediaPipe) z-values as a distance-to-wall signal — exploratory spike, inconclusive, not adopted | proposed — inconclusive, not adopted. Cross-reference D-016 (foot keypoints / prior MediaPipe exploration), D-015 and D-018 (why depth/distance-to-wall matters but stays out of scope). |

**Question:** May pixel-space displacement and velocity be used (a fixed, perpendicular camera, D-008)? What scale calibration would cross-session comparison need (D-013)? Can a depth or distance-to-wall signal be obtained from BlazePose z (D-021)?

**What was tried, in order (dates and commit hashes):** 
1. 2026-07-27, D-008 (`5181280`): assume a camera exactly perpendicular to the wall, fixed, with no pan, tilt or zoom. D-013 (dated 2026-08-02, first in git in `dd343be`): cross-session scale calibration left as an open question. D-020 (`6f1eb27`, 2026-10-07) proposes layout matching as its resolution.
2. 2026-10-07, `6f1eb27`: D-021 logged. A BlazePose (MediaPipe, "heavy") z-value spike ran in isolation (`experiments/blazepose_z_spike/`, branch `spike/blazepose-z`) and was inconclusive. The spike's source files were committed to that branch and pushed on 2026-10-28 (`cdf40f2`), with a README stating it cannot be reproduced from a clean clone.
3. 2026-10-28, `8f3f9bd` and `0b4a3f9`: the camera-setup claims in PROPOSAL §4 and D-008 were corrected after the sources were checked. Maschek & Schedl filmed with a handheld camera at 90° to an 8.5 m, non-inclined wall (their "Data acquisition" section); Reiff (2024) used a single ground-level, front-facing camera and does not say whether it was fixed or perpendicular. The fixed camera is therefore this project's own assumption.

**Data and sample sizes:** D-021: five clips, first 100 frames each, matched to the saved ViTPose-L frames (`validation/output/`), with 784-800 comparisons per clip (numbers in D-021). D-008 and D-013 have no data. Whether the author's own footage was filmed on a tripod, and the camera's angle to the wall: not recorded in the repo (the assumption is stated, not measured).

**Results (file paths):** D-021 in `DECISIONS.md` (findings quoted from `experiments/blazepose_z_spike/REPORT.md`, which exists on branch `spike/blazepose-z`, not on `main`). Camera wording: `DECISIONS.md` D-008, `PROPOSAL.md` §4 and register rows R1 and R18.

**Decision and status:** D-008 implemented (an assumption in the scope, not a tested result); D-013 proposed (open); D-021 proposed (inconclusive, not adopted). Statuses as in the table above.

**What it cannot show:** D-021 has no ground truth, covers 100 frames per clip, did not test the original hypothesis, and found raw z unusable without smoothing; it supports no accuracy claim either way. Nothing tests that this project's footage meets the perpendicular, fixed-camera assumption or how far the camera moved between sessions, so every pixel-space number rests on that assumption (D-008, D-013).

**References:** register rows R1, R12, R18, R24.

## E. Occlusion and gap handling

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-007 | Occlusion handling — three-tier framework | proposed — not yet validated against real occlusion events. |

**Question:** How should low-confidence or missing keypoints be handled, and are the tier boundaries (under 5, 5-20, over 20 frames) sensible for this footage? D-007 adopts a three-tier design (spline, Kalman, root-lock) that is this project's own, with no literature source for the boundaries.

**What was tried, in order (dates and commit hashes):** 
1. 2026-07-27, D-007 (`5181280`): three-tier framework adopted, keyed to per-joint confidence below 0.5.
2. 2026-10-04, `dd343be`: one occlusion event reviewed in detail (D-002a: IMG_2412 frame 64, confidence 0.257 with a smooth, plausible position).
3. 2026-10-28: `metrics.py` loads keypoint CSVs and treats confidence below 0.5, or an absent frame, as a gap, reporting the two causes separately (`f744622`). The gap-statistics script was written (`d23ddfa`) and extended with frame exclusions and truncation flags (`e0a597e`) and machine-readable outputs (`b43bb0b`). An early version counted whole-frame detector drops as occluded on every joint; this was corrected before the script was first committed (the script's comments explain why), so no commit records that state.
4. Results logged as D-007 addenda: 100-frame excerpts (`8bed60e`), whole clips (`00ec2a0`), pipeline-joint figures (`32ad2f2`); outputs committed in `34bdd08`. The Cheng et al. (2019) citation and what it does and does not support: `8870e73`.
5. The hide-and-fill test that D-007 names as its validation, and `occlusion_handling.py` (gap filling, step 3.2 in the author's plan labels), do not exist in the repo.

**Data and sample sizes:** Excerpts: five clips, 500 frames, 8,449 joint observations, 29 joint-level events. Whole clips: five clips, 5,886 frames (5,828 used after 40 detector-drop frames and 18 frames excluded by list), 99,076 joint observations, 558 joint-level events (17 detector-drop events); for the eight pipeline joints, 115 events. Exact counts and the independent recount are in the D-007 addenda.

**Results (file paths):** `evaluation/3.1/output_full/` and `evaluation/3.1/output/` (`events.csv`, `summary.csv`, `frame_losses.csv`, `manifest.txt`, `report.md`); `validation/analyze_occlusion_gaps.py`; `validation/frame_exclusions.csv`; the three D-007 addenda dated 2026-10-28.

**Decision and status:** D-007 proposed (not yet validated against real occlusion events). Statuses as in the table above.

**What it cannot show:** It cannot validate the 5 and 20 frame boundaries or the fill methods: five clips, no ground truth for a joint's true position during a gap, and no hide-and-fill test. A confidently wrong keypoint (confidence at or above 0.5 during an occlusion) is invisible to this statistic. Events next to a clip edge or a dropped or excluded frame are truncated, so their lengths are lower bounds. The 100-frame excerpts understated limb gaps (3.4 against 5.6 events per 1,000 observations on whole clips).

**References:** register rows R1, R11.

## F. Centre of mass

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-005 | Whole-body CoM as primary tracked quantity | proposed _(+2 addendum status line(s) in the file)_ |
| D-025 | Full 14-joint segment-model CoM not adopted as it stands | decided |

**Question:** What whole-body quantity should stand in as the climber's centre of mass (CoM) for velocity, smoothness and balance metrics, given 2D COCO-17 keypoints? D-005 makes the CoM the primary tracked quantity and left the weighting open; the follow-ups and D-025 record the options built and the decision on them.

**What was tried, in order (dates and commit hashes):** 
1. 2026-07-27, D-005: CoM as the primary tracked quantity, weighting TBD (first in git in `5181280`). `ARCHITECTURE.md` §4.3 allowed a simplified proxy if its exact weights were logged.
2. 2026-10-28, `f744622`: `metrics.py` with per-joint velocity and cumulative displacement and tests.
3. 2026-10-28, `c8e1f2b` and `68bafe0`: trunk-centroid proxy (equal weights on both shoulders and both hips), logged in a D-005 follow-up as the CoM "until a cited segment-mass table is chosen" (repo wording). [From author's notes, to verify: built first as a stopgap.] It omits the head because the nose had 4 gap events on the five clips against 0 for the four trunk joints (D-005 follow-up).
4. 2026-10-28, `4c576a2` and `81703a6`: review script `evaluation/2.2/report_metrics.py`, producing an overlay video, plots and per-clip CSVs (velocity, displacement, steps skipped) for the five whole clips. [From author's notes, to verify: the overlays showed the proxy sitting mid-torso and stable, and the speed peaks (IMG_2419 jump-down, IMG_2420 swing across a hold, IMG_2412 step-up) were reviewed as real movement.] The repo holds the files, not the review.
5. 2026-10-28, literature checked in chat by Claude via web search (register rows R5, R6, R26 to R29); [from author's notes, to verify: no source was found for a blended trunk-plus-limb estimator, so it would be this project's own design]. Register entries: `c00c508`, `b1459c9`.
6. 2026-10-28, `1ba4080`, `34ecb51`, `3739954`, `1f618d1`: de Leva (1996) Table 4 chosen for a 14-joint segment model (values read from the PDF and checked against a secondary file; all matched, as recorded in the D-005 follow-up), implemented in `metrics.py` next to the proxy and compared with it on the same clips: position difference, constant offset against time-varying part, speed correlation, ten largest speed differences. Overlay clips for those ten (`cb1f32c`, `8c258ad`, `a9b1c5b`) and a per-frame table for IMG_2420 frames 68-84 (`9b46484`).
7. 2026-10-28, `87c8f0f`: the author's review of the ten clips led to D-025: the segment model is not adopted as it stands; the proxy remains the working CoM; the final choice is a later entry.
8. [From author's notes, to verify: the next step is a comparison of candidates in step 2.2c (a fitted trunk point, a head-free reduced model, a legs-and-trunk model and a blended model), repeated after gap filling and smoothing. D-025 names the step and the revisit condition but not the four candidates.] The classifier does not use the CoM: `ARCHITECTURE.md` §4.4 defines its pelvis proxy as the hip midpoint [author's notes: to verify].

**Data and sample sizes:** Five whole own clips, 5,886 frames. Proxy defined on 5,828 frames (99.0%); segment model on 5,349 (90.9%); compared only where both are defined (5,349), with 479 proxy-only frames not compared. 5,201 consecutive-frame speed pairs. Male table (the author's statement that the current footage is male; per-clip sex: not recorded in the repo). The ten largest speed differences were selected as the largest disagreements.

**Results (file paths):** `evaluation/2.2/output_full/summary.csv` and `<clip>/joint_metrics.csv` (proxy review); `evaluation/2.2b/output_full/summary.csv`, `top_speed_differences.csv`, `<clip>/frames.csv`, `IMG_2420_frames_68-84.csv` with its manifest, `manifest.txt`, `overlay_clips_manifest.txt` (comparison); overlay videos and plots are gitignored (`evaluation/2.2/**`, `evaluation/2.2b/overlays/`). D-005 follow-ups and D-025 in `DECISIONS.md`.

**Decision and status:** D-005 proposed (all three status lines). D-025 decided: the full 14-joint segment model is not adopted as it stands, the trunk-centroid proxy remains the working CoM, and a later entry (D-026) will make the final choice. "decided" is the status word used in D-025; it is not one of the four in the `DECISIONS.md` template.

**What it cannot show:** There is no ground-truth CoM, so neither estimator is validated. The D-025 labels (jitter or dropouts in six of the ten, real leg-driven movement in three, one unlabelled) are the author's visual judgement on ten steps chosen as the largest disagreements, not an error rate. Problems cluster in IMG_2420. The proxy cannot show slow leg-driven shifts. Everything is raw, unsmoothed and not gap-filled, and the sex table is an assumption. The LDLJ difference between the two estimators has not been computed. The reviewed overlays are not in the repo.

**References:** register rows R2, R4, R5, R6, R7, R19, R26, R27, R28, R29.

## G. Smoothness metric

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-006 | Smoothness metric — Log Dimensionless Jerk (LDLJ) | implemented (formula, secondary-source verified) — primary-source identity confirmed; primary-source formula text unread due to access barriers, not treated as a blocker given the existing secondary verification. |

**Question:** Which movement-smoothness metric should be reported, and is the formula right? D-006 adopts log dimensionless jerk (LDLJ), applied to CoM velocity.

**What was tried, in order (dates and commit hashes):** 
1. 2026-07-27, D-006 (`5181280`): LDLJ with the formula written in the entry; later "formula verified" 2026-07-27 by secondary sources (the Gulde & Hermsdörfer corrigendum and a comparison study that used Balasubramanian's code).
2. 2026-10-04: primary-source identity confirmed (Balasubramanian et al. 2015, via the Imperial Spiral record); the formula text in the primary PDF could not be read (access barriers). First in git in `6f1eb27` (2026-10-07).
3. 2026-10-28, `8870e73`: the entry's description of the corrigendum was corrected (it amends Gulde & Hermsdörfer's own article, not Balasubramanian's); the corrigendum's exact corrected form is unconfirmed, and the (t2 - t1)^3 form rests on the author's own dimension check.
4. 2026-10-28, web check by Claude in chat: the comparison study is Bayle et al. (2023), which used Balasubramanian's Python code for LDLJ and SPARC (register R10).
5. No LDLJ computation exists in the repo. The only mentions of LDLJ in the code are the placeholder column `ldlj_difference` ("n/a") in the 2.2b comparison script and its tests.

**Data and sample sizes:** None. No smoothness value has been computed on any clip.

**Results (file paths):** None. The only artefacts are the formula and its checks in D-006 and register rows R8 to R10.

**Decision and status:** D-006 implemented (formula, secondary-source verified), as the file states. I found no LDLJ computation behind that status; it refers to the formula having been checked, not to an implementation.

**What it cannot show:** Nothing about the data: no smoothness result exists. The formula is not verified against the primary text. The choice of LDLJ over SPARC rests on the entry's rationale, not on a comparison on this footage. How smoothing choices would affect it is open (author's plan step 5.1a; the plan file is not in the repo).

**References:** register rows R8, R9, R10.

## H. Four-state classifier and thresholds

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-003 | Four-state motion detection method — z-score thresholding, not cusum | proposed |
| D-004 | Threshold and duration values — empirical sweep methodology | proposed — actual swept values to be logged as sub-entries once footage is available (e.g. D-004a for the four-state z-score threshold, D-004b for hysteresis duration). |
| D-015 | Known-groups validation as a supplementary check for threshold selection | proposed — not required for the core pipeline (`ARCHITECTURE.md` §1–§6); required as a prerequisite check before `threshold_z` (D-004) is finalised. |

**Question:** How can per-joint moving/static states be classified into the four Boulanger-derived states without ground truth, and how are the thresholds chosen? D-003 adopts z-score thresholding on joint velocity; D-004 says all thresholds are swept on real footage against the author's judgement; D-015 adds a known-groups check.

**What was tried, in order (dates and commit hashes):** 
1. 2026-07-27, D-003 and D-004 (`5181280`).
2. 2026-10-04: D-003 addendum, Beltrán et al.'s method confirmed from their methodology text (Section 4.5, Eq. 9) and a possible refinement noted (comparison against 50% of the peak); first in git in `6f1eb27` (2026-10-07). D-015 (known-groups validation), reconstructed from `ARCHITECTURE.md` §7 because that section already referenced it (`e7140a7`, 2026-10-04).
3. Nothing has been run: there is no `state_classifier.py`, no threshold sweep and no known-groups check in the repo.

**Data and sample sizes:** None. Climber experience level per clip, needed for the known-groups check: not recorded in the repo.

**Results (file paths):** None.

**Decision and status:** D-003 proposed; D-004 proposed (the planned sub-entries D-004a and D-004b were never written); D-015 proposed. Statuses as in the table above.

**What it cannot show:** Nothing about classification quality: no states have been computed, no threshold chosen, no known-groups result obtained. D-015 itself states the Hold Interaction state is only an approximate proxy for the exploratory/performatory distinction, with no validated mapping.

**References:** register rows R1, R2, R3, R17, R20.

## I. Hold detection and colour calibration

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-009 | Hold detection — confirmed out of scope | superseded by D-017 (2026-10-04) — original reasoning (annotation cost, generalisation) stands as a historical record of why hold detection was excluded at the time; D-017 reverses this on the grounds that assisted-annotation tool … |
| D-017 | Hold detection — hybrid gym-specific detector + colour-matching, superseding D-009 | proposed — author has authorized starting implementation (2026-10-04). **Update (2026-10-04):** this entry originally flagged supervisor sign-off (`PROPOSAL.md` §9) as a required gate before/while building. The author has since ma … |
| D-017a | Gym colour→grade mapping — partially supplied (grade bands known, HSV calibration still pending) | proposed — grade-band mapping implemented; HSV calibration outstanding, blocking real (non-"unassigned") colour-matching output. |
| D-022 | Hold detector model — YOLOv8n-seg instance segmentation with polygon labels, three classes | proposed _(+3 addendum status line(s) in the file)_ |

**Question:** Can holds be localised and assigned to routes (by colour) with a gym-specific detector, and what model type and class set should it use? D-009 first ruled hold detection out; D-017 reversed that; D-022 changed the model to instance segmentation with three classes.

**What was tried, in order (dates and commit hashes):** 
1. 2026-07-27, D-009 (`5181280`): hold detection out of scope (annotation cost, generalisation), from three sources (R12, R13, R1).
2. 2026-10-04, `9983bbe`: D-017 reverses D-009 (fine-tuned YOLOv8n box detector plus HSV colour matching) after the author authorised starting implementation; D-017a records the gym's colour-to-grade bands (the author's knowledge). `hold_detection.py`, `hold_colours_config.py`, `training/train_hold_detector.py` and `training/calibrate_hold_colours.py` were added.
3. 2026-10-21, `7974bea`, `d96fcf4`, `c7e8254`: D-022 switches to YOLOv8n-seg with polygon labels and three classes (`hold`, `descent_hold`, `volume`); the code was rewritten for segmentation, and mask and eroded-mask pixel counts were added per detection.
4. 2026-10-28, `2b09949`: two D-022 addenda (cut-off holds are annotated; overlapping polygons, smallest polygon wins, to be checked at task 0.5.7).
5. No detector has been trained: no checkpoint, annotated dataset or training run is in the repo.

**Data and sample sizes:** None for results. Number of annotated images, annotations per class, Roboflow project details, and the number of sample photos per colour for HSV calibration: not recorded in the repo. D-022 says the code was verified only against synthetic or hand-written polygon fixtures; those fixtures are not recorded in the repo (no hold-detection tests are in `tests/`).

**Results (file paths):** None. `hold_colours_config.py` holds the grade bands (D-017a); the HSV reference is still `None`, so colour matching returns "unassigned". `DEFAULT_MASK_ERODE_PX` is an unvalidated placeholder (D-022 addendum).

**Decision and status:** D-009 superseded (by D-017); D-017 proposed; D-017a proposed (grade bands implemented, HSV calibration outstanding); D-022 proposed. Statuses as in the table above. The planned sub-entries D-017b and D-017c were never written.

**What it cannot show:** No detection accuracy, colour-matching accuracy or generalisation figure exists: nothing has been trained or run on real footage. The detector will be gym-specific by design (D-017 limitation 1). Whether YOLOv8n-seg is accurate enough on this gym's holds, and whether background holds will be detected, are open questions in D-022.

**References:** register rows R1, R12, R13, R25.

## J. Hold-relative analysis and attempt comparison

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-018 | Promote hold-relative feedback — contact timeline, hold-to-hold segmentation, and derived timing/foot/balance metrics | proposed — build blocked on Phase 4 and Phase 0.5.2–0.5.5. |
| D-019 | Route auto-definition and route progress | proposed — blocked on D-018's dependencies. See 2026-10-21 addendum below (route-membership/start/finish/progress wording updated for the three-class segmentation detector, D-022). _(+1 addendum status line(s) in the file)_ |
| D-020 | Per-move attempt-to-attempt comparison by hold sequence — stretch goal; layout matching proposed as the D-013 resolution | proposed — stretch, after the core pipeline and D-018/D-019 features work. |

**Question:** Can the climb be analysed relative to the holds: a keypoint-to-hold contact timeline, hold-to-hold segmentation of the metrics, timing, foot and balance measures, route definition and progress (D-018, D-019), and per-move comparison of attempts (D-020, a stretch goal)?

**What was tried, in order (dates and commit hashes):** 
1. 2026-10-07, `6f1eb27`: D-018, D-019 and D-020 logged. D-018 states the build is blocked on the four-state classifier and on the trained, colour-calibrated hold detector.
2. 2026-10-21, `7974bea`: D-019's route-definition wording updated for the three-class segmentation detector.
3. Nothing is built: there is no `hold_contact.py`, contact timeline or route logic in the repo.

**Data and sample sizes:** None.

**Results (file paths):** None.

**Decision and status:** D-018 proposed (build blocked); D-019 proposed (blocked on D-018's dependencies); D-020 proposed (stretch). Statuses as in the table above.

**What it cannot show:** Nothing: no result exists. The entries list their own risks (hold IDs depending on the layout merge, wrist and ankle keypoints not being contact points, foot readjustment and the balance proxy having no literature source). The 2026 preprint cited in D-018 (R16) and the fluency paper (R15) are not checked against their bodies.

**References:** register rows R1, R2, R15, R16.

---

## Reference register

Every source cited in `DECISIONS.md`, `ARCHITECTURE.md` and `PROPOSAL.md`. **Verification status is only what the repo records.** "Not recorded" means the repo does not say; it does not mean unverified or verified. The "Role in write-up" column says how a source is expected to be used (related work, method source, dataset, measurement tool, background); "not decided" where that is not clear yet. Rows marked **FLAG** have no source traceable from the repo, or a repo-recorded caveat. Rows R4, R5, R6, R10, R12, R15–R20, R25 and the new rows R26–R29 were updated on 2026-10-28; R4, R6, R10, R15–R20, R25 and R26–R29 with web checks made by Claude in chat via web search (not re-run by Claude Code); anything marked unverified there stays unverified here. The cited documents themselves (`DECISIONS.md`, `ARCHITECTURE.md`, `PROPOSAL.md`) are not edited by these updates.

| Ref | Citation as recorded in the repo | Where cited | Role in write-up | Verification status as recorded | Flag |
|---|---|---|---|---|---|
| R1 | Maschek, A. & Schedl, D.C. (2025). "The Way Up: A Dataset for Hold Usage Detection in Sport Climbing." arXiv:2505.12854. | D-002, D-004, D-007, D-008, D-009, D-010, D-012 (addendum), D-013, D-014, D-016 (addendum), D-017, D-020, D-022, D-023; ARCHITECTURE §2, §3, §4.4; PROPOSAL §4, §8 | related work (hold usage); dataset; method precedent (camera, tIoU, threshold sweep) | The author re-checked the model name "vitpose-l-coco_25" and Figure 4a on 2026-10-28 (D-016, D-023: "not independently re-verified in this session"). The camera setup was read by the author in the "Data acquisition" section (quote and details supplied by the author on 2026-10-28; not re-checked by Claude Code): "Filming was conducted using a handheld GoPro Hero 9 camera positioned at a 90-degree angle to the wall. This angle was chosen after preliminary tests, shown in Figure 2, to capture the climber's movements best." The wall is 8.5 m and not inclined. The camera was handheld, so not a fixed camera. Whether the other cited parts (Tables 4–6 in D-002, the 0.5 s sweep in D-004, the occlusion statement in D-007, the homography section in D-013, the tIoU methodology in D-014) were read directly: not recorded. | One source supports many claims; the coco_25 / Figure 4a claim and the camera setup ("Data acquisition" section) have a recorded check. D-002's accuracy numbers come from the coco_25 checkpoint, not the one this project runs (D-002 addendum).  |
| R2 | Boulanger, J., Seifert, L., Hérault, R. & Coeurjolly, J-F. (2015). "Automatic Sensor-Based Detection and Classification of Climbing Activities." IEEE Sensors Journal 16(3), 742–749. | D-003, D-005, D-015, D-018; ARCHITECTURE §4.3, §4.4; PROPOSAL §3, §5 | method source (four-state framework); related work (climbing movement states) | Cited by section (III-A cusum method, IV-A four-state definitions). Whether read directly: not recorded. | Year (2015) and volume (16(3)) are as given; not checked against a publisher record in the repo.  |
| R3 | Beltrán, R., Richter, J., Köstermeyer, D. & Heinkel, U. (2023). "Climbing Technique Evaluation by Means of Skeleton Video Stream Analysis." Sensors 23(19), 8216. PMC10574944. | D-003; ARCHITECTURE §4.4 | method source (z-score segmentation); related work (climbing movement states) | Section 4.5 and Eq. 9 (z = (v − μ)/σ) confirmed directly from the methodology text (D-003 addendum, 2026-10-04). | D-003's first Source line names "Beltrán, R., Richter, J. & Heinkel, U." as the method's authors, the confirmation line gives four authors; the repo uses both forms.  |
| R4 | Dech, S. & Kittel, R. (2026). "Biomechanical Principles and Techniques — A Systematization for Sport Climbing." J. Funct. Morphol. Kinesiol. 11(1):103, doi 10.3390/jfmk11010103. Section 3.1.3. | D-005; ARCHITECTURE §4.3 | background (biomechanical principles, CoM shift) | Existence, volume, issue, article number and DOI confirmed (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code). The CoM-distance statement in Section 3.1.3 and its reference [51]: see R5. |  |
| R5 | Werner, I., Gebert, W. & Kauer, B. (1999). "Three Dimensional Analysis of Rock Climbing Techniques." Proceedings of the 1st International Conference on Science and Technology in Climbing and Mountaineering, Leeds, UK, 7–9 April 1999 (reference [51] of R4). | D-005; ARCHITECTURE §4.3 | background (cited in R4) | **Verified as cited in R4.** Dech & Kittel (2026), Section 3.1.3: "The biomechanical sub-principle of a low CoM-distance to the wall was found to be positively associated with ranks in a climbing competition [51]." Reference [51] is Werner et al. as above (quote and reference supplied by the author on 2026-10-28; not re-checked by Claude Code). Werner et al. itself has **not been read**: recorded as cited in R4. An earlier web check in chat, whose readable text was cut off before reference entry 41, did not find the citation; this entry supersedes it. D-005 and ARCHITECTURE §4.3 attribute it correctly; no edit needed. | Secondary citation (Werner et al. not read).  |
| R6 | Sibella, F., Frosio, I., Schena, F. & Borghese, N.A. (2007). "3D analysis of the body center of mass in rock climbing." Human Movement Science 26(6), 841–852. | D-005; ARCHITECTURE §4.3 | related work (CoM in climbing); cited in D-005 via R2 and R4 | Cited in D-005 as the precedent named by both R2 and R4 (secondary). Publication page read (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code); the CoM method is not stated on that page. Whether the paper itself was read: not recorded. |  |
| R7 | de Leva, P. (1996). "Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters." J. Biomechanics 29(9), 1223–1230 (printed identifier 0021-9290(95)00178-6). Table 4, p. 1228. | D-005 follow-up (2026-10-28) | method source (segment parameters) | Values read from the PDF and checked against a secondary data file: all matched. Local PDF copy is outside the repo; its SHA-256 (685916ec…dc72811) is recorded in the D-005 follow-up (commit `1f618d1`). | The "secondary data file" is unnamed and not in the repo.  |
| R8 | Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A. & Burdet, E. (2015). "On the analysis of movement smoothness." J. NeuroEngineering and Rehabilitation 12:112. DOI 10.1186/s12984-015-0090-9. | D-006; ARCHITECTURE §4.5 | method source (LDLJ) | Identity (title, journal, volume, year, authors) confirmed 2026-10-04 via the Imperial Spiral record. Formula text **not read** (access barriers; D-006). ARCHITECTURE §4.5 says to verify constants against the primary source before implementing. | Formula rests on secondary checks (R9, R10).  |
| R9 | Gulde, P. & Hermsdörfer, J. (2023). "Corrigendum: Smoothness metrics in complex movement tasks." Frontiers in Neurology, DOI 10.3389/fneur.2023.1279682. | D-006 | not decided | Not read directly. Exact corrected form **not confirmed** (D-006 correction, 2026-10-28). | The original Gulde & Hermsdörfer article that the corrigendum amends is not cited with its own details.  |
| R10 | Bayle, Lempereur, Hutin, Motavasseli, Remy-Neris, Gracies & Cornec (2023). "Comparison of Various Smoothness Metrics for Upper Limb Movements in Middle-Aged Healthy Subjects." Sensors 23(3):1158, doi 10.3390/s23031158. (D-006 cites it only as "a comparison study, Sensors 23(3), 1158, 2023".) | D-006 | method source (secondary check of the LDLJ implementation); related work (smoothness metrics) | Page read (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code); it confirms that LDLJ and SPARC were computed with Balasubramanian et al.'s Python code. Initials not recorded. The full text beyond the page read: not recorded. | Identity resolved. D-006 itself still carries the unnamed citation (not edited).  |
| R11 | Cheng, Y., Yang, B., Wang, B., Yan, W. & Tan, R.T. (2019). "Occlusion-Aware Networks for 3D Human Pose Estimation in Video." ICCV 2019. | D-007 (addendum, 2026-10-28) | related work (occlusion-aware pose) | Author list supplied by the user on 2026-10-28; no DOI or pages recorded. The addendum states what it does and does not support. Whether the paper was read in this project: not recorded. |  |
| R12 | Reiff, D. (2024). "Using Computer Vision to Assess Bouldering Performance." Roboflow blog (non-peer-reviewed; "implementation precedent only"). | D-008, D-009, D-017, D-022 | related work (see R18) | Web check of the blog post: see R18. Read record for the uses in D-009, D-017 and D-022 (custom detection and colour models, hand annotation): not recorded. | URL not recorded.  |
| R13 | Ludford, G. (2024). "Development of a climbing performance analysis tool using computer vision." The Plymouth Student Scientist 17(2). | D-009, D-017, D-019, D-022 | related work (hold detection) | D-009 quotes figures from it (1717 images, 63,897 annotations, 87.6% mAP, colour-classification accuracy 72–88%). Read record: not recorded. | No URL or DOI recorded.  |
| R14 | Heilbron et al. (2015), "ActivityNet", cited via R1 as the source of the tIoU metric. | D-014 | background (origin of tIoU) | Secondary citation; not read (as far as the repo shows). | Full title and venue not recorded.  |
| R15 | Rochat, Hacques, Ganiere, Seifert, Hauw, Iodice & Ade (2020). "Dynamics of Experience in a Learning Protocol: A Case Study in Climbing." Frontiers in Psychology 11, doi 10.3389/fpsyg.2020.00249. (D-018 cites it only as "Frontiers in Psychology (2020) climbing fluency study".) | D-018 | related work (climbing fluency indicators) | Abstract read (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code); it lists climbing time and immobility ratio among four fluency indicators. Not the same paper as Reveret et al. (2020), article 2188 (R29). Body of the paper read: not recorded here (D-018 records a fetched page, first 100,000 of about 102,000 characters). | Identity resolved. D-018 still says "authors and exact title not yet confirmed" (not edited). Initials not recorded.  |
| R16 | Bakar, Aftab & Hamza. "Training-Free Hold-Usage Detection in Sport Climbing with Foundation Pose Models." arXiv:2609.30026v1, 24 Sep 2026 (preprint; first names not recorded). | D-018 | related work (hold usage) | Abstract read (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code): it uses Sapiens-1B, fingertip and toe keypoints and The Way Up dataset (R1); it does not state a perpendicular camera. **Unverified:** the metrics D-018 attributes to it (climb time, pace, dwell, foot-usage ratio, flight time) were not checked against the paper's body. | **FLAG:** attribution of the metrics unverified. Preprint, not peer reviewed.  |
| R17 | Orth, Kerr, Davids & Seifert (2017). Frontiers in Psychology 8:1744 (title not recorded here). Defines performatory and exploratory actions, citing Pijpers et al. (2006). | D-003, D-015; ARCHITECTURE §7 | background (exploratory / performatory actions) | Checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code: the paper defines performatory and exploratory actions. **Unverified:** the Pijpers et al. (2006) citation. How much of the paper was read: not recorded. D-015's mapping of the Hold Interaction state onto this distinction stays this project's own approximation. | Partly resolved; Pijpers et al. (2006) unverified.  |
| R18 | "BoulderVision case study" (PROPOSAL §4), identified by the author as Reiff, D. (4 Dec 2024), "Using Computer Vision to Assess Bouldering Performance", Roboflow blog (web source, not peer reviewed). The repo itself does not state that the two are the same. | PROPOSAL §4 (as "BoulderVision"); D-008 and others as R12 | related work | Checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code: it describes a ground-level, front-facing camera; it does **not** say the camera was fixed or perpendicular. PROPOSAL §4 says both Maschek & Schedl and this source "use a perpendicular/front-facing static setup", and D-008's Source line says "fixed front-facing camera setup": neither is supported for this source by the check. **Related-work description of the post** (industry web source, not peer reviewed; to be used as related work; checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code): a single ground-level, front-facing camera; a custom hold detector (straight 90-degree walls) and a gym-specific hold-colour classifier; YOLOv8x pose; no dedicated tracker (it picks the keypoint set closest to the earlier trajectory); metrics are keypoint displacement and a velocity ratio over the last 10 frames, climb duration, a climb path and a time-spent heatmap; no accuracy or validation figures; no centre of mass, smoothness or technique score; stated limitations: hold detection fails when the climber occludes holds, poor generalisation to new walls, and the author proposes adding 3D depth. Maschek & Schedl's 90° camera angle is recorded in D-008 (Fig. 2); a direct read of that figure: not recorded. The corrected wording for PROPOSAL §4 and D-008 was applied in `8f3f9bd`. | Resolved in `8f3f9bd`: PROPOSAL §4 and D-008 previously overstated what this source shows.  |
| R19 | Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement, 4th ed., Wiley. (ARCHITECTURE §4.3 names "Dempster/Winter body segment parameter tables" without a publication.) | ARCHITECTURE §4.3 | background; not used (superseded by R7) | Existence of the book confirmed from catalogue records (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code). The Dempster basis of its segment table was not checked. **ARCHITECTURE §4.3 is out of date:** it names Dempster/Winter as the conventional default, but de Leva (1996) (R7) was chosen in the D-005 follow-up. The corrected ARCHITECTURE §4.3 sentence was applied in `8f3f9bd`. | **FLAG:** Dempster not traceable (the out-of-date ARCHITECTURE sentence was corrected in `8f3f9bd`).  |
| R20 | "Procedures for assessing the validities of tests using the known-groups method" (title only). The author believes this is Hattie and Cooksey (1984). | D-015 | method source (validation technique); not yet used | Checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code: the title exists. Authors, year and journal: **unverified**. D-015 itself cites only the general technique. | **FLAG:** authors, year and journal unverified.  |
| R21 | easy_ViTPose (github.com/JunkyByte/easy_ViTPose, by Adriano Donninelli); model card wording "finetuned on 25 keypoints COCO skeleton + feet". | D-012, D-016, D-023 (addenda, 2026-10-28); ARCHITECTURE §2 | measurement tool (candidate, D-023) | Author's re-check, 2026-10-28; "not independently re-verified" in the session. Keypoint layout, weight loading, licence and compute cost: unverified (D-023). |  |
| R22 | google-ai-edge/mediapipe issue #6356 (github.com/google-ai-edge/mediapipe/issues/6356). | D-016 | background (engineering note) | Described as a "confirmed upstream bug". The crash with mediapipe 1.0.x on macOS Apple Silicon was hit locally on 2026-10-04 (D-016). Whether the issue page was read: not recorded. |  |
| R23 | `usyd-community/vitpose-plus-large` (Hugging Face checkpoint, 434M parameters). | D-011, D-012, D-016, D-023 | measurement tool | Downloaded, loaded and its config inspected in this environment on 2026-07-27 (D-011, D-012). Model-card and licence record: not recorded. |  |
| R24 | Software and versions: `transformers` 5.14.1, `torch` 2.11.0, `ultralytics` 8.4.108 (D-011, 2026-07-27); `mediapipe` 0.10.35 pinned, Apache 2.0 confirmed via `pip show` (D-021). | D-011, D-016, D-021 | measurement tool | Recorded from direct inspection. `evaluation/3.1/*/manifest.txt` records library versions at analysis time. |  |
| R25 | Methods and tools named in the repo without a citation, now partly checked: ViTPose method — Xu, Zhang, Zhang & Tao, arXiv:2204.12484 (v3, 13 Oct 2022); BlazePose — Bazarevsky, Grishchenko, Raveendran, Zhu, Zhang & Grundmann, arXiv:2006.10204 (17 Jun 2020); YOLOv8 (Ultralytics, YOLOv8n / YOLOv8-seg / YOLOv8-pose); Roboflow Smart Select; the HSV colour space. | D-002, D-011, D-016, D-017, D-021, D-022 | measurement tool and method source (models used); not decided per item | Checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code: the two arXiv entries exist with the identifiers and dates given. YOLOv8 is Ultralytics software with no paper; citation details **not read**. Roboflow and HSV: **not checked**. The repo's own docs still cite none of these. | **FLAG:** not cited in the repo; YOLOv8, Roboflow and HSV unverified.  |
| R26 | Jamkrajang, Robinson, Limroongreungrat & Vanrenterghem (2017). "Can segmental model reductions quantify whole-body balance accurately during dynamic activities?" Gait & Posture (volume, pages and DOI not recorded). | not yet cited in `DECISIONS.md`, `ARCHITECTURE.md` or `PROPOSAL.md` | related work (reduced-segment CoM) | Full text read via the LJMU repository copy (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code). What it shows for this project: not recorded here. | Initials not recorded.  |
| R27 | Peng, Alizadeh Noghani & Bolivar-Nieto (2024), arXiv:2411.18774 (preprint; title not recorded here). | not yet cited in the repo docs | related work (reduced-segment CoM) | Abstract read (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code). | Preprint; only the abstract read.  |
| R28 | "A simplified marker set to define the center of mass for stability analysis in dynamic situations." Gait & Posture, 2016. | not yet cited in the repo docs | related work (reduced-segment CoM) | Abstract only (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code). | **FLAG:** authors unverified.  |
| R29 | Reveret, Chapelle, Quaine & Legreneur (2020). "3D Visualization of Body Motion in Speed Climbing." Frontiers in Psychology 11:2188. | not yet cited in the repo docs | related work (CoM in climbing) | Abstract and text read (checked by Claude in chat via web search, 2026-10-28; not re-run by Claude Code); it reports no validation numbers. Not the same paper as R15. |  |

Not in the register: the original proposal PDF of 26 April 2026 (the formal submitted document, not a literature source), and sources that appear only inside R7 (Zatsiorsky et al., Gordon et al., Webb Associates and others), which the repo does not cite itself.

---

## Part 3. Related work map (skeleton)

Groups of register rows, for the related-work chapter. The "how this project differs" column is left empty on purpose: it is to be written, not inferred.

| Group | Register rows | How this project differs |
|---|---|---|
| 1. Climbing movement states | R2 Boulanger et al. (2015); R3 Beltrán et al. (2023) | |
| 2. Hold usage | R1 Maschek & Schedl (2025); R16 Bakar, Aftab & Hamza (arXiv:2609.30026) | |
| 3. Centre of mass in climbing | R6 Sibella et al. (2007); R29 Reveret et al. (2020); R5 Werner et al. (1999), as cited in R4; R4 Dech & Kittel (2026) | |
| 4. Reduced-segment centre of mass | R26 Jamkrajang et al. (2017); R27 Peng et al. (2024); R28 "A simplified marker set to define the center of mass ..." (Gait & Posture, 2016) | |
| 5. Smoothness | R8 Balasubramanian et al. (2015); R10 Bayle et al. (2023) | |
| 6. Climbing fluency and exploratory movement | R15 Rochat et al. (2020); R17 Orth et al. (2017) | |
| 7. Industry example | R18 Reiff (2024), Roboflow blog | |

---

## Gaps table

Things the repo does not record, to be filled in from memory. Rows 1 to 13 come from the register and the dangling IDs; rows 14 to 27 were found while filling the topic sections.

| # | Topic | Item | What is missing | Where I looked |
|---|---|---|---|---|
| 1 | H | D-004a | Referenced as the planned sub-entry for the four-state z-score threshold (D-004 status line; `CLAUDE.md` rule 9) but never written. No threshold sweep is recorded. | `DECISIONS.md`, `CLAUDE.md`, git log, `evaluation/`, `validation/` |
| 2 | H | D-004b | Referenced as the planned sub-entry for the hysteresis duration (D-004 status line) but never written. No duration sweep is recorded. | same |
| 3 | I | D-017b | Referenced only in `ARCHITECTURE.md` §8 (epoch count and hyperparameters to be logged "once training actually happens"). Not written; no training run is recorded. | `ARCHITECTURE.md`, `DECISIONS.md`, git log |
| 4 | I | D-017c | Referenced in `ARCHITECTURE.md` and in D-022's pixel-count addendum (merge-IoU, colour-distance and mask-erosion thresholds to be swept). Not written; no sweep is recorded. | same |
| 5 | F | ARCHITECTURE §4.3 | Names Dempster/Winter tables as the conventional default; the D-005 follow-up chose de Leva (1996). Sentence was out of date; the correction was applied in `8f3f9bd`. | register R19 |
| 6 | D | PROPOSAL §4 / D-008 | Whether the "BoulderVision" / Reiff (2024) source used a fixed, perpendicular camera. PROPOSAL §4 says perpendicular static and D-008 says fixed front-facing; the web check found a ground-level, front-facing camera and no statement of fixed or perpendicular. Corrected wording applied in `8f3f9bd` (PROPOSAL §4 and D-008). | register R18 |
| 7 | J | D-018 / R15 | D-018 still says authors and title of the fluency study are unconfirmed; the register now records the identity (Rochat et al. 2020). D-018 not edited. | register R15 |
| 8 | J | D-018 / R16 | The metrics attributed to arXiv:2609.30026 (climb time, pace, dwell, foot-usage ratio, flight time) were not checked against the paper's body. | register R16 |
| 9 | H | R17 | Pijpers et al. (2006), cited by Orth et al. (2017) for the performatory/exploratory definition, is unverified. | register R17 |
| 10 | H | R20 | Authors, year and journal of the known-groups method paper are unverified (believed Hattie and Cooksey 1984). | register R20 |
| 11 | B / I | R25 | YOLOv8 (Ultralytics) citation details, Roboflow Smart Select and the HSV colour space have no checked source. | register R25 |
| 12 | F | R26–R28 | Volume, pages and DOI of R26, the title of R27, and the authors of R28 are not recorded; what each shows for the CoM work is not recorded. | register |
| 13 | G | D-006 / R10 | D-006 still cites the comparison study without authors or title; the register now records Bayle et al. (2023). D-006 not edited. | register R10 |
| 14 | A | D-001 | The text of the supervisor's original "manual labelling" comment, and whether or when it was answered. | `DECISIONS.md`, `PROPOSAL.md`, git log |
| 15 | B | D-002a / footage | Recording device, resolution, climber, route and grade for each of the five clips, and why these five were chosen. Only the file names are recorded. | `DECISIONS.md`, `validation/`, `evaluation/` |
| 16 | B | D-016 | The MediaPipe foot-keypoint exploration was a scratchpad-only script (not committed). Its numbers are in D-016 only; the script and its raw output are not in the repo. | `DECISIONS.md`, git log |
| 17 | B | D-002a | The hardware or device that ran the 2026-10-04 validation. The summary records processing seconds but not the device. | `validation/output/summary.csv` |
| 18 | C | D-024 | Per-clip counts of frames with more than one person box (only in the gitignored keypoint CSVs), and the images, notes and re-run script behind the spot checks (IMG_2419 frame 605, IMG_2420 frames 1850 and 2390, IMG_2419 detector re-runs). | `DECISIONS.md`, `validation/frame_exclusions.csv` |
| 19 | D | D-008 | Whether the author's footage was filmed on a tripod or otherwise fixed, and the camera's angle to the wall. The assumption is stated, not measured. | `DECISIONS.md`, `PROPOSAL.md` |
| 20 | D | D-021 | The spike's code and report exist only on branch `spike/blazepose-z` (`cdf40f2`), not on `main`, and cannot be reproduced from a clean clone (its README says so). | git log, branch list |
| 21 | E | D-007 | The hide-and-fill test and `occlusion_handling.py` (gap filling) do not exist. The first version of the gap script was never committed, so no commit records its early counting error. | git log, `validation/`, `evaluation/3.1/` |
| 22 | F | D-005 / D-025 | What the author saw in each of the ten reviewed overlay clips (only the counts and rank numbers are in D-025), and the author's review of the proxy overlays (mid-torso, stable; speed peaks real). The reviewed videos are gitignored. The sex of each climber (only "the current footage is male", the author's statement). | `DECISIONS.md` D-025, `evaluation/2.2b/` |
| 23 | F | step 2.2c | The four candidate estimators (fitted trunk point, head-free reduced model, legs-and-trunk model, blended model) and the statement that no source exists for a blended estimator are the author's notes, not recorded in the repo (D-025 names the step and the revisit condition only). | `DECISIONS.md`, `evaluation/`, register |
| 24 | G | D-006 | No LDLJ computation exists although the status line says "implemented (formula, secondary-source verified)". The primary-source formula text was never read. | `DECISIONS.md`, `*.py` |
| 25 | H | D-003 / D-004 / D-015 | Climber experience level per clip (the input of the known-groups check). Nothing was run. | `DECISIONS.md`, `ARCHITECTURE.md` §7 |
| 26 | I | D-017 / D-017a / D-022 | Annotated image and annotation counts, class counts, Roboflow project details, training runs, and sample photos per colour for HSV calibration. The synthetic fixtures D-022 says it was checked against are not in the repo (no hold-detection tests in `tests/`). | `DECISIONS.md`, `training/`, `tests/`, git log |
| 27 | all | plan labels | Step and task labels used in the entries (0.5.x, 2.2c, 3.2, 5.1a, D-026) come from the author's plan file, which is not in the repo (D-020 cites `PROJECT_PLAN.md` 7.2). | `DECISIONS.md` |

D-025 is committed (`87c8f0f`) and listed in topic F. Its status word "decided" is not one of the four in the `DECISIONS.md` template (proposed, implemented, validated, superseded); it is recorded as written.

Citation caveats are listed in the register (rows marked FLAG), not repeated here.
