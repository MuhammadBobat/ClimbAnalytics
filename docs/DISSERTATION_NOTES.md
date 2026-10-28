# Dissertation notes

Working notes for the dissertation write-up: one section per topic, one reference register, one gaps table. **This file records; it does not decide.** Decisions live in `DECISIONS.md`; this file points to them.

## How to read and maintain this file

- **Status** is copied from `DECISIONS.md` as it appears there (statuses as at the last commit touching that file, `1f618d1`). It is never upgraded here. Where an entry carries several status lines (D-005, D-019, D-022), the entry's own first status line is shown.
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
| F. Centre of mass | D-005 |
| G. Smoothness metric | D-006 |
| H. Four-state classifier and thresholds | D-003, D-004, D-015 |
| I. Hold detection and colour calibration | D-009, D-017, D-017a, D-022 |
| J. Hold-relative analysis and attempt comparison | D-018, D-019, D-020 |

D-025 (centre of mass: trunk-centroid proxy primary, segment model as a sensitivity check) is not committed yet; it will be added to topic F once it is. D-024 (tracking failure) is in topic C; the tracking fix will be an addendum or a later ID.

---

## A. Scope and evaluation plan

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-001 | Clarify manual-labelling requirement with supervisor | closed (2026-10-04) — author has made an executive decision not to pursue supervisor clarification on this point; scope is being decided by the author's own judgement going forward. Recorded as closed, not deleted, since it docume … |
| D-010 | Evaluation — qualitative now, quantitative deferred | proposed |
| D-014 | tIoU as the planned metric if quantitative evaluation (D-010) is picked up | proposed — dependent on D-010 being promoted from deferred to active; no action needed until then. |

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

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

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

**References:** register rows R1, R21, R22, R23, R24, R25.

## C. Person detection and single-climber tracking

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-024 | Single-climber lock-on — observed failure and options | proposed |

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

**References:** register rows R25.

## D. Camera assumption, scale calibration and depth

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-008 | Camera assumption — perpendicular, fixed | implemented (assumption baked into scope, see `PROPOSAL.md` §4) |
| D-013 | Cross-session scale calibration — open question, needed before attempt comparison is built | proposed — not required for core pipeline (which only needs internal, single-session consistency); required before implementing the attempt-comparison feature specifically. Revisit when that feature is picked up. **Update (2026-10 … |
| D-021 | BlazePose (MediaPipe) z-values as a distance-to-wall signal — exploratory spike, inconclusive, not adopted | proposed — inconclusive, not adopted. Cross-reference D-016 (foot keypoints / prior MediaPipe exploration), D-015 and D-018 (why depth/distance-to-wall matters but stays out of scope). |

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

**References:** register rows R1, R12, R18, R24.

## E. Occlusion and gap handling

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-007 | Occlusion handling — three-tier framework | proposed — not yet validated against real occlusion events. |

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

**References:** register rows R1, R11.

## F. Centre of mass

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-005 | Whole-body CoM as primary tracked quantity | proposed _(+2 addendum status line(s) in the file)_ |

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

**References:** register rows R2, R4, R5, R6, R7, R19.

## G. Smoothness metric

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-006 | Smoothness metric — Log Dimensionless Jerk (LDLJ) | implemented (formula, secondary-source verified) — primary-source identity confirmed; primary-source formula text unread due to access barriers, not treated as a blocker given the existing secondary verification. |

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

**References:** register rows R8, R9, R10.

## H. Four-state classifier and thresholds

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-003 | Four-state motion detection method — z-score thresholding, not cusum | proposed |
| D-004 | Threshold and duration values — empirical sweep methodology | proposed — actual swept values to be logged as sub-entries once footage is available (e.g. D-004a for the four-state z-score threshold, D-004b for hysteresis duration). |
| D-015 | Known-groups validation as a supplementary check for threshold selection | proposed — not required for the core pipeline (`ARCHITECTURE.md` §1–§6); required as a prerequisite check before `threshold_z` (D-004) is finalised. |

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

**References:** register rows R1, R2, R3, R17, R20.

## I. Hold detection and colour calibration

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-009 | Hold detection — confirmed out of scope | superseded by D-017 (2026-10-04) — original reasoning (annotation cost, generalisation) stands as a historical record of why hold detection was excluded at the time; D-017 reverses this on the grounds that assisted-annotation tool … |
| D-017 | Hold detection — hybrid gym-specific detector + colour-matching, superseding D-009 | proposed — author has authorized starting implementation (2026-10-04). **Update (2026-10-04):** this entry originally flagged supervisor sign-off (`PROPOSAL.md` §9) as a required gate before/while building. The author has since ma … |
| D-017a | Gym colour→grade mapping — partially supplied (grade bands known, HSV calibration still pending) | proposed — grade-band mapping implemented; HSV calibration outstanding, blocking real (non-"unassigned") colour-matching output. |
| D-022 | Hold detector model — YOLOv8n-seg instance segmentation with polygon labels, three classes | proposed _(+3 addendum status line(s) in the file)_ |

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

**References:** register rows R1, R12, R13, R25.

## J. Hold-relative analysis and attempt comparison

**Decisions in this topic**

| ID | Title | Status as in `DECISIONS.md` |
|---|---|---|
| D-018 | Promote hold-relative feedback — contact timeline, hold-to-hold segmentation, and derived timing/foot/balance metrics | proposed — build blocked on Phase 4 and Phase 0.5.2–0.5.5. |
| D-019 | Route auto-definition and route progress | proposed — blocked on D-018's dependencies. See 2026-10-21 addendum below (route-membership/start/finish/progress wording updated for the three-class segmentation detector, D-022). _(+1 addendum status line(s) in the file)_ |
| D-020 | Per-move attempt-to-attempt comparison by hold sequence — stretch goal; layout matching proposed as the D-013 resolution | proposed — stretch, after the core pipeline and D-018/D-019 features work. |

**Question:** _to be filled in part 2._

**What was tried, in order (dates and commit hashes):** _to be filled in part 2._

**Data and sample sizes:** _to be filled in part 2._

**Results (file paths):** _to be filled in part 2._

**Decision and status:** _to be filled in part 2 (IDs and statuses as above)._

**What it cannot show:** _to be filled in part 2._

**References:** register rows R1, R2, R15, R16.

---

## Reference register

Every source cited in `DECISIONS.md`, `ARCHITECTURE.md` and `PROPOSAL.md`. **Verification status is only what the repo records.** "Not recorded" means the repo does not say; it does not mean unverified or verified. Rows marked **FLAG** have no source traceable from the repo, or a repo-recorded caveat.

| Ref | Citation as recorded in the repo | Where cited | Verification status as recorded | Flag |
|---|---|---|---|---|
| R1 | Maschek, A. & Schedl, D.C. (2025). "The Way Up: A Dataset for Hold Usage Detection in Sport Climbing." arXiv:2505.12854. | D-002, D-004, D-007, D-008, D-009, D-010, D-012 (addendum), D-013, D-014, D-016 (addendum), D-017, D-020, D-022, D-023; ARCHITECTURE §2, §3, §4.4; PROPOSAL §4, §8 | The author re-checked the model name "vitpose-l-coco_25" and Figure 4a on 2026-10-28 (D-016, D-023: "not independently re-verified in this session"). Whether the other cited parts (Tables 4–6 in D-002, the 0.5 s sweep in D-004, the occlusion statement in D-007, Fig. 2 in D-008, the homography section in D-013, the tIoU methodology in D-014) were read directly: not recorded. | One source supports many claims; only the coco_25 / Figure 4a claim has a recorded check. D-002's accuracy numbers come from the coco_25 checkpoint, not the one this project runs (D-002 addendum). |
| R2 | Boulanger, J., Seifert, L., Hérault, R. & Coeurjolly, J-F. (2015). "Automatic Sensor-Based Detection and Classification of Climbing Activities." IEEE Sensors Journal 16(3), 742–749. | D-003, D-005, D-015, D-018; ARCHITECTURE §4.3, §4.4; PROPOSAL §3, §5 | Cited by section (III-A cusum method, IV-A four-state definitions). Whether read directly: not recorded. | Year (2015) and volume (16(3)) are as given; not checked against a publisher record in the repo. |
| R3 | Beltrán, R., Richter, J., Köstermeyer, D. & Heinkel, U. (2023). "Climbing Technique Evaluation by Means of Skeleton Video Stream Analysis." Sensors 23(19), 8216. PMC10574944. | D-003; ARCHITECTURE §4.4 | Section 4.5 and Eq. 9 (z = (v − μ)/σ) confirmed directly from the methodology text (D-003 addendum, 2026-10-04). | D-003's first Source line names "Beltrán, R., Richter, J. & Heinkel, U." as the method's authors, the confirmation line gives four authors; the repo uses both forms. |
| R4 | Dech, S. & Kittel, R. (2026). "Biomechanical Principles and Techniques — A Systematization for Sport Climbing." J. Funct. Morphol. Kinesiol. 11, 103. Section 3.1.3. | D-005; ARCHITECTURE §4.3 | Cited by section. Read status: not recorded. | |
| R5 | Werner, Gebert & Kauer (1999), CoM distance vs competition rank, as cited by R4. | D-005; ARCHITECTURE §4.3 | Secondary citation only. Title and venue: not recorded. Not read (as far as the repo shows). | **FLAG:** no source traceable from the repo. |
| R6 | Sibella, F., Frosio, I., Schena, F. & Borghese, N.A. (2007). "3D analysis of the body center of mass in rock climbing." Human Movement Science 26(6), 841–852. | D-005; ARCHITECTURE §4.3 | Cited as the precedent named by both R2 and R4 (secondary). Read status: not recorded. | |
| R7 | de Leva, P. (1996). "Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters." J. Biomechanics 29(9), 1223–1230 (printed identifier 0021-9290(95)00178-6). Table 4, p. 1228. | D-005 follow-up (2026-10-28) | Values read from the PDF and checked against a secondary data file: all matched. Local PDF copy is outside the repo; its SHA-256 (685916ec…dc72811) is recorded in the D-005 follow-up (commit `1f618d1`). | The "secondary data file" is unnamed and not in the repo. |
| R8 | Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A. & Burdet, E. (2015). "On the analysis of movement smoothness." J. NeuroEngineering and Rehabilitation 12:112. DOI 10.1186/s12984-015-0090-9. | D-006; ARCHITECTURE §4.5 | Identity (title, journal, volume, year, authors) confirmed 2026-10-04 via the Imperial Spiral record. Formula text **not read** (access barriers; D-006). ARCHITECTURE §4.5 says to verify constants against the primary source before implementing. | Formula rests on secondary checks (R9, R10). |
| R9 | Gulde, P. & Hermsdörfer, J. (2023). "Corrigendum: Smoothness metrics in complex movement tasks." Frontiers in Neurology, DOI 10.3389/fneur.2023.1279682. | D-006 | Not read directly. Exact corrected form **not confirmed** (D-006 correction, 2026-10-28). | The original Gulde & Hermsdörfer article that the corrigendum amends is not cited with its own details. |
| R10 | Comparison study, Sensors 23(3), 1158 (2023), computing LDLJ via "the Python code provided by Balasubramanian et al." on upper-limb data. | D-006 | Not recorded. | **FLAG:** authors and title not recorded; not traceable. |
| R11 | Cheng, Y., Yang, B., Wang, B., Yan, W. & Tan, R.T. (2019). "Occlusion-Aware Networks for 3D Human Pose Estimation in Video." ICCV 2019. | D-007 (addendum, 2026-10-28) | Author list supplied by the user on 2026-10-28; no DOI or pages recorded. The addendum states what it does and does not support. Whether the paper was read in this project: not recorded. | |
| R12 | Reiff, D. (2024). "Using Computer Vision to Assess Bouldering Performance." Roboflow blog (non-peer-reviewed; "implementation precedent only"). | D-008, D-009, D-017, D-022 | Not recorded (no URL, access date or read record). | URL not recorded. |
| R13 | Ludford, G. (2024). "Development of a climbing performance analysis tool using computer vision." The Plymouth Student Scientist 17(2). | D-009, D-017, D-019, D-022 | D-009 quotes figures from it (1717 images, 63,897 annotations, 87.6% mAP, colour-classification accuracy 72–88%). Read record: not recorded. | No URL or DOI recorded. |
| R14 | Heilbron et al. (2015), "ActivityNet", cited via R1 as the source of the tIoU metric. | D-014 | Secondary citation; not read (as far as the repo shows). | Full title and venue not recorded. |
| R15 | "Frontiers in Psychology (2020) climbing fluency study", DOI 10.3389/fpsyg.2020.00249. | D-018 | Page read via fetch (first 100,000 of about 102,000 characters). Authors and exact title: "not yet confirmed — confirm before citing" (the repo's own flag). | **FLAG:** repo marks it unconfirmed. |
| R16 | "Training-Free Hold-Usage Detection in Sport Climbing with Foundation Pose Models", arXiv:2609.30026v1 (preprint). | D-018 | Authors not recorded. Read status: not recorded. | **FLAG:** authors not recorded; not traceable beyond the arXiv identifier. |
| R17 | The exploratory / performatory climbing motor-control literature "associated with Seifert". | D-003, D-015; ARCHITECTURE §7 | No specific paper cited. D-015 states the mapping onto the Hold Interaction state is this project's own approximation. | **FLAG:** no source traceable. |
| R18 | "BoulderVision case study". | PROPOSAL §4 | No details anywhere in the repo. D-008 cites R12 for a fixed front-facing setup; whether the two are the same item is not stated. | **FLAG:** no source traceable. |
| R19 | "Dempster/Winter body segment parameter tables". | ARCHITECTURE §4.3 | Named as "the conventional default"; no publication cited. Not used (the D-005 follow-up uses R7). | **FLAG:** no source traceable. |
| R20 | "Known-groups validity" (general measurement technique). | D-015 | D-015 states there is no single climbing-specific source; no citation given. | **FLAG:** no source traceable. |
| R21 | easy_ViTPose (github.com/JunkyByte/easy_ViTPose, by Adriano Donninelli); model card wording "finetuned on 25 keypoints COCO skeleton + feet". | D-012, D-016, D-023 (addenda, 2026-10-28); ARCHITECTURE §2 | Author's re-check, 2026-10-28; "not independently re-verified" in the session. Keypoint layout, weight loading, licence and compute cost: unverified (D-023). | |
| R22 | google-ai-edge/mediapipe issue #6356 (github.com/google-ai-edge/mediapipe/issues/6356). | D-016 | Described as a "confirmed upstream bug". The crash with mediapipe 1.0.x on macOS Apple Silicon was hit locally on 2026-10-04 (D-016). Whether the issue page was read: not recorded. | |
| R23 | `usyd-community/vitpose-plus-large` (Hugging Face checkpoint, 434M parameters). | D-011, D-012, D-016, D-023 | Downloaded, loaded and its config inspected in this environment on 2026-07-27 (D-011, D-012). Model-card and licence record: not recorded. | |
| R24 | Software and versions: `transformers` 5.14.1, `torch` 2.11.0, `ultralytics` 8.4.108 (D-011, 2026-07-27); `mediapipe` 0.10.35 pinned, Apache 2.0 confirmed via `pip show` (D-021). | D-011, D-016, D-021 | Recorded from direct inspection. `evaluation/3.1/*/manifest.txt` records library versions at analysis time. | |
| R25 | Methods and tools named without any paper or documentation citation: the ViTPose method (only the checkpoint R23 is cited), YOLOv8n / YOLOv8-seg / YOLOv8-pose (Ultralytics), BlazePose / MediaPipe Pose, Roboflow Smart Select, the HSV colour space. | D-002, D-011, D-016, D-017, D-021, D-022 | Not applicable: no source is cited. | **FLAG:** no source cited in the repo. |

Not in the register: the original proposal PDF of 26 April 2026 (the formal submitted document, not a literature source), and sources that appear only inside R7 (Zatsiorsky et al., Gordon et al., Webb Associates and others), which the repo does not cite itself.

---

## Gaps table

Things the repo does not record, to be filled in from memory. Part 2 adds the gaps found in each topic section.

| # | Topic | Item | What is missing | Where I looked |
|---|---|---|---|---|
| 1 | H | D-004a | Referenced as the planned sub-entry for the four-state z-score threshold (D-004 status line; `CLAUDE.md` rule 9) but never written. No threshold sweep is recorded. | `DECISIONS.md`, `CLAUDE.md`, git log, `evaluation/`, `validation/` |
| 2 | H | D-004b | Referenced as the planned sub-entry for the hysteresis duration (D-004 status line) but never written. No duration sweep is recorded. | same |
| 3 | I | D-017b | Referenced only in `ARCHITECTURE.md` §8 (epoch count and hyperparameters to be logged "once training actually happens"). Not written; no training run is recorded. | `ARCHITECTURE.md`, `DECISIONS.md`, git log |
| 4 | I | D-017c | Referenced in `ARCHITECTURE.md` and in D-022's pixel-count addendum (merge-IoU, colour-distance and mask-erosion thresholds to be swept). Not written; no sweep is recorded. | same |

Pending, not a gap: D-025 (centre of mass: proxy primary, segment model as a sensitivity check) is not committed yet and will be added to topic F once it is.

Citation caveats are listed in the register (rows marked FLAG), not repeated here.
