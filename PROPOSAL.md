# PROPOSAL.md — Project Reference

**Project:** Movement Analysis System for Indoor Bouldering Performance
**Author:** Muhammad Bobat (Student ID: 11526968)
**Type:** Third-year CS dissertation project, 40/120 credits
**Base document:** Original proposal PDF, 26 April 2026 (this file is a condensed, amended working reference — the PDF remains the formal submitted document)

This file is ground truth for *what the system does and does not do*. `ARCHITECTURE.md` describes *how*. `DECISIONS.md` records *why* each specific choice was made, with sources. Read all three before making design changes.

---

## 1. Problem statement

Build a computer-vision system that processes fixed-camera video of a climber on a bouldering route, extracts 2D body keypoints via pretrained pose estimation, computes movement metrics from the keypoint time-series, and generates rule-based natural-language feedback on movement efficiency and technique.

## 2. Pipeline (unchanged from original proposal)

```
Video Input → Pose Estimation → Temporal Tracking → Feature Extraction
→ Behaviour Interpretation → Feedback Generation → Output
```

See `ARCHITECTURE.md` for the data contract at each stage.

## 3. Goals

Original proposal goals, plus amendments made during pre-term planning (each amendment has a full entry in `DECISIONS.md`):

- Modular CV pipeline extracting 2D keypoints from bouldering video using a pretrained pose model. *(original)*
- Quantitative movement metrics from keypoint time-series: velocity, cumulative displacement, movement smoothness. *(original — smoothness now formally specified, see §5 and `ARCHITECTURE.md`)*
- **Four-state motion classification** (immobility / postural regulation / hold interaction / traction), adapting Boulanger et al.'s (2015) IMU-based climbing activity framework to 2D pose-derived velocity. *(amendment — this fully replaces the originally-proposed standalone static-to-dynamic ratio, not just refines it. Resolution: the ratio is not a separate deliverable. It is trivially derivable from the four-state timeline as a summary statistic if needed later — e.g. `% time in Immobility` vs. `% time in Traction` — but is not implemented as an independent metric. See `DECISIONS.md` D-003.)*
- Behaviour interpretation layer mapping computed metrics to performance descriptors (e.g. controlled vs. dynamic style). *(original)*
- Rule-based, metric-driven natural language feedback. *(original — explicitly NOT a learned/LLM-based generator)*
- Qualitative evaluation on real bouldering footage. *(original, see §7 for evaluation status)*
- Generalisable architecture, in principle transferable to other dynamic sports. *(original)*
- **Hold detection and route/grade assignment** via a hybrid approach: a single-class object detector (fine-tuned YOLOv8n) to localise holds, combined with classical HSV colour-matching within each detected hold to assign route/grade membership (this gym grades by hold colour). *(amendment, reverses D-009 — see `DECISIONS.md` D-017. Build has started; supervisor sign-off is still pending, see §9. Gym-specific, does not generalise without retraining — tension with this section's own generalisability goal, see D-017 Limitation 1. Reintroduces model fine-tuning, see `CLAUDE.md` rule 4, updated accordingly.)*

## 4. Assumptions (load-bearing — do not violate without a logged decision)

- **Camera is fixed and mounted exactly perpendicular (90°) to the climbing wall plane**, following the precedent in Maschek & Schedl (2025) and the BoulderVision case study, both of which use a perpendicular/front-facing static setup. This assumption underlies every 2D displacement/velocity metric, the heatmap (future work, §6), and the planned occlusion-handling framework. If footage is ever off-axis, metrics derived from pixel displacement are not valid without a homography correction — this is out of scope unless separately decided.
- Camera does not pan, tilt, or zoom during a clip.
- Single climber per clip (no multi-person disambiguation required as a core feature — see `ARCHITECTURE.md` §2 for how this constrains pose model choice).
- Test footage source: **own gym footage, footage of friends climbing, and other self-shot video** filmed specifically for this project. **Not** sourced from the internet or third-party datasets, at least as of project start. (This affects licensing/ethics — no dataset consent forms needed since Bobat is filming consenting participants directly; get informal consent from anyone filmed who isn't the author.)

## 5. Core metrics (see `ARCHITECTURE.md` §4 for full mathematical definitions)

- Per-joint and whole-body (centre of mass) velocity
- Cumulative displacement
- Four-state motion classification (Boulanger-derived) — supersedes the standalone static-to-dynamic ratio (see §3 above)
- Movement smoothness via Log Dimensionless Jerk (LDLJ)
- Whole-body centre of mass (CoM) as the primary performance-tracking quantity (justification in `DECISIONS.md`)

**No standalone static-to-dynamic ratio metric exists in this design.** If a single summary number is wanted for a report, derive it from the four-state timeline rather than computing it independently.

## 6. Scope: nice-to-have / deferred (do NOT build unless explicitly promoted to core scope in this file)

- **Time-spent heatmap** derived from keypoint dwell time (reuses existing keypoint data, no new model needed — genuinely low-cost, but not core deliverable until this section says so).
- **Attempt-to-attempt comparison**, whole-climb summary level only (deltas between two runs' aggregate metrics, fed into feedback generator). Localised/hold-by-hold comparison (would need DTW temporal alignment and/or hold detection) is explicitly **not** planned — see §7 out-of-scope.
- Quantitative pose-accuracy evaluation (see §7).

## 7. Explicit scope boundaries — OUT OF SCOPE

- **Model fine-tuning / training**, except the single-class hold detector in §3 (`DECISIONS.md` D-017). Pose models remain pretrained-only.
- **Hold-relative feedback** (e.g. "you hesitated at hold 4"), and any other feature that requires linking pose/metric data to *individual* holds over time. Not achievable from hold localisation + colour/route assignment alone (§3) — that only gives static hold positions and route membership, not a hold-interaction timeline. Remains out of scope; do not assume §3's hold detection amendment covers this too.
- **Real-time or mobile deployment.**
- **Learned/LLM-based feedback generation.** Feedback stays rule-based and metric-driven.
- **Localised (hold-by-hold or DTW-aligned) attempt comparison** — see §6.
- **3D pose lifting** (e.g. MotionBERT, MeTRAbs). This project uses 2D pose estimation only; a 3D lifting stage is a different pipeline component and is not required by any current goal.

## 8. Evaluation status

- **Current plan: qualitative evaluation only** — visual inspection of pose tracking accuracy against real footage, and qualitative correlation of computed metrics against observed climb difficulty/fluency.
- **Quantitative pose-accuracy evaluation is explicitly deferred**, not ruled out. If pursued later, it would require manually annotating ground-truth joint positions on a sample of frames (see Maschek & Schedl's coach-annotation methodology for a template). This decision can be revisited later in the project without contradicting current scope — log the revisit as a new entry in `DECISIONS.md` if/when it happens.

## 9. Open questions to raise with supervisor

- Clarify what specifically prompted the "manual labelling" comment (quantitative pose evaluation? hold detection? learned classifier? human-rated climb comparison for validation?) — see `DECISIONS.md` entry D-001.
- Sign-off on adding the four-state classifier, heatmap, and whole-climb comparison to the formal proposal document.
- **Hold detection reverses D-009 — raise before/while building, not retroactively.** `DECISIONS.md` D-017 proposes a hybrid gym-specific detector + colour-matching approach, superseding D-009's "confirmed out of scope" decision; implementation has started. It reintroduces model fine-tuning (`CLAUDE.md` rule 4 updated to carve out this one exception) and creates a gym-specific component in tension with §3's generalisability goal. Do not treat D-009's reversal as final until this conversation happens.
