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
- **Hold detection and route/grade assignment** via a hybrid approach: a single-class object detector (fine-tuned YOLOv8n) to localise holds, combined with classical HSV colour-matching within each detected hold to assign route/grade membership (this gym grades by hold colour). *(amendment, reverses D-009 — see `DECISIONS.md` D-017. Build has started; supervisor sign-off waived by the author's executive decision, see §9. Gym-specific, does not generalise without retraining — tension with this section's own generalisability goal, see D-017 Limitation 1. Reintroduces model fine-tuning, see `CLAUDE.md` rule 4, updated accordingly.)*
- **Hold-relative analysis** built on hold detection *(amendment, 2026-10-07 — `DECISIONS.md` D-018, D-019; build blocked until the four-state classifier and the hold detector are done)*: a keypoint-to-hold contact timeline; hold-to-hold segmentation of the existing metrics (CoM, velocity, four-state, LDLJ per move); timing metrics derived from the timeline (climb time, pace, pre-climb pause, dwell per hold, flight time); foot usage ratio and foot readjustments; a balance proxy; automatic route definition and route progress (no manual start/finish marking). Hold-anchored feedback is built on these. Foot readjustment and the balance proxy have no literature source and are validated empirically only.

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
- **Hold-relative metrics (D-018, D-019):** all of the above computed per hold-to-hold segment as well as per clip, plus the timing, foot-usage, balance-proxy and route-progress measures listed in §3. Contingent on hold detection and the contact timeline.

**No standalone static-to-dynamic ratio metric exists in this design.** If a single summary number is wanted for a report, derive it from the four-state timeline rather than computing it independently.

## 6. Scope: nice-to-have / deferred (do NOT build unless explicitly promoted to core scope in this file)

- **Time-spent heatmap** derived from keypoint dwell time (reuses existing keypoint data, no new model needed — genuinely low-cost, but not core deliverable until this section says so).
- **Attempt-to-attempt comparison, per move by hold sequence — stretch goal** (D-020; replaces the earlier whole-climb-only version). Two attempts are aligned by hold identity (no DTW needed), compared move by move, with whole-climb aggregate deltas as the summary line, and fed into the feedback generator. Needs cross-session hold alignment (layout matching / homography, D-013, D-020). Nothing in the core pipeline depends on it.
- **Spoken/verbal feedback output**: piping the generated text report through an off-the-shelf text-to-speech API. Trivial to add once the text report exists (no new model or metric needed) — a presentation-layer addition, not a research contribution. Not core scope until promoted here.
- Quantitative pose-accuracy evaluation (see §7).

## 7. Explicit scope boundaries — OUT OF SCOPE

- **Model fine-tuning / training**, except the single-class hold detector in §3 (`DECISIONS.md` D-017). Pose models remain pretrained-only.
- **Distance-to-wall / any depth (z) quantity.** Not measurable with a single perpendicular 2D camera (D-008, D-015). Hold detection and per-move comparison do not replace it. A 2.5D route (e.g. BlazePose root-relative z) is unproven for this use and only an optional exploratory spike, not a goal. The spike ran (`DECISIONS.md` D-021) and was inconclusive — no ground truth was available — so this stays out of scope.
- **Hold-relative feedback is no longer out of scope** (D-018, 2026-10-07) — see §3. It remains blocked until its dependencies exist; do not build it ahead of the four-state classifier and the hold detector.
- **Real-time or mobile deployment.**
- **Learned/LLM-based feedback generation.** Feedback stays rule-based and metric-driven.
- **DTW-aligned attempt comparison.** Per-move comparison by hold sequence is a stretch goal instead (§6, D-020).
- **3D pose lifting** (e.g. MotionBERT, MeTRAbs). This project uses 2D pose estimation only; a 3D lifting stage is a different pipeline component and is not required by any current goal.

## 8. Evaluation status

- **Current plan: qualitative evaluation only** — visual inspection of pose tracking accuracy against real footage, and qualitative correlation of computed metrics against observed climb difficulty/fluency.
- **Quantitative pose-accuracy evaluation is explicitly deferred**, not ruled out. If pursued later, it would require manually annotating ground-truth joint positions on a sample of frames (see Maschek & Schedl's coach-annotation methodology for a template). This decision can be revisited later in the project without contradicting current scope — log the revisit as a new entry in `DECISIONS.md` if/when it happens.

## 9. Open questions to raise with supervisor

- Sign-off on adding the four-state classifier, heatmap, and per-move attempt comparison to the formal proposal document. The hold-relative metrics (D-018, D-019) also go beyond the submitted PDF.
- **Hold detection reverses D-009.** `DECISIONS.md` D-017 proposes a hybrid gym-specific detector + colour-matching approach, superseding D-009's "confirmed out of scope" decision; implementation has started. It reintroduces model fine-tuning (`CLAUDE.md` rule 4 updated to carve out this one exception) and creates a gym-specific component in tension with §3's generalisability goal. **Update (2026-10-04):** this was previously flagged as needing supervisor sign-off before/while building. The author has made an explicit executive decision to proceed without it — scope calls in this area are the author's own to make. Left here as a record of the tension (fine-tuning exception, generalisability goal), not as an open item blocking anything.
