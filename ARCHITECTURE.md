# ARCHITECTURE.md — High-Level Design Document (HLDD)

Describes *what the system is*: pipeline stages, data contracts, the pose model decision, occlusion handling, and the mathematical definitions of every metric. Read alongside `PROPOSAL.md` (scope) and `DECISIONS.md` (why, with sources).

---

## 1. Pipeline stages and data contracts

```
Video Input → Pose Estimation → Temporal Tracking → Feature Extraction
→ Behaviour Interpretation → Feedback Generation → Output
```

| Stage | Input | Output |
|---|---|---|
| Video Input | Video file path (fixed camera, perpendicular to wall — see `PROPOSAL.md` §4) | Decoded frame sequence |
| Pose Estimation | Frame sequence | Per-frame array of keypoint `(x, y, confidence)` tuples for the tracked climber only (single-person, see §2) |
| Temporal Tracking | Per-frame keypoints | Per-joint time-series, gap-filled per the occlusion framework (§3) |
| Feature Extraction | Gap-filled time-series | Velocity, cumulative displacement, CoM trajectory, four-state classification, LDLJ smoothness (§4) |
| Behaviour Interpretation | Computed metrics | Rule-based performance descriptors (thresholds sourced from `DECISIONS.md`) |
| Feedback Generation | Descriptors | Natural language report (template-based, not learned) |
| Output | Report + metrics | Annotated video w/ skeleton overlay, time-series graphs, text report |

Keypoint format: standard 17-keypoint COCO convention (confirmed as the actual output of the chosen ViTPose-L checkpoint — see D-012; do not remap unnecessarily). Joint names referenced throughout this document assume standard COCO naming (`left_hip`, `right_hip`, `left_wrist`, etc.).

---

## 2. Pose estimation model — decision made, value pending on-footage validation

**Status: the decision *process* is complete (a specific model is chosen and justified below); the *value* is not yet locked, since it depends on a validation step against real footage that hasn't run yet.** Treat this as "proceed with ViTPose-L unless/until the validation step in this section says otherwise" — not as an unconditionally fixed fact. See `DECISIONS.md` D-002 for the authoritative status field (currently `proposed`, not `implemented`/`validated`).

**Currently chosen: ViTPose-L, standard 17-keypoint COCO format** (see D-012 — an earlier version of this document incorrectly stated "COCO 25-keypoint"; no such ViTPose variant exists), paired with a simple single-target bounding-box selector (nearest-to-previous-frame / largest-confidence-box heuristic) to lock onto one climber and reject bystanders walking through frame. Implementation route: HuggingFace `transformers` (`VitPoseForPoseEstimation`, checkpoint `usyd-community/vitpose-plus-large`), with YOLOv8n (`ultralytics`) as the upstream person-bounding-box source — see D-011.

Reasoning (full citations in `DECISIONS.md` D-002):
- No real-time requirement exists in this project (`PROPOSAL.md` §7), so ViTPose's slower inference (~0.25s/frame vs. MediaPipe's ~0.12s/frame) is not a real cost.
- On climbing-specific footage, ViTPose achieved the highest accuracy of the three benchmarked models (86.6% overall, vs. MediaPipe 83.5% and YOLOv8-pose 75.3%) and had the fewest missing-detection frames (2–10 missed frames across seven videos, vs. MediaPipe's 150+ consecutive missed frames in three videos).
- YOLOv8-pose was specifically worse on feet because it relies on ankle keypoints rather than toe position, requiring larger, less precise overlap margins — a real weakness for "good limb position" accuracy.
- ViTPose is a top-down method requiring an external person detector/bounding box, which is actually convenient here: it gives an explicit point to enforce single-climber tracking (reject/ignore secondary detections), rather than relying on a model's built-in (and less controllable) multi-person handling.

**Documented fallback:** if ViTPose proves too slow or complex to integrate in the available time, YOLOv8-pose has native multi-person tracking (ID persistence via ByteTrack in Ultralytics) which simplifies single-climber locking at an accuracy cost, particularly for feet. If this fallback is used, log it as a new `DECISIONS.md` entry with the reasoning at the time.

**Action required before committing further:** replicate a short version of the Maschek & Schedl accuracy comparison on your *own* gym footage before finalising — their numbers were measured on their two specific routes/gym, not yours. Log the result as a `DECISIONS.md` entry regardless of outcome.

---

## 3. Occlusion handling framework

Single fixed RGB camera means occlusion (hands behind torso, limbs crossing, climber facing away) is a certainty, not an edge case — Maschek & Schedl specifically found longer hand occlusions behind the upper body degraded detection accuracy. This project uses a three-tier policy based on confidence drop duration, keyed to a per-joint confidence threshold `C < 0.5`.

> **Note on provenance:** this specific three-tier framework (tier boundaries, mirroring, root-locking) is an original design for this project, not lifted directly from the surveyed literature. The individual techniques it composes are standard in tracking/kinematics (cubic spline interpolation, Kalman filtering, joint-limit clamping), but the tier thresholds (5 frames, 20 frames) are engineering heuristics to be validated empirically on real footage, not values with a literature citation behind them. State this honestly in the dissertation methodology chapter — do not retroactively imply these numbers come from a paper.

### Tier 1 — Short occlusion (< 5 frames, ≈ 0.15s)
**Method:** bi-directional cubic spline interpolation over the joint's pixel-position time series, using the trajectory before and after the gap (not linear interpolation — limbs move along arcs).
**Implementation note:** buffer 5–10 frames so the interpolator can see the "landing" trajectory, not just the approach.

### Tier 2 — Mid occlusion (5–20 frames, ≈ 0.15–0.6s)
**Method:** physics-constrained Kalman filter (constant velocity/acceleration model), clamped to human joint-limit constraints (max angular range per joint — do not let a knee "unbend" past shin length, etc.).
**Alternative for symmetric joints:** if a paired joint (e.g. right hip) is fully visible while its mirror (left hip) is occluded, estimate the occluded joint via rigid transformation from the visible one plus torso/shoulder orientation, rather than running the Kalman filter.

### Tier 3 — Long occlusion (> 20 frames, > 0.6s)
**Method:** flag joint "stale" — freeze its position rigidly relative to the hip/shoulder root rather than letting it drift. Exclude the joint from any per-limb metric (e.g. elbow angle) while stale. For CoM computation specifically, **do not drop the frame** — substitute a statistical default limb position (e.g. resting ~40° from torso) so the whole-body mass formula does not break. Document the exact default values chosen as a `DECISIONS.md` entry once implemented, since they are a modelling assumption that affects CoM accuracy during long occlusions.

**Required before implementation:** validate all three tiers against real occlusion events in your own test footage (e.g. a climber turning fully sideways, or a hand disappearing behind the head during a dynamic move) — do not assume the tier boundaries transfer without checking.

---

## 4. Metric definitions (mathematical)

### 4.1 Per-joint velocity
For joint `j`, frame `t`, sampling interval `Δt`:
```
v_j(t) = || p_j(t) − p_j(t−1) || / Δt
```
where `p_j(t)` is the (smoothed, gap-filled) 2D pixel position.

### 4.2 Cumulative displacement
```
D_j = Σ_t || p_j(t) − p_j(t−1) ||   over the full clip or a chosen window
```

### 4.3 Whole-body centre of mass (CoM)
```
CoM(t) = Σ_i w_i · p_i(t)  /  Σ_i w_i
```
using standard anthropometric segment mass-weighting (Dempster/Winter body segment parameter tables are the conventional default in biomechanics literature) applied to the tracked joint set. **If full segment-mass weighting is too heavy an implementation lift, a simplified proxy (e.g. weighted average of hip midpoint, shoulder midpoint, and head) is an acceptable fallback — but this simplification must be logged as a `DECISIONS.md` entry with the exact weights used**, since it is a genuine modelling choice that affects every downstream CoM-based metric.

**Justification for using whole-body CoM as the primary tracked quantity** (not just one metric among many): both Boulanger et al. and the JFMK biomechanics review treat pelvis/CoM motion as the central variable distinguishing climbing activity states — Boulanger's entire four-state model is built on pelvis motion vs. limb motion as the two defining axes, and the JFMK paper identifies "Center of Mass Shift" as one of five fundamental climbing principles, further noting that a low CoM-distance to the wall has been empirically associated with competition ranking (citing Werner et al.'s 3D climbing technique analysis). Sibella et al.'s dedicated 3D body-CoM-in-climbing study (cited by both Boulanger and the JFMK paper) is the direct precedent for treating CoM as a first-class climbing-specific measurement, not just a convenience metric.

### 4.4 Four-state motion classification (Boulanger-derived)

Source: Boulanger, Seifert, Hérault & Coeurjolly (2015), Section IV-A — exact original terminology, adapted from IMU sensors to 2D pose-derived velocity (method: Beltrán et al.'s RGB video z-score approach, not Boulanger's cusum method — see `DECISIONS.md` D-003 for why).

**Per-joint moving/static classification** (pelvis proxy = hip midpoint; limbs = both wrists, both ankles):
```
z_j(t) = ( v_j(t) − mean(v_j) ) / std(v_j)     [computed within-clip, per joint]
moving_j(t) = 1  if z_j(t) > threshold_z   else 0
```
`threshold_z` is an empirical value — do not hardcode a guess; sweep candidate values on test footage per the methodology in `DECISIONS.md` D-004.

**Combine into the four exclusive states** (exact Boulanger definitions, Section IV-A):
| State | Pelvis | ≥1 limb moving |
|---|---|---|
| Immobility | immobile | no |
| Postural Regulation | moving | no |
| Hold Interaction | immobile | yes |
| Traction | moving | yes |

Apply minimum-duration/hysteresis smoothing to prevent frame-to-frame flicker (methodology precedent: Maschek & Schedl's empirical 0.5s duration sweep for their hold-usage confirmation task — same *type* of methodology, not the same threshold value, since that 0.5s was tuned for a different problem; this project's duration threshold must be swept independently).

### 4.5 Smoothness — Log Dimensionless Jerk (LDLJ)

Standard form (Balasubramanian et al.'s widely used velocity/jerk-based smoothness metric — **verify exact constants against the primary source before implementing**, the form below is the general shape, not guaranteed letter-perfect):
```
DJ = − (t_f − t_0)^3 / v_peak^2  ·  ∫_{t0}^{tf} | d²v(t)/dt² |² dt
LDLJ = − ln( −DJ )
```
Applied per movement segment to the velocity profile of a chosen signal — recommend starting with CoM velocity (whole-body smoothness) as the primary signal, with per-limb LDLJ as a secondary/optional metric.

---

## 5. Suggested module boundaries

```
pose_extraction.py       — wraps ViTPose inference + single-person tracking
occlusion_handling.py    — three-tier gap-filling (§3)
metrics.py                — velocity, displacement, CoM, LDLJ (§4)
state_classifier.py       — four-state z-score classifier (§4.4)
behaviour_interpretation.py — thresholds → descriptors
feedback_generator.py     — descriptors → text report
output.py                 — video overlay, graphs, report assembly
```

## 6. Explicitly out of scope for this file
See `PROPOSAL.md` §7 — do not add hold detection, model training/fine-tuning, real-time processing, or 3D lifting to this architecture without a corresponding scope change logged there first.
