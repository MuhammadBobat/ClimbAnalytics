# CLAUDE.md — Working Rules

This is a third-year university dissertation project (40/120 credits). Correctness, defensibility, and traceability of every design choice matter more than speed. Read `PROPOSAL.md`, `ARCHITECTURE.md`, and `DECISIONS.md` in full before writing or modifying any code.

## Non-negotiable rules

1. **Never make an undocumented decision.** If a task requires choosing a threshold, algorithm, library function, or approach that is not already specified in `ARCHITECTURE.md` or logged in `DECISIONS.md`, STOP and ask a specific question rather than picking a reasonable-sounding default. This applies even to small things (e.g. "should missing keypoints be `NaN` or `None`?") if there's genuine ambiguity affecting correctness.

2. **Every decision made gets logged.** After settling any design choice — during this session, not retroactively — add an entry to `DECISIONS.md` using the template at the top of that file, including a source citation or an explicit "no literature source — engineering heuristic, validate empirically" if none exists. Do not fabricate or guess at a citation.

3. **Do not deviate from documented decisions without confirmation — but check the Status field first, since "documented" does not mean "unconditionally fixed."** Most entries in `DECISIONS.md` are currently `proposed`, not `implemented`/`validated` — this means they are the *operative working assumption* (build against them, don't silently pick something else) but are explicitly expected to be checked against real data before being treated as final. Concretely:
   - **Pre-approved validation work is not a deviation.** Running the D-002a comparison (testing ViTPose-L vs. its documented fallback on real footage), sweeping threshold values per D-004, or checking the LDLJ formula against its primary source per D-006 are the exact next steps these entries call for — do this without asking permission first, since the entries already authorise it.
   - **A pre-approved fallback is not a deviation either.** `ARCHITECTURE.md` §2 explicitly names YOLOv8-pose as a documented fallback if ViTPose-L proves impractical — switching to it, *if the stated trigger condition is met*, only requires logging why in a new `DECISIONS.md` entry, not a separate confirmation request.
   - **What does require stopping and asking:** changing a decision's substance for a reason *not* already anticipated in its `DECISIONS.md` entry (e.g. wanting a different pose model entirely, a different smoothness metric, a different classification framework). If you think a `proposed` decision is wrong even on its own terms, say so and explain why — don't silently substitute something else.
   - Once an entry's Status is updated to `validated`, treat it as materially harder to change — a real deviation at that point should prompt an explicit conversation, not just a new log entry.

4. **Stay inside scope boundaries** (`PROPOSAL.md` §7). Do not suggest or implement: hold detection, model fine-tuning/training of any kind, real-time processing, a learned/LLM-based feedback generator, localised hold-by-hold attempt comparison, or 3D pose lifting (MotionBERT, MeTRAbs, etc.) — none of these are in scope. If a task seems to require one of these, stop and flag it rather than quietly implementing a workaround.

5. **Nice-to-have features (heatmap, whole-climb attempt comparison, `PROPOSAL.md` §6) are not core deliverables** until promoted explicitly in that file. Build the core pipeline first unless told otherwise.

6. **Cite in code, not just in the decision log.** When implementing a metric or algorithm that derives from a specific source, add a short docstring/comment referencing the `DECISIONS.md` entry ID, e.g.:
   ```python
   def classify_motion_state(pelvis_velocity, limb_velocities, threshold):
       """
       Four-state motion classifier adapted from Boulanger et al. (2015),
       using z-score thresholding per Beltrán et al.'s RGB-video method
       rather than Boulanger's cusum approach. See DECISIONS.md D-003, D-004.
       """
   ```
   Keep the comment short — the full reasoning lives in `DECISIONS.md`, not duplicated in every file.

7. **Camera assumption is load-bearing.** Every displacement/velocity-based metric assumes a fixed, perpendicular camera (D-008). Do not add homography correction, multi-camera support, or off-axis handling unless the assumption itself changes in `PROPOSAL.md` — that would be a scope change, not a bug fix.

8. **When uncertain about a design choice, ask one specific question and wait**, rather than proceeding on an assumption. Prefer "Should the z-score threshold be computed per-clip or globally across all test footage?" over silently picking one and moving on.

9. **Do not fabricate empirical results.** Threshold values, accuracy numbers, and swept-parameter results must come from actually running code against real footage, not estimated or invented. If a `DECISIONS.md` entry says a value is "TBD, pending footage," leave it as a clearly marked placeholder in code (e.g. `THRESHOLD_Z = None  # TBD — see DECISIONS.md D-004a`) rather than filling in a plausible-looking number.

10. **Flag drift.** If, partway through implementation, something in `ARCHITECTURE.md` turns out not to make sense (e.g. ViTPose is too slow to be practical even offline, or the four-state model doesn't cleanly map onto 2D data), say so explicitly and propose the update as a new/superseding `DECISIONS.md` entry — don't just quietly work around it.
