# Step 5.1a protocol — fixed 2026-11-26 BEFORE the sweep was run (D-006 addendum, D-004 method)

What this is: a speed-profile stability check of LDLJ and SPARC on windows of real CoM speed profiles. It is NOT a measure of move
smoothness: fixed windows are not movement segments (hold-to-hold segments wait for the hold detector and 4.1). The A vs E-dL
comparison on movement segments also waits for 4.1; here A is only a check that the conclusions do not depend on which CoM is used.
No known-groups check (no climber or experience labels exist; skipped at the author's instruction). No p-values, no pass/fail
thresholds. Every number below is an engineering heuristic with no literature source, validate empirically (CLAUDE.md rule 9).

## Input
- Clips: IMG_2412, 2416, 2417, 2419, 2420 (5 clips, one gym, one climber group not recorded in the repo), 30 fps (real timestamps used).
- Joints: validation/output_full_lock1 re-run through occlusion_handling.fill_gaps (D-030 defaults), compared row by row with
  evaluation/3.2/output_full_lock1_d030/*_filled.csv (result in manifest.txt).
- CoM: E-dL at 0.5 (`metrics.compute_com_working`, male) and proxy A (`metrics.compute_com`), same windows for both.

## Windows
- Non-overlapping tiles of 1 s, 2 s, 3 s (30, 60, 90 frames) from frame 0. A tile is kept when the CoM is defined on every frame
  and no body joint is flagged tier3 on any frame of it. Rejected tiles are dropped, not re-slid. Speeds = the L-1 step speeds inside
  the tile (`metrics.speed_profile` convention; checked numerically against it).
- Slow / middle / fast = lower / middle / upper tercile of each window's peak CoM speed (E-dL, filled, unfiltered, noise-free, in torso
  lengths per second, clip median torso length), terciles taken per duration pooled over the five clips. Class is fixed once and used for
  A, every filter and every noise level.

## Noise (levels bracket the measured noise, noise_estimates.csv)
- Measured before the sweep (`measure_noise.py`, slow-stretch residual of a cubic fit and a second-difference estimate): see noise_estimates.csv.
  The proposed levels 0.005 to 0.04 did not reach the second-difference estimates (0.0017 to 0.0026), so one lower level was added.
- Gaussian, independent per frame and per axis, added to the CoM positions BEFORE any filter, sd = level x clip median torso length.
  Levels (torso lengths): 0.0025, 0.005, 0.01, 0.02, 0.04. 20 seeds, seed = 20261126 + k (k = 0 to 19), the same unit-normal array
  scaled by each level (nested), generated per clip with numpy default_rng. Added noise is on top of the clips' own noise.

## Filters (applied to CoM positions, per maximal run of defined frames, before the finite-difference speed; chosen in advance, not tuned)
- none
- Savitzky-Golay, polynomial order 3 (not swept), smoothing only (no SG derivative), window 0.2 s, 0.4 s, 0.8 s = 7, 13, 25 frames
  (the nearest odd frame counts at 30 fps; actual 0.23, 0.43, 0.83 s), `scipy.signal.savgol_filter(mode="interp")`.
- Butterworth low-pass, zero-phase (`sosfiltfilt`), order 2 (so 4th-order magnitude after the forward-backward pass), cut-off 3, 6, 12 Hz
  (nominal fs = 30 Hz; -6 dB at the cut-off after the double pass).
- SPARC interaction (stated before running): SPARC integrates the speed spectrum only up to an adaptive cut-off capped at 10 Hz
  (omega_c_max = 20 pi rad/s). A Butterworth cut-off below 10 Hz (3 and 6 Hz) attenuates the spectrum above it, so the adaptive cut-off
  will be set by the filter and not by the data; at 12 Hz the filter lies above the cap and SPARC's own 10 Hz cap acts. The share of windows whose adaptive
  cut-off reaches the cap is recorded for every setting.

## SPARC parameters, one at a time around the reference defaults (amp_th 0.05, 10 Hz, padlevel 4), unfiltered, E-dL only
- amp_th: 0.01, 0.02, 0.05 (reference), 0.10, 0.20. fc: 3, 5, 10 (reference), 15 Hz (15 = Nyquist at 30 fps). padlevel: 0, 2, 4 (reference), 6, 8.
  All 11 distinct settings are reported (the reference once). Each setting gets the full noise grid.

## Reported
Noise-free LDLJ and SPARC per clip (descriptive) first; rank order of the five clips (rank 1 = smoothest, higher LDLJ / SPARC);
change by noise alone, by filtering alone; Spearman rank correlation of the clip order against the noise-free unfiltered order (n = 5, no
p-value) and the share of seeds with an identical order; slow versus fast windows at the same noise; metric versus peak speed (Spearman,
descriptive). Counts of windows and clips throughout.
