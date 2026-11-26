# Step 5.1a report — how LDLJ and SPARC react to noise, filtering and speed (D-006 addendum of 2026-11-26)

Written for: the author, as the working record behind the D-006 addendum. Every number is in the CSVs of `output/` (tables copied by code into
`output/tables.md`); protocol fixed and committed before the sweep (`PROTOCOL.md`, commit `78cacd7`); run provenance in `output/manifest.txt`.

**What this is and is not.** A speed-profile stability check: LDLJ and SPARC (the real `metrics.compute_ldlj` / `compute_sparc`) on non-overlapping
fixed windows of 1, 2 and 3 s of the CoM speed profile of the five own clips (E-dL at 0.5 on the D-030 gap-filled data; proxy A as a check). Fixed windows
are **not movement segments**, so this tests how stable the metrics are under noise and filtering, **not how smooth a move is**. The A against E-dL
comparison on movement segments waits for 4.1. No ground truth, no known-groups check (skipped at the author's instruction), no p-values, no pass/fail
thresholds. Five clips of one gym; IMG_2420 supplies 80 of the 188 one-second windows.

## 0. Counts
Windows kept (CoM defined on every frame, no tier3 flag): 1 s 188 (IMG_2412 21, 2416 30, 2417 29, 2419 28, 2420 80), 2 s 92 (10, 15, 13, 14, 40), 3 s 58 (6, 10, 7, 9, 26).
Terciles of peak CoM speed per duration: 1 s 63 slow / 62 middle / 63 fast; 2 s 31 / 30 / 31; 3 s 20 / 19 / 19. Seeds 20, noise levels 5, filters 7 (none, 3 SG, 3 Butterworth),
SPARC settings 12, two CoMs: 338 windows x 101 noise conditions. Re-run fills equal the 3.2 CSVs row for row (0 of 70,632 rows differ); the speed arrays equal `metrics.speed_profile` on all 338 windows.

## 1. Noise measured first (`noise_estimates.csv`)
Slow stretches (lower tercile of 1 s windows), cubic-fit residual sd: 0.0037 to 0.0063 torso lengths (0.9 to 1.4 px); robust second-difference estimate: 0.0017 to 0.0026 (0.4 to 0.5 px).
The proposed levels 0.005 to 0.04 did not reach the second-difference estimates from below, so 0.0025 was added: levels 0.0025, 0.005, 0.01, 0.02, 0.04 torso lengths
(about 0.5 to 10 px). Added noise is on top of the clips' own noise. Pose-model error is not white, so these are scales, not a noise model.

## 2. Noise-free values per clip (descriptive; `output/noise_free_per_clip.csv`, table 1 of `output/tables.md`)
Unfiltered E-dL, 1 s windows, medians: LDLJ −9.47 (2412), −9.59 (2416), −9.54 (2417), −9.58 (2419), −9.85 (2420); SPARC −2.42, −2.42, −2.49, −2.41, −2.53. Rank order (1 = smoothest)
LDLJ 1 4 2 3 5, SPARC 2 3 4 1 5 (clip order 2412 2416 2417 2419 2420). The window-to-window spread (IQR 0.65 to 1.1 LDLJ, 0.4 to 0.6 SPARC at 1 s) is larger than the
spread between clip medians (range 0.38 LDLJ, 0.13 SPARC at 1 s). Both metrics read lower for longer windows of the same clips (LDLJ about −9.5, −11.6, −13.0 and SPARC about −2.4, −3.0, −3.2
at 1, 2, 3 s): values are not comparable across window lengths. The cause was not investigated. Rank orders differ between 1, 2 and 3 s and between the two metrics.

## 3. Noise alone (unfiltered E-dL; table 3)
Median change from the same window's noise-free value (1 s): LDLJ −0.15, −0.33, −0.44, −0.43, −0.40 and SPARC −0.03, −0.09, −0.16, −0.20, −0.23 for the five levels; 2 s and 3 s are in the CSV (SPARC −0.54 at 3 s, level 0.04).
Both metrics read "less smooth" with added noise and flatten above level 0.01 for LDLJ (the clips' own noise already puts LDLJ near a floor). Seed-to-seed sd of a window: LDLJ about 0.3 to 0.5, SPARC 0.14 to 0.32.
At the level that matches the larger measured noise (0.005, 0.0037 to 0.0063 measured) the median change is of the same size as the whole range of clip medians: LDLJ 0.33 against 0.38 (1 s), 0.42 against 0.63 (2 s), 0.40 against 0.86 (3 s); SPARC 0.09 against 0.13, 0.15 against 0.29, 0.20 against 0.64.
Proxy A behaves the same way (1 s LDLJ −0.14 to −0.38, SPARC −0.04 to −0.21 over the levels).

## 4. Filtering alone (noise-free; table 4)
Median filtered-minus-unfiltered change, 1 s, LDLJ: Butterworth 12 / 6 / 3 Hz +0.26 / +1.31 / +2.90; Savitzky-Golay 7 / 13 / 25 frames +1.18 / +2.26 / +3.24. SPARC (1 s): +0.01 / +0.17 / +0.32 and +0.11 / +0.28 / +0.34; 3 s SPARC up to +0.92.
The filter setting moves a window's value by more than the whole between-clip range for every filter except Butterworth 12 Hz on LDLJ, and for SPARC at 1 s only the weakest filters stay below the between-clip range.
With noise added after filtering, the filtered scores fall by more than the unfiltered ones (1 s LDLJ at level 0.01: −0.44 unfiltered, −0.88 Butterworth 6 Hz, −1.25 SG 13 frames).
No filter was chosen: they are a sweep fixed in advance, and no ground truth exists to prefer one.

## 5. Does the clip order survive? (five clips, no p-value; tables 5 and 6, `ranking_stability.csv`)
Noise alone, unfiltered E-dL, mean Spearman rho of the five-clip order against the noise-free order over 20 seeds, level 0.0025 / 0.005 / 0.01 / 0.02 / 0.04:
LDLJ 1 s 0.56 / 0.36 / 0.10 / −0.12 / −0.03, 2 s 0.57 / 0.52 / 0.29 / 0.02 / −0.01, 3 s 0.85 / 0.75 / 0.52 / 0.11 / 0.10; SPARC 1 s 0.82 / 0.74 / 0.54 / 0.22 / −0.09, 2 s 0.84 / 0.63 / 0.52 / 0.12 / −0.24, 3 s 0.86 / 0.74 / 0.54 / 0.16 / −0.14.
The share of the 20 seeds that reproduce the noise-free order exactly is at most 0.25 at every level, and mostly 0 to 0.1.
Filtering alone (noise-free), rho against the unfiltered order, 1 s: LDLJ between 0.1 (Butterworth 3 Hz) and 0.9; SPARC between 0.5 and 0.9; other durations in table 6.
So with only a clip median to go on, neither metric keeps the order of these five clips once the added noise reaches the clips' own noise level, and the order also changes with the filter.
Part of this is that the clips are close together (section 2); it is not shown that the clips truly differ in smoothness.
E-dL against A (unfiltered, noise-free): the clip orders agree only partly (rho 0.3 / 0.7 / 0.8 for LDLJ and 0.2 / 0.1 / 0.6 for SPARC at 1 / 2 / 3 s); the median window difference A minus E-dL is within ±0.06.

## 6. Speed (`metric_vs_peak_speed.csv`, `slow_vs_fast.csv`, table 7)
Noise-free, unfiltered, 1 s: the Spearman correlation of a window's metric with its peak speed is +0.36 for LDLJ and +0.32 for SPARC over the 188 windows (per clip 0.09 to 0.42 and 0.16 to 0.51): faster windows score smoother.
Slow windows are scored less smooth than fast ones with no noise added (1 s medians, 63 windows each: LDLJ −9.85 against −9.37, SPARC −2.60 against −2.30; 5 clips each). For the same added noise:
- the slow-minus-fast gap in the absolute score stays negative at every level for LDLJ with SG 13 frames and Butterworth 6 Hz (1 s, for example −1.60 at level 0.01 and −0.78 at 0.04 for SG 13), so slow windows are scored less smooth at the same noise;
- unfiltered, the gap closes as noise rises (LDLJ −0.60, −0.46, −0.12, +0.02, −0.03 over the five levels), because the slow windows are already near the floor and the fast ones fall to it;
- in the change from a window's own noise-free value, slow windows lose more than fast ones at the lowest level in 21 of 21 LDLJ cells and 18 of 21 SPARC cells (filter x duration cells); over all 105 cells per metric, 73 (LDLJ) and 75 (SPARC) have the slow change more negative.
With a filter and noise added, LDLJ is close to a function of speed: its rho with peak speed rises to 0.82 to 0.87 at level 0.01 (Butterworth 6 Hz and SG 13, all durations; unfiltered 0.17 to 0.77). SPARC stays at 0.25 to 0.44.

## 7. SPARC internal 10 Hz cut-off against the filters (`sparc_cutoff_at_cap.csv`, table 8; stated before running)
Unfiltered, SPARC's adaptive cut-off sits at the 10 Hz cap in 99% of noise-free windows (100% with noise): the spectrum above 10 Hz has not fallen under 0.05 of its DC value, so SPARC is an arc length over 0 to 10 Hz, and V-bar values below 0.05 change nothing (amp 0.01 and 0.02 reproduce the reference value exactly).
A low-pass below 10 Hz moves the cut-off from the data to the filter: noise-free the cap is reached in 17% (Butterworth 3 Hz), 28% (6 Hz), 93% (12 Hz), 53% / 31% / 23% (SG 7 / 13 / 25 frames); added noise pushes these back towards 100% (Butterworth 6 Hz: 0.96 at level 0.04). The SPARC value under such a filter therefore reflects the filter's roll-off and the noise floor more than the movement.

## 8. SPARC parameters, one at a time (all 12 settings in `sparc_parameter_sweep.csv`, table 9)
Reference (amp 0.05, 10 Hz, padlevel 4), 1 s: clip medians −2.42 −2.42 −2.49 −2.41 −2.53. amp 0.01, 0.02: identical. amp 0.10: median change 0.00 / 0.00 / +0.20 at 1 / 2 / 3 s, order rho 1.0 / 0.9 / 0.5; amp 0.20: +0.74 / +1.29 / +1.54, order rho 0.8 / −0.2 / 0.9.
fc 3 / 5 / 15 Hz: +0.68 / +0.44 / −0.37 median change at 1 s, order rho 0.9 / 0.8 / 0.6 at 1 s and 0.3 / 0.6 / 0.7 at 2 s. padlevel 0: +0.63 and order rho 1.0 / 0.9 / 0.9; padlevel 2 within 0.14 of the reference, padlevels 6 and 8 within 0.015. No setting is selected or tuned.

## 9. Recommendation (a judgement from the descriptive results above, no agreed thresholds)
Report **both, with their limits**; no change is made to D-006's primary designation, and no filter is adopted.
- On 30 fps pose-derived CoM, at the clips' own noise level, neither metric holds the order of the five clips under added noise at or above that level, and neither is stable to the filter. The noise-induced change is as large as the between-clip range.
- SPARC changed less in noise relative to its between-clip range than LDLJ (section 3), kept the clip order better at levels up to 0.005 at 1 and 2 s (section 5), and was less tied to speed once a filter and noise were present (section 6). That is a modest, descriptive edge, not a validation.
- LDLJ moves most with the filter (+0.3 to +3.2 against SPARC's +0.0 to +0.9) and, with filtering and noise, tracks speed (rho 0.8). Its reading must not be compared across filters or window lengths.
- SPARC with the reference parameters is mostly a 0 to 10 Hz arc length of a noise-dominated spectrum unless a low-pass is applied, in which case the filter decides its cut-off.
- Neither is shown to measure smoothness of a move; windows are not moves. Use either only to compare like with like (same filter, same window length, same noise conditions).
What would change this: the 4.1 movement segments (the metrics have not been run on them), a filter choice made against something other than these metrics, ground-truth or independently judged smooth versus jerky moves, or 60 fps footage.

## 10. Limits (every one)
Five clips of one gym and a climber group the repo does not record; IMG_2420 contributes 80 of 188 one-second windows; windows are fixed tiles, not moves; terciles are pooled over clips so 3 s slow windows come from 3 clips only; frame timestamps are rounded to 1 ms (33 and 34 ms alternating) and speeds use them as `metrics.compute_velocity` does (not isolated here); the filters assume uniform 30 Hz sampling; noise is white Gaussian on the CoM while real pose error is correlated and not independent between axes; added noise is on top of unknown existing noise; the second-difference noise estimate assumes white noise and the cubic residual includes any path curvature the cubic misses; the choice of levels, windows, filters and tercile rule are the assistant's heuristics (author-approved, not from a source); SPARC sweep run for E-dL unfiltered only; a ranking over five clips cannot support any significance claim; known-groups not done.
