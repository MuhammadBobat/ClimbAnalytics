# This run (D-030) against the previous 3.2 run (D-028 tiers) on the same input

previous: evaluation/3.2/output_full_lock1; this run: evaluation/3.2/output_full_lock1_d030. Same clips, same keypoint CSVs (compare the sha256 lines of the two manifests).

## Joint-frames by flag and gap events by tier, previous -> this run

| clip | observed | tier1 | tier2 | tier3 | unavailable | events_tier1_filled | events_tier1_not_filled | events_tier2_filled | events_tier2_not_filled | events_tier3_filled | events_tier3_not_filled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IMG_2412.MOV | 8471 -> 8471 | 61 -> 0 | 51 -> 112 | 29 -> 29 | 4 -> 4 | 36 -> 0 | 1 -> 0 | 5 -> 41 | 0 -> 1 | 1 -> 0 | 0 -> 1 |
| IMG_2416.MOV | 10919 -> 10919 | 50 -> 0 | 32 -> 82 | 0 -> 0 | 3 -> 3 | 29 -> 0 | 1 -> 0 | 5 -> 34 | 0 -> 1 | 0 -> 0 | 0 -> 0 |
| IMG_2417.MOV | 11603 -> 11603 | 23 -> 0 | 61 -> 84 | 73 -> 73 | 0 -> 0 | 16 -> 0 | 0 -> 0 | 7 -> 23 | 0 -> 0 | 3 -> 0 | 0 -> 3 |
| IMG_2419.MOV | 10126 -> 10126 | 44 -> 0 | 30 -> 74 | 0 -> 0 | 0 -> 0 | 30 -> 0 | 0 -> 0 | 5 -> 35 | 0 -> 0 | 0 -> 0 | 0 -> 0 |
| IMG_2420.MOV | 28935 -> 28935 | 46 -> 0 | 71 -> 117 | 0 -> 0 | 0 -> 0 | 29 -> 0 | 0 -> 0 | 10 -> 39 | 0 -> 0 | 0 -> 0 | 0 -> 0 |
| ALL | 70054 -> 70054 | 224 -> 0 | 245 -> 469 | 102 -> 102 | 7 -> 7 | 140 -> 0 | 2 -> 0 | 32 -> 172 | 0 -> 2 | 4 -> 0 | 0 -> 4 |

Gap events: previous 178, this run 178, identical (clip, joint, first, last): 178. A gap is defined by the observed data only, so the set should be identical.
Events that were filled before and are not now: 4 (expected: the limb gaps over 20 frames); events filled now and not before: 0.

## Old spline fills against new Kalman fills on the frames the previous run flagged tier1 (px, a difference between methods, NOT an error)
frames compared 224; median 4.47, 90th percentile 21.49, maximum 93.35; previous tier1 frames that are not a tier2 fill now: 0.
