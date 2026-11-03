# Gated-lock run report (validation/output_full_lock1), setting `g0=0.25 v=0 s=1.75 K=10 c=0.1 N=off`

## 1. Replay comparison (run vs offline replay of the selected setting)

| clip | frames | boxes in both | box differs | box in only one | tier differs | first differing frame |
|---|---|---|---|---|---|---|
| IMG_2412 | 718 | 718 | 0 | 0 | 0 | None |
| IMG_2416 | 917 | 917 | 0 | 0 | 0 | None |
| IMG_2417 | 980 | 980 | 0 | 0 | 0 | None |
| IMG_2419 | 850 | 848 | 0 | 0 | 0 | None |
| IMG_2420 | 2421 | 2421 | 0 | 0 | 0 | None |

## 2. First 100 frames (frames the legacy rule tracked must be identical in every column)

| clip | reference | legacy-tracked frames | frames differing | rows differing | newly tracked | legacy-tracked, run not |
|---|---|---|---|---|---|---|
| IMG_2412 | validation/output (100 frames) | 98 | 0 | 0 | 41-42 |  |
| IMG_2412 | validation/output_full | 98 | 0 | 0 | 41-42 |  |
| IMG_2416 | validation/output (100 frames) | 100 | 0 | 0 |  |  |
| IMG_2416 | validation/output_full | 100 | 0 | 0 |  |  |
| IMG_2417 | validation/output (100 frames) | 100 | 0 | 0 |  |  |
| IMG_2417 | validation/output_full | 100 | 0 | 0 |  |  |
| IMG_2419 | validation/output (100 frames) | 100 | 0 | 0 |  |  |
| IMG_2419 | validation/output_full | 100 | 0 | 0 |  |  |
| IMG_2420 | validation/output (100 frames) | 99 | 0 | 0 | 86 |  |
| IMG_2420 | validation/output_full | 99 | 0 | 0 | 86 |  |

## 3. Per clip

| clip | frames | legacy dropped | legacy longest run | gated dropped | gated longest run | jump max L/G | jump p99 L/G | jumps>0.3 L/G | h-ratio flags L/G | gained | lost | replaced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IMG_2412 | 718 | 2 | 2 | 0 | 0 | 0.1806 / 0.1806 | 0.0888 / 0.1065 | 0 / 0 | 0 / 0 | 2 | 0 | 0 |
| IMG_2416 | 917 | 24 | 8 | 0 | 0 | 0.2047 / 0.2047 | 0.0745 / 0.0883 | 0 / 0 | 0 / 0 | 24 | 0 | 0 |
| IMG_2417 | 980 | 0 | 0 | 0 | 0 | 0.1236 / 0.1236 | 0.0605 / 0.0605 | 0 / 0 | 0 / 0 | 0 | 0 | 0 |
| IMG_2419 | 850 | 7 | 5 | 2 | 2 | 3.1514 / 0.1813 | 0.0665 / 0.0648 | 1 / 0 | 11 / 0 | 7 | 2 | 16 |
| IMG_2420 | 2421 | 7 | 3 | 0 | 0 | 0.1769 / 0.1769 | 0.0456 / 0.0492 | 0 / 0 | 0 / 0 | 7 | 0 | 0 |

## 4. Pose quality: share of keypoints at or above 0.5

| clip | frame set | frames | % all 17 | % 8 pipeline joints |
|---|---|---|---|---|
| IMG_2412 | all frames, gated run | 718 | 95.3 | 98.3 |
| IMG_2412 | all frames, legacy run | 716 | 95.3 | 98.3 |
| IMG_2412 | newly tracked (legacy had no box) | 2 | 100.0 | 100.0 |
| IMG_2412 | rescue-tier frames | 2 | 100.0 | 100.0 |
| IMG_2416 | all frames, gated run | 917 | 98.3 | 99.0 |
| IMG_2416 | all frames, legacy run | 893 | 98.3 | 99.0 |
| IMG_2416 | newly tracked (legacy had no box) | 24 | 99.0 | 100.0 |
| IMG_2416 | rescue-tier frames | 24 | 99.0 | 100.0 |
| IMG_2417 | all frames, gated run | 980 | 94.9 | 98.5 |
| IMG_2417 | all frames, legacy run | 980 | 94.9 | 98.5 |
| IMG_2419 | all frames, gated run | 848 | 97.7 | 99.4 |
| IMG_2419 | all frames, legacy run | 843 | 97.8 | 99.5 |
| IMG_2419 | newly tracked (legacy had no box) | 7 | 95.8 | 96.4 |
| IMG_2419 | rescue-tier frames | 10 | 94.1 | 95.0 |
| IMG_2419 | 2419 window 592-614 frames with a changed box | 16 | 92.3 | 97.7 |
| IMG_2420 | all frames, gated run | 2421 | 98.8 | 99.6 |
| IMG_2420 | all frames, legacy run | 2414 | 98.8 | 99.6 |
| IMG_2420 | newly tracked (legacy had no box) | 7 | 85.7 | 94.6 |
| IMG_2420 | rescue-tier frames | 7 | 85.7 | 94.6 |

## 5. Rejection log (<clip>_lock.csv)

- IMG_2412: 0 rejection row(s)
- IMG_2416: 0 rejection row(s)
- IMG_2417: 0 rejection row(s)
- IMG_2419: 5 rejection row(s); gate+size: 5 frame(s) (597-601)
- IMG_2420: 0 rejection row(s)
