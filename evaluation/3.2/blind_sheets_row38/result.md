# Blind sheets, gaps row 38: result (a visual judgement by one person, not an error measurement)

| set | new Kalman closer | old spline closer | equal | cannot tell |
|---|---|---|---|---|
| random10 | 1 | 3 | 5 | 1 |
| top10_disagreement | 4 | 2 | 2 | 2 |

Rule (fixed before the sheets were made): D-030 stands if the new Kalman fill is better or equal on at least 5 of the 10 random sheets. Random set: 6 of 10 -> **D-030 stands** under that rule.

| sheet | set | clip | joint | frame | judgement | closer | note |
|---|---|---|---|---|---|---|---|
| R01 | random10 | IMG_2419.MOV | left_elbow | 599 | equal | equal | B is more on it, but the author called them equal since ring size does not matter |
| R02 | random10 | IMG_2419.MOV | left_wrist | 599 | B | old_spline |  |
| R03 | random10 | IMG_2417.MOV | left_wrist | 774 | B | old_spline |  |
| R04 | random10 | IMG_2420.MOV | right_elbow | 2238 | B | old_spline | both a bit far off, B closer |
| R05 | random10 | IMG_2420.MOV | right_wrist | 145 | equal | equal | both a bit far off, equally far off |
| R06 | random10 | IMG_2412.MOV | left_wrist | 631 | cannot_tell | cannot_tell | an anomaly, cannot tell |
| R07 | random10 | IMG_2416.MOV | left_wrist | 613 | equal | equal |  |
| R08 | random10 | IMG_2416.MOV | left_elbow | 492 | equal | equal |  |
| R09 | random10 | IMG_2412.MOV | right_ankle | 2 | B | new_kalman |  |
| R10 | random10 | IMG_2417.MOV | left_wrist | 767 | equal | equal |  |
| T01 | top10_disagreement | IMG_2420.MOV | left_elbow | 2318 | cannot_tell | cannot_tell | no single verdict given: the two following frames' observed points are far apart; A predicts the next frame (2319) better, B predicts the one after (2320) better. Mapped to cannot_tell by the assistant; top set does not enter the rule |
| T02 | top10_disagreement | IMG_2420.MOV | left_knee | 77 | B | new_kalman |  |
| T03 | top10_disagreement | IMG_2416.MOV | left_wrist | 652 | A | new_kalman | the white rings on the following frames are not on the body |
| T04 | top10_disagreement | IMG_2420.MOV | left_wrist | 2318 | A | old_spline | B is far off |
| T05 | top10_disagreement | IMG_2416.MOV | left_wrist | 653 | B | new_kalman | B much closer; the white ring on the following frames is not on the body |
| T06 | top10_disagreement | IMG_2420.MOV | left_wrist | 2319 | B | old_spline | B is bang on, A is far |
| T07 | top10_disagreement | IMG_2420.MOV | left_wrist | 102 | equal | equal | both equally far off |
| T08 | top10_disagreement | IMG_2420.MOV | left_elbow | 2317 | equal | equal | same as T07 |
| T09 | top10_disagreement | IMG_2417.MOV | left_knee | 163 | cannot_tell | cannot_tell | fast movement; both are off but in the direction of motion, between the earlier and later observed rings; hard to tell, both close to each other |
| T10 | top10_disagreement | IMG_2420.MOV | left_knee | 78 | B | new_kalman |  |
