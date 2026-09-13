Learned procedure knowledge (train split):

| plate | steps (graph nodes) | precedence edges | mandatory steps | steps with duration bounds |
|---|---|---|---|---|
| cylinder | 17 | 5 | 5 | 10 |
| gear | 11 | 8 | 8 | 10 |
| general | 13 | 1 | 6 | 12 |

Synthetic violations on `gt` sequences (41 test recordings, seed 0); recall = perturbed step flagged by the matching check, false alarm = unperturbed recording flagged by that check; Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 37 | 37 | 100.0 [91, 100] | 7 / 41 | 17.1 [9, 31] | 84.1 | 31 / 531 = 5.8 [4, 8] |
| omission | 41 | 41 | 100.0 [91, 100] | 4 / 41 | 9.8 [4, 23] | 91.1 | — |
| duration | 41 | 41 | 100.0 [91, 100] | 23 / 41 | 56.1 [41, 70] | 64.1 | 46 / 531 = 8.7 [7, 11] |

Unperturbed `gt` recordings with any finding: 26 / 41.

Synthetic violations on `pred` sequences (41 test recordings, seed 0); recall = perturbed step flagged by the matching check, false alarm = unperturbed recording flagged by that check; Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 23 | 23 | 100.0 [86, 100] | 28 / 41 | 68.3 [53, 80] | 45.1 | 157 / 696 = 22.6 [20, 26] |
| omission | 22 | 22 | 100.0 [85, 100] | 40 / 41 | 97.6 [87, 100] | 35.5 | — |
| duration | 39 | 39 | 100.0 [91, 100] | 39 / 41 | 95.1 [84, 99] | 50.0 | 389 / 696 = 55.9 [52, 60] |

Unperturbed `pred` recordings with any finding: 41 / 41.

Native `w` (wrong) segments on test, both hands, predicted `w` overlapping a ground-truth `w` counts as detected — underpowered (fewer than 20 segments):

| ground-truth `w` segments | detected | recall | predicted `w` segments | false predicted | precision |
|---|---|---|---|---|---|
| 16 | 0 | 0.0 [0, 19] | 2 | 2 | 0.0 [0, 66] |

Recording level, unperturbed sequences: recordings flagged by each check, split by whether the ground truth contains a native `w` segment (counts only; `w` need not break order, completeness or timing):

| source | group | recordings | order | omission | duration | any |
|---|---|---|---|---|---|---|
| gt | with wrong | 7 | 2 | 1 | 5 | 5 |
| gt | without wrong | 34 | 5 | 3 | 18 | 21 |
| pred | with wrong | 7 | 4 | 7 | 7 | 7 |
| pred | without wrong | 34 | 24 | 33 | 32 | 34 |
