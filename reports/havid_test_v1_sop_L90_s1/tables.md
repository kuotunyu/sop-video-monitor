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

Synthetic violations on `pred` sequences (40 test recordings, seed 0); recall = perturbed step flagged by the matching check, false alarm = unperturbed recording flagged by that check; Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 23 | 23 | 100.0 [86, 100] | 33 / 40 | 82.5 [68, 91] | 41.1 | 123 / 533 = 23.1 [20, 27] |
| omission | 30 | 30 | 100.0 [89, 100] | 40 / 40 | 100.0 [91, 100] | 42.9 | — |
| duration | 40 | 40 | 100.0 [91, 100] | 39 / 40 | 97.5 [87, 100] | 50.6 | 217 / 533 = 40.7 [37, 45] |

Unperturbed `pred` recordings with any finding: 40 / 40.

Native `w` (wrong) segments on test, both hands, predicted `w` overlapping a ground-truth `w` counts as detected — underpowered (fewer than 20 segments):

| ground-truth `w` segments | detected | recall | predicted `w` segments | false predicted | precision |
|---|---|---|---|---|---|
| 16 | 0 | 0.0 [0, 19] | 0 | 0 | — [0, 0] |

Recording level, unperturbed sequences: recordings flagged by each check, split by whether the ground truth contains a native `w` segment (counts only; `w` need not break order, completeness or timing):

| source | group | recordings | order | omission | duration | any |
|---|---|---|---|---|---|---|
| gt | with wrong | 7 | 2 | 1 | 5 | 5 |
| gt | without wrong | 34 | 5 | 3 | 18 | 21 |
| pred | with wrong | 6 | 5 | 6 | 6 | 6 |
| pred | without wrong | 34 | 28 | 34 | 33 | 34 |
