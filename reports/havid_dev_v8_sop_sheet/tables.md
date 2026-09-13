Learned procedure knowledge (train split):

| plate | steps (graph nodes) | precedence edges | mandatory steps | steps with duration bounds |
|---|---|---|---|---|
| cylinder | 17 | 5 | 5 | 10 |
| gear | 11 | 8 | 8 | 10 |
| general | 13 | 1 | 6 | 12 |

Synthetic violations on `gt` sequences (18 val recordings, seed 0); recall = perturbed step flagged by the matching check, false alarm = unperturbed recording flagged by that check; Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 17 | 17 | 100.0 [82, 100] | 4 / 18 | 22.2 [9, 45] | 81.0 | 15 / 234 = 6.4 [4, 10] |
| omission | 17 | 17 | 100.0 [82, 100] | 1 / 18 | 5.6 [1, 26] | 94.4 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 8 / 18 | 44.4 [25, 66] | 69.2 | 10 / 234 = 4.3 [2, 8] |

Unperturbed `gt` recordings with any finding: 10 / 18.

Synthetic violations on `pred` sequences (18 val recordings, seed 0); recall = perturbed step flagged by the matching check, false alarm = unperturbed recording flagged by that check; Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 12 | 12 | 100.0 [76, 100] | 12 / 18 | 66.7 [44, 84] | 50.0 | 56 / 354 = 15.8 [12, 20] |
| omission | 13 | 13 | 100.0 [77, 100] | 17 / 18 | 94.4 [74, 99] | 43.3 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 18 / 18 | 100.0 [82, 100] | 50.0 | 196 / 354 = 55.4 [50, 60] |

Unperturbed `pred` recordings with any finding: 18 / 18.

Native `w` (wrong) segments on val, both hands, predicted `w` overlapping a ground-truth `w` counts as detected — underpowered (fewer than 20 segments):

| ground-truth `w` segments | detected | recall | predicted `w` segments | false predicted | precision |
|---|---|---|---|---|---|
| 12 | 0 | 0.0 [0, 24] | 0 | 0 | — [0, 0] |

Recording level, unperturbed sequences: recordings flagged by each check, split by whether the ground truth contains a native `w` segment (counts only; `w` need not break order, completeness or timing):

| source | group | recordings | order | omission | duration | any |
|---|---|---|---|---|---|---|
| gt | with wrong | 4 | 2 | 1 | 1 | 3 |
| gt | without wrong | 14 | 2 | 0 | 7 | 7 |
| pred | with wrong | 4 | 3 | 4 | 4 | 4 |
| pred | without wrong | 14 | 9 | 13 | 14 | 14 |
