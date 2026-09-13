Learned procedure knowledge (train split):

| plate | steps (graph nodes) | precedence edges | mandatory steps | steps with duration bounds |
|---|---|---|---|---|
| cylinder | 31 | 10 | 4 | 23 |
| gear | 16 | 15 | 5 | 15 |
| general | 16 | 1 | 4 | 15 |

Synthetic violations on `gt` sequences (18 val recordings, seed 0); recall = perturbed step flagged by the matching check, false alarm = unperturbed recording flagged by that check; Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 17 | 17 | 100.0 [82, 100] | 8 / 18 | 44.4 [25, 66] | 68.0 | 20 / 234 = 8.5 [6, 13] |
| omission | 17 | 17 | 100.0 [82, 100] | 1 / 18 | 5.6 [1, 26] | 94.4 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 11 / 18 | 61.1 [39, 80] | 62.1 | 14 / 234 = 6.0 [4, 10] |

Unperturbed `gt` recordings with any finding: 14 / 18.

Synthetic violations on `pred` sequences (18 val recordings, seed 0); recall = perturbed step flagged by the matching check, false alarm = unperturbed recording flagged by that check; Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 12 | 12 | 100.0 [76, 100] | 15 / 18 | 83.3 [61, 94] | 44.4 | 63 / 189 = 33.3 [27, 40] |
| omission | 11 | 11 | 100.0 [74, 100] | 18 / 18 | 100.0 [82, 100] | 37.9 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 17 / 18 | 94.4 [74, 99] | 51.4 | 58 / 189 = 30.7 [25, 38] |

Unperturbed `pred` recordings with any finding: 18 / 18.

Native `w` (wrong) segments on val, both hands, predicted `w` overlapping a ground-truth `w` counts as detected — underpowered (fewer than 20 segments):

| ground-truth `w` segments | detected | recall | predicted `w` segments | false predicted | precision |
|---|---|---|---|---|---|
| 12 | 0 | 0.0 [0, 24] | 0 | 0 | — [0, 0] |
