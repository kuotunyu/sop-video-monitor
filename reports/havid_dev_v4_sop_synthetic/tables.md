Learned procedure knowledge (train split):

| plate | steps (graph nodes) | precedence edges | mandatory steps | steps with duration bounds |
|---|---|---|---|---|
| cylinder | 31 | 53 | 4 | 23 |
| gear | 16 | 18 | 5 | 15 |
| general | 16 | 7 | 4 | 15 |

Synthetic violations on `gt` sequences (18 val recordings, seed 0); recall = perturbed step flagged by the matching check, false alarm = unperturbed recording flagged by that check; Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 18 | 18 | 100.0 [82, 100] | 16 / 18 | 88.9 [67, 97] | 52.9 | 103 / 234 = 44.0 [38, 50] |
| omission | 17 | 17 | 100.0 [82, 100] | 1 / 18 | 5.6 [1, 26] | 94.4 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 14 / 18 | 77.8 [55, 91] | 56.2 | 27 / 234 = 11.5 [8, 16] |

Unperturbed `gt` recordings with any finding: 17 / 18.

Synthetic violations on `pred` sequences (18 val recordings, seed 0); recall = perturbed step flagged by the matching check, false alarm = unperturbed recording flagged by that check; Wilson 95 % intervals in percent:

| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |
|---|---|---|---|---|---|---|---|
| order | 15 | 15 | 100.0 [80, 100] | 17 / 18 | 94.4 [74, 99] | 46.9 | 207 / 354 = 58.5 [53, 63] |
| omission | 10 | 10 | 100.0 [72, 100] | 17 / 18 | 94.4 [74, 99] | 37.0 | — |
| duration | 18 | 18 | 100.0 [82, 100] | 18 / 18 | 100.0 [82, 100] | 50.0 | 224 / 354 = 63.3 [58, 68] |

Unperturbed `pred` recordings with any finding: 18 / 18.

Native `w` (wrong) segments on val, both hands, predicted `w` overlapping a ground-truth `w` counts as detected — underpowered (fewer than 20 segments):

| ground-truth `w` segments | detected | recall | predicted `w` segments | false predicted | precision |
|---|---|---|---|---|---|
| 12 | 0 | 0.0 [0, 24] | 0 | 0 | — [0, 0] |
