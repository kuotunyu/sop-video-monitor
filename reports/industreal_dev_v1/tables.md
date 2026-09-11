Development result on `val` (16 videos, 5 participants, 38036 sampled frames); features `dinov2_vits14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 20.0 [15.5, 25.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| frame | 32.4 [31.7, 33.0] | 11.0 [10.1, 11.8] | 11.9 [11.1, 12.8] | 7.5 [6.8, 8.3] | 4.1 [3.3, 5.0] |
| causal | 34.0 [32.7, 35.0] | 23.1 [21.6, 24.9] | 24.1 [22.2, 26.3] | 18.9 [17.1, 20.9] | 10.6 [8.9, 12.6] |
| offline | 35.7 [34.3, 36.9] | 27.6 [25.5, 30.0] | 29.2 [27.0, 31.6] | 24.5 [22.0, 27.3] | 15.4 [13.2, 17.8] |

| run | what it uses |
|---|---|
| majority | constant most frequent training label |
| frame | argmax of the per-frame posterior (no temporal context) |
| causal | argmax of the mean posterior over the last 5 sampled frames (past only) |
| offline | argmax of the mean posterior over +-5 sampled frames (uses future frames; not an online result) |

Per-video MoF:

| video | majority | frame | causal | offline |
|---|---|---|---|---|
| 05_assy_0_1 | 25.2 | 32.9 | 36.3 | 37.0 |
| 05_assy_2_2 | 22.7 | 33.8 | 34.2 | 38.2 |
| 05_main_0_1 | 17.0 | 32.8 | 34.9 | 34.4 |
| 14_assy_0_1 | 8.4 | 32.2 | 32.1 | 34.1 |
| 14_main_0_1 | 15.1 | 28.8 | 28.7 | 28.5 |
| 14_main_2_2 | 19.6 | 29.3 | 31.8 | 33.3 |
| 14_main_2_3 | 12.7 | 32.6 | 33.1 | 35.3 |
| 20_assy_0_1 | 31.2 | 26.6 | 28.0 | 29.2 |
| 20_assy_3_6 | 27.8 | 35.4 | 37.8 | 39.7 |
| 20_main_0_1 | 26.8 | 35.0 | 36.2 | 38.7 |
| 24_assy_0_1 | 16.3 | 36.6 | 37.5 | 40.7 |
| 24_assy_2_4 | 14.7 | 31.4 | 32.9 | 33.5 |
| 24_main_0_1 | 17.2 | 30.1 | 30.6 | 31.9 |
| 26_assy_0_1 | 15.6 | 27.4 | 29.6 | 31.3 |
| 26_assy_1_5 | 22.7 | 35.1 | 37.5 | 40.4 |
| 26_main_0_1 | 18.6 | 37.4 | 39.5 | 39.9 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| -1 | 7 | 1084 |
| 7 | -1 | 503 |
| -1 | 6 | 502 |
| 7 | 29 | 357 |
| 1 | -1 | 349 |
| 34 | 6 | 342 |
| -1 | 2 | 311 |
| 1 | 7 | 308 |
| 44 | -1 | 302 |
| -1 | 43 | 288 |
| -1 | 1 | 265 |
| 3 | 7 | 258 |
| -1 | 4 | 253 |
| 6 | 34 | 240 |
| -1 | 71 | 209 |
