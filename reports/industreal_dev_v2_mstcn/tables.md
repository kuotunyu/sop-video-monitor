Development result on `val` (16 videos, 5 participants, 38036 sampled frames); features `dinov2_vits14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 20.0 [15.5, 25.3] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| mstcn_causal | 34.8 [31.8, 38.0] | 30.2 [29.9, 30.6] | 28.3 [25.1, 31.5] | 23.6 [19.9, 27.0] | 15.4 [12.7, 18.2] |
| mstcn_offline | 39.1 [36.8, 42.5] | 35.2 [32.1, 37.4] | 37.4 [35.0, 40.2] | 33.1 [30.3, 36.5] | 23.5 [20.5, 27.0] |

| run | what it uses |
|---|---|
| majority | constant most frequent training label |
| mstcn_causal | MS-TCN++ with left-only padding: frame t sees frames <= t (online) |
| mstcn_offline | MS-TCN++ with symmetric padding: sees future frames (not an online result) |

Per-video MoF:

| video | majority | mstcn_causal | mstcn_offline |
|---|---|---|---|
| 05_assy_0_1 | 25.2 | 32.2 | 37.0 |
| 05_assy_2_2 | 22.7 | 40.3 | 42.1 |
| 05_main_0_1 | 17.0 | 32.2 | 40.8 |
| 14_assy_0_1 | 8.4 | 31.4 | 33.6 |
| 14_main_0_1 | 15.1 | 31.5 | 38.3 |
| 14_main_2_2 | 19.6 | 40.0 | 33.9 |
| 14_main_2_3 | 12.7 | 36.7 | 45.5 |
| 20_assy_0_1 | 31.2 | 32.9 | 40.7 |
| 20_assy_3_6 | 27.8 | 45.9 | 52.8 |
| 20_main_0_1 | 26.8 | 41.0 | 40.5 |
| 24_assy_0_1 | 16.3 | 39.9 | 45.9 |
| 24_assy_2_4 | 14.7 | 36.5 | 33.9 |
| 24_main_0_1 | 17.2 | 27.4 | 32.2 |
| 26_assy_0_1 | 15.6 | 23.8 | 33.6 |
| 26_assy_1_5 | 22.7 | 30.8 | 34.2 |
| 26_main_0_1 | 18.6 | 40.8 | 46.4 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 7 | -1 | 884 |
| -1 | 7 | 682 |
| 1 | -1 | 439 |
| -1 | 4 | 361 |
| -1 | 1 | 358 |
| 34 | 6 | 357 |
| 44 | -1 | 350 |
| -1 | 17 | 320 |
| -1 | 5 | 279 |
| -1 | 21 | 262 |
| 7 | 17 | 259 |
| 7 | 21 | 256 |
| 6 | -1 | 254 |
| -1 | 6 | 248 |
| -1 | 32 | 248 |
