Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 2).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.7, 34.1] | 8.0 [6.8, 9.4] | 8.9 [5.9, 11.9] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 38.1 [29.8, 49.7] | 31.7 [26.2, 36.5] | 30.3 [23.7, 39.5] | 26.2 [19.4, 36.4] | 16.1 [10.6, 24.2] |
| fusion_conf_causal | 37.8 [29.5, 49.2] | 30.9 [26.2, 34.6] | 30.2 [23.6, 39.2] | 24.7 [18.6, 33.4] | 14.7 [8.8, 22.6] |
| fusion_geo_causal | 39.8 [32.2, 50.2] | 35.2 [28.0, 41.6] | 32.8 [25.3, 42.4] | 25.0 [17.2, 35.9] | 16.9 [9.3, 27.4] |
| view0_causal | 29.5 [22.0, 39.5] | 30.5 [21.5, 40.7] | 28.1 [20.1, 38.9] | 21.9 [14.0, 33.2] | 12.2 [7.4, 19.7] |
| view1_causal | 33.0 [26.2, 42.4] | 35.8 [28.8, 42.3] | 30.1 [23.7, 37.8] | 24.1 [19.1, 30.6] | 14.9 [8.7, 22.9] |
| view2_causal | 36.9 [30.8, 46.6] | 34.4 [28.1, 42.2] | 33.3 [27.8, 41.8] | 24.9 [20.1, 32.7] | 15.3 [10.3, 24.1] |

| run | what it uses |
|---|---|
| majority | constant most frequent training label |
| fusion_causal | mean of the per-view causal posteriors (late fusion, online) |
| fusion_conf_causal | per-frame confidence-weighted mean of the per-view causal posteriors, weight = each view's max posterior, online |
| fusion_geo_causal | normalised geometric mean of the per-view causal posteriors (product of experts, online) |
| view0_causal | MS-TCN++ on the side view (view 0), left-only padding, output delayed by 45 frames: frame t is labelled from frames <= t + 45 (3.0 s latency) |
| view1_causal | MS-TCN++ on the front view (view 1), left-only padding, output delayed by 45 frames: frame t is labelled from frames <= t + 45 (3.0 s latency) |
| view2_causal | MS-TCN++ on the top view (view 2), left-only padding, output delayed by 45 frames: frame t is labelled from frames <= t + 45 (3.0 s latency) |

Per-video MoF:

| video | majority | fusion_causal | fusion_conf_causal | fusion_geo_causal | view0_causal | view1_causal | view2_causal |
|---|---|---|---|---|---|---|---|
| S01A04I01 | 22.0 | 22.0 | 21.6 | 18.8 | 9.7 | 16.8 | 23.3 |
| S01A05I01 | 55.7 | 46.3 | 43.2 | 56.1 | 36.4 | 35.1 | 55.3 |
| S01A06I01 | 19.4 | 19.5 | 19.2 | 18.3 | 14.5 | 20.3 | 19.6 |
| S08A04I01 | 34.3 | 46.9 | 46.4 | 52.9 | 38.5 | 37.4 | 52.7 |
| S08A05I01 | 47.9 | 58.3 | 53.9 | 59.0 | 31.8 | 32.5 | 54.7 |
| S08A06I01 | 36.7 | 67.1 | 67.6 | 65.3 | 58.1 | 64.3 | 65.0 |
| S10A04I01 | 40.8 | 31.9 | 32.8 | 34.5 | 26.8 | 31.3 | 30.6 |
| S10A05I01 | 27.3 | 40.4 | 39.7 | 42.4 | 35.6 | 28.0 | 34.7 |
| S10A06I01 | 27.9 | 19.8 | 19.6 | 23.9 | 15.8 | 18.4 | 26.2 |
| S12A04I01 | 17.5 | 15.5 | 17.7 | 15.8 | 21.6 | 17.2 | 15.5 |
| S12A05I01 | 23.2 | 51.6 | 51.6 | 55.1 | 44.0 | 43.8 | 51.6 |
| S12A06I01 | 27.8 | 17.3 | 16.4 | 25.9 | 13.5 | 22.3 | 18.2 |
| S18A04I01 | 10.9 | 15.5 | 15.2 | 18.1 | 7.3 | 8.8 | 24.4 |
| S18A05I01 | 15.5 | 32.0 | 31.8 | 17.0 | 10.1 | 11.0 | 32.7 |
| S18A06I01 | 25.9 | 80.3 | 81.6 | 81.2 | 49.1 | 81.6 | 54.6 |
| S30A04I01 | 16.7 | 24.2 | 24.7 | 23.6 | 35.1 | 24.9 | 41.3 |
| S30A05I01 | 50.1 | 67.3 | 67.3 | 66.7 | 56.9 | 52.8 | 48.2 |
| S30A06I01 | 15.4 | 65.4 | 64.7 | 63.1 | 48.8 | 65.0 | 34.5 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 41 | 27 | 399 |
| 72 | 27 | 268 |
| 61 | 27 | 255 |
| 1 | 27 | 253 |
| 51 | 52 | 244 |
| 53 | 27 | 244 |
| 67 | 27 | 223 |
| 27 | 71 | 207 |
| 27 | 51 | 187 |
| 74 | 27 | 176 |
| 58 | 68 | 175 |
| 14 | 27 | 170 |
| 61 | 68 | 166 |
| 49 | 47 | 156 |
| 47 | 49 | 156 |
