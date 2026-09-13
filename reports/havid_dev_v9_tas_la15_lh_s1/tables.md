Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 1).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [27.5, 45.4] | 8.3 [6.7, 10.1] | 9.4 [5.0, 14.0] | 3.6 [0.7, 7.6] | 0.7 [0.0, 2.5] |
| fusion_causal | 40.8 [37.6, 44.6] | 30.4 [25.0, 35.6] | 29.7 [23.2, 35.7] | 24.9 [19.7, 29.4] | 15.3 [10.8, 20.1] |
| fusion_conf_causal | 39.4 [35.4, 44.0] | 31.8 [25.8, 37.8] | 28.0 [20.2, 35.1] | 22.8 [17.3, 28.0] | 14.9 [9.1, 20.9] |
| fusion_geo_causal | 41.6 [38.4, 45.5] | 33.9 [25.9, 42.5] | 36.6 [28.8, 42.9] | 28.5 [22.2, 33.6] | 16.5 [10.4, 22.2] |
| view0_causal | 33.6 [29.7, 37.5] | 33.2 [29.6, 36.8] | 27.0 [22.6, 31.4] | 22.0 [16.8, 27.0] | 13.5 [8.9, 18.3] |
| view1_causal | 37.5 [34.6, 41.2] | 33.2 [28.9, 37.7] | 31.1 [25.9, 36.5] | 22.6 [19.2, 26.8] | 13.4 [11.0, 16.2] |
| view2_causal | 37.4 [33.4, 42.1] | 33.7 [28.5, 38.0] | 32.5 [23.6, 40.5] | 26.1 [18.4, 33.2] | 17.1 [11.6, 21.8] |

| run | what it uses |
|---|---|
| majority | constant most frequent training label |
| fusion_causal | mean of the per-view causal posteriors (late fusion, online) |
| fusion_conf_causal | per-frame confidence-weighted mean of the per-view causal posteriors, weight = each view's max posterior, online |
| fusion_geo_causal | normalised geometric mean of the per-view causal posteriors (product of experts, online) |
| view0_causal | MS-TCN++ on the side view (view 0), left-only padding, output delayed by 15 frames: frame t is labelled from frames <= t + 15 (1.0 s latency) |
| view1_causal | MS-TCN++ on the front view (view 1), left-only padding, output delayed by 15 frames: frame t is labelled from frames <= t + 15 (1.0 s latency) |
| view2_causal | MS-TCN++ on the top view (view 2), left-only padding, output delayed by 15 frames: frame t is labelled from frames <= t + 15 (1.0 s latency) |

Per-video MoF:

| video | majority | fusion_causal | fusion_conf_causal | fusion_geo_causal | view0_causal | view1_causal | view2_causal |
|---|---|---|---|---|---|---|---|
| S01A04I01 | 41.2 | 45.9 | 45.9 | 45.6 | 46.4 | 41.4 | 41.4 |
| S01A05I01 | 47.4 | 36.1 | 35.5 | 37.4 | 33.7 | 24.8 | 29.0 |
| S01A06I01 | 63.7 | 19.4 | 16.7 | 24.2 | 2.9 | 39.1 | 13.9 |
| S08A04I01 | 18.1 | 50.4 | 51.1 | 52.2 | 47.3 | 52.6 | 53.8 |
| S08A05I01 | 25.4 | 35.5 | 35.6 | 34.4 | 37.4 | 23.6 | 28.9 |
| S08A06I01 | 23.6 | 43.2 | 44.5 | 42.6 | 32.1 | 46.5 | 41.0 |
| S10A04I01 | 33.7 | 46.4 | 37.2 | 45.0 | 31.2 | 38.2 | 44.5 |
| S10A05I01 | 35.0 | 41.6 | 41.2 | 41.0 | 32.9 | 50.8 | 45.0 |
| S10A06I01 | 26.0 | 23.8 | 24.1 | 26.0 | 19.7 | 20.7 | 26.1 |
| S12A04I01 | 62.5 | 59.8 | 41.1 | 62.5 | 26.7 | 46.8 | 62.5 |
| S12A05I01 | 70.6 | 69.7 | 76.1 | 73.7 | 77.3 | 47.7 | 42.6 |
| S12A06I01 | 28.7 | 14.4 | 14.3 | 17.0 | 13.2 | 24.4 | 12.1 |
| S18A04I01 | 14.7 | 15.9 | 16.2 | 13.6 | 7.1 | 16.5 | 33.6 |
| S18A05I01 | 31.6 | 26.5 | 24.9 | 36.1 | 6.7 | 20.0 | 44.5 |
| S18A06I01 | 22.9 | 78.4 | 78.8 | 77.2 | 80.5 | 62.5 | 71.0 |
| S30A04I01 | 48.5 | 49.2 | 47.7 | 47.4 | 32.5 | 49.9 | 45.5 |
| S30A05I01 | 41.1 | 40.2 | 39.5 | 44.9 | 29.7 | 43.2 | 43.7 |
| S30A06I01 | 30.1 | 54.2 | 53.4 | 47.8 | 43.6 | 41.1 | 25.1 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 41 | 27 | 361 |
| 27 | 73 | 346 |
| 74 | 27 | 294 |
| 1 | 27 | 282 |
| 58 | 27 | 276 |
| 27 | 71 | 275 |
| 49 | 27 | 255 |
| 6 | 27 | 254 |
| 65 | 27 | 243 |
| 70 | 71 | 214 |
| 12 | 27 | 183 |
| 64 | 27 | 177 |
| 27 | 49 | 166 |
| 55 | 27 | 160 |
| 24 | 27 | 149 |
