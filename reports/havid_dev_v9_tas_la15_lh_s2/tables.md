Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 2).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [26.9, 45.4] | 8.3 [6.6, 10.1] | 9.4 [4.8, 14.0] | 3.6 [0.6, 7.9] | 0.7 [0.0, 2.6] |
| fusion_causal | 38.4 [33.1, 44.6] | 30.7 [25.4, 38.7] | 31.7 [23.7, 43.5] | 25.6 [18.1, 35.6] | 16.8 [10.5, 25.0] |
| fusion_conf_causal | 36.8 [31.5, 44.3] | 31.1 [24.4, 39.5] | 31.7 [22.8, 43.5] | 26.1 [17.7, 37.4] | 16.1 [9.4, 24.8] |
| fusion_geo_causal | 40.1 [35.0, 45.7] | 37.5 [31.3, 45.1] | 35.5 [27.0, 45.8] | 29.4 [21.7, 38.2] | 18.3 [9.9, 27.1] |
| view0_causal | 32.5 [28.1, 35.6] | 33.3 [27.9, 38.2] | 28.1 [22.3, 33.6] | 22.6 [18.3, 27.5] | 11.1 [8.7, 14.5] |
| view1_causal | 36.0 [33.2, 39.1] | 31.3 [26.5, 36.5] | 30.3 [23.9, 37.5] | 25.5 [19.4, 32.5] | 13.6 [11.7, 16.2] |
| view2_causal | 33.9 [27.2, 42.4] | 30.0 [23.5, 38.7] | 30.5 [21.4, 42.0] | 24.7 [14.7, 37.7] | 14.1 [7.4, 22.6] |

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
| S01A04I01 | 41.2 | 43.6 | 37.0 | 46.8 | 41.2 | 38.6 | 33.8 |
| S01A05I01 | 47.4 | 39.6 | 35.3 | 41.0 | 54.7 | 48.3 | 14.1 |
| S01A06I01 | 63.7 | 11.5 | 11.2 | 11.1 | 7.0 | 21.2 | 11.9 |
| S08A04I01 | 18.1 | 61.4 | 62.2 | 61.6 | 29.9 | 43.9 | 61.2 |
| S08A05I01 | 25.4 | 32.3 | 34.4 | 32.8 | 23.4 | 31.0 | 32.1 |
| S08A06I01 | 23.6 | 49.3 | 47.9 | 43.3 | 37.3 | 37.7 | 48.8 |
| S10A04I01 | 33.7 | 30.4 | 29.4 | 32.9 | 38.3 | 34.9 | 28.9 |
| S10A05I01 | 35.0 | 40.4 | 40.4 | 42.7 | 45.9 | 58.1 | 39.6 |
| S10A06I01 | 26.0 | 22.3 | 22.2 | 23.6 | 16.8 | 19.0 | 22.9 |
| S12A04I01 | 62.5 | 62.5 | 62.5 | 62.5 | 31.0 | 58.6 | 62.5 |
| S12A05I01 | 70.6 | 50.6 | 42.4 | 54.8 | 51.1 | 57.4 | 32.5 |
| S12A06I01 | 28.7 | 24.4 | 23.8 | 27.5 | 28.7 | 22.7 | 24.1 |
| S18A04I01 | 14.7 | 20.4 | 20.5 | 20.4 | 12.2 | 19.3 | 20.0 |
| S18A05I01 | 31.6 | 35.1 | 35.9 | 31.6 | 6.7 | 8.2 | 39.6 |
| S18A06I01 | 22.9 | 66.8 | 65.7 | 71.6 | 38.0 | 63.3 | 65.3 |
| S30A04I01 | 48.5 | 50.3 | 48.9 | 55.7 | 35.9 | 55.5 | 44.0 |
| S30A05I01 | 41.1 | 44.1 | 42.6 | 46.3 | 49.1 | 35.7 | 35.1 |
| S30A06I01 | 30.1 | 34.2 | 33.4 | 38.9 | 21.9 | 12.3 | 30.3 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 27 | 71 | 416 |
| 27 | 73 | 398 |
| 1 | 27 | 306 |
| 55 | 27 | 296 |
| 74 | 27 | 282 |
| 41 | 27 | 273 |
| 64 | 27 | 202 |
| 70 | 71 | 193 |
| 68 | 27 | 167 |
| 12 | 27 | 164 |
| 24 | 27 | 161 |
| 47 | 50 | 157 |
| 49 | 50 | 152 |
| 58 | 27 | 151 |
| 27 | 51 | 150 |
