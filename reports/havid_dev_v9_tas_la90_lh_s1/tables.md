Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 1).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [27.5, 45.4] | 8.3 [6.7, 10.1] | 9.4 [5.0, 14.0] | 3.6 [0.7, 7.6] | 0.7 [0.0, 2.5] |
| fusion_causal | 37.1 [32.1, 42.9] | 33.2 [26.7, 39.9] | 32.1 [24.4, 41.4] | 22.9 [15.6, 31.9] | 11.9 [7.5, 17.5] |
| fusion_conf_causal | 37.1 [32.0, 43.2] | 33.4 [27.7, 39.0] | 30.8 [23.9, 38.8] | 22.9 [15.9, 31.6] | 12.8 [8.6, 17.5] |
| fusion_geo_causal | 37.8 [33.2, 42.9] | 33.2 [24.9, 42.3] | 31.6 [22.8, 41.7] | 23.3 [14.9, 33.6] | 12.1 [7.2, 17.6] |
| view0_causal | 32.9 [29.2, 37.1] | 31.3 [24.1, 37.8] | 30.9 [22.8, 39.8] | 22.2 [14.7, 31.0] | 10.9 [5.5, 17.4] |
| view1_causal | 33.4 [26.3, 40.7] | 29.6 [22.4, 34.8] | 30.5 [21.3, 40.8] | 23.2 [13.8, 32.3] | 10.3 [3.4, 18.6] |
| view2_causal | 36.1 [32.9, 38.9] | 35.6 [27.0, 43.1] | 33.0 [27.2, 39.0] | 24.4 [18.4, 29.2] | 14.5 [12.7, 16.4] |

| run | what it uses |
|---|---|
| majority | constant most frequent training label |
| fusion_causal | mean of the per-view causal posteriors (late fusion, online) |
| fusion_conf_causal | per-frame confidence-weighted mean of the per-view causal posteriors, weight = each view's max posterior, online |
| fusion_geo_causal | normalised geometric mean of the per-view causal posteriors (product of experts, online) |
| view0_causal | MS-TCN++ on the side view (view 0), left-only padding, output delayed by 90 frames: frame t is labelled from frames <= t + 90 (6.0 s latency) |
| view1_causal | MS-TCN++ on the front view (view 1), left-only padding, output delayed by 90 frames: frame t is labelled from frames <= t + 90 (6.0 s latency) |
| view2_causal | MS-TCN++ on the top view (view 2), left-only padding, output delayed by 90 frames: frame t is labelled from frames <= t + 90 (6.0 s latency) |

Per-video MoF:

| video | majority | fusion_causal | fusion_conf_causal | fusion_geo_causal | view0_causal | view1_causal | view2_causal |
|---|---|---|---|---|---|---|---|
| S01A04I01 | 41.2 | 29.1 | 30.0 | 30.0 | 28.9 | 25.0 | 39.6 |
| S01A05I01 | 47.4 | 27.7 | 28.2 | 31.5 | 26.4 | 19.2 | 33.3 |
| S01A06I01 | 63.7 | 42.8 | 42.6 | 43.2 | 42.0 | 10.6 | 45.5 |
| S08A04I01 | 18.1 | 55.2 | 54.9 | 56.0 | 53.2 | 53.5 | 49.5 |
| S08A05I01 | 25.4 | 35.1 | 34.3 | 35.9 | 30.5 | 33.8 | 25.4 |
| S08A06I01 | 23.6 | 42.8 | 41.3 | 43.9 | 36.7 | 37.4 | 33.6 |
| S10A04I01 | 33.7 | 38.5 | 35.2 | 40.4 | 31.9 | 26.3 | 40.2 |
| S10A05I01 | 35.0 | 35.0 | 35.0 | 35.0 | 35.0 | 38.1 | 41.2 |
| S10A06I01 | 26.0 | 20.3 | 20.2 | 20.0 | 17.4 | 28.7 | 17.0 |
| S12A04I01 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 |
| S12A05I01 | 70.6 | 60.2 | 60.4 | 56.3 | 49.4 | 47.0 | 41.4 |
| S12A06I01 | 28.7 | 27.4 | 27.3 | 27.3 | 14.9 | 28.1 | 28.1 |
| S18A04I01 | 14.7 | 11.0 | 9.1 | 13.2 | 14.1 | 12.0 | 12.7 |
| S18A05I01 | 31.6 | 31.6 | 31.2 | 31.6 | 9.5 | 24.7 | 37.4 |
| S18A06I01 | 22.9 | 64.5 | 64.5 | 62.6 | 67.8 | 63.6 | 45.1 |
| S30A04I01 | 48.5 | 54.0 | 54.9 | 56.1 | 36.5 | 61.9 | 55.1 |
| S30A05I01 | 41.1 | 22.0 | 30.8 | 25.5 | 38.9 | 28.7 | 32.4 |
| S30A06I01 | 30.1 | 33.6 | 33.2 | 33.8 | 19.6 | 30.5 | 29.5 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 55 | 27 | 467 |
| 27 | 51 | 363 |
| 27 | 52 | 362 |
| 41 | 27 | 311 |
| 74 | 27 | 310 |
| 58 | 27 | 299 |
| 24 | 27 | 266 |
| 65 | 68 | 253 |
| 27 | 73 | 241 |
| 12 | 27 | 230 |
| 64 | 27 | 225 |
| 70 | 71 | 222 |
| 27 | 59 | 220 |
| 24 | 73 | 213 |
| 1 | 27 | 212 |
