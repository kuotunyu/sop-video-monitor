Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [27.3, 45.4] | 8.3 [6.7, 10.0] | 9.4 [4.8, 13.9] | 3.6 [0.6, 7.5] | 0.7 [0.0, 2.5] |
| fusion_causal | 40.2 [36.3, 45.9] | 37.5 [32.2, 43.2] | 36.8 [29.4, 44.3] | 30.8 [22.0, 40.3] | 16.1 [9.4, 24.5] |
| fusion_conf_causal | 40.5 [36.4, 45.8] | 35.5 [30.3, 42.4] | 34.8 [27.6, 42.1] | 28.6 [21.9, 35.1] | 16.7 [10.7, 24.2] |
| fusion_geo_causal | 39.1 [35.5, 44.2] | 35.9 [29.2, 43.7] | 36.0 [29.2, 42.9] | 30.3 [22.1, 38.2] | 16.6 [10.9, 23.6] |
| view0_causal | 30.8 [23.3, 37.4] | 35.6 [31.2, 39.9] | 30.7 [25.0, 37.2] | 23.0 [17.4, 28.1] | 14.2 [9.1, 19.1] |
| view1_causal | 34.2 [28.7, 40.7] | 36.9 [29.5, 44.4] | 33.6 [26.4, 42.4] | 26.0 [19.7, 33.2] | 14.4 [10.1, 19.6] |
| view2_causal | 33.2 [29.2, 39.1] | 37.5 [31.6, 43.9] | 34.7 [27.5, 42.2] | 26.0 [18.7, 34.9] | 15.2 [7.7, 24.0] |

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
| S01A04I01 | 41.2 | 27.4 | 27.4 | 31.8 | 9.2 | 26.7 | 22.1 |
| S01A05I01 | 47.4 | 30.3 | 29.9 | 31.0 | 33.2 | 22.7 | 28.7 |
| S01A06I01 | 63.7 | 56.1 | 56.2 | 53.3 | 1.4 | 58.4 | 36.5 |
| S08A04I01 | 18.1 | 67.3 | 67.8 | 63.3 | 42.8 | 48.6 | 58.7 |
| S08A05I01 | 25.4 | 33.0 | 28.3 | 28.3 | 35.9 | 30.5 | 22.7 |
| S08A06I01 | 23.6 | 47.3 | 45.3 | 48.2 | 31.6 | 49.3 | 47.7 |
| S10A04I01 | 33.7 | 41.9 | 39.4 | 40.3 | 33.8 | 32.6 | 32.4 |
| S10A05I01 | 35.0 | 38.2 | 37.8 | 38.1 | 35.6 | 33.0 | 37.8 |
| S10A06I01 | 26.0 | 26.2 | 27.7 | 23.9 | 22.1 | 18.8 | 25.3 |
| S12A04I01 | 62.5 | 62.5 | 62.5 | 62.5 | 53.2 | 31.0 | 56.6 |
| S12A05I01 | 70.6 | 47.9 | 51.5 | 45.9 | 51.1 | 36.1 | 42.6 |
| S12A06I01 | 28.7 | 16.9 | 16.7 | 16.1 | 17.8 | 13.6 | 10.3 |
| S18A04I01 | 14.7 | 34.4 | 33.6 | 33.1 | 13.2 | 22.9 | 29.0 |
| S18A05I01 | 31.6 | 25.2 | 26.9 | 28.6 | 31.6 | 32.0 | 20.6 |
| S18A06I01 | 22.9 | 72.2 | 71.9 | 72.1 | 74.0 | 64.8 | 67.4 |
| S30A04I01 | 48.5 | 47.0 | 47.0 | 47.1 | 42.3 | 53.7 | 45.5 |
| S30A05I01 | 41.1 | 52.0 | 53.0 | 43.7 | 32.0 | 51.7 | 34.6 |
| S30A06I01 | 30.1 | 29.7 | 39.1 | 29.1 | 35.5 | 24.3 | 9.9 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 41 | 27 | 289 |
| 27 | 71 | 280 |
| 27 | 52 | 248 |
| 27 | 55 | 246 |
| 58 | 27 | 235 |
| 27 | 46 | 229 |
| 68 | 65 | 229 |
| 27 | 65 | 229 |
| 1 | 27 | 222 |
| 70 | 71 | 222 |
| 55 | 27 | 215 |
| 74 | 27 | 211 |
| 47 | 49 | 201 |
| 64 | 27 | 189 |
| 12 | 27 | 179 |
