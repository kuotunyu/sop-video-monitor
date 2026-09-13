Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.6, 34.1] | 8.0 [6.8, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 38.6 [30.0, 49.9] | 33.0 [25.9, 39.6] | 33.2 [24.7, 44.5] | 26.2 [18.5, 37.7] | 17.0 [10.2, 27.1] |
| fusion_conf_causal | 38.1 [29.4, 49.6] | 33.2 [25.8, 40.0] | 31.3 [23.6, 42.2] | 25.4 [17.6, 36.5] | 15.8 [8.9, 26.4] |
| fusion_geo_causal | 38.4 [30.7, 48.6] | 32.6 [25.3, 41.0] | 33.8 [27.3, 42.4] | 28.0 [22.1, 36.0] | 17.2 [11.0, 24.8] |
| view0_causal | 35.5 [27.1, 45.6] | 31.2 [24.7, 38.2] | 29.0 [21.0, 39.7] | 24.0 [16.6, 34.4] | 14.6 [8.1, 23.8] |
| view1_causal | 32.9 [26.0, 41.7] | 28.3 [22.7, 34.0] | 30.6 [24.1, 37.2] | 21.1 [15.0, 27.3] | 12.0 [8.5, 15.6] |
| view2_causal | 33.7 [25.6, 44.0] | 32.7 [25.1, 40.1] | 30.8 [22.7, 40.3] | 24.3 [16.4, 34.1] | 14.3 [8.4, 22.1] |

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
| S01A04I01 | 22.0 | 19.9 | 19.4 | 19.2 | 15.5 | 19.6 | 22.3 |
| S01A05I01 | 55.7 | 53.8 | 54.2 | 51.8 | 43.1 | 22.3 | 27.1 |
| S01A06I01 | 19.4 | 15.2 | 15.2 | 15.5 | 14.3 | 17.0 | 17.2 |
| S08A04I01 | 34.3 | 55.3 | 53.9 | 54.1 | 47.4 | 40.5 | 44.9 |
| S08A05I01 | 47.9 | 55.8 | 54.7 | 54.2 | 56.1 | 50.3 | 44.6 |
| S08A06I01 | 36.7 | 68.1 | 68.6 | 68.2 | 62.8 | 60.4 | 53.8 |
| S10A04I01 | 40.8 | 30.4 | 28.3 | 38.5 | 29.7 | 34.5 | 22.6 |
| S10A05I01 | 27.3 | 33.6 | 34.5 | 32.3 | 33.2 | 24.0 | 29.5 |
| S10A06I01 | 27.9 | 23.2 | 23.3 | 24.4 | 31.0 | 23.0 | 18.1 |
| S12A04I01 | 17.5 | 17.5 | 16.3 | 17.5 | 17.2 | 17.5 | 16.9 |
| S12A05I01 | 23.2 | 45.4 | 46.4 | 47.2 | 45.6 | 33.9 | 48.8 |
| S12A06I01 | 27.8 | 25.9 | 21.9 | 27.4 | 24.7 | 31.6 | 23.1 |
| S18A04I01 | 10.9 | 28.8 | 29.4 | 17.4 | 10.6 | 26.1 | 37.4 |
| S18A05I01 | 15.5 | 28.2 | 28.0 | 21.3 | 15.1 | 35.3 | 29.9 |
| S18A06I01 | 25.9 | 64.3 | 64.3 | 61.4 | 55.2 | 54.1 | 66.4 |
| S30A04I01 | 16.7 | 41.5 | 42.8 | 41.4 | 37.0 | 16.9 | 69.3 |
| S30A05I01 | 50.1 | 61.2 | 61.8 | 58.6 | 59.3 | 59.3 | 37.6 |
| S30A06I01 | 15.4 | 53.2 | 52.2 | 52.0 | 51.7 | 40.8 | 38.1 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 41 | 27 | 389 |
| 72 | 27 | 347 |
| 51 | 27 | 317 |
| 61 | 27 | 271 |
| 74 | 27 | 255 |
| 67 | 27 | 237 |
| 53 | 27 | 231 |
| 27 | 47 | 230 |
| 14 | 27 | 212 |
| 55 | 27 | 206 |
| 1 | 27 | 174 |
| 9 | 27 | 142 |
| 53 | 54 | 141 |
| 58 | 65 | 140 |
| 11 | 27 | 139 |
