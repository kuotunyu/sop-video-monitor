Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [27.3, 45.4] | 8.3 [6.7, 10.0] | 9.4 [4.8, 13.9] | 3.6 [0.6, 7.5] | 0.7 [0.0, 2.5] |
| fusion_causal | 42.2 [36.2, 49.1] | 34.5 [27.6, 41.2] | 34.4 [27.0, 42.0] | 28.9 [21.2, 37.9] | 18.9 [11.3, 26.7] |
| fusion_conf_causal | 41.9 [36.1, 48.6] | 33.7 [27.8, 39.2] | 34.4 [27.5, 42.3] | 27.4 [19.8, 35.8] | 17.1 [10.0, 24.3] |
| fusion_geo_causal | 43.0 [37.1, 50.0] | 34.2 [25.9, 41.9] | 36.7 [28.2, 45.6] | 30.7 [20.9, 40.9] | 18.5 [10.7, 25.1] |
| view0_causal | 35.2 [29.1, 41.6] | 29.5 [24.0, 35.4] | 28.2 [20.6, 35.7] | 22.7 [16.1, 29.3] | 13.1 [9.0, 17.1] |
| view1_causal | 33.4 [30.7, 36.3] | 30.7 [26.7, 34.9] | 30.8 [26.0, 36.5] | 23.7 [20.6, 27.0] | 14.0 [11.7, 16.9] |
| view2_causal | 37.6 [32.0, 44.3] | 30.8 [26.9, 35.2] | 33.8 [27.2, 41.9] | 26.9 [19.3, 36.4] | 15.8 [9.5, 23.5] |

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
| S01A04I01 | 41.2 | 35.2 | 34.8 | 42.6 | 21.9 | 37.3 | 32.7 |
| S01A05I01 | 47.4 | 44.1 | 44.9 | 38.2 | 41.2 | 31.8 | 27.7 |
| S01A06I01 | 63.7 | 16.1 | 16.2 | 15.9 | 14.3 | 21.9 | 15.2 |
| S08A04I01 | 18.1 | 67.3 | 64.9 | 67.3 | 54.7 | 47.3 | 64.8 |
| S08A05I01 | 25.4 | 32.1 | 32.4 | 34.7 | 33.0 | 28.8 | 31.8 |
| S08A06I01 | 23.6 | 54.0 | 54.2 | 55.2 | 44.5 | 31.3 | 40.9 |
| S10A04I01 | 33.7 | 43.2 | 41.3 | 42.2 | 37.6 | 28.3 | 38.3 |
| S10A05I01 | 35.0 | 51.3 | 50.4 | 49.0 | 41.5 | 51.6 | 41.0 |
| S10A06I01 | 26.0 | 22.6 | 23.2 | 23.1 | 22.1 | 21.4 | 22.1 |
| S12A04I01 | 62.5 | 59.2 | 58.6 | 62.5 | 56.0 | 50.7 | 50.9 |
| S12A05I01 | 70.6 | 55.3 | 55.9 | 52.2 | 53.3 | 51.3 | 34.5 |
| S12A06I01 | 28.7 | 31.0 | 29.6 | 33.0 | 20.4 | 21.9 | 34.5 |
| S18A04I01 | 14.7 | 18.5 | 18.8 | 13.3 | 5.0 | 9.9 | 35.4 |
| S18A05I01 | 31.6 | 22.2 | 24.1 | 42.2 | 6.7 | 9.2 | 46.7 |
| S18A06I01 | 22.9 | 71.3 | 71.2 | 69.8 | 62.0 | 66.8 | 67.4 |
| S30A04I01 | 48.5 | 49.6 | 49.5 | 51.6 | 55.3 | 44.9 | 43.7 |
| S30A05I01 | 41.1 | 51.8 | 51.5 | 50.1 | 35.6 | 44.4 | 46.2 |
| S30A06I01 | 30.1 | 52.5 | 53.1 | 55.0 | 44.1 | 18.7 | 34.5 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 41 | 27 | 358 |
| 1 | 27 | 315 |
| 27 | 54 | 274 |
| 27 | 71 | 270 |
| 74 | 27 | 264 |
| 58 | 27 | 263 |
| 64 | 27 | 211 |
| 55 | 27 | 211 |
| 27 | 73 | 200 |
| 27 | 14 | 196 |
| 71 | 27 | 182 |
| 49 | 27 | 163 |
| 12 | 27 | 161 |
| 24 | 27 | 161 |
| 27 | 52 | 154 |
