Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 1).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.6, 33.9] | 8.0 [6.9, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.4] | 0.0 [0.0, 0.0] |
| fusion_causal | 36.9 [29.1, 46.9] | 32.4 [27.2, 37.5] | 31.0 [24.8, 38.4] | 22.6 [17.7, 28.7] | 14.1 [10.0, 19.1] |
| fusion_conf_causal | 37.0 [29.1, 47.1] | 32.8 [27.6, 37.6] | 31.9 [25.3, 39.7] | 24.3 [18.0, 32.0] | 14.4 [10.0, 20.4] |
| fusion_geo_causal | 37.7 [30.5, 46.7] | 34.0 [27.3, 39.9] | 31.9 [25.6, 39.0] | 23.0 [17.5, 28.2] | 15.3 [11.1, 20.4] |
| view0_causal | 32.1 [27.5, 37.5] | 32.2 [27.3, 36.6] | 28.9 [22.7, 36.5] | 23.1 [16.8, 30.2] | 13.6 [9.0, 18.7] |
| view1_causal | 33.0 [24.8, 43.0] | 30.4 [24.4, 36.4] | 28.4 [20.3, 38.4] | 24.5 [16.1, 34.3] | 14.5 [8.3, 22.6] |
| view2_causal | 35.3 [27.8, 44.2] | 30.3 [25.6, 34.3] | 29.2 [22.3, 35.9] | 25.0 [19.2, 30.7] | 14.6 [9.8, 20.3] |

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
| S01A04I01 | 22.0 | 19.4 | 19.2 | 22.0 | 14.0 | 13.7 | 17.3 |
| S01A05I01 | 55.7 | 51.3 | 56.1 | 49.2 | 59.7 | 31.4 | 60.4 |
| S01A06I01 | 19.4 | 19.4 | 19.4 | 19.4 | 25.8 | 19.4 | 15.3 |
| S08A04I01 | 34.3 | 46.8 | 49.3 | 48.2 | 44.0 | 35.1 | 50.9 |
| S08A05I01 | 47.9 | 48.7 | 46.9 | 49.2 | 40.2 | 38.0 | 43.8 |
| S08A06I01 | 36.7 | 63.7 | 63.8 | 64.4 | 42.0 | 64.2 | 59.4 |
| S10A04I01 | 40.8 | 25.0 | 23.4 | 32.9 | 33.8 | 24.4 | 23.6 |
| S10A05I01 | 27.3 | 32.1 | 31.7 | 31.3 | 27.4 | 29.6 | 32.0 |
| S10A06I01 | 27.9 | 27.6 | 27.5 | 27.4 | 14.5 | 18.5 | 27.7 |
| S12A04I01 | 17.5 | 17.5 | 17.5 | 17.5 | 28.1 | 17.5 | 17.5 |
| S12A05I01 | 23.2 | 51.0 | 51.1 | 55.1 | 45.0 | 45.2 | 49.0 |
| S12A06I01 | 27.8 | 24.4 | 26.0 | 27.7 | 20.7 | 24.7 | 26.3 |
| S18A04I01 | 10.9 | 20.7 | 21.2 | 15.8 | 11.6 | 31.6 | 13.1 |
| S18A05I01 | 15.5 | 32.0 | 32.3 | 30.5 | 14.8 | 23.0 | 27.7 |
| S18A06I01 | 25.9 | 43.6 | 36.7 | 41.7 | 53.5 | 63.7 | 33.6 |
| S30A04I01 | 16.7 | 38.1 | 39.2 | 35.0 | 33.3 | 43.2 | 46.6 |
| S30A05I01 | 50.1 | 57.0 | 57.0 | 59.3 | 44.6 | 49.0 | 55.5 |
| S30A06I01 | 15.4 | 65.3 | 65.0 | 59.0 | 38.2 | 57.3 | 47.2 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 41 | 27 | 330 |
| 61 | 27 | 305 |
| 51 | 27 | 295 |
| 53 | 27 | 291 |
| 1 | 27 | 285 |
| 55 | 27 | 261 |
| 74 | 27 | 254 |
| 46 | 27 | 251 |
| 67 | 27 | 244 |
| 47 | 27 | 233 |
| 58 | 65 | 219 |
| 27 | 71 | 203 |
| 14 | 27 | 181 |
| 72 | 27 | 175 |
| 27 | 47 | 153 |
