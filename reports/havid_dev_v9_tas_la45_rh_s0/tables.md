Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.6, 34.1] | 8.0 [6.8, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 35.8 [28.2, 46.0] | 35.8 [29.8, 41.1] | 33.4 [26.0, 42.5] | 24.6 [18.4, 32.2] | 16.4 [11.3, 22.9] |
| fusion_conf_causal | 35.8 [27.8, 46.5] | 35.5 [28.4, 41.8] | 33.4 [25.6, 43.5] | 25.0 [18.3, 33.6] | 16.2 [10.6, 23.6] |
| fusion_geo_causal | 35.5 [28.7, 44.5] | 33.9 [28.4, 39.5] | 35.3 [27.2, 44.2] | 25.4 [19.3, 32.7] | 16.3 [11.2, 22.3] |
| view0_causal | 29.4 [23.1, 37.8] | 28.9 [21.6, 34.3] | 27.3 [19.1, 36.9] | 21.7 [15.8, 28.2] | 14.2 [9.9, 19.5] |
| view1_causal | 32.5 [27.5, 39.7] | 31.2 [25.2, 37.1] | 31.2 [24.5, 39.8] | 23.8 [18.0, 32.3] | 12.7 [9.6, 17.8] |
| view2_causal | 32.1 [26.0, 39.9] | 34.5 [28.6, 40.8] | 30.0 [23.2, 38.6] | 24.0 [19.6, 29.9] | 13.8 [10.7, 17.9] |

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
| S01A04I01 | 22.0 | 14.8 | 14.9 | 15.0 | 15.3 | 17.6 | 9.6 |
| S01A05I01 | 55.7 | 44.7 | 45.7 | 50.4 | 33.3 | 43.3 | 43.9 |
| S01A06I01 | 19.4 | 19.0 | 19.2 | 18.7 | 12.3 | 19.7 | 17.5 |
| S08A04I01 | 34.3 | 49.2 | 49.0 | 47.6 | 45.1 | 39.2 | 35.9 |
| S08A05I01 | 47.9 | 45.8 | 50.3 | 48.1 | 38.2 | 47.6 | 28.7 |
| S08A06I01 | 36.7 | 70.8 | 71.2 | 71.2 | 54.2 | 59.5 | 58.6 |
| S10A04I01 | 40.8 | 30.4 | 29.4 | 33.3 | 28.9 | 30.6 | 27.1 |
| S10A05I01 | 27.3 | 37.6 | 35.7 | 36.9 | 28.1 | 27.0 | 38.2 |
| S10A06I01 | 27.9 | 18.9 | 18.1 | 20.5 | 14.7 | 20.9 | 19.5 |
| S12A04I01 | 17.5 | 15.8 | 15.8 | 17.5 | 16.0 | 15.5 | 17.5 |
| S12A05I01 | 23.2 | 64.5 | 64.3 | 63.7 | 61.5 | 56.6 | 55.0 |
| S12A06I01 | 27.8 | 17.2 | 17.3 | 17.9 | 11.3 | 20.0 | 14.8 |
| S18A04I01 | 10.9 | 19.6 | 19.5 | 20.4 | 15.6 | 28.8 | 14.8 |
| S18A05I01 | 15.5 | 32.0 | 32.3 | 15.5 | 12.0 | 14.2 | 30.5 |
| S18A06I01 | 25.9 | 53.6 | 50.1 | 49.5 | 46.0 | 51.2 | 61.7 |
| S30A04I01 | 16.7 | 40.3 | 41.5 | 38.8 | 31.1 | 39.1 | 37.2 |
| S30A05I01 | 50.1 | 50.1 | 50.0 | 49.8 | 40.7 | 33.8 | 59.7 |
| S30A06I01 | 15.4 | 48.7 | 49.0 | 37.8 | 35.3 | 32.8 | 41.1 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 53 | 27 | 314 |
| 27 | 59 | 311 |
| 27 | 46 | 305 |
| 72 | 27 | 294 |
| 41 | 27 | 282 |
| 61 | 27 | 271 |
| 67 | 27 | 227 |
| 74 | 46 | 221 |
| 1 | 27 | 202 |
| 49 | 47 | 197 |
| 27 | 52 | 182 |
| 61 | 65 | 179 |
| 27 | 51 | 169 |
| 55 | 27 | 168 |
| 14 | 27 | 156 |
