Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 1).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [27.5, 45.4] | 8.3 [6.7, 10.1] | 9.4 [5.0, 14.0] | 3.6 [0.7, 7.6] | 0.7 [0.0, 2.5] |
| fusion_causal | 41.9 [38.2, 45.9] | 32.8 [26.2, 39.1] | 34.3 [28.0, 41.2] | 27.8 [21.9, 34.5] | 16.4 [11.5, 22.8] |
| fusion_conf_causal | 41.4 [37.4, 45.7] | 32.7 [24.9, 39.8] | 33.2 [26.6, 40.6] | 26.9 [20.8, 34.0] | 18.1 [13.2, 23.6] |
| fusion_geo_causal | 44.5 [41.9, 46.8] | 34.9 [26.5, 42.7] | 36.8 [30.5, 42.0] | 29.9 [22.9, 36.7] | 17.1 [12.5, 22.4] |
| view0_causal | 34.8 [29.2, 41.1] | 32.5 [28.1, 37.5] | 30.2 [23.5, 37.7] | 24.4 [18.5, 31.3] | 16.0 [11.2, 21.1] |
| view1_causal | 35.0 [30.4, 39.4] | 33.7 [29.8, 37.7] | 33.4 [28.0, 38.7] | 23.8 [18.0, 29.5] | 13.7 [10.0, 18.0] |
| view2_causal | 39.2 [36.2, 42.1] | 34.7 [30.4, 39.0] | 35.3 [29.9, 40.6] | 29.3 [24.8, 35.6] | 15.7 [12.4, 19.4] |

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
| S01A04I01 | 41.2 | 46.9 | 47.2 | 46.7 | 36.9 | 31.1 | 45.2 |
| S01A05I01 | 47.4 | 34.9 | 36.2 | 46.3 | 26.9 | 27.5 | 28.9 |
| S01A06I01 | 63.7 | 45.4 | 40.0 | 47.3 | 2.4 | 51.6 | 52.9 |
| S08A04I01 | 18.1 | 51.6 | 51.7 | 53.0 | 49.1 | 39.8 | 50.7 |
| S08A05I01 | 25.4 | 34.7 | 34.7 | 35.1 | 32.9 | 27.5 | 32.3 |
| S08A06I01 | 23.6 | 41.3 | 40.9 | 41.4 | 36.0 | 36.8 | 48.4 |
| S10A04I01 | 33.7 | 51.5 | 50.6 | 55.3 | 44.3 | 39.2 | 43.0 |
| S10A05I01 | 35.0 | 31.3 | 30.1 | 38.4 | 29.9 | 25.5 | 41.6 |
| S10A06I01 | 26.0 | 22.5 | 20.6 | 27.8 | 15.9 | 13.7 | 23.8 |
| S12A04I01 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 |
| S12A05I01 | 70.6 | 75.1 | 75.1 | 72.0 | 67.8 | 47.7 | 50.0 |
| S12A06I01 | 28.7 | 24.4 | 24.2 | 24.2 | 18.2 | 22.2 | 22.1 |
| S18A04I01 | 14.7 | 17.0 | 17.3 | 19.5 | 10.7 | 15.0 | 20.3 |
| S18A05I01 | 31.6 | 19.6 | 15.3 | 31.6 | 8.8 | 17.4 | 37.4 |
| S18A06I01 | 22.9 | 69.8 | 69.5 | 68.8 | 69.9 | 64.5 | 51.8 |
| S30A04I01 | 48.5 | 54.2 | 56.1 | 60.9 | 53.4 | 48.2 | 57.7 |
| S30A05I01 | 41.1 | 40.0 | 40.1 | 40.0 | 38.9 | 46.8 | 31.3 |
| S30A06I01 | 30.1 | 34.2 | 34.2 | 36.3 | 31.8 | 36.3 | 24.6 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 1 | 27 | 352 |
| 41 | 27 | 297 |
| 24 | 27 | 278 |
| 58 | 27 | 267 |
| 74 | 27 | 249 |
| 47 | 49 | 239 |
| 64 | 27 | 223 |
| 27 | 51 | 222 |
| 27 | 55 | 189 |
| 6 | 27 | 188 |
| 12 | 27 | 185 |
| 27 | 52 | 180 |
| 27 | 71 | 175 |
| 27 | 49 | 163 |
| 27 | 54 | 158 |
