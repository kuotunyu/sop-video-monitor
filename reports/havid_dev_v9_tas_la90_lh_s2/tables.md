Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 2).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [26.9, 45.4] | 8.3 [6.6, 10.1] | 9.4 [4.8, 14.0] | 3.6 [0.6, 7.9] | 0.7 [0.0, 2.6] |
| fusion_causal | 38.3 [32.7, 44.3] | 36.4 [28.0, 44.4] | 35.3 [26.5, 44.5] | 26.4 [19.0, 33.8] | 13.6 [9.0, 18.3] |
| fusion_conf_causal | 38.4 [32.4, 44.6] | 32.0 [26.3, 37.4] | 32.9 [25.9, 40.9] | 25.9 [18.9, 32.4] | 14.1 [9.4, 18.2] |
| fusion_geo_causal | 39.5 [34.3, 45.3] | 36.0 [28.9, 43.3] | 37.9 [28.5, 47.6] | 28.9 [21.1, 37.7] | 14.7 [10.1, 19.5] |
| view0_causal | 30.7 [24.3, 35.9] | 30.6 [24.4, 37.3] | 30.6 [22.5, 38.7] | 24.7 [17.2, 31.2] | 13.5 [8.3, 17.8] |
| view1_causal | 28.8 [21.7, 36.0] | 33.0 [25.2, 39.2] | 30.2 [23.3, 38.0] | 21.2 [15.1, 29.0] | 9.5 [6.0, 13.9] |
| view2_causal | 34.3 [29.1, 42.1] | 33.5 [25.3, 42.8] | 30.6 [21.4, 43.8] | 21.9 [15.0, 31.5] | 12.4 [8.4, 17.4] |

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
| S01A04I01 | 41.2 | 24.0 | 24.1 | 28.2 | 12.3 | 6.7 | 25.6 |
| S01A05I01 | 47.4 | 40.4 | 41.1 | 40.6 | 27.3 | 28.9 | 35.0 |
| S01A06I01 | 63.7 | 24.0 | 23.0 | 24.1 | 20.5 | 13.8 | 23.5 |
| S08A04I01 | 18.1 | 51.2 | 51.9 | 51.5 | 51.7 | 48.9 | 51.0 |
| S08A05I01 | 25.4 | 35.9 | 36.1 | 37.3 | 25.4 | 26.6 | 37.1 |
| S08A06I01 | 23.6 | 55.9 | 55.7 | 54.9 | 32.6 | 38.7 | 55.6 |
| S10A04I01 | 33.7 | 38.6 | 36.6 | 40.1 | 40.9 | 32.9 | 19.2 |
| S10A05I01 | 35.0 | 58.1 | 57.2 | 56.3 | 34.5 | 33.5 | 60.4 |
| S10A06I01 | 26.0 | 23.3 | 27.6 | 26.4 | 19.6 | 14.1 | 24.5 |
| S12A04I01 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 | 22.3 | 60.0 |
| S12A05I01 | 70.6 | 57.0 | 60.6 | 62.4 | 47.2 | 43.1 | 39.7 |
| S12A06I01 | 28.7 | 25.9 | 25.6 | 26.0 | 13.7 | 27.2 | 18.6 |
| S18A04I01 | 14.7 | 18.1 | 16.9 | 15.6 | 6.7 | 17.8 | 19.6 |
| S18A05I01 | 31.6 | 41.7 | 41.9 | 45.2 | 8.6 | 27.3 | 45.6 |
| S18A06I01 | 22.9 | 71.3 | 72.7 | 71.6 | 57.2 | 48.1 | 71.9 |
| S30A04I01 | 48.5 | 43.9 | 41.0 | 45.9 | 41.8 | 58.7 | 30.7 |
| S30A05I01 | 41.1 | 32.3 | 31.7 | 33.9 | 39.3 | 26.2 | 28.7 |
| S30A06I01 | 30.1 | 25.6 | 22.4 | 24.5 | 29.7 | 28.5 | 25.9 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 27 | 73 | 359 |
| 27 | 51 | 301 |
| 41 | 27 | 299 |
| 74 | 27 | 297 |
| 27 | 52 | 293 |
| 58 | 27 | 274 |
| 27 | 46 | 263 |
| 55 | 27 | 249 |
| 64 | 27 | 245 |
| 24 | 27 | 228 |
| 27 | 68 | 220 |
| 1 | 27 | 208 |
| 68 | 27 | 208 |
| 49 | 47 | 191 |
| 12 | 27 | 177 |
