Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [27.3, 45.4] | 8.3 [6.7, 10.0] | 9.4 [4.8, 13.9] | 3.6 [0.6, 7.5] | 0.7 [0.0, 2.5] |
| fusion_causal | 37.7 [33.4, 42.3] | 35.8 [30.2, 39.2] | 32.7 [26.6, 38.5] | 24.2 [17.7, 30.7] | 9.0 [5.9, 11.9] |
| fusion_conf_causal | 37.8 [33.8, 42.5] | 37.0 [32.0, 40.5] | 31.3 [25.7, 36.9] | 23.5 [17.5, 30.1] | 9.1 [6.2, 12.1] |
| fusion_geo_causal | 38.1 [33.9, 42.2] | 33.4 [26.7, 39.7] | 33.1 [23.9, 42.6] | 24.0 [15.3, 33.4] | 10.1 [6.9, 13.8] |
| view0_causal | 31.9 [29.0, 35.6] | 33.8 [26.1, 39.9] | 31.9 [26.2, 37.1] | 23.5 [19.9, 26.8] | 10.1 [6.8, 13.4] |
| view1_causal | 33.8 [29.3, 40.2] | 32.6 [27.5, 37.4] | 33.8 [26.5, 41.3] | 23.7 [15.7, 32.2] | 11.4 [6.5, 17.7] |
| view2_causal | 35.0 [31.4, 38.7] | 38.6 [31.5, 44.6] | 34.7 [26.4, 43.2] | 26.4 [21.0, 32.4] | 14.4 [10.2, 19.8] |

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
| S01A04I01 | 41.2 | 39.7 | 39.5 | 40.1 | 26.4 | 41.2 | 31.5 |
| S01A05I01 | 47.4 | 22.6 | 21.6 | 19.2 | 21.9 | 15.5 | 24.3 |
| S01A06I01 | 63.7 | 62.6 | 64.5 | 61.0 | 43.2 | 40.8 | 65.4 |
| S08A04I01 | 18.1 | 43.7 | 41.1 | 47.8 | 37.2 | 44.5 | 40.7 |
| S08A05I01 | 25.4 | 29.3 | 29.2 | 32.5 | 28.0 | 23.3 | 27.8 |
| S08A06I01 | 23.6 | 42.7 | 40.2 | 46.5 | 33.4 | 22.7 | 53.6 |
| S10A04I01 | 33.7 | 33.9 | 36.8 | 35.2 | 33.6 | 29.3 | 31.0 |
| S10A05I01 | 35.0 | 38.4 | 39.3 | 38.5 | 37.2 | 32.0 | 50.2 |
| S10A06I01 | 26.0 | 23.3 | 23.2 | 22.2 | 21.8 | 21.1 | 17.9 |
| S12A04I01 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 | 62.5 |
| S12A05I01 | 70.6 | 50.3 | 51.5 | 48.7 | 55.1 | 45.7 | 43.4 |
| S12A06I01 | 28.7 | 17.8 | 18.0 | 17.5 | 16.6 | 16.4 | 16.8 |
| S18A04I01 | 14.7 | 14.7 | 14.7 | 14.7 | 12.5 | 16.7 | 15.4 |
| S18A05I01 | 31.6 | 28.8 | 28.8 | 43.9 | 11.6 | 28.8 | 44.5 |
| S18A06I01 | 22.9 | 51.0 | 50.4 | 51.0 | 51.3 | 46.7 | 34.9 |
| S30A04I01 | 48.5 | 53.1 | 54.9 | 55.2 | 26.1 | 64.8 | 33.1 |
| S30A05I01 | 41.1 | 40.9 | 40.7 | 41.4 | 42.4 | 44.9 | 25.9 |
| S30A06I01 | 30.1 | 49.6 | 48.4 | 41.6 | 28.6 | 38.8 | 48.5 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 41 | 27 | 348 |
| 27 | 52 | 321 |
| 58 | 27 | 319 |
| 55 | 27 | 315 |
| 1 | 27 | 307 |
| 49 | 27 | 296 |
| 27 | 59 | 291 |
| 24 | 27 | 288 |
| 74 | 27 | 281 |
| 68 | 27 | 249 |
| 64 | 27 | 239 |
| 12 | 27 | 230 |
| 46 | 27 | 226 |
| 27 | 55 | 224 |
| 71 | 27 | 224 |
