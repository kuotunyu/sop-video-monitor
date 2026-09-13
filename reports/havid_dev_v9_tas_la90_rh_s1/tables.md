Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 1).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.6, 33.9] | 8.0 [6.9, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.4] | 0.0 [0.0, 0.0] |
| fusion_causal | 35.6 [28.2, 45.1] | 34.8 [28.3, 40.8] | 32.6 [25.7, 42.8] | 23.6 [15.9, 33.5] | 14.3 [9.5, 20.9] |
| fusion_conf_causal | 35.5 [27.8, 45.5] | 34.9 [27.5, 41.8] | 30.6 [21.8, 42.5] | 24.0 [16.4, 35.1] | 13.5 [9.2, 19.8] |
| fusion_geo_causal | 36.9 [30.8, 45.6] | 37.6 [32.9, 41.9] | 34.0 [25.6, 45.6] | 25.8 [17.2, 37.3] | 14.8 [8.6, 23.2] |
| view0_causal | 27.2 [20.3, 36.3] | 27.4 [24.4, 29.9] | 26.2 [19.8, 34.3] | 20.0 [12.9, 28.9] | 13.7 [9.6, 18.3] |
| view1_causal | 32.6 [23.5, 43.5] | 29.8 [25.5, 34.3] | 28.6 [20.5, 39.0] | 20.1 [13.2, 28.8] | 11.2 [6.0, 17.7] |
| view2_causal | 31.0 [25.8, 36.2] | 34.8 [26.5, 42.7] | 31.6 [24.6, 40.4] | 23.6 [16.9, 31.4] | 14.1 [11.0, 16.8] |

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
| S01A04I01 | 22.0 | 7.7 | 7.2 | 9.9 | 10.0 | 21.9 | 6.4 |
| S01A05I01 | 55.7 | 50.4 | 51.4 | 68.3 | 37.5 | 29.5 | 46.9 |
| S01A06I01 | 19.4 | 20.6 | 21.1 | 19.1 | 15.5 | 21.9 | 18.4 |
| S08A04I01 | 34.3 | 47.4 | 47.7 | 47.2 | 41.8 | 42.6 | 41.3 |
| S08A05I01 | 47.9 | 46.0 | 52.2 | 48.9 | 36.6 | 40.6 | 30.9 |
| S08A06I01 | 36.7 | 75.2 | 75.2 | 76.2 | 58.3 | 65.3 | 52.9 |
| S10A04I01 | 40.8 | 39.5 | 37.3 | 36.3 | 23.4 | 38.8 | 39.2 |
| S10A05I01 | 27.3 | 39.9 | 40.1 | 32.6 | 20.7 | 20.7 | 44.7 |
| S10A06I01 | 27.9 | 19.5 | 19.2 | 23.4 | 12.3 | 15.8 | 16.4 |
| S12A04I01 | 17.5 | 17.5 | 17.5 | 24.3 | 14.9 | 25.3 | 17.5 |
| S12A05I01 | 23.2 | 54.0 | 54.3 | 53.4 | 44.9 | 20.8 | 55.1 |
| S12A06I01 | 27.8 | 15.2 | 13.9 | 19.8 | 11.9 | 16.2 | 9.3 |
| S18A04I01 | 10.9 | 20.7 | 21.1 | 18.5 | 10.9 | 15.5 | 26.0 |
| S18A05I01 | 15.5 | 29.0 | 28.8 | 31.4 | 10.8 | 27.3 | 29.5 |
| S18A06I01 | 25.9 | 56.6 | 56.3 | 52.6 | 48.4 | 44.3 | 45.4 |
| S30A04I01 | 16.7 | 28.3 | 26.5 | 28.1 | 32.0 | 36.5 | 40.6 |
| S30A05I01 | 50.1 | 54.0 | 54.4 | 54.8 | 32.6 | 58.9 | 48.6 |
| S30A06I01 | 15.4 | 45.9 | 43.8 | 40.3 | 44.9 | 63.7 | 15.7 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 27 | 59 | 391 |
| 61 | 27 | 301 |
| 41 | 27 | 271 |
| 27 | 50 | 266 |
| 27 | 46 | 241 |
| 49 | 47 | 209 |
| 67 | 27 | 203 |
| 14 | 27 | 200 |
| 51 | 27 | 196 |
| 74 | 72 | 191 |
| 53 | 27 | 191 |
| 72 | 55 | 186 |
| 13 | 27 | 174 |
| 1 | 27 | 172 |
| 70 | 27 | 169 |
