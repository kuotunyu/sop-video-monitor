Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 2).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.7, 34.1] | 8.0 [6.8, 9.4] | 8.9 [5.9, 11.9] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 34.6 [27.8, 44.0] | 34.4 [29.1, 38.7] | 30.0 [22.0, 40.1] | 23.1 [15.7, 33.3] | 12.2 [9.0, 16.7] |
| fusion_conf_causal | 34.2 [27.4, 43.1] | 33.9 [28.9, 38.5] | 30.9 [23.3, 40.7] | 23.6 [16.7, 32.4] | 12.6 [8.5, 17.8] |
| fusion_geo_causal | 35.3 [29.6, 42.4] | 32.6 [25.2, 38.9] | 30.9 [22.7, 40.4] | 24.0 [17.1, 33.0] | 14.2 [11.0, 18.4] |
| view0_causal | 31.3 [26.1, 37.9] | 30.8 [25.8, 36.0] | 29.7 [22.2, 40.3] | 22.8 [15.5, 31.2] | 13.0 [8.7, 18.3] |
| view1_causal | 29.9 [24.3, 36.3] | 28.4 [23.7, 34.7] | 26.5 [20.9, 34.2] | 19.5 [15.6, 24.7] | 8.2 [5.8, 10.8] |
| view2_causal | 32.6 [27.2, 39.7] | 33.1 [27.4, 38.3] | 31.4 [25.7, 40.2] | 24.5 [19.1, 31.4] | 14.5 [10.7, 20.0] |

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
| S01A04I01 | 22.0 | 5.5 | 5.7 | 4.3 | 12.6 | 16.7 | 8.7 |
| S01A05I01 | 55.7 | 53.2 | 53.7 | 63.9 | 51.4 | 41.8 | 51.2 |
| S01A06I01 | 19.4 | 20.2 | 19.9 | 19.1 | 18.6 | 17.1 | 16.8 |
| S08A04I01 | 34.3 | 48.0 | 47.7 | 45.6 | 53.8 | 38.7 | 47.6 |
| S08A05I01 | 47.9 | 33.9 | 33.9 | 33.0 | 21.3 | 39.7 | 25.4 |
| S08A06I01 | 36.7 | 65.6 | 66.2 | 60.2 | 54.7 | 49.4 | 55.1 |
| S10A04I01 | 40.8 | 35.9 | 37.4 | 39.5 | 40.4 | 32.2 | 26.3 |
| S10A05I01 | 27.3 | 37.6 | 37.6 | 46.8 | 42.8 | 22.4 | 49.2 |
| S10A06I01 | 27.9 | 17.6 | 17.6 | 22.9 | 14.2 | 21.0 | 23.5 |
| S12A04I01 | 17.5 | 18.4 | 18.8 | 17.5 | 17.4 | 17.5 | 25.9 |
| S12A05I01 | 23.2 | 56.8 | 52.8 | 56.9 | 46.0 | 51.9 | 52.2 |
| S12A06I01 | 27.8 | 13.6 | 13.8 | 13.4 | 14.1 | 14.1 | 9.8 |
| S18A04I01 | 10.9 | 22.4 | 21.6 | 13.7 | 8.7 | 16.7 | 31.2 |
| S18A05I01 | 15.5 | 32.7 | 32.7 | 32.7 | 30.3 | 12.0 | 32.7 |
| S18A06I01 | 25.9 | 55.8 | 51.6 | 57.2 | 32.4 | 26.8 | 62.2 |
| S30A04I01 | 16.7 | 37.7 | 35.4 | 33.5 | 34.8 | 27.2 | 47.3 |
| S30A05I01 | 50.1 | 46.7 | 45.2 | 50.8 | 31.5 | 47.9 | 34.3 |
| S30A06I01 | 15.4 | 52.4 | 51.9 | 50.5 | 47.3 | 41.8 | 30.4 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 27 | 47 | 373 |
| 72 | 55 | 299 |
| 41 | 27 | 295 |
| 53 | 27 | 274 |
| 27 | 71 | 248 |
| 61 | 27 | 242 |
| 27 | 59 | 242 |
| 27 | 46 | 227 |
| 49 | 47 | 221 |
| 1 | 27 | 211 |
| 27 | 51 | 200 |
| 51 | 27 | 193 |
| 53 | 55 | 193 |
| 67 | 27 | 188 |
| 58 | 65 | 185 |
