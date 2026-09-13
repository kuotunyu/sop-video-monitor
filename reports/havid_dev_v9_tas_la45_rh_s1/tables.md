Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 1).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.6, 33.9] | 8.0 [6.9, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.4] | 0.0 [0.0, 0.0] |
| fusion_causal | 36.7 [28.8, 46.6] | 37.1 [29.8, 42.4] | 33.8 [25.1, 42.3] | 26.0 [20.3, 32.3] | 14.8 [9.8, 20.8] |
| fusion_conf_causal | 36.4 [28.7, 46.1] | 38.1 [33.5, 42.5] | 34.4 [25.6, 43.3] | 25.3 [19.4, 31.7] | 16.3 [12.3, 21.3] |
| fusion_geo_causal | 36.2 [29.1, 44.8] | 34.8 [25.0, 42.6] | 34.9 [26.8, 43.3] | 26.1 [19.5, 33.6] | 14.7 [9.9, 20.5] |
| view0_causal | 32.2 [24.8, 41.3] | 30.9 [26.7, 35.4] | 28.3 [22.2, 37.3] | 20.1 [14.0, 29.4] | 11.5 [7.8, 16.7] |
| view1_causal | 32.7 [25.6, 41.4] | 33.6 [22.8, 43.7] | 30.8 [22.8, 40.7] | 23.5 [17.5, 30.9] | 14.5 [8.9, 21.3] |
| view2_causal | 36.3 [27.7, 46.9] | 34.8 [27.8, 41.1] | 32.9 [24.6, 42.7] | 28.8 [20.7, 38.6] | 18.3 [11.2, 27.1] |

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
| S01A04I01 | 22.0 | 15.5 | 15.1 | 14.6 | 18.5 | 15.3 | 13.1 |
| S01A05I01 | 55.7 | 44.8 | 46.8 | 46.2 | 18.5 | 42.5 | 39.5 |
| S01A06I01 | 19.4 | 19.4 | 19.4 | 19.4 | 19.2 | 19.4 | 18.9 |
| S08A04I01 | 34.3 | 58.2 | 58.5 | 54.4 | 47.2 | 41.5 | 55.4 |
| S08A05I01 | 47.9 | 48.0 | 47.2 | 47.9 | 49.6 | 46.7 | 43.8 |
| S08A06I01 | 36.7 | 59.4 | 59.8 | 59.2 | 56.7 | 64.2 | 63.1 |
| S10A04I01 | 40.8 | 26.0 | 24.8 | 30.9 | 35.0 | 37.8 | 21.7 |
| S10A05I01 | 27.3 | 34.5 | 34.5 | 34.5 | 27.3 | 30.8 | 47.0 |
| S10A06I01 | 27.9 | 23.7 | 24.4 | 23.5 | 22.7 | 20.4 | 22.5 |
| S12A04I01 | 17.5 | 17.5 | 17.5 | 17.5 | 17.5 | 15.8 | 15.5 |
| S12A05I01 | 23.2 | 63.3 | 62.7 | 65.6 | 42.6 | 24.2 | 71.9 |
| S12A06I01 | 27.8 | 30.7 | 29.0 | 34.4 | 22.8 | 24.5 | 23.4 |
| S18A04I01 | 10.9 | 21.8 | 22.6 | 15.6 | 16.7 | 11.0 | 28.3 |
| S18A05I01 | 15.5 | 32.0 | 32.3 | 26.7 | 10.8 | 15.5 | 32.5 |
| S18A06I01 | 25.9 | 53.6 | 50.7 | 52.2 | 63.7 | 70.1 | 47.0 |
| S30A04I01 | 16.7 | 40.2 | 41.0 | 38.5 | 40.2 | 26.5 | 48.8 |
| S30A05I01 | 50.1 | 40.9 | 40.3 | 43.8 | 41.7 | 49.2 | 40.9 |
| S30A06I01 | 15.4 | 52.2 | 50.5 | 39.1 | 40.2 | 45.4 | 53.3 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 72 | 27 | 635 |
| 53 | 27 | 421 |
| 47 | 49 | 409 |
| 41 | 27 | 342 |
| 74 | 27 | 339 |
| 1 | 27 | 299 |
| 27 | 62 | 295 |
| 27 | 71 | 248 |
| 55 | 27 | 241 |
| 61 | 27 | 222 |
| 27 | 59 | 204 |
| 24 | 27 | 186 |
| 27 | 49 | 174 |
| 58 | 65 | 169 |
| 51 | 52 | 156 |
