Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 2).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.7, 34.1] | 8.0 [6.8, 9.4] | 8.9 [5.9, 11.9] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 37.0 [29.5, 47.0] | 31.6 [25.9, 37.5] | 31.0 [23.7, 40.3] | 23.8 [17.6, 31.7] | 13.2 [8.6, 19.7] |
| fusion_conf_causal | 37.6 [30.3, 47.4] | 33.1 [26.9, 38.5] | 30.1 [22.7, 40.1] | 24.2 [18.7, 30.5] | 14.1 [9.9, 19.5] |
| fusion_geo_causal | 37.9 [30.4, 48.3] | 35.1 [29.7, 40.0] | 35.1 [27.3, 42.6] | 26.5 [20.3, 33.2] | 15.2 [9.7, 22.0] |
| view0_causal | 32.4 [25.1, 42.1] | 30.4 [24.6, 35.5] | 27.6 [21.2, 35.1] | 22.5 [16.6, 28.0] | 11.5 [6.6, 16.6] |
| view1_causal | 29.3 [22.8, 37.5] | 29.5 [24.3, 34.5] | 27.1 [19.4, 36.9] | 21.2 [15.0, 29.0] | 13.1 [9.9, 17.2] |
| view2_causal | 36.2 [27.4, 47.6] | 31.2 [25.6, 36.6] | 28.6 [22.5, 36.2] | 23.3 [18.1, 30.2] | 14.5 [11.3, 19.0] |

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
| S01A04I01 | 22.0 | 20.6 | 20.1 | 19.6 | 20.1 | 10.8 | 22.1 |
| S01A05I01 | 55.7 | 44.9 | 48.1 | 49.8 | 36.7 | 26.0 | 36.9 |
| S01A06I01 | 19.4 | 16.8 | 16.8 | 16.9 | 15.1 | 20.4 | 14.2 |
| S08A04I01 | 34.3 | 52.9 | 53.3 | 55.0 | 40.2 | 37.2 | 61.8 |
| S08A05I01 | 47.9 | 52.8 | 52.1 | 55.8 | 39.3 | 37.9 | 48.4 |
| S08A06I01 | 36.7 | 62.8 | 64.8 | 67.0 | 48.0 | 53.4 | 58.6 |
| S10A04I01 | 40.8 | 34.7 | 35.2 | 36.6 | 33.9 | 24.0 | 26.3 |
| S10A05I01 | 27.3 | 27.0 | 29.2 | 28.1 | 29.8 | 21.3 | 21.3 |
| S10A06I01 | 27.9 | 28.4 | 28.8 | 28.1 | 22.7 | 22.2 | 22.7 |
| S12A04I01 | 17.5 | 20.0 | 19.1 | 25.3 | 25.7 | 33.8 | 16.3 |
| S12A05I01 | 23.2 | 41.3 | 45.6 | 39.7 | 31.2 | 45.0 | 56.4 |
| S12A06I01 | 27.8 | 25.7 | 25.3 | 31.0 | 21.4 | 15.8 | 35.4 |
| S18A04I01 | 10.9 | 27.3 | 27.2 | 27.2 | 10.9 | 32.5 | 26.5 |
| S18A05I01 | 15.5 | 24.7 | 25.2 | 15.5 | 15.5 | 15.5 | 28.4 |
| S18A06I01 | 25.9 | 52.1 | 52.2 | 48.5 | 42.8 | 61.1 | 49.1 |
| S30A04I01 | 16.7 | 38.0 | 38.0 | 38.1 | 48.4 | 37.4 | 38.0 |
| S30A05I01 | 50.1 | 61.4 | 61.9 | 61.5 | 41.7 | 44.2 | 56.6 |
| S30A06I01 | 15.4 | 46.4 | 46.0 | 43.8 | 70.5 | 20.5 | 42.6 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 55 | 27 | 389 |
| 41 | 27 | 382 |
| 1 | 27 | 294 |
| 47 | 49 | 290 |
| 53 | 27 | 258 |
| 51 | 27 | 245 |
| 67 | 27 | 226 |
| 61 | 27 | 225 |
| 72 | 27 | 200 |
| 53 | 54 | 189 |
| 14 | 27 | 185 |
| 27 | 51 | 174 |
| 27 | 71 | 172 |
| 74 | 27 | 167 |
| 27 | 47 | 142 |
