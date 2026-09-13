Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 2).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [26.9, 45.4] | 8.3 [6.6, 10.1] | 9.4 [4.8, 14.0] | 3.6 [0.6, 7.9] | 0.7 [0.0, 2.6] |
| fusion_causal | 44.5 [38.3, 49.6] | 36.5 [30.7, 43.5] | 38.2 [31.0, 46.8] | 30.2 [23.3, 38.7] | 17.3 [10.3, 26.3] |
| fusion_conf_causal | 44.7 [38.4, 49.7] | 37.0 [29.8, 45.1] | 37.2 [29.4, 46.8] | 29.7 [23.0, 37.9] | 16.2 [9.3, 24.8] |
| fusion_geo_causal | 45.2 [38.6, 50.8] | 36.7 [32.3, 41.1] | 37.5 [31.0, 45.1] | 31.6 [26.3, 38.1] | 18.4 [11.0, 26.8] |
| view0_causal | 35.6 [32.2, 39.5] | 34.3 [29.6, 38.9] | 29.5 [23.3, 37.2] | 25.0 [20.1, 31.3] | 13.7 [9.3, 18.8] |
| view1_causal | 36.5 [29.6, 45.0] | 32.2 [24.5, 39.8] | 33.7 [27.8, 41.6] | 26.3 [17.3, 36.9] | 15.2 [8.8, 22.9] |
| view2_causal | 39.1 [33.9, 44.4] | 34.8 [26.3, 43.8] | 33.6 [25.8, 43.6] | 26.0 [19.0, 35.4] | 15.9 [11.5, 21.9] |

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
| S01A04I01 | 41.2 | 40.6 | 39.6 | 38.8 | 33.3 | 12.3 | 43.6 |
| S01A05I01 | 47.4 | 52.5 | 58.8 | 62.4 | 39.4 | 21.4 | 64.8 |
| S01A06I01 | 63.7 | 55.6 | 52.9 | 66.7 | 25.8 | 63.7 | 19.5 |
| S08A04I01 | 18.1 | 64.0 | 63.6 | 64.1 | 42.1 | 65.1 | 58.6 |
| S08A05I01 | 25.4 | 31.8 | 32.0 | 31.0 | 37.8 | 31.8 | 28.4 |
| S08A06I01 | 23.6 | 49.1 | 48.0 | 45.1 | 37.8 | 49.3 | 44.0 |
| S10A04I01 | 33.7 | 36.9 | 36.1 | 40.1 | 35.7 | 31.2 | 30.4 |
| S10A05I01 | 35.0 | 46.2 | 46.7 | 40.9 | 30.5 | 32.7 | 50.1 |
| S10A06I01 | 26.0 | 24.1 | 24.0 | 24.7 | 23.0 | 16.8 | 23.3 |
| S12A04I01 | 62.5 | 56.9 | 56.3 | 61.2 | 46.0 | 42.3 | 53.0 |
| S12A05I01 | 70.6 | 69.3 | 70.6 | 65.2 | 53.3 | 58.3 | 59.6 |
| S12A06I01 | 28.7 | 35.4 | 34.8 | 35.6 | 29.8 | 28.6 | 23.3 |
| S18A04I01 | 14.7 | 29.9 | 30.1 | 15.8 | 7.3 | 12.7 | 31.8 |
| S18A05I01 | 31.6 | 21.9 | 21.9 | 23.9 | 26.0 | 22.8 | 21.5 |
| S18A06I01 | 22.9 | 79.4 | 78.4 | 78.4 | 67.0 | 69.9 | 80.6 |
| S30A04I01 | 48.5 | 42.1 | 44.3 | 43.2 | 42.9 | 36.1 | 43.2 |
| S30A05I01 | 41.1 | 38.7 | 37.3 | 39.9 | 31.8 | 41.2 | 29.4 |
| S30A06I01 | 30.1 | 43.0 | 47.2 | 49.7 | 41.8 | 52.8 | 18.3 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 1 | 27 | 327 |
| 41 | 27 | 310 |
| 24 | 27 | 285 |
| 58 | 27 | 278 |
| 74 | 27 | 242 |
| 27 | 52 | 238 |
| 65 | 68 | 237 |
| 27 | 71 | 222 |
| 27 | 51 | 213 |
| 64 | 27 | 210 |
| 70 | 71 | 201 |
| 55 | 27 | 195 |
| 6 | 27 | 184 |
| 27 | 73 | 171 |
| 12 | 27 | 163 |
