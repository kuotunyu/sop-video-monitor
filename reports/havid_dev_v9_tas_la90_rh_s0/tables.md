Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.6, 34.1] | 8.0 [6.8, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 34.9 [27.1, 45.1] | 32.2 [27.6, 37.1] | 30.1 [24.6, 37.7] | 25.4 [20.7, 31.9] | 14.1 [7.8, 23.4] |
| fusion_conf_causal | 34.6 [27.0, 45.1] | 30.7 [25.0, 35.4] | 29.9 [24.1, 37.4] | 24.8 [18.8, 31.7] | 13.8 [7.4, 23.0] |
| fusion_geo_causal | 35.5 [28.4, 45.1] | 32.7 [27.7, 37.1] | 31.9 [25.1, 39.4] | 25.4 [19.8, 31.9] | 15.5 [8.2, 24.8] |
| view0_causal | 30.1 [25.4, 36.2] | 30.9 [27.3, 35.1] | 27.2 [20.9, 34.7] | 20.8 [16.0, 26.5] | 13.2 [9.0, 18.9] |
| view1_causal | 28.6 [21.0, 38.4] | 28.9 [21.9, 36.0] | 27.5 [20.6, 36.0] | 20.0 [13.0, 28.6] | 13.3 [7.5, 19.7] |
| view2_causal | 31.2 [26.3, 38.0] | 32.8 [27.9, 38.1] | 30.7 [25.0, 37.5] | 23.7 [19.0, 30.3] | 12.4 [7.8, 17.5] |

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
| S01A04I01 | 22.0 | 10.8 | 10.9 | 13.3 | 13.3 | 5.4 | 21.1 |
| S01A05I01 | 55.7 | 48.0 | 49.0 | 48.9 | 51.3 | 36.6 | 35.9 |
| S01A06I01 | 19.4 | 14.6 | 14.4 | 14.4 | 14.6 | 14.8 | 14.3 |
| S08A04I01 | 34.3 | 56.9 | 57.6 | 58.6 | 56.0 | 36.6 | 45.9 |
| S08A05I01 | 47.9 | 39.4 | 40.2 | 40.7 | 22.2 | 36.5 | 31.6 |
| S08A06I01 | 36.7 | 62.8 | 61.9 | 61.5 | 42.9 | 66.5 | 52.2 |
| S10A04I01 | 40.8 | 36.2 | 36.0 | 39.7 | 31.4 | 31.0 | 30.7 |
| S10A05I01 | 27.3 | 39.3 | 39.3 | 41.5 | 34.7 | 24.4 | 38.8 |
| S10A06I01 | 27.9 | 21.4 | 21.6 | 19.2 | 21.4 | 18.5 | 16.7 |
| S12A04I01 | 17.5 | 24.3 | 23.7 | 17.5 | 24.7 | 12.7 | 17.5 |
| S12A05I01 | 23.2 | 38.2 | 39.0 | 48.6 | 33.6 | 35.6 | 37.9 |
| S12A06I01 | 27.8 | 19.8 | 19.4 | 20.8 | 19.0 | 15.3 | 22.1 |
| S18A04I01 | 10.9 | 11.8 | 12.9 | 10.9 | 8.4 | 7.5 | 17.1 |
| S18A05I01 | 15.5 | 28.6 | 27.7 | 21.1 | 10.1 | 10.8 | 29.0 |
| S18A06I01 | 25.9 | 66.0 | 56.3 | 64.2 | 50.4 | 50.9 | 58.4 |
| S30A04I01 | 16.7 | 18.7 | 18.9 | 23.1 | 16.7 | 28.1 | 38.5 |
| S30A05I01 | 50.1 | 60.0 | 60.1 | 55.8 | 45.5 | 46.3 | 42.4 |
| S30A06I01 | 15.4 | 53.9 | 53.8 | 52.2 | 47.4 | 50.1 | 33.8 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 27 | 59 | 430 |
| 27 | 47 | 350 |
| 61 | 27 | 266 |
| 41 | 27 | 256 |
| 72 | 54 | 244 |
| 51 | 27 | 231 |
| 72 | 55 | 229 |
| 1 | 27 | 209 |
| 27 | 52 | 195 |
| 14 | 27 | 194 |
| 27 | 71 | 181 |
| 49 | 47 | 179 |
| 67 | 27 | 178 |
| 58 | 59 | 177 |
| 9 | 27 | 174 |
