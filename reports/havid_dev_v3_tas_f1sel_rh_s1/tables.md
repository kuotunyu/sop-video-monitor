Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 1).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.6, 33.9] | 8.0 [6.9, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.4] | 0.0 [0.0, 0.0] |
| fusion_causal | 37.0 [28.6, 46.2] | 33.4 [27.9, 38.1] | 30.3 [22.1, 38.6] | 24.5 [18.0, 31.4] | 13.4 [8.5, 18.3] |
| fusion_conf_causal | 36.6 [28.5, 45.7] | 34.2 [27.2, 39.9] | 31.0 [22.6, 40.0] | 24.1 [17.7, 30.9] | 14.1 [8.7, 20.0] |
| fusion_conf_offline | 42.2 [30.6, 56.9] | 40.9 [32.0, 50.2] | 42.3 [32.3, 55.3] | 34.8 [23.8, 48.4] | 23.8 [14.1, 36.5] |
| fusion_geo_causal | 37.3 [29.4, 45.9] | 31.6 [24.2, 38.3] | 31.8 [24.9, 38.1] | 22.9 [16.7, 27.8] | 13.6 [8.0, 19.1] |
| fusion_geo_offline | 42.4 [31.6, 56.1] | 40.9 [31.5, 50.2] | 44.7 [32.9, 58.6] | 36.3 [25.2, 49.6] | 23.4 [13.1, 36.1] |
| fusion_offline | 41.6 [29.9, 56.3] | 39.8 [31.2, 49.2] | 42.5 [33.1, 54.2] | 35.5 [24.8, 48.7] | 23.9 [14.3, 35.8] |
| view0_causal | 33.8 [26.3, 42.5] | 30.4 [24.4, 35.7] | 28.8 [21.0, 37.8] | 22.0 [15.5, 28.8] | 10.2 [5.3, 16.6] |
| view0_offline | 34.0 [22.6, 47.1] | 39.7 [31.1, 47.1] | 37.8 [26.4, 52.5] | 30.4 [18.7, 45.1] | 20.1 [10.0, 32.3] |
| view1_causal | 34.0 [25.0, 44.8] | 30.3 [24.5, 36.0] | 27.8 [20.8, 35.8] | 21.3 [16.1, 28.1] | 12.4 [8.9, 17.3] |
| view1_offline | 36.4 [26.6, 48.8] | 40.3 [31.0, 48.7] | 38.5 [26.9, 51.8] | 30.3 [19.4, 43.4] | 18.9 [10.1, 29.8] |
| view2_causal | 32.4 [25.5, 41.2] | 30.7 [26.3, 34.9] | 28.2 [23.6, 35.0] | 19.1 [14.5, 25.6] | 8.0 [4.9, 11.8] |
| view2_offline | 40.2 [31.0, 51.9] | 45.2 [34.9, 54.7] | 45.0 [36.5, 56.3] | 35.5 [26.3, 46.8] | 25.2 [17.6, 35.8] |

| run | what it uses |
|---|---|
| majority | constant most frequent training label |
| fusion_causal | mean of the per-view causal posteriors (late fusion, online) |
| fusion_conf_causal | per-frame confidence-weighted mean of the per-view causal posteriors, weight = each view's max posterior, online |
| fusion_conf_offline | per-frame confidence-weighted mean of the per-view offline posteriors, weight = each view's max posterior |
| fusion_geo_causal | normalised geometric mean of the per-view causal posteriors (product of experts, online) |
| fusion_geo_offline | normalised geometric mean of the per-view offline posteriors (product of experts) |
| fusion_offline | mean of the per-view offline posteriors (late fusion) |
| view0_causal | MS-TCN++ on the side view (view 0), left-only padding: frame t sees frames <= t (online) |
| view0_offline | MS-TCN++ on the side view (view 0), symmetric padding: sees future frames (not an online result) |
| view1_causal | MS-TCN++ on the front view (view 1), left-only padding: frame t sees frames <= t (online) |
| view1_offline | MS-TCN++ on the front view (view 1), symmetric padding: sees future frames (not an online result) |
| view2_causal | MS-TCN++ on the top view (view 2), left-only padding: frame t sees frames <= t (online) |
| view2_offline | MS-TCN++ on the top view (view 2), symmetric padding: sees future frames (not an online result) |

Per-video MoF:

| video | majority | fusion_causal | fusion_conf_causal | fusion_conf_offline | fusion_geo_causal | fusion_geo_offline | fusion_offline | view0_causal | view0_offline | view1_causal | view1_offline | view2_causal | view2_offline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S01A04I01 | 22.0 | 19.9 | 19.4 | 16.6 | 20.0 | 17.9 | 16.1 | 9.6 | 13.0 | 9.9 | 15.8 | 21.9 | 16.8 |
| S01A05I01 | 55.7 | 36.9 | 36.1 | 34.4 | 44.1 | 38.8 | 28.9 | 53.5 | 27.4 | 30.0 | 26.2 | 28.3 | 37.5 |
| S01A06I01 | 19.4 | 18.9 | 18.8 | 27.8 | 19.4 | 23.3 | 27.7 | 15.9 | 14.3 | 19.1 | 19.4 | 16.8 | 33.7 |
| S08A04I01 | 34.3 | 42.9 | 42.3 | 59.5 | 42.0 | 61.6 | 58.6 | 35.9 | 55.4 | 32.2 | 45.6 | 40.4 | 58.3 |
| S08A05I01 | 47.9 | 57.6 | 56.9 | 53.6 | 57.5 | 46.4 | 50.1 | 46.5 | 39.1 | 44.7 | 39.6 | 58.5 | 46.9 |
| S08A06I01 | 36.7 | 63.2 | 62.8 | 83.7 | 65.4 | 83.9 | 83.7 | 58.7 | 61.4 | 59.3 | 71.8 | 60.7 | 79.8 |
| S10A04I01 | 40.8 | 45.0 | 42.1 | 23.8 | 45.1 | 24.8 | 23.4 | 35.3 | 16.4 | 44.5 | 32.3 | 31.5 | 23.6 |
| S10A05I01 | 27.3 | 36.1 | 35.9 | 36.4 | 35.6 | 37.0 | 36.4 | 30.7 | 34.4 | 28.0 | 18.2 | 32.9 | 39.0 |
| S10A06I01 | 27.9 | 27.2 | 28.1 | 24.5 | 27.9 | 23.5 | 24.3 | 26.4 | 13.6 | 17.9 | 21.6 | 25.0 | 24.2 |
| S12A04I01 | 17.5 | 17.1 | 17.1 | 47.9 | 17.1 | 37.4 | 46.8 | 17.5 | 53.2 | 17.2 | 17.5 | 16.9 | 17.5 |
| S12A05I01 | 23.2 | 39.9 | 39.9 | 75.2 | 41.9 | 76.9 | 75.7 | 36.0 | 67.0 | 41.6 | 59.3 | 33.6 | 71.9 |
| S12A06I01 | 27.8 | 27.0 | 27.1 | 25.4 | 26.6 | 27.6 | 25.8 | 22.9 | 23.7 | 16.1 | 17.9 | 27.9 | 28.7 |
| S18A04I01 | 10.9 | 12.5 | 12.5 | 19.9 | 12.7 | 31.8 | 21.1 | 11.8 | 9.0 | 38.5 | 47.3 | 12.5 | 38.2 |
| S18A05I01 | 15.5 | 28.8 | 29.2 | 31.8 | 23.2 | 57.8 | 34.0 | 15.5 | 8.6 | 31.6 | 32.9 | 23.7 | 46.2 |
| S18A06I01 | 25.9 | 46.4 | 46.4 | 80.5 | 45.1 | 75.5 | 78.9 | 52.6 | 74.9 | 55.7 | 79.7 | 42.9 | 60.3 |
| S30A04I01 | 16.7 | 32.5 | 32.7 | 57.9 | 32.8 | 59.6 | 59.3 | 38.3 | 62.0 | 32.1 | 58.6 | 30.1 | 39.1 |
| S30A05I01 | 50.1 | 57.8 | 57.9 | 58.8 | 55.0 | 46.4 | 56.0 | 55.8 | 33.9 | 61.5 | 40.6 | 45.6 | 63.0 |
| S30A06I01 | 15.4 | 62.4 | 61.2 | 56.0 | 62.3 | 53.5 | 55.7 | 58.3 | 53.1 | 62.5 | 52.6 | 42.3 | 40.1 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 51 | 27 | 483 |
| 46 | 27 | 430 |
| 74 | 27 | 427 |
| 41 | 27 | 395 |
| 72 | 27 | 390 |
| 53 | 27 | 379 |
| 61 | 27 | 366 |
| 55 | 27 | 363 |
| 1 | 27 | 315 |
| 51 | 52 | 284 |
| 67 | 27 | 247 |
| 27 | 47 | 247 |
| 14 | 27 | 237 |
| 71 | 27 | 218 |
| 65 | 27 | 168 |
