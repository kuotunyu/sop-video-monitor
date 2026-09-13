Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 2).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.7, 34.1] | 8.0 [6.8, 9.4] | 8.9 [5.9, 11.9] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 32.4 [25.0, 42.1] | 33.2 [27.4, 39.2] | 29.2 [22.0, 37.7] | 21.3 [15.5, 27.8] | 10.5 [6.8, 14.5] |
| fusion_conf_causal | 32.2 [25.4, 41.1] | 35.5 [29.0, 42.9] | 30.7 [24.0, 39.5] | 21.3 [16.5, 29.2] | 10.6 [7.6, 15.5] |
| fusion_conf_offline | 44.6 [34.3, 57.8] | 42.0 [30.1, 52.7] | 43.9 [32.8, 57.5] | 40.6 [29.7, 53.8] | 24.7 [18.0, 35.1] |
| fusion_geo_causal | 32.9 [25.9, 42.0] | 34.0 [29.3, 39.9] | 30.0 [24.4, 39.0] | 20.7 [15.9, 29.0] | 8.9 [4.6, 13.7] |
| fusion_geo_offline | 45.2 [34.6, 58.6] | 44.4 [33.6, 54.1] | 45.3 [33.6, 58.8] | 40.9 [28.7, 55.6] | 26.4 [20.0, 35.7] |
| fusion_offline | 44.6 [34.7, 57.4] | 42.1 [29.9, 52.5] | 43.2 [31.8, 57.7] | 40.3 [29.5, 54.4] | 25.4 [19.3, 35.1] |
| view0_causal | 26.8 [17.0, 38.9] | 28.2 [20.1, 35.9] | 25.7 [16.7, 36.0] | 19.2 [12.7, 27.0] | 11.1 [4.1, 19.3] |
| view0_offline | 36.6 [25.3, 50.7] | 39.2 [29.4, 51.5] | 38.4 [26.3, 52.6] | 32.1 [21.2, 46.1] | 21.6 [13.4, 32.0] |
| view1_causal | 33.1 [27.5, 40.3] | 37.0 [30.5, 43.6] | 29.3 [23.5, 37.8] | 21.6 [17.1, 28.4] | 11.6 [8.1, 16.3] |
| view1_offline | 39.0 [31.5, 49.2] | 41.1 [34.2, 48.2] | 39.6 [31.7, 48.3] | 33.2 [24.3, 42.5] | 21.2 [14.3, 28.9] |
| view2_causal | 28.8 [20.3, 39.6] | 31.0 [26.1, 36.2] | 28.7 [22.8, 36.8] | 20.0 [14.3, 27.9] | 9.6 [5.4, 14.1] |
| view2_offline | 42.4 [34.2, 53.7] | 42.4 [34.5, 49.9] | 44.2 [35.8, 54.6] | 40.2 [31.8, 50.6] | 26.7 [19.6, 36.2] |

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
| S01A04I01 | 22.0 | 17.0 | 18.1 | 23.6 | 15.0 | 24.2 | 24.3 | 4.3 | 16.4 | 26.4 | 24.2 | 13.7 | 25.6 |
| S01A05I01 | 55.7 | 29.9 | 29.3 | 44.7 | 53.2 | 55.2 | 46.5 | 13.6 | 29.5 | 39.8 | 35.3 | 16.3 | 43.5 |
| S01A06I01 | 19.4 | 18.7 | 18.7 | 34.1 | 18.7 | 19.9 | 35.1 | 18.3 | 20.2 | 18.7 | 22.4 | 19.7 | 26.9 |
| S08A04I01 | 34.3 | 42.9 | 39.1 | 62.5 | 47.2 | 65.0 | 61.6 | 47.8 | 63.9 | 34.6 | 44.3 | 47.0 | 69.6 |
| S08A05I01 | 47.9 | 56.7 | 57.7 | 57.1 | 60.1 | 59.5 | 56.7 | 43.1 | 45.6 | 51.2 | 39.6 | 56.2 | 47.8 |
| S08A06I01 | 36.7 | 56.5 | 52.7 | 77.4 | 53.3 | 84.3 | 79.4 | 56.4 | 70.4 | 47.0 | 71.9 | 57.8 | 72.7 |
| S10A04I01 | 40.8 | 37.1 | 38.5 | 38.6 | 38.2 | 40.3 | 38.8 | 28.1 | 22.0 | 38.2 | 38.9 | 35.9 | 45.7 |
| S10A05I01 | 27.3 | 31.4 | 32.1 | 32.6 | 28.6 | 32.4 | 32.7 | 30.4 | 31.1 | 27.3 | 17.6 | 28.4 | 44.9 |
| S10A06I01 | 27.9 | 20.6 | 20.8 | 19.0 | 20.8 | 19.3 | 18.9 | 16.2 | 19.9 | 21.4 | 19.0 | 17.4 | 15.0 |
| S12A04I01 | 17.5 | 17.5 | 17.5 | 43.1 | 17.5 | 43.1 | 42.3 | 17.5 | 42.6 | 16.6 | 44.2 | 17.2 | 16.3 |
| S12A05I01 | 23.2 | 36.8 | 33.1 | 70.6 | 32.1 | 72.4 | 71.0 | 32.3 | 63.6 | 34.9 | 57.0 | 28.1 | 67.9 |
| S12A06I01 | 27.8 | 23.2 | 23.4 | 28.7 | 23.2 | 31.5 | 28.4 | 14.1 | 26.2 | 16.9 | 30.1 | 22.2 | 28.1 |
| S18A04I01 | 10.9 | 19.5 | 19.7 | 23.5 | 15.4 | 19.6 | 24.8 | 9.8 | 7.8 | 17.0 | 22.4 | 20.1 | 30.2 |
| S18A05I01 | 15.5 | 30.3 | 30.5 | 35.9 | 15.5 | 35.1 | 37.4 | 15.5 | 8.2 | 24.5 | 52.5 | 23.7 | 27.5 |
| S18A06I01 | 25.9 | 35.7 | 39.1 | 63.7 | 33.3 | 63.4 | 63.4 | 37.7 | 53.3 | 59.1 | 78.4 | 27.9 | 61.1 |
| S30A04I01 | 16.7 | 14.8 | 20.4 | 57.2 | 14.8 | 57.9 | 57.7 | 28.1 | 60.8 | 25.4 | 51.5 | 13.9 | 59.3 |
| S30A05I01 | 50.1 | 64.9 | 64.2 | 63.7 | 63.1 | 65.2 | 64.7 | 42.1 | 46.9 | 68.3 | 61.8 | 52.3 | 49.0 |
| S30A06I01 | 15.4 | 42.0 | 38.9 | 63.0 | 41.2 | 56.3 | 56.5 | 47.4 | 56.3 | 40.8 | 31.7 | 29.3 | 51.1 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 51 | 27 | 485 |
| 72 | 27 | 463 |
| 53 | 27 | 437 |
| 74 | 27 | 402 |
| 27 | 71 | 361 |
| 44 | 27 | 300 |
| 27 | 59 | 298 |
| 55 | 27 | 296 |
| 41 | 27 | 289 |
| 58 | 59 | 286 |
| 61 | 27 | 266 |
| 1 | 27 | 262 |
| 14 | 27 | 232 |
| 67 | 27 | 226 |
| 61 | 59 | 217 |
