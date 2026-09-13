Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 1).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [27.5, 45.4] | 8.3 [6.7, 10.1] | 9.4 [5.0, 14.0] | 3.6 [0.7, 7.6] | 0.7 [0.0, 2.5] |
| fusion_causal | 38.6 [34.1, 43.1] | 34.0 [29.9, 38.0] | 29.6 [25.0, 34.0] | 20.5 [16.5, 24.2] | 12.3 [9.3, 14.5] |
| fusion_conf_causal | 38.0 [33.2, 42.8] | 33.8 [26.9, 39.5] | 27.3 [22.2, 32.5] | 20.5 [15.7, 25.5] | 11.0 [6.9, 15.0] |
| fusion_conf_offline | 47.0 [42.4, 52.6] | 40.3 [34.3, 45.7] | 46.8 [38.2, 54.5] | 41.9 [33.3, 49.8] | 27.3 [20.5, 33.8] |
| fusion_geo_causal | 38.8 [35.1, 42.8] | 30.2 [22.8, 36.2] | 31.8 [23.7, 39.3] | 19.1 [13.6, 23.2] | 8.7 [5.2, 11.7] |
| fusion_geo_offline | 46.1 [42.0, 51.4] | 41.8 [35.7, 47.5] | 44.9 [37.1, 52.2] | 40.3 [31.9, 48.6] | 26.9 [20.3, 33.1] |
| fusion_offline | 47.2 [42.6, 53.0] | 41.0 [35.2, 46.3] | 46.1 [38.5, 53.1] | 41.3 [33.4, 48.6] | 28.0 [21.2, 34.7] |
| view0_causal | 35.5 [31.0, 39.8] | 29.4 [24.4, 35.0] | 27.1 [21.8, 32.4] | 19.9 [13.7, 25.4] | 9.7 [4.5, 14.1] |
| view0_offline | 33.9 [30.8, 36.6] | 39.1 [34.6, 43.7] | 37.7 [30.7, 43.6] | 32.0 [27.0, 36.7] | 17.1 [12.3, 22.1] |
| view1_causal | 32.1 [26.6, 37.5] | 29.0 [22.8, 34.8] | 26.7 [20.9, 32.4] | 21.1 [17.3, 24.9] | 9.0 [6.3, 11.5] |
| view1_offline | 36.8 [30.3, 44.6] | 40.3 [32.7, 48.0] | 41.8 [32.6, 51.1] | 34.9 [26.2, 43.9] | 18.4 [11.7, 25.5] |
| view2_causal | 35.6 [33.0, 38.1] | 36.4 [31.6, 40.8] | 31.4 [27.0, 35.2] | 22.6 [18.0, 26.7] | 12.0 [9.3, 14.7] |
| view2_offline | 43.1 [38.4, 49.2] | 38.6 [33.0, 44.4] | 44.6 [36.6, 53.3] | 39.6 [31.1, 48.8] | 28.0 [21.1, 35.6] |

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
| S01A04I01 | 41.2 | 41.2 | 41.2 | 42.2 | 41.2 | 40.5 | 41.8 | 43.7 | 31.0 | 29.3 | 26.4 | 31.2 | 40.4 |
| S01A05I01 | 47.4 | 29.3 | 28.2 | 31.3 | 32.0 | 33.0 | 32.5 | 31.4 | 25.3 | 26.6 | 29.7 | 27.3 | 21.6 |
| S01A06I01 | 63.7 | 61.9 | 61.8 | 44.7 | 63.7 | 50.2 | 45.7 | 27.5 | 23.9 | 63.7 | 34.7 | 61.7 | 44.4 |
| S08A04I01 | 18.1 | 53.6 | 51.9 | 66.8 | 46.6 | 66.0 | 69.4 | 48.8 | 36.0 | 45.2 | 63.5 | 49.6 | 67.7 |
| S08A05I01 | 25.4 | 23.0 | 21.5 | 38.4 | 25.5 | 35.6 | 37.9 | 32.5 | 38.4 | 19.1 | 32.8 | 32.5 | 31.5 |
| S08A06I01 | 23.6 | 31.9 | 30.8 | 56.6 | 33.9 | 64.1 | 58.7 | 29.0 | 35.2 | 39.2 | 46.2 | 29.1 | 62.2 |
| S10A04I01 | 33.7 | 34.0 | 33.0 | 45.7 | 33.3 | 45.3 | 46.7 | 33.7 | 44.1 | 21.0 | 41.3 | 33.6 | 41.9 |
| S10A05I01 | 35.0 | 41.6 | 41.3 | 34.2 | 42.2 | 34.5 | 34.7 | 41.3 | 42.1 | 32.1 | 21.5 | 46.8 | 31.6 |
| S10A06I01 | 26.0 | 22.5 | 21.1 | 39.4 | 26.0 | 37.0 | 38.9 | 17.0 | 21.9 | 18.4 | 23.7 | 21.8 | 39.8 |
| S12A04I01 | 62.5 | 62.2 | 62.2 | 58.4 | 61.4 | 54.6 | 58.3 | 50.5 | 39.1 | 28.2 | 36.9 | 62.5 | 49.9 |
| S12A05I01 | 70.6 | 46.1 | 49.2 | 61.7 | 43.3 | 57.8 | 59.3 | 58.4 | 51.3 | 33.1 | 35.5 | 38.4 | 54.2 |
| S12A06I01 | 28.7 | 25.8 | 24.7 | 30.9 | 26.7 | 31.8 | 30.2 | 23.8 | 22.2 | 23.1 | 18.4 | 24.9 | 31.4 |
| S18A04I01 | 14.7 | 11.0 | 11.0 | 29.0 | 11.6 | 35.6 | 29.9 | 10.3 | 13.5 | 12.8 | 29.3 | 9.5 | 23.1 |
| S18A05I01 | 31.6 | 18.9 | 17.8 | 34.2 | 33.8 | 14.2 | 33.5 | 6.7 | 11.6 | 18.9 | 11.6 | 18.5 | 37.4 |
| S18A06I01 | 22.9 | 70.5 | 69.9 | 89.8 | 69.0 | 89.6 | 88.5 | 67.4 | 87.1 | 67.9 | 75.7 | 62.3 | 87.3 |
| S30A04I01 | 48.5 | 44.3 | 43.9 | 49.9 | 45.2 | 48.4 | 48.5 | 47.1 | 26.1 | 42.1 | 53.1 | 38.9 | 47.0 |
| S30A05I01 | 41.1 | 45.2 | 45.3 | 44.6 | 41.7 | 47.2 | 45.0 | 37.4 | 39.1 | 39.3 | 42.7 | 46.3 | 40.1 |
| S30A06I01 | 30.1 | 50.7 | 49.5 | 63.9 | 46.7 | 51.5 | 65.1 | 41.4 | 33.6 | 41.1 | 53.6 | 31.6 | 34.6 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 74 | 27 | 438 |
| 55 | 27 | 407 |
| 49 | 27 | 397 |
| 1 | 27 | 346 |
| 41 | 27 | 306 |
| 27 | 51 | 303 |
| 65 | 27 | 291 |
| 68 | 27 | 274 |
| 27 | 71 | 266 |
| 64 | 27 | 245 |
| 46 | 27 | 241 |
| 58 | 27 | 238 |
| 24 | 27 | 234 |
| 71 | 27 | 231 |
| 6 | 27 | 230 |
