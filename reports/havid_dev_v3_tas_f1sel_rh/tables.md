Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.6, 34.1] | 8.0 [6.8, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 34.3 [26.3, 44.0] | 32.4 [25.6, 39.3] | 30.2 [23.4, 38.4] | 25.1 [18.3, 33.3] | 13.1 [8.3, 19.3] |
| fusion_conf_causal | 33.9 [25.2, 44.7] | 33.8 [26.6, 42.1] | 30.8 [23.1, 40.8] | 26.0 [18.7, 34.6] | 14.7 [9.9, 21.5] |
| fusion_conf_offline | 41.0 [32.2, 53.8] | 39.7 [32.7, 46.4] | 39.8 [32.6, 50.2] | 35.7 [28.3, 46.3] | 23.0 [17.8, 30.5] |
| fusion_geo_causal | 35.4 [28.3, 43.3] | 33.0 [26.3, 40.1] | 33.3 [24.1, 43.8] | 26.7 [19.2, 35.5] | 14.7 [9.3, 22.7] |
| fusion_geo_offline | 39.6 [31.3, 51.3] | 40.5 [34.1, 47.3] | 42.1 [34.7, 53.5] | 35.4 [28.0, 46.7] | 23.2 [17.9, 30.3] |
| fusion_offline | 41.0 [32.5, 53.3] | 38.3 [31.9, 44.4] | 40.2 [33.0, 49.7] | 35.5 [27.4, 46.4] | 22.8 [17.4, 29.9] |
| view0_causal | 29.5 [23.8, 36.9] | 31.9 [26.4, 37.7] | 29.3 [20.9, 40.1] | 22.0 [15.7, 29.5] | 12.5 [6.1, 19.8] |
| view0_offline | 35.4 [28.1, 44.7] | 37.0 [30.2, 42.6] | 37.4 [31.4, 45.5] | 29.3 [21.8, 38.7] | 19.2 [13.0, 27.7] |
| view1_causal | 29.8 [22.2, 37.2] | 32.0 [25.7, 37.5] | 28.5 [19.0, 39.5] | 19.5 [12.9, 26.7] | 12.1 [8.1, 17.5] |
| view1_offline | 35.4 [27.7, 45.7] | 39.6 [32.2, 46.7] | 38.0 [30.4, 48.6] | 33.8 [25.5, 44.3] | 20.5 [13.0, 30.2] |
| view2_causal | 32.2 [25.3, 42.2] | 31.0 [25.6, 36.0] | 27.7 [21.3, 36.1] | 20.9 [16.7, 27.6] | 11.3 [6.1, 18.9] |
| view2_offline | 37.1 [26.2, 51.2] | 43.7 [38.2, 49.1] | 42.6 [33.9, 53.0] | 36.9 [28.0, 47.0] | 24.7 [15.9, 35.0] |

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
| S01A04I01 | 22.0 | 17.1 | 16.6 | 20.7 | 17.1 | 17.6 | 18.9 | 12.7 | 17.8 | 14.5 | 19.3 | 24.5 | 13.1 |
| S01A05I01 | 55.7 | 43.4 | 39.5 | 53.9 | 53.6 | 57.4 | 57.5 | 37.4 | 58.4 | 35.9 | 42.2 | 30.2 | 28.0 |
| S01A06I01 | 19.4 | 17.6 | 17.0 | 20.6 | 19.0 | 20.6 | 20.4 | 15.8 | 15.6 | 19.4 | 22.2 | 15.3 | 21.3 |
| S08A04I01 | 34.3 | 43.8 | 43.9 | 69.1 | 43.4 | 64.8 | 66.5 | 40.9 | 55.6 | 34.4 | 40.1 | 45.7 | 61.0 |
| S08A05I01 | 47.9 | 57.5 | 57.9 | 51.3 | 57.9 | 51.1 | 51.1 | 37.3 | 38.7 | 46.9 | 48.4 | 51.5 | 52.5 |
| S08A06I01 | 36.7 | 60.2 | 61.0 | 77.0 | 57.8 | 74.8 | 77.4 | 54.3 | 60.4 | 48.4 | 79.0 | 57.7 | 64.6 |
| S10A04I01 | 40.8 | 34.4 | 32.1 | 30.6 | 43.8 | 31.6 | 28.4 | 28.1 | 29.1 | 48.2 | 28.0 | 23.0 | 17.1 |
| S10A05I01 | 27.3 | 30.2 | 30.1 | 42.8 | 29.8 | 40.0 | 42.4 | 27.7 | 32.0 | 25.9 | 13.2 | 31.3 | 37.3 |
| S10A06I01 | 27.9 | 29.2 | 29.9 | 26.1 | 30.3 | 19.9 | 26.2 | 29.5 | 25.8 | 23.9 | 23.0 | 22.7 | 23.2 |
| S12A04I01 | 17.5 | 17.5 | 17.5 | 17.1 | 17.5 | 17.1 | 17.1 | 17.5 | 17.1 | 10.9 | 22.9 | 17.4 | 18.9 |
| S12A05I01 | 23.2 | 31.8 | 26.6 | 60.9 | 37.5 | 64.2 | 63.4 | 34.1 | 45.1 | 26.9 | 51.2 | 36.5 | 61.4 |
| S12A06I01 | 27.8 | 15.2 | 16.0 | 20.1 | 15.3 | 26.7 | 22.4 | 16.6 | 16.7 | 12.3 | 25.6 | 17.1 | 27.4 |
| S18A04I01 | 10.9 | 13.7 | 13.9 | 25.9 | 13.3 | 19.0 | 25.6 | 11.8 | 13.2 | 11.7 | 18.8 | 18.9 | 35.8 |
| S18A05I01 | 15.5 | 24.7 | 24.7 | 29.7 | 15.5 | 15.5 | 30.3 | 15.5 | 10.8 | 15.5 | 17.8 | 28.0 | 40.0 |
| S18A06I01 | 25.9 | 63.1 | 64.5 | 67.3 | 62.2 | 71.5 | 69.0 | 54.4 | 61.9 | 34.1 | 65.9 | 66.7 | 64.5 |
| S30A04I01 | 16.7 | 35.9 | 36.5 | 45.8 | 35.2 | 44.3 | 45.9 | 32.4 | 55.3 | 13.9 | 47.0 | 30.6 | 41.0 |
| S30A05I01 | 50.1 | 49.4 | 49.4 | 60.6 | 49.8 | 50.0 | 59.1 | 43.7 | 38.7 | 60.5 | 47.4 | 43.5 | 60.1 |
| S30A06I01 | 15.4 | 51.4 | 56.0 | 45.3 | 43.4 | 46.2 | 44.9 | 30.5 | 58.2 | 43.2 | 42.7 | 46.7 | 50.9 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 72 | 27 | 449 |
| 41 | 27 | 425 |
| 51 | 27 | 385 |
| 74 | 27 | 381 |
| 53 | 27 | 358 |
| 61 | 27 | 266 |
| 55 | 27 | 252 |
| 74 | 46 | 229 |
| 67 | 27 | 229 |
| 27 | 71 | 227 |
| 51 | 52 | 224 |
| 27 | 46 | 219 |
| 46 | 27 | 216 |
| 14 | 27 | 207 |
| 71 | 27 | 193 |
