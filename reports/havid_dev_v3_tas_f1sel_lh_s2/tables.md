Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 2).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [26.9, 45.4] | 8.3 [6.6, 10.1] | 9.4 [4.8, 14.0] | 3.6 [0.6, 7.9] | 0.7 [0.0, 2.6] |
| fusion_causal | 40.5 [34.2, 45.9] | 31.4 [27.1, 36.1] | 29.8 [22.0, 37.0] | 22.7 [16.8, 27.9] | 12.9 [8.4, 17.1] |
| fusion_conf_causal | 39.2 [33.5, 44.7] | 32.8 [28.2, 36.0] | 28.2 [21.3, 34.0] | 22.5 [16.6, 27.1] | 11.7 [6.8, 15.8] |
| fusion_conf_offline | 42.6 [36.8, 50.8] | 38.9 [34.4, 43.2] | 41.5 [33.0, 50.7] | 35.2 [24.7, 48.1] | 25.6 [15.3, 37.3] |
| fusion_geo_causal | 41.8 [34.7, 48.5] | 28.7 [23.2, 33.3] | 32.5 [25.1, 37.8] | 23.2 [18.4, 26.8] | 11.8 [8.1, 15.9] |
| fusion_geo_offline | 44.2 [38.9, 51.6] | 39.2 [33.3, 45.3] | 41.8 [32.2, 52.0] | 35.8 [24.2, 48.4] | 27.5 [17.9, 39.0] |
| fusion_offline | 43.3 [37.9, 50.8] | 40.2 [35.0, 45.3] | 42.2 [33.2, 51.1] | 36.2 [26.5, 47.6] | 26.7 [16.1, 38.4] |
| view0_causal | 34.2 [29.9, 39.5] | 27.1 [22.3, 31.0] | 25.6 [19.1, 31.6] | 18.8 [12.9, 23.6] | 10.2 [6.8, 13.2] |
| view0_offline | 34.6 [28.2, 42.9] | 40.1 [34.9, 44.5] | 38.4 [32.1, 45.1] | 31.6 [25.9, 38.2] | 21.8 [16.2, 28.6] |
| view1_causal | 36.1 [32.4, 39.8] | 33.4 [28.5, 37.2] | 30.7 [25.2, 35.2] | 21.5 [17.5, 26.7] | 11.8 [9.5, 14.6] |
| view1_offline | 32.9 [27.4, 41.6] | 37.8 [31.2, 44.6] | 36.5 [26.9, 47.5] | 28.4 [18.4, 40.8] | 20.0 [11.7, 29.3] |
| view2_causal | 38.5 [33.9, 43.2] | 31.1 [23.5, 38.4] | 30.8 [24.7, 36.6] | 21.3 [16.9, 25.4] | 10.3 [8.8, 11.8] |
| view2_offline | 44.0 [37.9, 50.6] | 39.6 [31.1, 47.9] | 43.4 [32.8, 55.6] | 37.6 [26.0, 52.1] | 26.9 [16.7, 39.8] |

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
| S01A04I01 | 41.2 | 46.4 | 42.6 | 33.8 | 43.3 | 36.1 | 33.9 | 34.1 | 26.4 | 35.5 | 19.2 | 41.2 | 33.6 |
| S01A05I01 | 47.4 | 30.9 | 30.5 | 37.5 | 38.3 | 41.1 | 40.6 | 29.0 | 35.8 | 37.4 | 25.7 | 36.9 | 49.9 |
| S01A06I01 | 63.7 | 57.7 | 48.7 | 48.1 | 62.3 | 52.8 | 50.6 | 16.2 | 5.5 | 63.7 | 48.2 | 58.2 | 53.1 |
| S08A04I01 | 18.1 | 48.2 | 49.2 | 69.7 | 37.9 | 66.3 | 67.0 | 42.1 | 49.3 | 46.6 | 70.6 | 41.4 | 64.1 |
| S08A05I01 | 25.4 | 29.4 | 29.1 | 39.8 | 29.8 | 38.8 | 39.3 | 29.2 | 30.0 | 30.7 | 23.0 | 26.2 | 40.3 |
| S08A06I01 | 23.6 | 41.7 | 40.9 | 59.5 | 41.3 | 61.5 | 60.3 | 39.1 | 43.5 | 34.6 | 47.4 | 26.3 | 59.7 |
| S10A04I01 | 33.7 | 33.7 | 33.9 | 32.4 | 34.5 | 40.8 | 34.8 | 33.5 | 31.5 | 46.1 | 26.3 | 37.3 | 34.0 |
| S10A05I01 | 35.0 | 37.2 | 36.4 | 42.1 | 35.7 | 37.5 | 39.7 | 37.0 | 44.6 | 37.5 | 32.7 | 60.9 | 36.4 |
| S10A06I01 | 26.0 | 23.0 | 21.8 | 27.9 | 23.6 | 27.5 | 30.8 | 21.9 | 23.9 | 22.2 | 23.0 | 16.3 | 30.8 |
| S12A04I01 | 62.5 | 62.2 | 61.9 | 59.1 | 62.5 | 52.6 | 57.4 | 60.6 | 31.3 | 62.5 | 20.3 | 62.5 | 61.4 |
| S12A05I01 | 70.6 | 69.1 | 67.8 | 46.8 | 71.5 | 49.1 | 45.6 | 63.5 | 33.2 | 39.4 | 43.9 | 57.7 | 49.1 |
| S12A06I01 | 28.7 | 26.2 | 24.7 | 29.6 | 31.6 | 29.7 | 29.5 | 18.8 | 23.0 | 22.3 | 15.5 | 27.8 | 30.4 |
| S18A04I01 | 14.7 | 9.7 | 8.7 | 26.1 | 10.2 | 42.4 | 26.4 | 7.1 | 31.8 | 17.6 | 31.3 | 17.3 | 26.3 |
| S18A05I01 | 31.6 | 15.7 | 12.3 | 24.5 | 31.6 | 15.5 | 24.5 | 8.6 | 11.6 | 13.1 | 17.8 | 26.7 | 24.1 |
| S18A06I01 | 22.9 | 71.9 | 71.5 | 85.6 | 71.2 | 85.0 | 85.3 | 71.8 | 79.1 | 55.2 | 72.4 | 58.0 | 87.8 |
| S30A04I01 | 48.5 | 41.0 | 41.0 | 54.4 | 52.0 | 55.5 | 54.6 | 33.9 | 56.4 | 27.0 | 37.6 | 62.2 | 52.5 |
| S30A05I01 | 41.1 | 44.0 | 44.5 | 49.3 | 44.1 | 44.9 | 49.4 | 39.4 | 42.5 | 32.8 | 37.3 | 39.2 | 52.2 |
| S30A06I01 | 30.1 | 52.6 | 52.5 | 35.0 | 53.5 | 41.8 | 38.3 | 42.7 | 49.2 | 28.8 | 20.9 | 27.3 | 37.1 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 55 | 27 | 387 |
| 49 | 27 | 380 |
| 41 | 27 | 361 |
| 74 | 27 | 326 |
| 1 | 27 | 308 |
| 58 | 27 | 250 |
| 64 | 27 | 245 |
| 71 | 27 | 236 |
| 27 | 71 | 233 |
| 73 | 27 | 228 |
| 51 | 27 | 214 |
| 65 | 27 | 208 |
| 24 | 27 | 205 |
| 6 | 27 | 200 |
| 24 | 55 | 197 |
