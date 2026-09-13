Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [27.3, 45.4] | 8.3 [6.7, 10.0] | 9.4 [4.8, 13.9] | 3.6 [0.6, 7.5] | 0.7 [0.0, 2.5] |
| fusion_causal | 39.4 [33.8, 45.2] | 35.4 [30.7, 39.8] | 32.0 [27.7, 37.8] | 22.5 [17.3, 27.9] | 14.1 [10.0, 18.6] |
| fusion_conf_causal | 38.5 [32.8, 44.2] | 35.8 [31.0, 41.4] | 30.0 [24.3, 36.3] | 21.5 [17.0, 26.1] | 11.6 [6.6, 15.7] |
| fusion_conf_offline | 42.2 [36.1, 49.5] | 42.0 [33.6, 49.4] | 43.1 [32.7, 52.0] | 36.0 [26.9, 45.7] | 23.9 [13.5, 34.4] |
| fusion_geo_causal | 38.3 [33.7, 43.0] | 33.3 [25.0, 41.4] | 32.2 [25.5, 39.7] | 22.9 [17.0, 29.4] | 13.3 [6.2, 20.0] |
| fusion_geo_offline | 42.2 [34.8, 50.4] | 42.1 [35.3, 48.9] | 43.1 [32.0, 53.6] | 36.2 [24.8, 47.6] | 25.4 [12.8, 37.1] |
| fusion_offline | 42.1 [35.4, 49.8] | 42.3 [34.7, 48.9] | 43.5 [33.0, 52.4] | 35.8 [25.9, 45.4] | 25.1 [14.4, 35.7] |
| view0_causal | 30.3 [26.6, 34.6] | 32.3 [27.2, 38.4] | 26.2 [19.3, 32.8] | 18.5 [14.1, 23.1] | 9.7 [6.8, 12.4] |
| view0_offline | 38.7 [32.0, 46.6] | 45.8 [40.7, 50.5] | 42.2 [33.9, 52.2] | 35.4 [26.0, 46.5] | 22.1 [14.8, 29.9] |
| view1_causal | 31.4 [27.1, 36.6] | 33.4 [26.1, 39.1] | 30.1 [24.3, 36.4] | 22.5 [17.6, 27.8] | 12.4 [7.2, 17.3] |
| view1_offline | 34.5 [27.9, 43.2] | 38.9 [33.3, 44.1] | 41.2 [33.3, 49.9] | 35.9 [27.1, 44.8] | 20.2 [11.7, 29.0] |
| view2_causal | 33.1 [29.4, 37.0] | 33.2 [26.9, 38.6] | 27.8 [22.4, 34.5] | 20.3 [13.8, 28.6] | 9.5 [7.4, 12.5] |
| view2_offline | 39.8 [33.5, 48.6] | 40.3 [33.8, 45.7] | 40.9 [32.6, 51.2] | 32.6 [21.8, 45.8] | 24.6 [15.4, 35.9] |

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
| S01A04I01 | 41.2 | 41.2 | 41.2 | 24.7 | 41.2 | 25.3 | 26.3 | 21.6 | 30.7 | 20.6 | 9.8 | 41.2 | 24.8 |
| S01A05I01 | 47.4 | 54.0 | 47.9 | 33.2 | 47.9 | 26.3 | 29.5 | 39.6 | 30.8 | 15.6 | 25.1 | 57.0 | 27.8 |
| S01A06I01 | 63.7 | 46.9 | 44.7 | 37.1 | 49.6 | 35.1 | 34.5 | 17.0 | 13.8 | 63.7 | 41.2 | 17.4 | 41.3 |
| S08A04I01 | 18.1 | 52.8 | 52.4 | 67.9 | 54.9 | 71.4 | 69.6 | 39.0 | 51.7 | 50.2 | 64.5 | 47.1 | 71.1 |
| S08A05I01 | 25.4 | 33.4 | 33.8 | 38.4 | 32.3 | 38.5 | 38.2 | 29.2 | 43.4 | 26.8 | 27.3 | 35.0 | 34.8 |
| S08A06I01 | 23.6 | 32.4 | 31.6 | 57.3 | 31.9 | 59.2 | 58.9 | 30.1 | 64.2 | 33.3 | 48.5 | 29.7 | 63.3 |
| S10A04I01 | 33.7 | 29.8 | 28.1 | 49.1 | 34.3 | 47.0 | 49.6 | 23.7 | 44.9 | 21.4 | 32.0 | 34.2 | 40.8 |
| S10A05I01 | 35.0 | 40.9 | 40.4 | 45.9 | 40.1 | 46.1 | 45.6 | 43.9 | 40.9 | 37.3 | 40.7 | 41.9 | 44.7 |
| S10A06I01 | 26.0 | 25.9 | 25.0 | 23.9 | 25.5 | 26.8 | 24.1 | 18.4 | 18.7 | 29.6 | 26.5 | 17.5 | 19.5 |
| S12A04I01 | 62.5 | 62.5 | 61.7 | 54.1 | 62.5 | 55.0 | 54.1 | 62.5 | 43.3 | 22.0 | 32.6 | 54.1 | 49.9 |
| S12A05I01 | 70.6 | 45.6 | 48.1 | 58.0 | 40.9 | 55.9 | 56.4 | 49.4 | 72.2 | 29.3 | 29.9 | 36.1 | 49.6 |
| S12A06I01 | 28.7 | 18.7 | 19.6 | 25.9 | 17.9 | 28.1 | 27.3 | 13.5 | 20.3 | 19.6 | 23.6 | 15.9 | 23.0 |
| S18A04I01 | 14.7 | 9.5 | 9.5 | 30.2 | 9.5 | 30.5 | 30.7 | 8.2 | 14.6 | 18.2 | 32.0 | 9.3 | 28.7 |
| S18A05I01 | 31.6 | 39.4 | 31.2 | 20.2 | 31.6 | 16.1 | 18.3 | 12.5 | 17.6 | 31.6 | 21.1 | 30.1 | 22.6 |
| S18A06I01 | 22.9 | 54.9 | 54.4 | 90.4 | 54.0 | 89.5 | 89.8 | 56.7 | 91.3 | 48.2 | 70.7 | 40.6 | 84.8 |
| S30A04I01 | 48.5 | 39.8 | 39.3 | 55.2 | 39.9 | 55.7 | 55.9 | 36.1 | 26.6 | 42.5 | 53.0 | 28.3 | 55.6 |
| S30A05I01 | 41.1 | 58.6 | 56.9 | 46.7 | 54.8 | 46.7 | 45.3 | 35.5 | 36.6 | 49.3 | 49.3 | 53.1 | 40.2 |
| S30A06I01 | 30.1 | 49.1 | 50.0 | 31.7 | 39.1 | 32.9 | 30.4 | 40.1 | 51.7 | 32.6 | 29.1 | 18.8 | 29.4 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 55 | 27 | 421 |
| 27 | 71 | 400 |
| 74 | 27 | 393 |
| 24 | 27 | 377 |
| 41 | 27 | 335 |
| 58 | 27 | 323 |
| 1 | 27 | 315 |
| 68 | 27 | 269 |
| 46 | 27 | 263 |
| 64 | 27 | 245 |
| 71 | 27 | 237 |
| 12 | 27 | 230 |
| 27 | 73 | 230 |
| 73 | 27 | 228 |
| 49 | 27 | 194 |
