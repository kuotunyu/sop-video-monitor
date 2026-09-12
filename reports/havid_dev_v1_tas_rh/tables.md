Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.8 [24.6, 34.1] | 8.0 [6.8, 9.3] | 8.9 [5.9, 11.8] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 38.6 [32.3, 47.2] | 33.5 [26.8, 39.0] | 30.3 [24.5, 36.9] | 23.8 [17.7, 30.8] | 13.6 [9.4, 18.7] |
| fusion_offline | 41.8 [33.6, 52.5] | 40.9 [32.2, 48.7] | 41.3 [34.0, 49.9] | 35.0 [26.9, 44.5] | 23.4 [17.1, 32.2] |
| view0_causal | 35.1 [28.9, 43.3] | 27.7 [22.0, 33.2] | 24.2 [17.6, 31.4] | 18.9 [12.6, 26.0] | 10.7 [6.6, 15.4] |
| view0_offline | 36.7 [29.1, 46.8] | 36.9 [28.3, 46.1] | 36.9 [29.2, 47.6] | 30.2 [21.1, 40.8] | 19.7 [12.4, 28.3] |
| view1_causal | 33.2 [25.4, 43.2] | 29.0 [25.1, 33.1] | 26.6 [20.5, 34.3] | 23.8 [17.7, 31.4] | 13.1 [8.1, 19.8] |
| view1_offline | 36.7 [29.6, 45.8] | 38.5 [33.3, 43.7] | 37.4 [31.8, 44.0] | 32.5 [26.1, 39.2] | 22.8 [18.2, 29.1] |
| view2_causal | 35.0 [29.2, 44.5] | 27.9 [24.8, 31.0] | 25.5 [21.7, 31.1] | 19.9 [15.4, 26.7] | 9.6 [6.0, 14.2] |
| view2_offline | 38.5 [32.0, 47.1] | 41.9 [36.6, 48.0] | 40.6 [35.8, 47.3] | 34.0 [27.3, 41.4] | 22.4 [16.6, 30.3] |

| run | what it uses |
|---|---|
| majority | constant most frequent training label |
| fusion_causal | mean of the per-view causal posteriors (late fusion, online) |
| fusion_offline | mean of the per-view offline posteriors (late fusion) |
| view0_causal | MS-TCN++ on the side view (view 0), left-only padding: frame t sees frames <= t (online) |
| view0_offline | MS-TCN++ on the side view (view 0), symmetric padding: sees future frames (not an online result) |
| view1_causal | MS-TCN++ on the front view (view 1), left-only padding: frame t sees frames <= t (online) |
| view1_offline | MS-TCN++ on the front view (view 1), symmetric padding: sees future frames (not an online result) |
| view2_causal | MS-TCN++ on the top view (view 2), left-only padding: frame t sees frames <= t (online) |
| view2_offline | MS-TCN++ on the top view (view 2), symmetric padding: sees future frames (not an online result) |

Per-video MoF:

| video | majority | fusion_causal | fusion_offline | view0_causal | view0_offline | view1_causal | view1_offline | view2_causal | view2_offline |
|---|---|---|---|---|---|---|---|---|---|
| S01A04I01 | 22.0 | 21.9 | 17.7 | 16.9 | 25.4 | 24.2 | 12.3 | 25.6 | 22.1 |
| S01A05I01 | 55.7 | 55.0 | 51.9 | 49.2 | 42.5 | 21.7 | 46.3 | 36.5 | 33.7 |
| S01A06I01 | 19.4 | 19.0 | 28.3 | 18.6 | 19.1 | 17.2 | 22.0 | 16.2 | 31.0 |
| S08A04I01 | 34.3 | 50.4 | 46.3 | 49.3 | 51.2 | 39.5 | 31.2 | 47.2 | 44.3 |
| S08A05I01 | 47.9 | 51.2 | 49.8 | 45.3 | 41.9 | 37.9 | 52.0 | 50.6 | 42.5 |
| S08A06I01 | 36.7 | 67.0 | 86.2 | 63.4 | 53.0 | 64.2 | 77.2 | 69.9 | 60.5 |
| S10A04I01 | 40.8 | 39.1 | 35.5 | 33.7 | 33.7 | 35.1 | 33.5 | 30.3 | 27.4 |
| S10A05I01 | 27.3 | 29.5 | 27.4 | 29.6 | 18.7 | 21.0 | 12.7 | 31.6 | 47.3 |
| S10A06I01 | 27.9 | 31.5 | 29.0 | 30.4 | 24.1 | 19.4 | 32.6 | 28.7 | 23.6 |
| S12A04I01 | 17.5 | 17.1 | 17.5 | 14.9 | 32.4 | 14.3 | 16.1 | 17.1 | 17.1 |
| S12A05I01 | 23.2 | 50.3 | 66.7 | 47.4 | 44.0 | 51.1 | 64.5 | 47.8 | 66.7 |
| S12A06I01 | 27.8 | 21.9 | 26.1 | 23.9 | 21.9 | 10.7 | 19.9 | 21.3 | 24.7 |
| S18A04I01 | 10.9 | 18.6 | 19.0 | 5.2 | 11.2 | 23.4 | 18.5 | 18.9 | 25.3 |
| S18A05I01 | 15.5 | 23.4 | 26.2 | 14.4 | 15.5 | 11.0 | 9.7 | 23.4 | 52.3 |
| S18A06I01 | 25.9 | 68.1 | 89.0 | 64.2 | 67.4 | 67.4 | 84.0 | 66.5 | 84.2 |
| S30A04I01 | 16.7 | 39.2 | 43.9 | 36.9 | 49.5 | 41.4 | 52.5 | 34.7 | 45.6 |
| S30A05I01 | 50.1 | 60.3 | 56.0 | 48.8 | 51.0 | 58.4 | 32.1 | 45.9 | 60.8 |
| S30A06I01 | 15.4 | 38.7 | 60.3 | 41.6 | 71.5 | 54.2 | 57.6 | 29.6 | 34.4 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 72 | 27 | 428 |
| 41 | 27 | 425 |
| 74 | 27 | 396 |
| 51 | 27 | 319 |
| 55 | 27 | 302 |
| 53 | 27 | 277 |
| 46 | 27 | 249 |
| 27 | 47 | 222 |
| 61 | 27 | 218 |
| 67 | 27 | 217 |
| 58 | 65 | 211 |
| 55 | 54 | 206 |
| 27 | 71 | 204 |
| 1 | 27 | 179 |
| 72 | 54 | 170 |
