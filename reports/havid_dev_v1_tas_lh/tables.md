Development result on `val` (18 videos, 6 participants, 17973 sampled frames); features `dinov2_vitb14` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.6 [27.3, 45.4] | 8.3 [6.7, 10.0] | 9.4 [4.8, 13.9] | 3.6 [0.6, 7.5] | 0.7 [0.0, 2.5] |
| fusion_causal | 42.8 [38.8, 46.9] | 28.3 [20.9, 34.3] | 29.1 [23.2, 34.1] | 21.5 [14.0, 28.3] | 12.6 [8.5, 16.4] |
| fusion_offline | 45.0 [39.4, 51.0] | 44.1 [37.6, 50.9] | 43.5 [34.0, 53.0] | 36.8 [24.6, 48.5] | 25.3 [14.3, 36.6] |
| view0_causal | 33.5 [27.6, 40.6] | 29.0 [23.3, 34.0] | 25.1 [19.5, 31.1] | 14.8 [10.3, 20.4] | 5.8 [3.1, 9.1] |
| view0_offline | 38.7 [32.0, 46.6] | 45.8 [40.7, 50.5] | 42.2 [33.9, 52.2] | 35.4 [26.0, 46.5] | 22.1 [14.8, 29.9] |
| view1_causal | 33.1 [26.1, 40.6] | 18.9 [16.5, 21.6] | 20.2 [17.9, 22.5] | 11.6 [7.3, 14.8] | 5.5 [2.9, 8.5] |
| view1_offline | 39.4 [34.1, 44.8] | 40.7 [34.6, 47.1] | 36.5 [25.9, 49.2] | 31.6 [22.8, 42.7] | 20.3 [12.3, 30.1] |
| view2_causal | 41.6 [38.5, 44.8] | 27.1 [21.1, 31.8] | 27.2 [23.2, 30.2] | 20.5 [14.4, 25.7] | 12.3 [9.6, 14.3] |
| view2_offline | 40.8 [34.2, 49.1] | 41.5 [35.8, 47.6] | 39.9 [31.0, 50.5] | 36.1 [27.4, 46.7] | 23.4 [13.1, 35.9] |

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
| S01A04I01 | 41.2 | 41.2 | 36.3 | 30.5 | 30.7 | 41.2 | 33.1 | 41.2 | 31.3 |
| S01A05I01 | 47.4 | 50.9 | 31.8 | 47.4 | 30.8 | 25.9 | 36.3 | 51.1 | 28.0 |
| S01A06I01 | 63.7 | 34.1 | 73.8 | 21.4 | 13.8 | 18.9 | 63.7 | 33.7 | 73.8 |
| S08A04I01 | 18.1 | 59.5 | 57.3 | 21.8 | 51.7 | 18.1 | 48.5 | 59.7 | 65.9 |
| S08A05I01 | 25.4 | 25.1 | 37.4 | 25.7 | 43.4 | 17.7 | 26.5 | 24.3 | 36.4 |
| S08A06I01 | 23.6 | 47.3 | 66.9 | 25.8 | 64.2 | 27.4 | 68.9 | 45.0 | 64.1 |
| S10A04I01 | 33.7 | 40.0 | 41.5 | 33.8 | 44.9 | 32.2 | 40.1 | 41.5 | 32.1 |
| S10A05I01 | 35.0 | 50.5 | 36.6 | 33.6 | 40.9 | 63.1 | 27.4 | 51.4 | 31.0 |
| S10A06I01 | 26.0 | 25.9 | 26.8 | 14.4 | 18.7 | 25.5 | 20.5 | 24.8 | 26.9 |
| S12A04I01 | 62.5 | 62.5 | 54.6 | 62.5 | 43.3 | 60.3 | 39.8 | 62.5 | 53.5 |
| S12A05I01 | 70.6 | 67.6 | 70.5 | 62.3 | 72.2 | 70.6 | 63.7 | 63.0 | 36.4 |
| S12A06I01 | 28.7 | 26.8 | 22.6 | 21.9 | 20.3 | 27.3 | 21.5 | 25.7 | 24.8 |
| S18A04I01 | 14.7 | 16.6 | 32.8 | 21.0 | 14.6 | 13.7 | 22.6 | 17.0 | 31.3 |
| S18A05I01 | 31.6 | 26.9 | 29.7 | 27.3 | 17.6 | 31.6 | 20.0 | 23.9 | 29.2 |
| S18A06I01 | 22.9 | 66.0 | 90.5 | 49.5 | 91.3 | 22.9 | 77.5 | 66.2 | 75.3 |
| S30A04I01 | 48.5 | 43.7 | 48.8 | 61.6 | 26.6 | 46.3 | 46.9 | 40.7 | 54.4 |
| S30A05I01 | 41.1 | 54.8 | 49.4 | 30.2 | 36.6 | 34.6 | 42.7 | 51.6 | 47.1 |
| S30A06I01 | 30.1 | 45.3 | 34.7 | 41.8 | 51.7 | 29.2 | 24.2 | 38.0 | 27.2 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 41 | 27 | 361 |
| 1 | 27 | 332 |
| 58 | 27 | 295 |
| 68 | 27 | 277 |
| 74 | 27 | 262 |
| 71 | 27 | 252 |
| 46 | 27 | 246 |
| 64 | 27 | 245 |
| 51 | 27 | 232 |
| 24 | 27 | 209 |
| 49 | 27 | 206 |
| 55 | 27 | 191 |
| 27 | 71 | 184 |
| 27 | 54 | 170 |
| 39 | 27 | 169 |
