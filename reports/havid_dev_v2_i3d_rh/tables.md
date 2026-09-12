Development result on `val` (18 videos, 6 participants, 17793 sampled frames); features `i3d_official` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 29.2 [23.8, 33.5] | 8.0 [6.8, 9.3] | 8.2 [5.1, 11.2] | 0.7 [0.0, 2.3] | 0.0 [0.0, 0.0] |
| fusion_causal | 39.5 [32.9, 48.5] | 21.0 [17.7, 24.2] | 23.0 [18.7, 29.1] | 18.7 [15.4, 23.3] | 10.3 [7.3, 13.3] |
| fusion_offline | 44.5 [35.6, 56.0] | 39.7 [33.6, 45.3] | 39.6 [30.8, 50.2] | 32.0 [24.6, 40.9] | 23.5 [16.5, 32.1] |
| view0_causal | 32.6 [25.1, 42.3] | 16.2 [12.0, 20.6] | 15.1 [10.8, 21.1] | 13.2 [9.6, 18.5] | 7.9 [5.5, 11.5] |
| view0_offline | 37.9 [29.4, 48.2] | 34.1 [27.6, 42.1] | 34.4 [27.2, 43.4] | 28.5 [21.0, 38.0] | 19.1 [12.9, 26.7] |
| view1_causal | 35.0 [28.6, 41.0] | 30.5 [27.6, 33.3] | 27.2 [21.8, 33.2] | 20.9 [16.9, 25.1] | 11.7 [7.5, 16.4] |
| view1_offline | 36.2 [26.7, 48.3] | 38.1 [29.4, 47.0] | 36.1 [27.3, 47.8] | 29.4 [21.7, 38.8] | 17.7 [10.1, 26.7] |
| view2_causal | 37.0 [30.3, 44.4] | 26.3 [20.9, 31.7] | 24.4 [18.5, 30.4] | 20.3 [15.4, 25.1] | 10.0 [7.3, 12.5] |
| view2_offline | 40.8 [32.8, 51.0] | 41.5 [32.9, 49.9] | 38.4 [30.4, 48.5] | 34.3 [25.1, 44.8] | 23.9 [15.7, 34.1] |

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
| S01A04I01 | 21.5 | 23.8 | 18.8 | 22.6 | 25.0 | 20.9 | 11.8 | 12.5 | 15.4 |
| S01A05I01 | 55.3 | 51.2 | 40.5 | 45.2 | 38.8 | 63.4 | 27.9 | 29.4 | 34.7 |
| S01A06I01 | 19.0 | 30.1 | 36.4 | 16.2 | 13.9 | 26.2 | 32.2 | 33.6 | 42.9 |
| S08A04I01 | 34.2 | 60.7 | 62.9 | 62.2 | 56.2 | 41.6 | 57.6 | 41.7 | 58.6 |
| S08A05I01 | 47.2 | 46.3 | 51.5 | 28.0 | 33.1 | 44.2 | 45.0 | 40.3 | 48.5 |
| S08A06I01 | 36.1 | 66.6 | 74.3 | 62.7 | 79.4 | 56.2 | 67.5 | 50.5 | 68.8 |
| S10A04I01 | 40.5 | 50.4 | 47.2 | 29.5 | 38.6 | 46.3 | 39.3 | 42.6 | 36.6 |
| S10A05I01 | 26.2 | 32.2 | 22.4 | 42.9 | 30.1 | 27.8 | 20.3 | 26.5 | 41.8 |
| S10A06I01 | 27.4 | 23.5 | 29.5 | 18.5 | 33.8 | 28.0 | 25.9 | 29.9 | 19.3 |
| S12A04I01 | 16.2 | 12.4 | 25.8 | 11.3 | 15.7 | 11.5 | 12.1 | 28.8 | 31.3 |
| S12A05I01 | 22.5 | 49.7 | 68.7 | 42.9 | 46.1 | 31.3 | 43.5 | 53.2 | 71.3 |
| S12A06I01 | 27.2 | 24.5 | 30.2 | 19.2 | 28.6 | 25.9 | 25.0 | 23.6 | 22.2 |
| S18A04I01 | 9.7 | 19.4 | 28.6 | 9.8 | 24.7 | 12.1 | 19.4 | 31.2 | 36.7 |
| S18A05I01 | 14.7 | 14.3 | 15.8 | 9.9 | 13.0 | 15.2 | 11.9 | 25.5 | 9.2 |
| S18A06I01 | 24.7 | 56.9 | 81.6 | 35.0 | 39.1 | 35.7 | 55.7 | 70.6 | 78.4 |
| S30A04I01 | 15.5 | 26.7 | 59.8 | 24.5 | 52.2 | 18.7 | 52.4 | 44.6 | 63.0 |
| S30A05I01 | 49.5 | 52.2 | 63.5 | 49.3 | 40.8 | 39.7 | 58.3 | 46.9 | 40.8 |
| S30A06I01 | 14.4 | 59.4 | 57.9 | 52.4 | 61.2 | 63.3 | 53.5 | 58.6 | 51.3 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 51 | 27 | 430 |
| 41 | 27 | 328 |
| 58 | 27 | 282 |
| 71 | 27 | 259 |
| 1 | 27 | 251 |
| 67 | 27 | 239 |
| 46 | 27 | 224 |
| 74 | 27 | 221 |
| 61 | 27 | 214 |
| 74 | 54 | 188 |
| 47 | 27 | 181 |
| 39 | 27 | 177 |
| 13 | 27 | 172 |
| 74 | 46 | 159 |
| 14 | 27 | 159 |
