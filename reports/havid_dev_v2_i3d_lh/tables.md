Development result on `val` (18 videos, 6 participants, 17793 sampled frames); features `i3d_official` stride 1; 95% CI = participant bootstrap (2000 draws, seed 0).

| run | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| majority | 36.1 [26.7, 44.8] | 8.3 [6.7, 10.0] | 9.4 [4.8, 13.9] | 3.6 [0.6, 7.5] | 0.7 [0.0, 2.5] |
| fusion_causal | 44.1 [40.8, 47.9] | 28.0 [25.9, 31.3] | 28.7 [23.1, 34.9] | 21.6 [16.2, 27.1] | 12.8 [9.9, 15.8] |
| fusion_offline | 46.9 [43.4, 51.1] | 40.6 [33.0, 48.1] | 41.8 [32.4, 52.3] | 35.1 [25.2, 46.2] | 22.5 [12.7, 31.8] |
| view0_causal | 36.9 [31.5, 42.4] | 26.6 [22.3, 31.1] | 23.0 [17.7, 29.0] | 19.3 [14.0, 25.2] | 10.5 [8.1, 13.2] |
| view0_offline | 38.0 [33.8, 43.4] | 32.7 [27.2, 38.6] | 31.3 [24.2, 39.8] | 26.2 [19.4, 35.2] | 15.3 [9.6, 22.8] |
| view1_causal | 35.8 [29.7, 41.7] | 30.1 [25.7, 34.6] | 26.5 [21.4, 31.9] | 19.8 [14.7, 24.2] | 10.5 [8.0, 13.0] |
| view1_offline | 38.4 [31.9, 46.1] | 35.7 [28.9, 42.4] | 34.8 [27.3, 43.0] | 29.5 [22.1, 37.5] | 19.1 [12.2, 26.5] |
| view2_causal | 36.8 [34.2, 39.6] | 22.0 [17.5, 26.8] | 22.6 [16.8, 30.5] | 17.2 [12.0, 23.6] | 9.1 [5.9, 12.8] |
| view2_offline | 45.2 [41.5, 50.6] | 39.8 [30.8, 49.8] | 43.9 [34.2, 54.1] | 36.0 [25.7, 46.9] | 20.9 [11.6, 30.4] |

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
| S01A04I01 | 40.8 | 40.8 | 41.3 | 40.3 | 40.8 | 22.4 | 16.6 | 25.1 | 30.8 |
| S01A05I01 | 46.9 | 47.6 | 50.9 | 46.9 | 47.6 | 37.2 | 31.9 | 35.0 | 39.0 |
| S01A06I01 | 63.3 | 52.1 | 43.7 | 40.0 | 9.3 | 26.4 | 35.7 | 53.0 | 68.7 |
| S08A04I01 | 17.4 | 55.8 | 57.5 | 40.9 | 52.3 | 30.3 | 34.1 | 51.6 | 63.3 |
| S08A05I01 | 24.4 | 29.6 | 38.9 | 29.6 | 35.0 | 32.9 | 35.1 | 21.1 | 37.4 |
| S08A06I01 | 22.8 | 40.7 | 59.8 | 37.3 | 42.9 | 43.4 | 58.0 | 39.1 | 62.4 |
| S10A04I01 | 33.3 | 42.8 | 61.0 | 35.1 | 33.0 | 44.7 | 55.0 | 33.9 | 56.2 |
| S10A05I01 | 34.7 | 43.9 | 20.8 | 47.4 | 23.3 | 59.2 | 20.0 | 33.2 | 29.0 |
| S10A06I01 | 25.5 | 30.7 | 31.7 | 16.6 | 30.5 | 20.5 | 27.7 | 29.0 | 28.1 |
| S12A04I01 | 61.9 | 63.3 | 61.9 | 36.9 | 28.2 | 60.8 | 58.9 | 51.5 | 60.2 |
| S12A05I01 | 70.3 | 60.6 | 48.8 | 57.4 | 53.2 | 60.4 | 40.9 | 45.1 | 52.0 |
| S12A06I01 | 28.2 | 26.2 | 26.0 | 22.3 | 26.4 | 18.2 | 22.8 | 24.9 | 21.9 |
| S18A04I01 | 13.5 | 41.2 | 31.2 | 13.0 | 27.7 | 17.7 | 17.8 | 46.6 | 29.9 |
| S18A05I01 | 31.2 | 13.2 | 47.9 | 7.5 | 17.4 | 11.6 | 55.8 | 12.7 | 31.2 |
| S18A06I01 | 21.7 | 61.1 | 82.4 | 54.8 | 73.5 | 31.7 | 42.4 | 57.2 | 81.3 |
| S30A04I01 | 47.8 | 54.4 | 66.1 | 58.7 | 58.2 | 51.5 | 79.8 | 36.0 | 60.1 |
| S30A05I01 | 40.4 | 46.6 | 43.2 | 44.2 | 37.3 | 40.7 | 45.6 | 40.4 | 44.3 |
| S30A06I01 | 29.3 | 55.3 | 49.6 | 43.9 | 48.8 | 53.7 | 47.1 | 42.8 | 41.1 |

Most frequent causal-run confusions (gt -> pred, frames; -1 = background):

| gt | pred | frames |
|---|---|---|
| 74 | 27 | 387 |
| 41 | 27 | 336 |
| 49 | 27 | 321 |
| 1 | 27 | 263 |
| 71 | 27 | 222 |
| 24 | 27 | 210 |
| 39 | 27 | 205 |
| 70 | 27 | 204 |
| 62 | 27 | 204 |
| 27 | 73 | 201 |
| 46 | 27 | 183 |
| 64 | 27 | 178 |
| 6 | 27 | 174 |
| 12 | 27 | 156 |
| 68 | 27 | 149 |
