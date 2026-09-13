Streaming benchmark, consumer `dinov2`, policy `drop_oldest`, buffer 16 per stream, batch ≤ 16 (wait ≤ 20.0 ms), 20.0 s per point after 5.0 s warm-up; machine: {'platform': 'Windows-11-10.0.26200-SP0', 'python': '3.12.13', 'torch': '2.13.0+cu130', 'gpu': 'NVIDIA GeForce RTX 4090'}.

| speed | streams | offered fps | consumed fps | dropped | p50 latency ms | p95 latency ms | mean batch | consumer busy |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1 | 15 | 15.1 | 0.0 % | 39 | 71 | 1.0 | 27 % |
| 1.0 | 3 | 45 | 45.1 | 0.0 % | 37 | 69 | 2.6 | 33 % |
| 1.0 | 6 | 90 | 90.3 | 0.0 % | 38 | 62 | 4.7 | 35 % |
| 1.0 | 12 | 180 | 180.7 | 0.0 % | 44 | 70 | 9.4 | 42 % |
| 1.0 | 18 | 270 | 270.0 | 0.0 % | 39 | 68 | 11.4 | 44 % |
| 1.0 | 24 | 360 | 360.6 | 0.0 % | 35 | 71 | 13.7 | 46 % |
| 1.0 | 36 | 540 | 468.7 | 0.0 % | 41 | 63 | 16.0 | 53 % |
