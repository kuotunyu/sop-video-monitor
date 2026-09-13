Streaming benchmark, consumer `decode`, policy `drop_oldest`, buffer 16 per stream, batch ≤ 16 (wait ≤ 20.0 ms), 20.0 s per point after 5.0 s warm-up; machine: {'platform': 'Windows-11-10.0.26200-SP0', 'python': '3.12.13', 'torch': '2.13.0+cu130', 'gpu': 'NVIDIA GeForce RTX 4090'}.

| speed | streams | offered fps | consumed fps | dropped | p50 latency ms | p95 latency ms | mean batch | consumer busy |
|---|---|---|---|---|---|---|---|---|
| 1.0 | 1 | 15 | 15.0 | 0.0 % | 24 | 49 | 1.0 | 0 % |
| 1.0 | 3 | 45 | 45.0 | 0.0 % | 21 | 49 | 2.6 | 0 % |
| 1.0 | 6 | 90 | 90.0 | 0.0 % | 22 | 48 | 5.1 | 0 % |
| 1.0 | 12 | 180 | 180.3 | 0.0 % | 23 | 46 | 9.1 | 0 % |
| 1.0 | 24 | 360 | 360.4 | 0.0 % | 14 | 35 | 11.0 | 0 % |
| 1.0 | 36 | 540 | 540.2 | 0.0 % | 14 | 34 | 11.9 | 0 % |
| 1.0 | 48 | 720 | 720.3 | 0.0 % | 14 | 36 | 14.8 | 0 % |
