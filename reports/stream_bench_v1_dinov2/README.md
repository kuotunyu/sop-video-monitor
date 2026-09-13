# stream_bench_v1 — how many 15 fps cameras one RTX 4090 keeps up with (systems measurement)

This README covers this directory (decode + DINOv2) and its twin
[`../stream_bench_v1_decode/`](../stream_bench_v1_decode/tables.md) (decode only).

- Tier: **systems measurement** on one machine at one moment; not a model result. Val videos only
  (the command refuses the test split).
- Date: 2026-09-14, 06:49–06:55. Author: kuotunyu. Machine: Windows 11, RTX 4090, 24 logical CPUs,
  Python 3.12, torch 2.13 cu130; no other job running.
- Question: with the streaming pipeline of `src/sop_monitor/stream/`, how many concurrent camera
  streams at the HA-ViD rate (15 fps) can be decoded and embedded by the frame branch of the online
  recogniser before frames are dropped or sources fall behind?

## 1. Setup

`sop-monitor stream-bench`: N sources replay the 54 val videos round robin, each paced to 15 fps on
its own decode thread (PyAV, rescaled to 392 × 224), into a ring buffer of 16 slots per stream with
`DROP_OLDEST`; one consumer collects micro-batches (≤ 16 frames, wait ≤ 20 ms). Consumers: `decode`
(no model) and `dinov2` (DINOv2 ViT-B/14 on the GPU, fp16, synchronised per batch). 5 s warm-up, then
20 s measured per point. Latency = time from a frame's decode to the end of the batch that consumed
it, measured with `time.perf_counter`.

**Clock bug found and fixed before these numbers.** The first run of this benchmark used
`time.monotonic`, which on Windows advances in 15.6 ms steps; every latency came out as a multiple
of 15.6 ms (16, 31, 47, 78 ms). The pipeline now uses `time.perf_counter` (commit `73b14ad`, with a
test); the flawed outputs were discarded, never committed.

## 2. Results (`tables.md` of each directory is authoritative)

| streams | offered fps | decode only: consumed fps / p95 ms | decode + DINOv2: consumed fps / p95 ms / GPU consumer busy |
|---|---|---|---|
| 1 | 15 | 15.0 / 49 | 15.1 / 71 / 27 % |
| 3 | 45 | 45.0 / 49 | 45.1 / 69 / 33 % |
| 6 | 90 | 90.0 / 48 | 90.3 / 62 / 35 % |
| 12 | 180 | 180.3 / 46 | 180.7 / 70 / 42 % |
| 18 | 270 | — | 270.0 / 68 / 44 % |
| 24 | 360 | 360.4 / 35 | 360.6 / 71 / 46 % |
| 36 | 540 | 540.2 / 34 | **468.7** / 63 / 53 % |
| 48 | 720 | 720.3 / 36 | — |

No frame was dropped from a ring buffer at any point (0.0 %), and the watchdog reported no stall.

Reading:

- **24 cameras at 15 fps (360 frames per second) are decoded and embedded in real time** with a p95
  latency of 71 ms and the GPU consumer busy 46 % of the time.
- **At 36 cameras the sources fall behind**: each decode thread produced 322 instead of about 375
  frames in the 25 s (86 % of the schedule), so only 469 of 540 frames per second arrive. The ring
  buffers never fill (mean occupancy 0.9 %) and the GPU consumer is busy only 53 % of the time, so
  the limit is on the producing side; the likely cause, not isolated here, is 36 decode threads
  competing with the consumer's Python thread for the CPU and the interpreter lock. The latency column does not show
  this lag, because latency is counted from decode; a frame that is decoded late looks fresh.
- **Decoding alone scales to 48 streams (720 fps)** without falling behind, so the saturation at 36
  comes from decoding and embedding together.
- HA-ViD needs three cameras per station: by these numbers one RTX 4090 could embed about eight
  stations' worth of video. The recogniser heads and the monitor are not in this benchmark; their
  streamed cost for one station is in `havid_dev_v11_online_head_features/` (62–134 ms per second of
  video, growing with the recording's length).

## 3. Reproduce

```bash
export PYTHONUTF8=1
make stream-bench-decode stream-bench-dinov2   # about 3 min each; needs an otherwise idle machine
```

## 4. Not done here

- Schedule lag (decode time minus nominal capture time) as its own column; points between 24 and 36
  streams; other drop policies.
- RTSP ingestion (mediamtx), hardware video decoding, a second GPU, or the heads inside the
  concurrent loop.
