# havid_dev_v11_online_head — frames to deviations, streamed (development result)

This README covers this directory (cached features) and its twin
[`../havid_dev_v11_online_head_video/`](../havid_dev_v11_online_head_video/online_head.json)
(decoded video).

- Tier: **development result / systems measurement** on `splits/ha-vid/val.csv`; the frozen test
  subjects were not read for this run.
- Date: 2026-09-14, 06:35–06:42. Author: kuotunyu. Machine: Windows 11, RTX 4090, otherwise idle.
- Question: does the whole online path (camera frames → DINOv2 → per-view causal MS-TCN++ → fusion
  → online SOP monitor) run as a stream, give the same labels as the offline evaluation of the same
  networks, and keep up with three 15 fps cameras?

## 1. Setup

`sop-monitor havid-online-head` (`src/sop_monitor/online_head.py`) with the checkpointed final
networks at look-ahead 45 frames, seed 0 (`artifacts/checkpoints/havid_final_L45_{lh,rh}_s0`, gated
identical to `havid_dev_v9_tas_la45_*_s0`), mean fusion, instruction-sheet knowledge
(`sop/ha-vid/sheet`), confirmation length 8 frames, chunks of 15 frames (one second) per view. Every
push recomputes each causal network on the prefix seen so far (exact by causality). Recordings
are replayed as fast as the machine allows, not paced to 15 fps.

| mode | input per view | recordings |
|---|---|---|
| features (this directory) | the cached DINOv2 ViT-B/14 features | all 18 val recordings |
| video (`_video`) | the mp4 decoded with PyAV, rescaled to 392 × 224, embedded by DINOv2 on the GPU | S18A05I01 (31 s), S08A04I01 (75 s), S10A04I01 (113 s) |

Files: `online_head.json` (per recording: deviations with detection frame, per-chunk latency
summary, agreement with the checkpoint run; machine), `stream_labels_val.csv` (streamed labels per
frame and hand).

## 2. Results

| measurement | features mode | video mode |
|---|---|---|
| frames streamed (per hand) | 17,973 (1,198 s of video) | 3,284 (219 s) |
| frames whose streamed label differs from the checkpoint run's saved posteriors | 9 of 35,946 (0.03 %) | 3 of 6,568 (0.05 %) |
| recogniser + monitor compute per 1 s chunk, p95 | 62–120 ms (grows with recording length) | 59–134 ms |
| largest single chunk | 430 ms (one chunk; per-chunk times are not stored) | 145 ms |
| wall time / video time, whole path | 0.064 | 0.32 (decode 38.3 s + DINOv2 14.4 s + heads and monitor 17.4 s = 70.1 s) |

Reading:

- **The streamed path reproduces the offline evaluation.** Of 35,946 hand-frames, 9 differ from the
  labels computed from the same networks on whole recordings. At the checked difference the saved
  posterior's top two classes are equal within float16 precision; the largest posterior difference
  between prefix and whole-recording inference is 0.0004 (GPU convolutions on different lengths). On
  CPU the unit tests find no difference at all. Re-decoding the video and recomputing DINOv2 changes
  3 of 6,568 frames.
- **Three cameras run in about a third of real time on one RTX 4090**, including decoding, with
  video decoding the largest cost (55 % of the wall time, single-threaded PyAV per view in lockstep).
  The heads' cost grows linearly with the prefix: a p95 of 62 ms per second of video for a 43 s
  recording, 116–120 ms for 107–113 s recordings; extrapolated, a recording of about 20 minutes
  would need a full second per chunk, which is where per-layer state caching becomes necessary.
- **The deviations match the label replay.** 155 deviations over 18 recordings at m = 8, i.e. 8.6
  per recording, as in `havid_dev_v10_sop_online` for L = 45 at m = 8; the systems path works, and
  the recogniser still limits its usefulness.
- These are single-machine, single-run timings (no pacing, no concurrent streams). Concurrency and
  pacing are measured by the streaming benchmark (`../stream_bench_v1_dinov2/`).

## 3. Reproduce

```bash
export PYTHONUTF8=1
make havid-final-lstar LSTAR=45           # checkpoints (gated in reports/havid_final_gate)
uv run sop-monitor havid-online-head --checkpoints-lh artifacts/checkpoints/havid_final_L45_lh_s0 \
  --checkpoints-rh artifacts/checkpoints/havid_final_L45_rh_s0 --mode features --chunk 15 \
  --min-frames 8 --device cuda --out reports/havid_dev_v11_online_head_features
uv run sop-monitor havid-online-head ... --mode video --recording S18A05I01 --recording S08A04I01 \
  --recording S10A04I01 --out reports/havid_dev_v11_online_head_video
```

## 4. Not done here

- Paced multi-recording concurrency through the heads (the benchmark stops at DINOv2).
- Per-layer state caching for constant-time pushes; RTSP input (mediamtx not installed).
- No number on the frozen test subjects.
