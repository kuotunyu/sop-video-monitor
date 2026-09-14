# Streaming: what is built, what is measured, and the gap between them

Two paths exist and they do not yet meet. The first is the concurrent camera pipeline that the
benchmark measures up to the frame embedder; the second is the single-station online head that
runs frames through the recogniser and the SOP monitor. Joining them (several stations' heads
behind one ring-buffer pipeline) is W4 work that is not built.

## 1. The measured pipeline (`src/sop_monitor/stream/`, `reports/stream_bench_v1_*`)

```mermaid
sequenceDiagram
    participant S as ReplaySource<br/>(a thread per camera)
    participant R as RingBuffer<br/>(DROP_OLDEST)
    participant C as BatchCollector
    participant E as DINOv2 consumer<br/>(GPU)
    participant W as Watchdog
    loop every frame, paced to 15 fps
        S->>R: put(frame, produced_at)
        alt buffer full
            R-->>R: evict the oldest, skipped_frames += 1
        end
    end
    loop micro-batch: up to 16 frames or 20 ms
        C->>R: take what is ready
        C->>E: embed_batch(frames)
        E-->>C: latency = done − produced_at, per frame
    end
    W-->>S: stall flagged if no frame for stall_after_s
```

On one RTX 4090, 24 replayed cameras at 15 fps are decoded and embedded without falling behind;
at 36 the decode threads lag (`reports/stream_bench_v1_dinov2/README.md`).

## 2. The online head for one station (`src/sop_monitor/online_head.py`, `reports/havid_dev_v11_*`)

```mermaid
sequenceDiagram
    participant V as decoder<br/>(3 views in lockstep)
    participant D as DINOv2
    participant H as StreamingHead<br/>(per view and hand)
    participant F as fusion +<br/>look-ahead emission
    participant M as OnlineSOPMonitor
    loop one chunk (1 s of frames)
        V->>D: decode + rescale, per view
        D-->>H: features per view
        H->>H: recompute the causal prefix (exact by causality)
        H-->>F: posteriors for the new frames
        F->>F: mean over views, emit frame f at time f + L
        F->>M: push(frame, lh label, rh label)
        M-->>M: order / unknown at confirmation, too_long while open
    end
    M->>M: finish(): too_short at close, omissions at the end
```

Three views run from mp4 to deviations at about a third of real time, decoding being more than
half of it; streamed labels equal the offline evaluation of the same networks on all but a few
float16 ties (`reports/havid_dev_v11_online_head_features/README.md`).

## 3. Not built

- RTSP ingestion (mediamtx is not installed) and a cross-process shared-memory ring buffer.
- The heads inside the concurrent loop: the benchmark stops at the embedder, the head runs one
  station at a time, unpaced.
- Per-layer state caching for the heads; each push recomputes the prefix, so cost grows with the
  recording's length.
