"""Streaming layer (spec 5.1-5.3): RTSP sources, ring buffer, batch collector, watchdog.

``ring_buffer`` holds the backpressure vocabulary; ``pipeline`` the replay sources (one decode
thread per camera), the batch collector and the watchdog; ``bench`` the stream x speed grid that
produces the throughput / latency / dropped-frame curves. RTSP ingest (mediamtx) and the
shared-memory buffer for cross-process decoders are not built yet.
"""

from sop_monitor.stream.ring_buffer import AdaptiveConfig, DropPolicy, RingBuffer

__all__ = ["AdaptiveConfig", "DropPolicy", "RingBuffer"]
