"""Streaming layer (spec 5.1-5.3): RTSP sources, ring buffer, batch collector, watchdog.

W0 ships only the ring buffer and its backpressure vocabulary; decode threads,
the batch collector and the watchdog are W4 work.
"""

from sop_monitor.stream.ring_buffer import AdaptiveConfig, DropPolicy, RingBuffer

__all__ = ["AdaptiveConfig", "DropPolicy", "RingBuffer"]
