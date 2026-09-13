"""Streaming pipeline with fake frame iterators: pacing, backpressure, batching, watchdog, report."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from sop_monitor.stream import DropPolicy, RingBuffer
from sop_monitor.stream.pipeline import (
    BatchCollector,
    Frame,
    ReplaySource,
    Watchdog,
    run_pipeline,
)

IMAGE = np.zeros((4, 4, 3), dtype=np.uint8)


def frames(n: int = 1_000_000) -> Iterator[tuple[int, np.ndarray]]:
    for index in range(n):
        yield index, IMAGE


def test_replay_source_paces_frames_and_stops() -> None:
    source = ReplaySource("cam", lambda: frames(), fps=100.0, loop=True)
    buffer: RingBuffer[Frame] = RingBuffer(10_000, DropPolicy.DROP_OLDEST)
    stop = threading.Event()
    thread = threading.Thread(target=source.run, args=(buffer, stop))
    thread.start()
    time.sleep(0.5)
    stop.set()
    thread.join(timeout=2.0)
    assert source.stats.finished and source.stats.error is None
    assert 30 <= source.stats.produced <= 70  # 100 fps for 0.5 s, generous for CI jitter
    first, second = buffer.get(timeout=0), buffer.get(timeout=0)
    assert (first.camera, first.index, second.index) == ("cam", 0, 1)
    with pytest.raises(ValueError):
        ReplaySource("cam", lambda: frames(), fps=0.0)


def test_replay_source_reports_a_crash() -> None:
    def broken() -> Iterator[tuple[int, np.ndarray]]:
        yield 0, IMAGE
        raise OSError("decoder died")

    source = ReplaySource("cam", broken, fps=1000.0, loop=False)
    source.run(RingBuffer(4, DropPolicy.DROP_OLDEST), threading.Event())
    assert source.stats.finished and source.stats.error == "OSError: decoder died"


def test_batch_collector_caps_batches_and_records_latency() -> None:
    buffers = {c: RingBuffer[Frame](16, DropPolicy.WAIT) for c in ("a", "b")}
    now = time.monotonic()
    for i in range(5):
        for camera, buffer in buffers.items():
            buffer.put(Frame(camera, i, now, IMAGE))
    seen: list[Sequence[Frame]] = []
    collector = BatchCollector(buffers, seen.append, max_batch=4, max_wait_s=0.01)
    stop = threading.Event()
    thread = threading.Thread(target=collector.run, args=(stop,))
    thread.start()
    time.sleep(0.2)
    stop.set()
    thread.join(timeout=2.0)
    assert collector.stats.frames == 10 and max(len(b) for b in seen) == 4
    assert {f.camera for f in seen[0]} == {"a", "b"}  # round robin across cameras
    assert len(collector.stats.latencies_s) == 10 and min(collector.stats.latencies_s) >= 0
    with pytest.raises(ValueError):
        BatchCollector(buffers, seen.append, max_batch=0)


def test_watchdog_flags_a_silent_source_once() -> None:
    clock = [0.0]
    source = ReplaySource("cam", lambda: frames(), fps=10.0, clock=lambda: clock[0])
    watchdog = Watchdog([source], stall_after_s=1.0, clock=lambda: clock[0])
    source.stats.last_put_at = 0.0
    clock[0] = 0.5
    assert watchdog.check(started_at=0.0) == []
    clock[0] = 2.0
    events = watchdog.check(started_at=0.0)
    assert [e.camera for e in events] == ["cam"] and events[0].silent_for_s == pytest.approx(2.0)
    assert watchdog.check(started_at=0.0) == []  # reported once while it stays silent
    source.stats.last_put_at = 2.5
    clock[0] = 2.6
    watchdog.check(started_at=0.0)
    clock[0] = 5.0
    assert len(watchdog.check(started_at=0.0)) == 1  # a new stall after recovery is reported again


def test_pipeline_drop_oldest_sheds_load_and_wait_does_not() -> None:
    def slow(batch: Sequence[Frame]) -> None:
        time.sleep(0.02 * len(batch))  # 50 fps consumer

    def sources() -> list[ReplaySource]:
        return [ReplaySource(f"cam{i}", lambda: frames(), fps=100.0) for i in range(3)]

    dropping = run_pipeline(
        sources(), slow, duration_s=1.0, capacity=4, policy=DropPolicy.DROP_OLDEST, max_batch=4
    )
    assert dropping.cameras == 3 and dropping.target_fps_per_camera == pytest.approx(100.0)
    assert dropping.skipped_frames > 0 and dropping.put_timeouts == 0
    assert dropping.consumed < dropping.produced
    assert (
        0 < dropping.throughput_fps < 120
        and dropping.latency_ms["p95"] >= dropping.latency_ms["p50"]
    )
    assert dropping.stalls == [] and dropping.occupancy_max > 0.5
    waiting = run_pipeline(
        sources(), slow, duration_s=1.0, capacity=4, policy=DropPolicy.WAIT, max_batch=4
    )
    assert waiting.skipped_frames == 0 and waiting.put_timeouts > 0
    assert set(waiting.to_dict()) >= {"throughput_fps", "latency_ms", "per_camera", "stalls"}
    with pytest.raises(ValueError, match="unique"):
        run_pipeline([sources()[0], sources()[0]], slow, duration_s=0.1)


def test_default_clocks_are_high_resolution() -> None:
    """``time.monotonic`` ticks in 15.6 ms steps on Windows, which quantises every latency."""
    import inspect
    import time

    from sop_monitor.stream import pipeline, ring_buffer

    for cls in (
        pipeline.ReplaySource,
        pipeline.BatchCollector,
        pipeline.Watchdog,
        ring_buffer.RingBuffer,
    ):
        assert inspect.signature(cls).parameters["clock"].default is time.perf_counter, cls
