"""Multi-camera streaming pipeline (spec 5.1-5.3, W4 first slice): replay → ring buffers → batches.

Topology::

    ReplaySource (one thread per camera) ──put──▶ RingBuffer (per camera, backpressure policy)
                                                        │ get
    BatchCollector (one thread) ◀───────────────────────┘ ──▶ consumer(batch) ──▶ PipelineReport
    Watchdog (one thread): flags a source that has not produced a frame for ``stall_after_s``

A :class:`ReplaySource` replays a decoded video at its nominal frame rate (or ``speed`` times it),
so decode cost and pacing are those of a live camera without the network hop an RTSP server would
add; the frame iterator is injectable, so tests run without PyAV. The collector takes up to
``max_batch`` frames from all cameras, waiting at most ``max_wait_s`` for a batch to fill — the
usual GPU micro-batching trade-off — and hands them to a consumer (decode-only, a frame embedder,
later the online head). Every frame carries the monotonic time it was produced, so the report gives
end-to-end latency percentiles next to throughput, per-camera ``skipped_frames`` and a sampled
buffer-occupancy series: the inputs of the backpressure curves.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from sop_monitor.stream.ring_buffer import AdaptiveConfig, DropPolicy, RingBuffer

FrameIterator = Callable[[], Iterator[tuple[int, np.ndarray]]]


@dataclass(frozen=True)
class Frame:
    camera: str
    index: int
    produced_at: float
    image: np.ndarray


@dataclass
class SourceStats:
    camera: str
    produced: int = 0
    put_timeouts: int = 0
    last_put_at: float | None = None
    finished: bool = False
    error: str | None = None


class ReplaySource:
    """Replay one camera's frames into a ring buffer at ``fps * speed`` frames per second."""

    def __init__(
        self,
        camera: str,
        frames: FrameIterator,
        fps: float,
        speed: float = 1.0,
        loop: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if fps <= 0 or speed <= 0:
            raise ValueError("fps and speed must be positive")
        self.camera = camera
        self.frames = frames
        self.period = 1.0 / (fps * speed)
        self.loop = loop
        self.clock = clock
        self.stats = SourceStats(camera)

    def run(self, buffer: RingBuffer[Frame], stop: threading.Event) -> None:
        start = self.clock()
        emitted = 0
        try:
            while not stop.is_set():
                for index, image in self.frames():
                    if stop.is_set():
                        break
                    due = start + emitted * self.period
                    delay = due - self.clock()
                    if delay > 0:
                        stop.wait(delay)
                    frame = Frame(self.camera, index, self.clock(), image)
                    if buffer.put(frame, timeout=self.period):
                        self.stats.produced += 1
                        self.stats.last_put_at = frame.produced_at
                    else:
                        self.stats.put_timeouts += 1
                    emitted += 1
                if not self.loop:
                    break
        except Exception as error:  # a crashed source is reported, never silently lost
            self.stats.error = f"{type(error).__name__}: {error}"
        finally:
            self.stats.finished = True


@dataclass
class ConsumerStats:
    batches: int = 0
    frames: int = 0
    latencies_s: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    busy_s: float = 0.0


class BatchCollector:
    """Round-robin micro-batches of up to ``max_batch`` frames across the camera buffers."""

    def __init__(
        self,
        buffers: Mapping[str, RingBuffer[Frame]],
        consumer: Callable[[Sequence[Frame]], None],
        max_batch: int = 8,
        max_wait_s: float = 0.02,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_batch < 1 or max_wait_s < 0:
            raise ValueError("max_batch must be >= 1 and max_wait_s >= 0")
        self.buffers = dict(buffers)
        self.consumer = consumer
        self.max_batch = max_batch
        self.max_wait_s = max_wait_s
        self.clock = clock
        self.stats = ConsumerStats()

    def _collect(self, stop: threading.Event) -> list[Frame]:
        batch: list[Frame] = []
        deadline = self.clock() + self.max_wait_s
        cameras = list(self.buffers)
        while len(batch) < self.max_batch and not stop.is_set():
            took = False
            for camera in cameras:
                if len(batch) >= self.max_batch:
                    break
                try:
                    batch.append(self.buffers[camera].get(timeout=0))
                    took = True
                except queue.Empty:
                    continue
            if took:
                continue
            remaining = deadline - self.clock()
            if remaining <= 0 and batch:
                break
            stop.wait(min(0.001, max(remaining, 0.0)) or 0.001)
            if remaining <= 0 and not batch:
                deadline = self.clock() + self.max_wait_s
        return batch

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            batch = self._collect(stop)
            if not batch:
                continue
            began = self.clock()
            self.consumer(batch)
            done = self.clock()
            self.stats.busy_s += done - began
            self.stats.batches += 1
            self.stats.frames += len(batch)
            self.stats.batch_sizes.append(len(batch))
            self.stats.latencies_s.extend(done - frame.produced_at for frame in batch)


@dataclass(frozen=True)
class StallEvent:
    camera: str
    detected_at: float
    silent_for_s: float


class Watchdog:
    """Flags sources that stopped producing (crashed, stuck decoder, starved by WAIT backpressure)."""

    def __init__(
        self,
        sources: Sequence[ReplaySource],
        stall_after_s: float = 1.0,
        interval_s: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sources = list(sources)
        self.stall_after_s = stall_after_s
        self.interval_s = interval_s
        self.clock = clock
        self.events: list[StallEvent] = []
        self._stalled: set[str] = set()

    def check(self, started_at: float) -> list[StallEvent]:
        now = self.clock()
        new: list[StallEvent] = []
        for source in self.sources:
            if source.stats.finished and source.stats.error is None:
                continue
            last = source.stats.last_put_at or started_at
            silent = now - last
            if silent > self.stall_after_s or source.stats.error is not None:
                if source.camera not in self._stalled:
                    self._stalled.add(source.camera)
                    new.append(StallEvent(source.camera, now, silent))
            else:
                self._stalled.discard(source.camera)
        self.events.extend(new)
        return new

    def run(self, stop: threading.Event, started_at: float) -> None:
        while not stop.wait(self.interval_s):
            self.check(started_at)


@dataclass(frozen=True)
class PipelineReport:
    duration_s: float
    cameras: int
    target_fps_per_camera: float
    policy: str
    capacity: int
    max_batch: int
    produced: int
    consumed: int
    skipped_frames: int
    put_timeouts: int
    throughput_fps: float
    latency_ms: dict[str, float]
    mean_batch: float
    consumer_busy_fraction: float
    occupancy_mean: float
    occupancy_max: float
    per_camera: dict[str, dict[str, object]]
    stalls: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    array = np.asarray(values) * 1000.0
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def run_pipeline(
    sources: Sequence[ReplaySource],
    consumer: Callable[[Sequence[Frame]], None],
    duration_s: float,
    capacity: int = 16,
    policy: DropPolicy = DropPolicy.DROP_OLDEST,
    max_batch: int = 8,
    max_wait_s: float = 0.02,
    stall_after_s: float = 1.0,
    warmup_s: float = 0.0,
) -> PipelineReport:
    """Run sources, collector and watchdog for ``duration_s`` seconds and report.

    Frames consumed during the first ``warmup_s`` seconds are excluded from latency and throughput
    (model warm-up, decoder start-up), but not from the counters of dropped frames.
    """
    if len({s.camera for s in sources}) != len(sources):
        raise ValueError("camera names must be unique")
    buffers: dict[str, RingBuffer[Frame]] = {
        s.camera: RingBuffer(capacity, policy, AdaptiveConfig()) for s in sources
    }
    collector = BatchCollector(buffers, consumer, max_batch, max_wait_s)
    watchdog = Watchdog(sources, stall_after_s)
    stop = threading.Event()
    started_at = time.monotonic()
    threads = [
        threading.Thread(target=s.run, args=(buffers[s.camera], stop), daemon=True) for s in sources
    ]
    threads.append(threading.Thread(target=collector.run, args=(stop,), daemon=True))
    threads.append(threading.Thread(target=watchdog.run, args=(stop, started_at), daemon=True))
    for thread in threads:
        thread.start()
    occupancy: list[float] = []
    frames_at_warmup = 0
    latency_start = 0
    warm = warmup_s <= 0
    while (elapsed := time.monotonic() - started_at) < duration_s:
        if not warm and elapsed >= warmup_s:
            warm = True
            frames_at_warmup = collector.stats.frames
            latency_start = len(collector.stats.latencies_s)
        occupancy.append(float(np.mean([b.occupancy for b in buffers.values()])))
        time.sleep(0.05)
    stop.set()
    for thread in threads:
        thread.join(timeout=5.0)
    measured_s = duration_s - (warmup_s if warmup_s > 0 else 0.0)
    consumed = collector.stats.frames - frames_at_warmup
    per_camera = {
        s.camera: {
            "produced": s.stats.produced,
            "skipped_frames": buffers[s.camera].skipped_frames,
            "put_timeouts": s.stats.put_timeouts,
            "error": s.stats.error,
        }
        for s in sources
    }
    total_duration = time.monotonic() - started_at
    return PipelineReport(
        duration_s=duration_s,
        cameras=len(sources),
        target_fps_per_camera=1.0 / sources[0].period if sources else 0.0,
        policy=DropPolicy(policy).value,
        capacity=capacity,
        max_batch=max_batch,
        produced=sum(s.stats.produced for s in sources),
        consumed=collector.stats.frames,
        skipped_frames=sum(b.skipped_frames for b in buffers.values()),
        put_timeouts=sum(s.stats.put_timeouts for s in sources),
        throughput_fps=consumed / measured_s if measured_s > 0 else 0.0,
        latency_ms=_percentiles(collector.stats.latencies_s[latency_start:]),
        mean_batch=float(np.mean(collector.stats.batch_sizes))
        if collector.stats.batch_sizes
        else 0.0,
        consumer_busy_fraction=collector.stats.busy_s / total_duration
        if total_duration > 0
        else 0.0,
        occupancy_mean=float(np.mean(occupancy)) if occupancy else 0.0,
        occupancy_max=float(np.max(occupancy)) if occupancy else 0.0,
        per_camera=per_camera,
        stalls=[e.__dict__ for e in watchdog.events],
    )
