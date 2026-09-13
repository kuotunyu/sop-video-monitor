"""Bounded frame ring buffer with explicit backpressure policies (spec 5.1 and 5.3).

Vocabulary (spec 5.3):

- ``WAIT``        producer blocks until a slot frees; no frame is ever dropped
- ``DROP_OLDEST`` when full, evict the oldest frame and increment ``skipped_frames``
- ``ADAPTIVE``    drop only while the producer outpaces the consumer by more than
                  ``fps_tolerance`` (5 fps, proposed) and for at most
                  ``max_consecutive_drops`` (16, proposed) frames in a row; otherwise WAIT

``skipped_frames`` is the counter the W4 backpressure benchmarks report against
(Frigate ``skipped_fps`` semantics). W0 keeps frames as in-process Python objects
in a deque; the shared-memory fixed-slot layout (16 slots, proposed) for
cross-process decode workers is W4 work and must keep this policy contract.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class DropPolicy(StrEnum):
    """Backpressure policy applied when the buffer is full."""

    WAIT = "wait"
    DROP_OLDEST = "drop_oldest"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class AdaptiveConfig:
    """Parameters of the ADAPTIVE policy (spec 5.3; defaults are the proposed values)."""

    fps_tolerance: float = 5.0
    max_consecutive_drops: int = 16
    rate_window_s: float = 1.0


class RingBuffer[T]:
    """Thread-safe bounded FIFO of frames with a configurable full-buffer policy.

    ``put`` returns ``True`` when the frame was stored (possibly after evicting an
    older one) and ``False`` when the WAIT path timed out; a timed-out frame is not
    counted in ``skipped_frames`` because the caller still holds it and decides.
    ``get`` raises :class:`queue.Empty` on timeout.

    ``clock`` is injectable so that producer/consumer rate estimation is
    deterministic in tests; it never affects the real-time waits of ``timeout``.
    """

    def __init__(
        self,
        capacity: int,
        policy: DropPolicy = DropPolicy.WAIT,
        adaptive: AdaptiveConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._policy = DropPolicy(policy)
        self._adaptive = adaptive if adaptive is not None else AdaptiveConfig()
        self._clock = clock
        self._items: deque[T] = deque()
        self._cond = threading.Condition()
        self._put_times: deque[float] = deque()
        self._get_times: deque[float] = deque()
        self._consecutive_drops = 0
        self.skipped_frames = 0
        self.adaptive_drops = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def policy(self) -> DropPolicy:
        return self._policy

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)

    @property
    def occupancy(self) -> float:
        """Fill ratio in ``[0, 1]``; W4 samples it as a time series."""
        return len(self) / self._capacity

    def put(self, item: T, timeout: float | None = None) -> bool:
        """Offer one frame according to the policy; see the class docstring."""
        with self._cond:
            now = self._clock()
            self._record(self._put_times, now)
            if len(self._items) >= self._capacity:
                if self._should_drop(now):
                    self._items.popleft()
                    self.skipped_frames += 1
                    self._consecutive_drops += 1
                    if self._policy is DropPolicy.ADAPTIVE:
                        self.adaptive_drops += 1
                else:
                    freed = self._cond.wait_for(
                        lambda: len(self._items) < self._capacity, timeout=timeout
                    )
                    if not freed:
                        return False
                    self._consecutive_drops = 0
            else:
                self._consecutive_drops = 0
            self._items.append(item)
            self._cond.notify_all()
            return True

    def get(self, timeout: float | None = None) -> T:
        """Pop the oldest frame, blocking up to ``timeout`` seconds."""
        with self._cond:
            if not self._cond.wait_for(lambda: bool(self._items), timeout=timeout):
                raise queue.Empty
            item = self._items.popleft()
            self._record(self._get_times, self._clock())
            self._cond.notify_all()
            return item

    def stats(self) -> dict[str, float | int | str]:
        """Snapshot of the counters W4 writes into ``reports/backpressure.json``."""
        with self._cond:
            now = self._clock()
            return {
                "policy": self._policy.value,
                "capacity": self._capacity,
                "size": len(self._items),
                "skipped_frames": self.skipped_frames,
                "adaptive_drops": self.adaptive_drops,
                "producer_fps": self._fps(self._put_times, now),
                "consumer_fps": self._fps(self._get_times, now),
            }

    # -- internals (caller holds the lock) -------------------------------------------------

    def _should_drop(self, now: float) -> bool:
        if self._policy is DropPolicy.WAIT:
            return False
        if self._policy is DropPolicy.DROP_OLDEST:
            return True
        surplus = self._fps(self._put_times, now) - self._fps(self._get_times, now)
        return (
            surplus > self._adaptive.fps_tolerance
            and self._consecutive_drops < self._adaptive.max_consecutive_drops
        )

    def _record(self, times: deque[float], now: float) -> None:
        times.append(now)
        self._prune(times, now)

    def _prune(self, times: deque[float], now: float) -> None:
        horizon = now - self._adaptive.rate_window_s
        while times and times[0] < horizon:
            times.popleft()

    def _fps(self, times: deque[float], now: float) -> float:
        self._prune(times, now)
        if len(times) < 2:
            return 0.0
        span = times[-1] - times[0]
        return (len(times) - 1) / span if span > 0 else 0.0
