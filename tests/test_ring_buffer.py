"""Ring buffer backpressure policies (spec 5.3): WAIT, DROP_OLDEST, ADAPTIVE and skipped_frames."""

import queue
import threading

import pytest

from sop_monitor.stream import AdaptiveConfig, DropPolicy, RingBuffer

FRAME_DT = 1 / 30  # a 30 fps producer


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RingBuffer(capacity=0)


def test_wait_never_drops_and_times_out_when_full() -> None:
    rb: RingBuffer[str] = RingBuffer(capacity=2, policy=DropPolicy.WAIT)
    assert rb.put("a") and rb.put("b")
    assert rb.put("c", timeout=0.01) is False
    assert rb.skipped_frames == 0
    assert len(rb) == 2 and rb.occupancy == 1.0
    assert rb.get() == "a" and rb.get() == "b"


def test_wait_unblocks_when_consumer_frees_a_slot() -> None:
    rb: RingBuffer[str] = RingBuffer(capacity=1, policy=DropPolicy.WAIT)
    assert rb.put("a")
    result: dict[str, bool] = {}

    def producer() -> None:
        result["stored"] = rb.put("b", timeout=5.0)

    thread = threading.Thread(target=producer)
    thread.start()
    assert rb.get(timeout=1.0) == "a"
    thread.join(timeout=5.0)
    assert result == {"stored": True}
    assert rb.get(timeout=1.0) == "b"
    assert rb.skipped_frames == 0


def test_drop_oldest_evicts_and_counts_skipped_frames() -> None:
    rb: RingBuffer[str] = RingBuffer(capacity=2, policy=DropPolicy.DROP_OLDEST)
    for frame in "abcd":
        assert rb.put(frame)
    assert rb.skipped_frames == 2
    assert rb.adaptive_drops == 0
    assert [rb.get(), rb.get()] == ["c", "d"]
    with pytest.raises(queue.Empty):
        rb.get(timeout=0.01)


def test_adaptive_drops_while_producer_outpaces_consumer_then_waits() -> None:
    clock = FakeClock()
    rb: RingBuffer[int] = RingBuffer(
        capacity=2,
        policy=DropPolicy.ADAPTIVE,
        adaptive=AdaptiveConfig(fps_tolerance=5.0, max_consecutive_drops=3),
        clock=clock,
    )
    # 30 fps producer, no consumer at all: buffer fills, then drops up to the cap.
    for frame in range(5):
        assert rb.put(frame)
        clock.advance(FRAME_DT)
    assert rb.skipped_frames == 3
    assert rb.adaptive_drops == 3
    # The cap was reached, so the next full-buffer put falls back to WAIT semantics.
    assert rb.put(5, timeout=0.01) is False
    assert rb.skipped_frames == 3
    assert [rb.get(), rb.get()] == [3, 4]


def test_adaptive_waits_when_rates_are_within_tolerance() -> None:
    clock = FakeClock()
    rb: RingBuffer[int] = RingBuffer(capacity=1, policy=DropPolicy.ADAPTIVE, clock=clock)
    # Producer and consumer both run at 30 fps for a while.
    for frame in range(10):
        assert rb.put(frame)
        assert rb.get() == frame
        clock.advance(FRAME_DT)
    assert rb.put(100)
    clock.advance(FRAME_DT)
    # Full buffer but no rate surplus: no drop, so the put times out instead.
    assert rb.put(101, timeout=0.01) is False
    assert rb.skipped_frames == 0
    stats = rb.stats()
    assert stats["policy"] == "adaptive"
    assert stats["producer_fps"] == pytest.approx(30.0)
    assert stats["consumer_fps"] == pytest.approx(30.0)


def test_adaptive_consecutive_counter_resets_after_a_clean_put() -> None:
    clock = FakeClock()
    rb: RingBuffer[int] = RingBuffer(
        capacity=1,
        policy=DropPolicy.ADAPTIVE,
        adaptive=AdaptiveConfig(fps_tolerance=5.0, max_consecutive_drops=1),
        clock=clock,
    )
    assert rb.put(0)
    clock.advance(FRAME_DT)
    assert rb.put(1)  # full, producer 30 fps vs consumer 0 -> one drop allowed
    assert rb.skipped_frames == 1
    clock.advance(FRAME_DT)
    assert rb.put(2, timeout=0.01) is False  # cap of 1 reached -> WAIT
    assert rb.get() == 1  # consumer frees the slot
    clock.advance(FRAME_DT)
    assert rb.put(3)  # clean put resets the consecutive counter
    clock.advance(FRAME_DT)
    assert rb.put(4)  # dropping is allowed again
    assert rb.skipped_frames == 2
