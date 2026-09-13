"""Online SOP monitor: matches the offline checker on finished recordings and detects early."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from sop_monitor.havid import TemporalSegment
from sop_monitor.havid_sop import (
    FPS,
    at_granularity,
    check_sequence,
    learn_duration_bounds,
    learn_plate_graphs,
    mandatory_steps,
    step_sequence,
)
from sop_monitor.online import OnlineSOPMonitor

S = TemporalSegment
VOCAB = ["pckbx", "icbck", "ibscb", "sshc1", "sshc2dh", "ickcb", "scccb", "null", "w"]
ORDER = ["pckbx", "icbck", "ibscb", "sshc1", "sshc2dh", "scccb"]


def _frames(segments: list[TemporalSegment]) -> list[str]:
    out: list[str] = []
    for segment in segments:
        out.extend([segment.label] * segment.n_frames)
    return out


def _random_hand(rng: random.Random, n_frames: int) -> list[TemporalSegment]:
    """Random segments without two adjacent equal labels (a frame stream cannot express those)."""
    segments: list[TemporalSegment] = []
    frame = 0
    while frame < n_frames:
        length = min(rng.randint(3, 40), n_frames - frame)
        choices = [label for label in VOCAB if not segments or label != segments[-1].label]
        segments.append(S(frame, frame + length - 1, rng.choice(choices)))
        frame += length
    return segments


def _knowledge(granularity: str):
    rng = random.Random(1)
    train = {}
    for n in range(12):
        lh = []
        frame = 0
        for label in ORDER:
            length = rng.randint(20, 45)
            lh.append(S(frame, frame + length - 1, label))
            frame += length
        rh = [S(0, frame - 1, "null")]
        train[f"S{n:02d}A04I01"] = at_granularity(step_sequence(lh, rh), granularity)
    plates = {rec: "cylinder" for rec in train}
    graph = learn_plate_graphs(train, plates, min_support=3)["cylinder"]
    bounds = learn_duration_bounds(train, plates, 0.05, 0.95, min_count=5)["cylinder"]
    mandatory = mandatory_steps(train, plates)["cylinder"]
    return graph, bounds, mandatory


def _run_online(monitor: OnlineSOPMonitor, lh: list[str], rh: list[str]) -> list:
    out = []
    for frame, (a, b) in enumerate(zip(lh, rh, strict=True)):
        out.extend(monitor.push(frame, {"lh": a, "rh": b}))
    out.extend(monitor.finish())
    return out


@pytest.mark.parametrize("granularity", ["pt", "sheet"])
def test_online_findings_equal_the_offline_checker(granularity: str) -> None:
    graph, bounds, mandatory = _knowledge(granularity)
    rng = random.Random(7)
    for _ in range(60):
        n = rng.randint(50, 400)
        lh_segments, rh_segments = _random_hand(rng, n), _random_hand(rng, n)
        steps = at_granularity(step_sequence(lh_segments, rh_segments), granularity)
        offline = check_sequence(steps, graph, bounds, mandatory)
        expected = Counter()
        expected.update(("order", v["step"]) for v in offline["violations"])
        expected.update(("omission", label) for label in offline["omissions"])
        expected.update(("duration_" + d["kind"], d["step"]) for d in offline["durations"])
        expected.update(("unknown", label) for label in set(offline["unknown"]))
        monitor = OnlineSOPMonitor(graph, bounds, mandatory, granularity, min_frames=1)
        found = Counter(
            (d.kind if d.kind != "duration" else "duration_" + d.evidence["kind"], d.step)
            for d in _run_online(monitor, _frames(lh_segments), _frames(rh_segments))
        )
        assert found == expected


def test_too_long_is_detected_while_the_step_runs_and_order_at_confirmation() -> None:
    graph, bounds, mandatory = _knowledge("sheet")
    high = bounds["pckbx"].high
    long_frames = int(high * FPS) + 30
    lh = ["icbck"] * 30 + ["pckbx"] * long_frames + ["null"] * 10
    rh = ["null"] * len(lh)
    monitor = OnlineSOPMonitor(graph, bounds, mandatory, "sheet", min_frames=1)
    events = _run_online(monitor, lh, rh)
    order = next(e for e in events if e.kind == "order")
    assert order.step == "icbck" and order.detected_at == 0 and order.latency_s == 0
    too_long = next(e for e in events if e.kind == "duration" and e.evidence["kind"] == "too_long")
    assert too_long.step == "pckbx" and too_long.end_frame is None
    assert too_long.detected_at < 30 + long_frames - 1  # before the step ended
    assert too_long.detected_at - 30 + 1 == int(high * FPS) + 1  # the first frame past the bound
    omissions = {e.step for e in events if e.kind == "omission"}
    assert "scccb" in omissions and monitor.finish() == []


def test_min_frames_absorbs_blips_and_delays_confirmation() -> None:
    graph, bounds, mandatory = _knowledge("sheet")
    lh = ["pckbx"] * 40 + ["icbck"] * 2 + ["pckbx"] * 20 + ["icbck"] * 40
    rh = ["null"] * len(lh)
    monitor = OnlineSOPMonitor(graph, bounds, mandatory, "sheet", min_frames=5)
    events = _run_online(monitor, lh, rh)
    assert [s.label for s in monitor.steps] == ["pckbx", "icbck"]  # the 2-frame blip is absorbed
    assert monitor.steps[1].start == 62
    assert all(e.kind != "order" for e in events)
    with pytest.raises(ValueError):
        OnlineSOPMonitor(graph, bounds, mandatory, min_frames=0)
    fresh = OnlineSOPMonitor(graph, bounds, mandatory)
    with pytest.raises(ValueError, match="expected frame 0"):
        fresh.push(3, {"lh": "null", "rh": "null"})


def test_adjacent_equal_segments_are_one_step_online() -> None:
    graph, bounds, mandatory = _knowledge("sheet")
    lh = _frames([S(0, 29, "pckbx"), S(30, 59, "pckbx"), S(60, 99, "null")])
    monitor = OnlineSOPMonitor(graph, bounds, mandatory, "sheet")
    _run_online(monitor, lh, ["null"] * len(lh))
    assert [(s.label, s.start, s.end) for s in monitor.steps] == [("pckbx", 0, 59)]
    offline = step_sequence(
        [S(0, 29, "pckbx"), S(30, 59, "pckbx"), S(60, 99, "null")], [S(0, 99, "null")]
    )
    assert len(offline) == 2  # the annotation keeps them apart; the frame stream cannot
