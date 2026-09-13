"""Online replay of HA-ViD recordings: stream construction, source parsing and the summary."""

from __future__ import annotations

from pathlib import Path

import pytest

from sop_monitor.havid import TemporalSegment
from sop_monitor.havid_online import (
    PredictionSource,
    frame_labels,
    offline_findings,
    replay,
    seed_groups,
    summarise_online,
)
from sop_monitor.havid_sop import (
    learn_duration_bounds,
    learn_plate_graphs,
    mandatory_steps,
    step_sequence,
)
from sop_monitor.online import OnlineSOPMonitor

S = TemporalSegment


def test_frame_labels_requires_contiguous_segments_from_zero() -> None:
    assert frame_labels([S(0, 1, "a"), S(2, 4, "b")]) == ["a", "a", "b", "b", "b"]
    with pytest.raises(ValueError, match="contiguous"):
        frame_labels([S(1, 3, "a")])
    with pytest.raises(ValueError, match="contiguous"):
        frame_labels([S(0, 1, "a"), S(3, 4, "b")])


def test_prediction_source_parse() -> None:
    source = PredictionSource.parse("la15_s0=runs/lh,runs/rh,fusion_causal,15")
    assert source == PredictionSource(
        "la15_s0", Path("runs/lh"), Path("runs/rh"), "fusion_causal", 15
    )
    for bad in ("la15=runs/lh,runs/rh,fusion_causal", "runs/lh,runs/rh,x,0", "gt=a,b,c,0"):
        with pytest.raises(ValueError):
            PredictionSource.parse(bad)


def test_replay_matches_offline_findings() -> None:
    order = ["pckbx", "icbck", "ibscb", "scccb"]
    train = {}
    for n in range(6):
        lh, frame = [], 0
        for label in order:
            lh.append(S(frame, frame + 29 + n, label))
            frame += 30 + n
        train[f"S{n:02d}A04I01"] = step_sequence(lh, [S(0, frame - 1, "null")])
    plates = dict.fromkeys(train, "cylinder")
    graph = learn_plate_graphs(train, plates, min_support=3)["cylinder"]
    bounds = learn_duration_bounds(train, plates, 0.05, 0.95, min_count=3)["cylinder"]
    mandatory = mandatory_steps(train, plates)["cylinder"]
    hands = {
        "lh": [S(0, 29, "icbck"), S(30, 59, "pckbx"), S(60, 299, "ibscb")],
        "rh": [S(0, 299, "null")],
    }
    monitor = OnlineSOPMonitor(graph, bounds, mandatory, "pt", min_frames=1)
    found = replay(hands, monitor)
    online = sorted(
        (d.kind if d.kind != "duration" else "duration_" + d.evidence["kind"], d.step)
        for d in found
    )
    offline = offline_findings(hands, graph, bounds, mandatory, "pt")
    assert online == sorted(offline.elements())
    assert ("order", "icbck") in online and ("omission", "scccb") in online
    assert ("duration_too_long", "ibscb") in online


def test_summarise_online_counts_and_timing() -> None:
    recordings = [
        {"recording": "r1", "plate": "cylinder", "n_frames": "300", "with_wrong": "1"},
        {"recording": "r2", "plate": "cylinder", "n_frames": "150", "with_wrong": "0"},
    ]

    def row(rec: str, kind: str, start: str, detected: int) -> dict[str, str]:
        return {
            "source": "p",
            "min_frames": "4",
            "recording": rec,
            "kind": kind,
            "step": "x",
            "start_frame": start,
            "end_frame": "",
            "detected_at": str(detected),
        }

    deviations = [
        row("r1", "order", "27", 30),
        row("r1", "duration_too_long", "0", 60),
        row("r1", "omission", "", 299),
        row("r2", "unknown", "10", 13),
    ]
    summary = summarise_online(deviations, recordings, {"p": 15}, [("p", 4)])["p@4"]
    level = summary["recording_level"]
    assert level["with_wrong"] == {
        "recordings": 1,
        "order": 1,
        "omission": 1,
        "duration": 1,
        "unknown": 0,
        "any": 1,
    }
    assert level["without_wrong"]["any"] == 0 and level["without_wrong"]["unknown"] == 1
    assert summary["alarms_per_recording"]["order"] == 0.5
    assert summary["first_alarm_s_median"] == (30 + 15) / 15  # unknown does not count as an alarm
    # lead before end: order (299-30-15)/15, too_long (299-60-15)/15, unknown (149-13-15)/15
    assert summary["lead_before_end_s_median"] == (299 - 60 - 15) / 15
    assert summary["order_latency_s_median"] == (30 - 27 + 15) / 15
    empty = summarise_online([], recordings, {"p": 0}, [("p", 1)])["p@1"]
    assert (
        empty["first_alarm_s_median"] is None and empty["recording_level"]["with_wrong"]["any"] == 0
    )


def test_seed_groups_strip_the_seed_suffix() -> None:
    summary = {
        f"{source}@{m}": {"source": source, "min_frames": m}
        for source in ("gt", "la15_s0", "la15_s1", "la15_s10")
        for m in (1, 8)
    }
    groups = seed_groups(summary)
    assert groups[("gt", 1)] == ["gt@1"]
    assert groups[("la15", 8)] == ["la15_s0@8", "la15_s1@8", "la15_s10@8"]
    assert len(groups) == 4
