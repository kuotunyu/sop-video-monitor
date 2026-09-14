"""Figure data: the timeline loader reads the committed report tables; styles cover every kind."""

from __future__ import annotations

from pathlib import Path

import pytest

from sop_monitor.figures import (
    ALARM_STYLES,
    STATUS,
    alarm_style,
    load_timeline,
    stack_slots,
)

STEPS = """recording,plate,source,hands,label,start,end
R1,gear,gt,lh,sftg,0,29
R1,gear,gt,lh+rh,sntft,30,59
R1,gear,pred,rh,sftg,15,44
R2,gear,gt,lh,sftg,0,9
"""
DEVIATIONS = """source,min_frames,recording,kind,step,start_frame,end_frame,detected_at
gt,1,R1,duration_too_short,sftg,0,29,29
la0_s0,1,R1,order,sftg,15,,15
la0_s0,1,R1,omission,sntft,,,59
la0_s0,8,R1,omission,sntft,,,59
la15_s0,1,R1,unknown,x,0,,3
gt,1,R2,order,sftg,0,,0
"""
RECORDINGS = """recording,plate,n_frames,with_wrong
R1,gear,60,0
R2,gear,10,1
"""


def _write(tmp_path: Path) -> tuple[Path, Path, Path]:
    paths = (tmp_path / "steps.csv", tmp_path / "dev.csv", tmp_path / "rec.csv")
    for path, text in zip(paths, (STEPS, DEVIATIONS, RECORDINGS), strict=True):
        path.write_text(text, encoding="utf-8")
    return paths


def test_load_timeline_converts_frames_and_filters_sources(tmp_path: Path) -> None:
    steps, dev, rec = _write(tmp_path)
    timeline = load_timeline(steps, dev, rec, "R1")
    assert timeline.plate == "gear" and timeline.duration_s == 4.0
    assert [(s.hands, s.label, s.start_s, s.end_s) for s in timeline.steps["gt"]] == [
        ("lh", "sftg", 0.0, 2.0),
        ("lh+rh", "sntft", 2.0, 4.0),
    ]
    assert [(s.label, s.start_s) for s in timeline.steps["pred"]] == [("sftg", 1.0)]
    assert [(a.kind, a.detected_s, a.start_s) for a in timeline.alarms["gt"]] == [
        ("duration_too_short", 29 / 15, 0.0)
    ]
    assert [(a.kind, a.detected_s, a.start_s) for a in timeline.alarms["pred"]] == [
        ("order", 1.0, 1.0),
        ("omission", 59 / 15, None),
    ]
    assert load_timeline(steps, dev, rec, "R1", min_frames=8).alarms["pred"][0].kind == "omission"
    assert load_timeline(steps, dev, rec, "R1", pred_source="la15_s0").alarms["pred"][0].kind == (
        "unknown"
    )
    with pytest.raises(ValueError, match="not in"):
        load_timeline(steps, dev, rec, "R9")


def test_alarm_styles_cover_every_kind_with_glyph_and_label() -> None:
    for kind in ("order", "omission", "duration_too_short", "duration_too_long", "unknown"):
        style = alarm_style(kind)
        assert style.glyph and style.label
        assert style.colour in STATUS.values() or style.role == "none"
    assert len({s.glyph for s in ALARM_STYLES.values() if s.role != "none"}) == 3  # shape != colour
    with pytest.raises(ValueError):
        alarm_style("late")


def test_stack_slots_separates_close_marks() -> None:
    assert stack_slots([0.0, 0.1, 0.2, 5.0, 5.05], min_gap_s=1.0) == [0, 1, 2, 0, 1]
    assert stack_slots([], 1.0) == []
