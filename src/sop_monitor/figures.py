"""Data and style shared by the animated figures under ``figures/`` (rendered with Manim).

Nothing here imports Manim, so the loaders are tested in CI like any other module. The timeline
figure is derived from the committed report tables, not drawn by hand: the step bars come from a
SOP run's ``steps_<split>.csv`` and the alarms from an online replay's ``deviations_<split>.csv``,
so the picture shows the same numbers as the reports. The palette is the dark instance of the
data-viz reference palette: status colours are reserved for alarms and never reused for series,
steps share one hue, and every alarm carries a glyph and a label, so meaning never rests on colour
alone.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

FPS = 15.0
FONT = "Arial"
"""Sans-serif face available on the rendering machine; the GIFs are committed, so viewers need no font."""

SURFACE = "#1a1a19"
INK = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
SERIES = ("#3987e5", "#d95926", "#199e70")
"""Categorical slots 1–3 (dark surface); validated all-pairs. Used for the three camera views."""
STEP_HUE = "#256abf"
"""Sequential blue, step 500: every step bar shares this one hue."""
STEP_HUE_SOFT = "#1c5cab"
STATUS = {"critical": "#d03b3b", "serious": "#ec835a", "warning": "#fab219"}


@dataclass(frozen=True)
class AlarmStyle:
    role: str  # critical | serious | warning | none
    glyph: str  # triangle | circle | square | diamond
    label: str

    @property
    def colour(self) -> str:
        return STATUS.get(self.role, INK_MUTED)


ALARM_STYLES: dict[str, AlarmStyle] = {
    "order": AlarmStyle("critical", "triangle", "order"),
    "omission": AlarmStyle("serious", "circle", "omitted"),
    "duration_too_short": AlarmStyle("warning", "square", "too short"),
    "duration_too_long": AlarmStyle("warning", "square", "too long"),
    "unknown": AlarmStyle("none", "diamond", "unknown"),
}


def alarm_style(kind: str) -> AlarmStyle:
    try:
        return ALARM_STYLES[kind]
    except KeyError as error:
        raise ValueError(f"unknown deviation kind {kind!r}") from error


@dataclass(frozen=True)
class StepBar:
    hands: str  # lh | rh | lh+rh
    label: str
    start_s: float
    end_s: float  # exclusive


@dataclass(frozen=True)
class AlarmMark:
    kind: str
    step: str
    detected_s: float
    start_s: float | None


@dataclass(frozen=True)
class Timeline:
    recording: str
    plate: str
    duration_s: float
    steps: dict[str, list[StepBar]]  # "gt" and "pred"
    alarms: dict[str, list[AlarmMark]]  # "gt" and "pred"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_timeline(
    steps_csv: Path,
    deviations_csv: Path,
    recordings_csv: Path,
    recording: str,
    pred_source: str = "la0_s0",
    min_frames: int = 1,
) -> Timeline:
    """One recording's ground-truth and predicted step bars with the online replay's alarms.

    ``pred_source`` names the predicted stream in the deviations table; its step bars are the
    ``pred`` rows of ``steps_csv``, which must come from the same prediction run.
    """
    meta = {row["recording"]: row for row in _rows(recordings_csv)}
    if recording not in meta:
        raise ValueError(f"{recording} is not in {recordings_csv}")
    steps: dict[str, list[StepBar]] = {"gt": [], "pred": []}
    for row in _rows(steps_csv):
        if row["recording"] != recording:
            continue
        steps[row["source"]].append(
            StepBar(
                row["hands"],
                row["label"],
                int(row["start"]) / FPS,
                (int(row["end"]) + 1) / FPS,
            )
        )
    alarms: dict[str, list[AlarmMark]] = {"gt": [], "pred": []}
    wanted = {"gt": "gt", pred_source: "pred"}
    for row in _rows(deviations_csv):
        if row["recording"] != recording or int(row["min_frames"]) != min_frames:
            continue
        if row["source"] not in wanted:
            continue
        alarm_style(row["kind"])  # validates the kind
        alarms[wanted[row["source"]]].append(
            AlarmMark(
                row["kind"],
                row["step"],
                int(row["detected_at"]) / FPS,
                int(row["start_frame"]) / FPS if row["start_frame"] else None,
            )
        )
    if not steps["gt"]:
        raise ValueError(f"{recording} has no ground-truth steps in {steps_csv}")
    for source in steps:
        steps[source].sort(key=lambda s: (s.start_s, s.end_s))
        alarms[source].sort(key=lambda a: (a.detected_s, a.kind))
    return Timeline(
        recording,
        meta[recording]["plate"],
        int(meta[recording]["n_frames"]) / FPS,
        steps,
        alarms,
    )


def stack_slots(times: Iterable[float], min_gap_s: float) -> list[int]:
    """Row index per mark so that marks closer than ``min_gap_s`` do not overlap (greedy)."""
    rows: list[float] = []  # last time placed in each row
    slots: list[int] = []
    for time in times:
        for index, last in enumerate(rows):
            if time - last >= min_gap_s:
                rows[index] = time
                slots.append(index)
                break
        else:
            rows.append(time)
            slots.append(len(rows) - 1)
    return slots
