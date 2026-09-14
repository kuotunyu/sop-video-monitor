"""One val recording's SOP checks, frame by frame, from the committed report tables.

Render (see ``make figures``)::

    uv run --group figures manim -r 960,540 --fps 15 --format mp4 figures/timeline.py TimelineScene

Environment: ``FIGURE_RECORDING`` picks the recording (default S18A06I01), ``FIGURE_PRED_SOURCE``
the predicted stream of the online replay (default ``la0_s0``: causal mean fusion, no output
delay, seed 0, whose step bars are the ``pred`` rows of ``havid_dev_v8_sop_sheet/steps_val.csv``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    FadeIn,
    GrowFromCenter,
    Line,
    Rectangle,
    Scene,
    Square,
    Text,
    Triangle,
    ValueTracker,
    VGroup,
    always_redraw,
    config,
    linear,
)

from sop_monitor.figures import (
    AXIS,
    FONT,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    STEP_HUE,
    SURFACE,
    AlarmMark,
    StepBar,
    alarm_style,
    load_timeline,
    stack_slots,
)

config.background_color = SURFACE

ROOT = Path(__file__).resolve().parents[1]
X0, X1 = -4.6, 6.2
LANE_H, LANE_GAP = 0.42, 0.08
GROUPS = {
    "gt": {"title": "ground truth", "heading": 3.0, "marks": 1.95, "lh": 1.5, "rh": 1.0},
    "pred": {
        "title": "predicted (causal MS-TCN++, no delay)",
        "heading": 0.55,
        "marks": -0.45,
        "lh": -1.0,
        "rh": -1.5,
    },
}
AXIS_Y = -2.2
MARK_ROWS, MARK_ROW_H = 5, 0.21
ANIMATION_S = 20.0


def glyph(kind: str, size: float = 0.11):
    style = alarm_style(kind)
    colour = style.colour
    if style.glyph == "triangle":
        shape = Triangle(color=colour, fill_color=colour, fill_opacity=1, stroke_width=0).scale(
            size * 1.3
        )
    elif style.glyph == "circle":
        shape = Circle(radius=size, color=colour, fill_color=colour, fill_opacity=1, stroke_width=0)
    elif style.glyph == "square":
        shape = Square(
            side_length=size * 1.8, color=colour, fill_color=colour, fill_opacity=1, stroke_width=0
        )
    else:
        shape = Square(side_length=size * 1.8, color=colour, fill_opacity=0, stroke_width=2).rotate(
            np.pi / 4
        )
    return shape


class TimelineScene(Scene):
    def construct(self) -> None:
        recording = os.environ.get("FIGURE_RECORDING", "S18A06I01")
        pred_source = os.environ.get("FIGURE_PRED_SOURCE", "la0_s0")
        timeline = load_timeline(
            ROOT / "reports/havid_dev_v8_sop_sheet/steps_val.csv",
            ROOT / "reports/havid_dev_v10_sop_online/deviations_val.csv",
            ROOT / "reports/havid_dev_v10_sop_online/recordings_val.csv",
            recording,
            pred_source=pred_source,
        )
        duration = timeline.duration_s

        def x_of(seconds: float) -> float:
            return X0 + (X1 - X0) * seconds / duration

        # ---- static chrome --------------------------------------------------------------------
        title = Text(
            f"SOP checks frame by frame · val recording {recording} ({timeline.plate} plate)",
            font=FONT,
            font_size=24,
            color=INK,
        ).to_edge(UP, buff=0.15)
        subtitle = Text(
            "instruction-sheet knowledge learned on train · alarms appear at the frame they were detected",
            font=FONT,
            font_size=15,
            color=INK_SECONDARY,
        ).next_to(title, DOWN, buff=0.12)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.6)

        chrome = VGroup()
        for layout in GROUPS.values():
            for hand in ("lh", "rh"):
                y = layout[hand]
                lane = Rectangle(
                    width=X1 - X0, height=LANE_H, stroke_color=GRID, stroke_width=1, fill_opacity=0
                ).move_to([(X0 + X1) / 2, y, 0])
                tag = Text(
                    "left hand" if hand == "lh" else "right hand",
                    font=FONT,
                    font_size=13,
                    color=INK_MUTED,
                ).next_to(lane, LEFT, buff=0.15)
                chrome.add(lane, tag)
            heading = Text(layout["title"], font=FONT, font_size=16, color=INK).move_to(
                [X0, layout["heading"], 0], aligned_edge=LEFT
            )
            chrome.add(heading)
        axis = Line([X0, AXIS_Y, 0], [X1, AXIS_Y, 0], stroke_width=1.5, color=AXIS)
        chrome.add(axis)
        for tick in range(0, int(duration) + 1, 10):
            x = x_of(tick)
            chrome.add(Line([x, AXIS_Y, 0], [x, AXIS_Y - 0.1, 0], stroke_width=1.5, color=AXIS))
            chrome.add(
                Text(str(tick), font=FONT, font_size=13, color=INK_MUTED).move_to(
                    [x, AXIS_Y - 0.3, 0]
                )
            )
        chrome.add(
            Text("seconds", font=FONT, font_size=13, color=INK_MUTED).move_to(
                [X1 + 0.05, AXIS_Y - 0.3, 0], aligned_edge=LEFT
            )
        )
        caption = Text(
            "from reports/havid_dev_v8_sop_sheet/steps_val.csv and havid_dev_v10_sop_online/deviations_val.csv "
            f"(stream {pred_source}, confirmation 1 frame)",
            font=FONT,
            font_size=11,
            color=INK_MUTED,
        ).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(chrome), FadeIn(caption), run_time=0.6)

        # ---- step bars that grow with the play head --------------------------------------------
        clock = ValueTracker(0.0)

        def bar_geometry(source: str, step: StepBar) -> tuple[float, float]:
            layout = GROUPS[source]
            if step.hands == "lh+rh":
                return (layout["lh"] + layout["rh"]) / 2, 2 * LANE_H + (
                    layout["lh"] - layout["rh"] - LANE_H
                )
            return layout[step.hands], LANE_H

        def make_bar(source: str, step: StepBar):
            y, height = bar_geometry(source, step)
            now = clock.get_value()
            right = min(step.end_s, now)
            if right <= step.start_s:
                return VGroup()
            width = x_of(right) - x_of(step.start_s)
            return Rectangle(
                width=width,
                height=height - 0.04,
                stroke_color=SURFACE,
                stroke_width=2,
                fill_color=STEP_HUE,
                fill_opacity=0.9,
            ).move_to([x_of(step.start_s) + width / 2, y, 0])

        labels = VGroup()
        for source, steps in timeline.steps.items():
            for step in steps:
                self.add(always_redraw(lambda s=source, st=step: make_bar(s, st)))
                full_width = x_of(step.end_s) - x_of(step.start_s)
                if full_width >= 0.5:
                    y, _height = bar_geometry(source, step)
                    label = Text(step.label, font=FONT, font_size=11, color=INK).move_to(
                        [x_of(step.start_s) + full_width / 2, y, 0]
                    )
                    label.add_updater(
                        lambda m, end=step.end_s: m.set_opacity(
                            1.0 if clock.get_value() >= end else 0.0
                        )
                    )
                    labels.add(label)
        self.add(labels)
        head = always_redraw(
            lambda: Line(
                [x_of(clock.get_value()), 2.8, 0],
                [x_of(clock.get_value()), AXIS_Y, 0],
                stroke_width=1.5,
                color=INK_SECONDARY,
            )
        )
        self.add(head)

        # ---- play, popping alarms at their detection frame ------------------------------------
        counters: dict[str, Text] = {}
        counts = {"gt": 0, "pred": 0}

        def counter_text(source: str) -> Text:
            layout = GROUPS[source]
            return Text(f"{counts[source]} alarms", font=FONT, font_size=16, color=INK).move_to(
                [X1, layout["heading"], 0], aligned_edge=RIGHT
            )

        for source in GROUPS:
            counters[source] = counter_text(source)
            self.add(counters[source])

        events: list[tuple[float, str, AlarmMark, int]] = []
        for source, alarms in timeline.alarms.items():
            slots = stack_slots([a.detected_s for a in alarms], min_gap_s=duration * 0.07)
            events.extend(
                (a.detected_s, source, a, slot) for a, slot in zip(alarms, slots, strict=True)
            )
        events.sort(key=lambda e: e[0])
        speed = duration / ANIMATION_S
        now = 0.0
        index = 0
        while index < len(events):
            at = events[index][0]
            if at > now:
                self.play(
                    clock.animate.set_value(at), run_time=(at - now) / speed, rate_func=linear
                )
                now = at
            pops = []
            while index < len(events) and events[index][0] == at:
                _at, source, alarm, slot = events[index]
                style = alarm_style(alarm.kind)
                mark = glyph(alarm.kind).move_to(
                    [x_of(at), GROUPS[source]["marks"] + MARK_ROW_H * (slot % MARK_ROWS), 0]
                )
                text = Text(style.label, font=FONT, font_size=9, color=INK_SECONDARY).next_to(
                    mark, RIGHT, buff=0.06
                )
                pops.append(VGroup(mark, text))
                counts[source] += 1
                self.remove(counters[source])
                counters[source] = counter_text(source)
                self.add(counters[source])
                index += 1
            self.play(*[GrowFromCenter(p) for p in pops], run_time=0.3)
        if now < duration:
            self.play(
                clock.animate.set_value(duration),
                run_time=(duration - now) / speed,
                rate_func=linear,
            )
        summary = Text(
            f"ground truth: {counts['gt']} alarms · predicted stream: {counts['pred']} alarms — "
            "on val every predicted recording is flagged; the recogniser is the bottleneck",
            font=FONT,
            font_size=14,
            color=INK,
        ).next_to(caption, UP, buff=0.12)
        self.play(FadeIn(summary), run_time=0.5)
        self.wait(2.5)
