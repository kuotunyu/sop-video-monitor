"""README hero: three cameras → frozen features → causal heads → fusion → online SOP monitor.

Render (see ``make figures``)::

    uv run --group figures manim -r 960,540 --fps 15 --format mp4 figures/pipeline.py PipelineScene

A mechanism diagram, not a result: the boxes are the components in ``src/sop_monitor`` and the
dots are frames flowing through them; the alarms that pop are illustrative.
"""

from __future__ import annotations

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    Indicate,
    LaggedStart,
    Line,
    MoveAlongPath,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    Triangle,
    VGroup,
    config,
    linear,
)

from sop_monitor.figures import (
    FONT,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    STEP_HUE,
    SURFACE,
    alarm_style,
)

config.background_color = SURFACE

VIEWS = ("side", "front", "top")
ROW_Y = (1.35, -0.15, -1.65)
MID_Y = -0.15


def box(text: str, width: float, height: float, colour: str, sub: str | None = None) -> VGroup:
    rect = RoundedRectangle(
        corner_radius=0.12,
        width=width,
        height=height,
        stroke_color=colour,
        stroke_width=2,
        fill_color=colour,
        fill_opacity=0.12,
    )
    label = Text(text, font=FONT, font_size=15, color=INK).move_to(rect)
    group = VGroup(rect, label)
    if sub:
        label.shift(UP * 0.14)
        group.add(
            Text(sub, font=FONT, font_size=11, color=INK_SECONDARY).next_to(label, DOWN, buff=0.08)
        )
    return group


def glyph(kind: str) -> VGroup:
    style = alarm_style(kind)
    colour = style.colour
    if style.glyph == "triangle":
        shape = Triangle(color=colour, fill_color=colour, fill_opacity=1, stroke_width=0).scale(
            0.13
        )
    elif style.glyph == "circle":
        shape = Circle(radius=0.1, color=colour, fill_color=colour, fill_opacity=1, stroke_width=0)
    else:
        shape = Square(
            side_length=0.18, color=colour, fill_color=colour, fill_opacity=1, stroke_width=0
        )
    return VGroup(shape)


def arrow(start: np.ndarray, end: np.ndarray, colour: str) -> Arrow:
    return Arrow(
        start, end, buff=0.06, stroke_width=2, max_tip_length_to_length_ratio=0.12, color=colour
    )


class PipelineScene(Scene):
    def construct(self) -> None:
        title = Text(
            "sop-video-monitor: three cameras → online SOP deviations",
            font=FONT,
            font_size=26,
            color=INK,
        ).to_edge(UP, buff=0.35)
        subtitle = Text(
            "development build on HA-ViD · every deviation is a suggestion for human review, "
            "never a verdict",
            font=FONT,
            font_size=14,
            color=INK_SECONDARY,
        ).next_to(title, DOWN, buff=0.12)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.6)

        cams = VGroup(
            *[
                box(f"{name} camera", 1.6, 0.9, SERIES[i], sub="15 fps").move_to([-5.75, y, 0])
                for i, (name, y) in enumerate(zip(VIEWS, ROW_Y, strict=True))
            ]
        )
        feats = box("frozen DINOv2", 2.3, 3.9, INK_MUTED, sub="ViT-B/14, 1536-d per frame")
        feats.move_to([-3.2, MID_Y, 0])
        heads = VGroup(
            *[
                box(
                    "causal MS-TCN++", 2.25, 0.9, SERIES[i], sub="output at t sees frames ≤ t"
                ).move_to([-0.2, y, 0])
                for i, y in enumerate(ROW_Y)
            ]
        )
        hands_note = Text(
            "one network per view and per hand (six in all)",
            font=FONT,
            font_size=11,
            color=INK_MUTED,
        ).next_to(heads, DOWN, buff=0.18)
        fusion = box("late fusion", 1.7, 1.2, STEP_HUE, sub="mean of the 3 views").move_to(
            [2.4, MID_Y, 0]
        )
        monitor_rect = RoundedRectangle(
            corner_radius=0.12,
            width=2.75,
            height=2.5,
            stroke_color=INK_SECONDARY,
            stroke_width=2,
            fill_color=INK_SECONDARY,
            fill_opacity=0.08,
        ).move_to([5.2, MID_Y, 0])
        monitor_title = Text("online SOP monitor", font=FONT, font_size=15, color=INK).move_to(
            monitor_rect.get_top() + DOWN * 0.3
        )
        rows = VGroup()
        for kind, text in (
            ("order", "order: learned precedence"),
            ("omission", "omission: mandatory steps"),
            ("duration_too_long", "duration: train windows"),
        ):
            row = VGroup(glyph(kind), Text(text, font=FONT, font_size=11, color=INK_SECONDARY))
            row.arrange(RIGHT, buff=0.12)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(monitor_rect).shift(DOWN * 0.15)
        monitor = VGroup(monitor_rect, monitor_title, rows)
        queue = Text(
            "→ deviation queue → human review", font=FONT, font_size=12, color=INK_SECONDARY
        ).next_to(monitor_rect, DOWN, buff=0.15)

        arrows_in = [
            arrow(cams[i].get_right(), [feats.get_left()[0], ROW_Y[i], 0], SERIES[i])
            for i in range(3)
        ]
        arrows_mid = [
            arrow([feats.get_right()[0], ROW_Y[i], 0], heads[i].get_left(), SERIES[i])
            for i in range(3)
        ]
        arrows_fuse = [
            arrow(
                heads[i].get_right(), fusion.get_left() + UP * (ROW_Y[i] - MID_Y) * 0.3, SERIES[i]
            )
            for i in range(3)
        ]
        arrow_out = arrow(fusion.get_right(), monitor_rect.get_left(), STEP_HUE)

        self.play(LaggedStart(*[FadeIn(c) for c in cams], lag_ratio=0.25), run_time=0.8)
        self.play(FadeIn(feats), *[Create(a) for a in arrows_in], run_time=0.8)
        self.play(
            LaggedStart(*[FadeIn(h) for h in heads], lag_ratio=0.25),
            *[Create(a) for a in arrows_mid],
            FadeIn(hands_note),
            run_time=0.9,
        )
        self.play(FadeIn(fusion), *[Create(a) for a in arrows_fuse], run_time=0.8)
        self.play(FadeIn(monitor), Create(arrow_out), FadeIn(queue), run_time=0.8)

        pops = {
            1: ("order", "order alarm at the frame the step is confirmed"),
            2: ("omission", "omission reported when the recording ends"),
        }
        note = None
        for pass_index in range(3):
            dots = [Dot(cams[i].get_right(), radius=0.07, color=SERIES[i]) for i in range(3)]
            self.add(*dots)
            for legs in (
                [Line(cams[i].get_right(), [feats.get_left()[0], ROW_Y[i], 0]) for i in range(3)],
                [Line([feats.get_right()[0], ROW_Y[i], 0], heads[i].get_left()) for i in range(3)],
                [
                    Line(heads[i].get_right(), fusion.get_left() + UP * (ROW_Y[i] - MID_Y) * 0.3)
                    for i in range(3)
                ],
            ):
                self.play(
                    *[MoveAlongPath(dots[i], legs[i]) for i in range(3)],
                    run_time=0.45,
                    rate_func=linear,
                )
            fused = Dot(fusion.get_right(), radius=0.08, color=INK)
            self.remove(*dots)
            self.add(fused)
            self.play(
                MoveAlongPath(fused, Line(fusion.get_right(), monitor_rect.get_left())),
                run_time=0.35,
                rate_func=linear,
            )
            self.remove(fused)
            if pass_index in pops:
                kind, text = pops[pass_index]
                row = rows[0 if kind == "order" else 1]
                if note is not None:
                    self.remove(note)
                note = Text(text, font=FONT, font_size=12, color=INK).next_to(
                    queue, DOWN, buff=0.12
                )
                self.play(
                    Indicate(row, scale_factor=1.12, color=alarm_style(kind).colour),
                    FadeIn(note),
                    run_time=0.5,
                )
        self.wait(1.8)
