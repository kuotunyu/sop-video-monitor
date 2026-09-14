"""Three short concept animations for the docs: causal padding, look-ahead, ring-buffer policies.

Render (see ``make figures``)::

    uv run --group figures manim -r 960,540 --fps 15 --format mp4 figures/concepts.py CausalPaddingScene
    uv run --group figures manim -r 960,540 --fps 15 --format mp4 figures/concepts.py LookAheadScene
    uv run --group figures manim -r 960,540 --fps 15 --format mp4 figures/concepts.py RingBufferScene

The numbers quoted in ``LookAheadScene`` are the val results of ``reports/havid_dev_v9_step_recall``.
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    Scene,
    Square,
    Text,
    VGroup,
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
    SERIES,
    STATUS,
    STEP_HUE,
    STEP_HUE_SOFT,
    SURFACE,
)

config.background_color = SURFACE


def heading(text: str, sub: str | None = None) -> VGroup:
    title = Text(text, font=FONT, font_size=24, color=INK).to_edge(UP, buff=0.3)
    group = VGroup(title)
    if sub:
        group.add(
            Text(sub, font=FONT, font_size=14, color=INK_SECONDARY).next_to(title, DOWN, buff=0.12)
        )
    return group


def note(text: str, size: int = 13, colour: str = INK_SECONDARY) -> Text:
    return Text(text, font=FONT, font_size=size, color=colour)


# ---- 1. causal vs offline padding ------------------------------------------------------------------


class CausalPaddingScene(Scene):
    """Two stacked dilated layers (kernel 3, dilations 1 and 2): which input frames reach the output at t."""

    N, T = 13, 6
    CELL = 0.4

    def panel(
        self, x_centre: float, title: str, causal: bool
    ) -> tuple[VGroup, list[VGroup], list[VGroup]]:
        layers: list[list[Square]] = []
        rows = VGroup()
        for level, y in enumerate((-2.0, -0.9, 0.2)):
            cells = []
            for j in range(self.N):
                cell = Square(
                    side_length=self.CELL - 0.06, stroke_color=GRID, stroke_width=1, fill_opacity=0
                )
                cell.move_to([x_centre + (j - self.N // 2) * self.CELL, y, 0])
                cells.append(cell)
            layers.append(cells)
            rows.add(VGroup(*cells))
            rows.add(
                note(("input", "layer 1", "output")[level], 11, INK_MUTED).next_to(
                    rows[-1], LEFT, buff=0.12
                )
            )
        for j in range(self.N):
            rows.add(note(str(j), 10, INK_MUTED).next_to(layers[0][j], DOWN, buff=0.08))
        rows.add(note(title, 15, INK).move_to([x_centre, 0.95, 0]))
        offsets = ((-2, -1, 0), (-4, -2, 0)) if causal else ((-1, 0, 1), (-2, 0, 2))
        deps2 = [self.T + o for o in offsets[1]]
        deps1 = sorted({j + o for j in deps2 for o in offsets[0]})
        edges2 = [
            Line(
                layers[2][self.T].get_bottom(),
                layers[1][j].get_top(),
                stroke_width=1.5,
                color=INK_SECONDARY,
            )
            for j in deps2
        ]
        edges1 = [
            Line(layers[1][j].get_bottom(), layers[0][k].get_top(), stroke_width=1, color=INK_MUTED)
            for j in deps2
            for k in (j + o for o in offsets[0])
        ]
        return (
            rows,
            [
                VGroup(*[layers[2][self.T]]),
                VGroup(*[layers[1][j] for j in deps2]),
                VGroup(*[layers[0][k] for k in deps1]),
            ],
            [VGroup(*edges2), VGroup(*edges1)],
        )

    def construct(self) -> None:
        head = heading(
            "Causal MS-TCN++: the output at frame t only sees frames ≤ t",
            "two dilated layers shown (kernel 3, dilations 1 and 2); the real network stacks eleven",
        )
        self.play(FadeIn(head), run_time=0.5)
        left, highlights_l, edges_l = self.panel(-3.4, "offline (symmetric padding)", causal=False)
        right, highlights_r, edges_r = self.panel(3.4, "causal (left-only padding)", causal=True)
        self.play(FadeIn(left), FadeIn(right), run_time=0.6)
        self.play(
            highlights_l[0].animate.set_fill(STEP_HUE, opacity=0.9),
            highlights_r[0].animate.set_fill(STEP_HUE, opacity=0.9),
            run_time=0.4,
        )
        t_label_l = note("t", 11, INK).next_to(highlights_l[0], UP, buff=0.06)
        t_label_r = note("t", 11, INK).next_to(highlights_r[0], UP, buff=0.06)
        self.play(FadeIn(t_label_l), FadeIn(t_label_r), run_time=0.3)
        for level in (0, 1):
            self.play(
                Create(edges_l[level]),
                Create(edges_r[level]),
                highlights_l[level + 1].animate.set_fill(STEP_HUE_SOFT, opacity=0.7),
                highlights_r[level + 1].animate.set_fill(STEP_HUE_SOFT, opacity=0.7),
                run_time=0.8,
            )
        future = Rectangle(
            width=6 * self.CELL,
            height=0.5,
            stroke_width=0,
            fill_color=STATUS["critical"],
            fill_opacity=0.18,
        )
        future.move_to([3.4 + 3.5 * self.CELL, -2.0, 0])
        future_label = note("not yet seen", 11, STATUS["critical"]).next_to(future, DOWN, buff=0.35)
        self.play(FadeIn(future), FadeIn(future_label), run_time=0.5)
        summary = (
            VGroup(
                note(
                    "offline: frames t−3 … t+3 reach the output; needs the future, so it is not an online result",
                    13,
                    INK,
                ),
                note(
                    "causal: frames t−6 … t; deployable frame by frame, at a cost of 12–13 F1@10 on val (havid_dev_v7)",
                    13,
                    INK,
                ),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            .to_edge(DOWN, buff=0.35)
        )
        self.play(FadeIn(summary), run_time=0.5)
        self.wait(3.0)


# ---- 2. look-ahead as an output delay ---------------------------------------------------------------


class LookAheadScene(Scene):
    """Training a causal network to name frame t - L at time t: a fixed output delay of L frames."""

    X0, X1 = -6.0, 2.6
    STEPS = ((0.0, 3.0), (3.0, 4.5), (4.5, 8.0), (8.0, 10.0))
    DURATION = 10.0
    L_S = 3.0  # 45 frames at 15 fps

    def x_of(self, seconds: float) -> float:
        return self.X0 + (self.X1 - self.X0) * seconds / (self.DURATION + self.L_S)

    def bars(self, y: float, shift_s: float = 0.0, upto_s: float | None = None) -> VGroup:
        group = VGroup()
        for index, (start, end) in enumerate(self.STEPS):
            end_shown = end if upto_s is None else min(end, upto_s - shift_s)
            if end_shown <= start:
                continue
            width = self.x_of(end_shown + shift_s) - self.x_of(start + shift_s)
            bar = Rectangle(
                width=width,
                height=0.42,
                stroke_color=SURFACE,
                stroke_width=2,
                fill_color=STEP_HUE if index % 2 == 0 else STEP_HUE_SOFT,
                fill_opacity=0.9,
            )
            bar.move_to([self.x_of(start + shift_s) + width / 2, y, 0])
            group.add(bar)
            if end_shown == end:
                group.add(note(f"step {index + 1}", 11, INK).move_to(bar))
        return group

    def construct(self) -> None:
        head = heading(
            "Look-ahead L: a causal network trained to name frame t − L at time t",
            "the label of every frame arrives L frames later; nothing else changes, so metrics stay frame-aligned",
        )
        self.play(FadeIn(head), run_time=0.5)
        rows = (
            ("true steps", 1.6),
            ("causal output, L = 0", 0.6),
            ("causal output, L = 45 frames (3 s)", -0.4),
        )
        for label, y in rows:
            self.add(note(label, 12, INK_MUTED).move_to([self.X0, y + 0.4, 0], aligned_edge=LEFT))
        axis_y = -1.1
        self.add(
            Line(
                [self.x_of(0), axis_y, 0],
                [self.x_of(self.DURATION + self.L_S), axis_y, 0],
                stroke_width=1.5,
                color=AXIS,
            )
        )
        for tick in range(0, 14, 2):
            self.add(
                Line(
                    [self.x_of(tick), axis_y, 0],
                    [self.x_of(tick), axis_y - 0.1, 0],
                    stroke_width=1.5,
                    color=AXIS,
                )
            )
            self.add(note(str(tick), 10, INK_MUTED).move_to([self.x_of(tick), axis_y - 0.28, 0]))
        self.add(
            note("seconds", 10, INK_MUTED).move_to(
                [self.x_of(self.DURATION + self.L_S) + 0.1, axis_y - 0.28, 0], aligned_edge=LEFT
            )
        )
        self.add(self.bars(1.6))
        from manim import ValueTracker, always_redraw

        clock = ValueTracker(0.0)
        self.add(always_redraw(lambda: self.bars(0.6, 0.0, clock.get_value())))
        self.add(always_redraw(lambda: self.bars(-0.4, self.L_S, clock.get_value())))
        self.add(
            always_redraw(
                lambda: Line(
                    [self.x_of(clock.get_value()), 1.9, 0],
                    [self.x_of(clock.get_value()), axis_y, 0],
                    stroke_width=1.5,
                    color=INK_SECONDARY,
                )
            )
        )
        self.play(clock.animate.set_value(self.DURATION + self.L_S), run_time=6.0, rate_func=linear)
        delay = Arrow(
            [self.x_of(0.0), -0.4, 0],
            [self.x_of(3.0), -0.4, 0],
            buff=0,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.2,
            color=STATUS["warning"],
        )
        delay_label = note("output delay L = 3 s", 11, STATUS["warning"]).next_to(
            delay, DOWN, buff=0.08
        )
        self.play(Create(delay), FadeIn(delay_label), run_time=0.6)

        # what it buys: val mandatory-step frame recall (reports/havid_dev_v9_step_recall)
        values = (
            ("L = 0", 32.8),
            ("L = 15", 45.0),
            ("L = 45", 49.7),
            ("L = 90", 43.4),
            ("offline", 55.8),
        )
        chart = VGroup()
        base_x, base_y, bar_w, gap, scale = 3.6, -1.1, 0.5, 0.18, 0.045
        for index, (label, value) in enumerate(values):
            x = base_x + index * (bar_w + gap)
            online = label != "offline"
            bar = Rectangle(
                width=bar_w,
                height=value * scale,
                stroke_color=INK_MUTED if not online else SURFACE,
                stroke_width=2 if not online else 0,
                fill_color=STEP_HUE,
                fill_opacity=0.9 if online else 0.0,
            )
            bar.move_to([x, base_y + value * scale / 2, 0])
            chart.add(
                bar,
                note(f"{value:.0f}", 10, INK).next_to(bar, UP, buff=0.05),
                note(label, 9, INK_MUTED).next_to(bar, DOWN, buff=0.06),
            )
        chart.add(
            Line(
                [base_x - bar_w, base_y, 0],
                [base_x + 4 * (bar_w + gap) + bar_w, base_y, 0],
                stroke_width=1.5,
                color=AXIS,
            )
        )
        chart_title = note("val: mandatory-step frames recognised (%)", 11, INK_SECONDARY).move_to(
            [base_x + 2 * (bar_w + gap), 1.9, 0]
        )
        self.play(FadeIn(chart_title), FadeIn(chart), run_time=0.7)
        summary = note(
            "3 s of delay recovers most of the gap to offline; 6 s is worse than 3 s (havid_dev_v9). Test: L = 45 also best.",
            13,
            INK,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(summary), run_time=0.5)
        self.wait(3.0)


# ---- 3. ring buffer backpressure -------------------------------------------------------------------


class RingBufferScene(Scene):
    """A producer faster than the consumer: WAIT blocks the camera thread, DROP_OLDEST skips frames."""

    CAPACITY = 6
    TICKS = 14

    def panel(self, y: float, title: str) -> tuple[VGroup, list[Square]]:
        slots = [
            Square(side_length=0.5, stroke_color=GRID, stroke_width=1.5, fill_opacity=0).move_to(
                [-1.5 + k * 0.6, y, 0]
            )
            for k in range(self.CAPACITY)
        ]
        group = VGroup(*slots)
        group.add(note(title, 15, INK).move_to([-4.6, y + 0.55, 0], aligned_edge=LEFT))
        group.add(
            note("decode thread, 15 fps", 11, INK_MUTED).move_to([-4.6, y, 0], aligned_edge=LEFT)
        )
        group.add(
            Arrow(
                [-2.7, y, 0],
                [-1.95, y, 0],
                buff=0,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.25,
                color=INK_MUTED,
            )
        )
        group.add(
            Arrow(
                [1.95, y, 0],
                [2.7, y, 0],
                buff=0,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.25,
                color=INK_MUTED,
            )
        )
        group.add(note("consumer (slower)", 11, INK_MUTED).move_to([2.85, y, 0], aligned_edge=LEFT))
        return group, slots

    def construct(self) -> None:
        head = heading(
            "Backpressure per camera: a ring buffer of 16 slots, here 6",
            "the consumer (DINOv2 batches) is slower than the producer (decode); two of the three policies",
        )
        self.play(FadeIn(head), run_time=0.5)
        panels = {}
        for policy, y in (("WAIT", 1.2), ("DROP_OLDEST", -1.3)):
            group, slots = self.panel(y, policy)
            panels[policy] = {"slots": slots, "dots": [], "y": y, "counter": 0}
            self.add(group)
        counters = {
            "WAIT": note("producer stalls: 0 (frames arrive late)", 12, STATUS["warning"]).move_to(
                [-4.6, 1.2 - 0.6, 0], aligned_edge=LEFT
            ),
            "DROP_OLDEST": note(
                "skipped_frames: 0 (the newest frame always gets in)", 12, STATUS["warning"]
            ).move_to([-4.6, -1.3 - 0.6, 0], aligned_edge=LEFT),
        }
        for text in counters.values():
            self.add(text)
        frame_id = {"WAIT": 0, "DROP_OLDEST": 0}
        for tick in range(self.TICKS):
            anims = []
            for policy, state in panels.items():
                slots, dots, y = state["slots"], state["dots"], state["y"]
                colour = SERIES[frame_id[policy] % 3]
                frame_id[policy] += 1
                if len(dots) == self.CAPACITY:
                    if policy == "WAIT":
                        state["counter"] += 1
                        new = counters[policy]
                        replacement = note(
                            f"producer stalls: {state['counter']} (frames arrive late)",
                            12,
                            STATUS["warning"],
                        ).move_to(new, aligned_edge=LEFT)
                        self.remove(new)
                        counters[policy] = replacement
                        self.add(replacement)
                        continue
                    oldest = dots.pop(0)
                    state["counter"] += 1
                    anims.append(FadeOut(oldest, shift=DOWN * 0.3))
                    for k, dot in enumerate(dots):
                        anims.append(dot.animate.move_to(slots[k]))
                    replacement = note(
                        f"skipped_frames: {state['counter']} (the newest frame always gets in)",
                        12,
                        STATUS["warning"],
                    ).move_to(counters[policy], aligned_edge=LEFT)
                    self.remove(counters[policy])
                    counters[policy] = replacement
                    self.add(replacement)
                dot = Dot([-2.3, y, 0], radius=0.14, color=colour)
                self.add(dot)
                dots.append(dot)
                anims.append(dot.animate.move_to(slots[len(dots) - 1]))
            self.play(*anims, run_time=0.28, rate_func=linear)
            if tick % 2 == 1:  # the consumer takes one frame every second tick
                anims = []
                for state in panels.values():
                    if not state["dots"]:
                        continue
                    taken = state["dots"].pop(0)
                    anims.append(FadeOut(taken, shift=RIGHT * 0.8))
                    for k, dot in enumerate(state["dots"]):
                        anims.append(dot.animate.move_to(state["slots"][k]))
                self.play(*anims, run_time=0.22, rate_func=linear)
        summary = note(
            "WAIT keeps every frame but the camera falls behind; DROP_OLDEST keeps the buffer fresh and counts what it skipped. ADAPTIVE raises the stride instead.",
            12,
            INK,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(summary), run_time=0.5)
        self.wait(2.5)
