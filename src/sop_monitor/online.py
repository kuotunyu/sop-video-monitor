"""Online SOP monitoring: frame-by-frame hand labels in, deviations out as soon as they are knowable.

:class:`OnlineSOPMonitor` is the streaming counterpart of :func:`sop_monitor.havid_sop.check_sequence`.
It receives, for every frame, one primitive-task label per hand (from the recogniser or the
annotations) and keeps the same state the offline checker reconstructs after the fact:

- per hand, a run-length segmenter that confirms a label once it has lasted ``min_frames`` frames
  and absorbs shorter blips into the previous label (the online form of
  :func:`sop_monitor.havid_sop.smooth_segments`);
- two-hand steps: a segment that starts while the other hand is still inside a segment with the
  same label joins that step (the online form of :func:`sop_monitor.havid_sop.step_sequence`);
- the observed steps, mapped to the knowledge's granularity, for the precedence check.

Deviations are emitted at the earliest frame they are certain: an **order** violation when a step
is confirmed before one of its learned predecessors was observed; **too_long** as soon as a running
step passes its upper bound (not when it ends); **too_short** when a step ends below its lower
bound; **unknown** the first time a step the plate never showed is confirmed; **omission** of a
mandatory step when the recording finishes. Each deviation carries the frame it was detected at, so
detection latency is measurable. With ``min_frames=1`` the set of findings on a finished recording
equals the offline checker's (``tests/test_online.py`` checks this on random sequences) with one
deliberate exception: a frame-level label stream cannot tell two adjacent segments with the same
label apart, so they are one step online (149 of 5,871 primitive-task boundaries in the HA-ViD
annotations are such pairs; a recogniser's output never contains them).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from sop_monitor.havid import NULL, WRONG
from sop_monitor.havid_sop import FPS, sheet_step
from sop_monitor.sop_graph import DurationBounds, TaskGraph

HANDS = ("lh", "rh")
NOT_STEPS = (NULL, WRONG)


@dataclass
class _Segmenter:
    """Run-length labels of one hand with minimum-duration confirmation."""

    min_frames: int
    confirmed: str | None = None
    confirmed_start: int = 0
    candidate: str | None = None
    candidate_start: int = 0
    candidate_length: int = 0

    def push(self, frame: int, label: str) -> list[tuple[str, int, int | None]]:
        """Return transitions ``(label, start, previous_end)``: a newly confirmed label starting at
        ``start`` whose predecessor ended at ``previous_end`` (``None`` for the first label)."""
        if self.confirmed is None:
            self.confirmed, self.confirmed_start = label, frame
            return [(label, frame, None)]
        if label == self.confirmed:
            self.candidate, self.candidate_length = None, 0
            return []
        if label == self.candidate:
            self.candidate_length += 1
        else:
            self.candidate, self.candidate_start, self.candidate_length = label, frame, 1
        if self.candidate_length >= self.min_frames:
            previous_end = self.candidate_start - 1
            self.confirmed, self.confirmed_start = self.candidate, self.candidate_start
            self.candidate, self.candidate_length = None, 0
            return [(self.confirmed, self.confirmed_start, previous_end)]
        return []


@dataclass
class OnlineStep:
    raw_label: str
    label: str
    start: int
    open_hands: set[str] = field(default_factory=set)
    hands: set[str] = field(default_factory=set)
    end: int | None = None
    flagged_long: bool = False


@dataclass(frozen=True)
class Deviation:
    kind: str
    step: str
    start_frame: int | None
    end_frame: int | None
    detected_at: int
    evidence: dict[str, object]

    @property
    def latency_s(self) -> float | None:
        return None if self.start_frame is None else (self.detected_at - self.start_frame) / FPS

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "step": self.step,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "detected_at": self.detected_at,
            "latency_s": self.latency_s,
            "evidence": self.evidence,
        }


class OnlineSOPMonitor:
    """Frame-by-frame SOP checks for one recording of one plate."""

    def __init__(
        self,
        graph: TaskGraph,
        bounds: Mapping[str, DurationBounds],
        mandatory: Iterable[str],
        granularity: str = "sheet",
        min_frames: int = 1,
    ) -> None:
        if min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        if granularity not in ("pt", "sheet"):
            raise ValueError("granularity must be pt or sheet")
        self.ancestors = graph.ancestors("PT")
        self.bounds = dict(bounds)
        self.mandatory = set(mandatory)
        self.granularity = granularity
        self.segmenters = {hand: _Segmenter(min_frames) for hand in HANDS}
        self.open_step: dict[str, OnlineStep | None] = {hand: None for hand in HANDS}
        self.steps: list[OnlineStep] = []
        self.seen: set[str] = set()
        self.unknown_seen: set[str] = set()
        self.deviations: list[Deviation] = []
        self.frame = -1
        self.finished = False

    def _map(self, raw: str) -> str:
        return sheet_step(raw) if self.granularity == "sheet" else raw

    def _emit(self, deviation: Deviation) -> None:
        self.deviations.append(deviation)
        self._new.append(deviation)

    def _close(self, hand: str, end: int) -> None:
        step = self.open_step[hand]
        if step is None:
            return
        self.open_step[hand] = None
        step.open_hands.discard(hand)
        step.end = end if step.end is None else max(step.end, end)
        if step.open_hands:
            return
        window = self.bounds.get(step.label)
        if window is None:
            return
        duration = (step.end - step.start + 1) / FPS
        if duration < window.low:
            self._emit(
                Deviation(
                    "duration",
                    step.label,
                    step.start,
                    step.end,
                    self.frame,
                    {
                        "kind": "too_short",
                        "duration_s": duration,
                        "window_s": [window.low, window.high],
                    },
                )
            )

    def _open(self, hand: str, raw: str, start: int) -> None:
        other = HANDS[1 - HANDS.index(hand)]
        joined = self.open_step[other]
        if joined is not None and joined.raw_label == raw and hand not in joined.hands:
            joined.open_hands.add(hand)
            joined.hands.add(hand)
            self.open_step[hand] = joined
            return
        step = OnlineStep(raw, self._map(raw), start, {hand}, {hand})
        self.open_step[hand] = step
        self.steps.append(step)
        self._opened.append(step)

    def _check_opened(self) -> None:
        """Order-check the steps confirmed on this frame against the steps seen before it: steps
        starting on the same frame are simultaneous and do not precede each other."""
        added: list[str] = []
        for step in self._opened:
            if step.label not in self.ancestors:
                if step.label not in self.unknown_seen:
                    self.unknown_seen.add(step.label)
                    self._emit(Deviation("unknown", step.label, step.start, None, self.frame, {}))
                continue
            missing = sorted(self.ancestors[step.label] - self.seen)
            if missing:
                self._emit(
                    Deviation(
                        "order",
                        step.label,
                        step.start,
                        None,
                        self.frame,
                        {"missing_predecessors": missing},
                    )
                )
            added.append(step.label)
        self.seen.update(added)
        self._opened = []

    def push(self, frame: int, labels: Mapping[str, str]) -> list[Deviation]:
        """Advance to ``frame`` (consecutive, from 0) with one label per hand; return new deviations."""
        if self.finished:
            raise RuntimeError("recording already finished")
        if frame != self.frame + 1:
            raise ValueError(f"expected frame {self.frame + 1}, got {frame}")
        self.frame = frame
        self._new: list[Deviation] = []
        self._opened: list[OnlineStep] = []
        transitions = {hand: self.segmenters[hand].push(frame, labels[hand]) for hand in HANDS}
        for hand in HANDS:  # close first, so a step that ends where another begins is complete
            for _label, _start, previous_end in transitions[hand]:
                if previous_end is not None:
                    self._close(hand, previous_end)
        for hand in HANDS:
            for label, start, _previous_end in transitions[hand]:
                if label not in NOT_STEPS:
                    self._open(hand, label, start)
        self._check_opened()
        for step in {id(s): s for s in self.open_step.values() if s is not None}.values():
            window = self.bounds.get(step.label)
            if window is None or step.flagged_long:
                continue
            duration = (frame - step.start + 1) / FPS
            if duration > window.high:
                step.flagged_long = True
                self._emit(
                    Deviation(
                        "duration",
                        step.label,
                        step.start,
                        None,
                        frame,
                        {
                            "kind": "too_long",
                            "duration_s": duration,
                            "window_s": [window.low, window.high],
                        },
                    )
                )
        return self._new

    def finish(self) -> list[Deviation]:
        """End of the recording: close open steps and report omitted mandatory steps."""
        if self.finished:
            return []
        self._new = []
        for hand in HANDS:
            self._close(hand, self.frame)
        for label in sorted(self.mandatory - self.seen):
            if label in self.ancestors:
                self._emit(Deviation("omission", label, None, None, self.frame, {}))
        self.finished = True
        return self._new

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for deviation in self.deviations:
            key = (
                deviation.kind
                if deviation.kind != "duration"
                else f"duration_{deviation.evidence['kind']}"
            )
            counts[key] = counts.get(key, 0) + 1
        return counts
