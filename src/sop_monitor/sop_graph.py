"""Task precedence graph and the three SOP checks (spec sections 5.1 "SOP engine" and 6.4).

HA-ViD ships a subject-agnostic task precedence graph as an OWL ontology. W1 converts
it to ``sop/graph.json`` (nodes = primitive tasks, edges = must-precede constraints);
W3 drives a state machine from that file with three checks:

- precedence: all prerequisites of a step are complete before the step is entered
- duration:   a step's elapsed time stays inside train-split q05-q95 bounds (proposed)
- omission:   a step still missing after the largest gap observed in train (proposed)

Deviations are suggestions and always go through human review (spec 2.2).
W0 provides typed stubs only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class DeviationType(StrEnum):
    """Deviation vocabulary shared by the SOP engine, the queue and the synthetic generator."""

    ORDER = "order"
    DURATION = "duration"
    OMISSION = "omission"


@dataclass(frozen=True)
class Deviation:
    """One SOP deviation event (spec 6.4).

    Only the fields the state machine itself produces live here; evidence windows,
    per-view scores and the VLM opinion are attached by the queue layer (W5).
    """

    t: float
    type: DeviationType
    step: str
    detail: str = ""


@dataclass
class SopGraph:
    """Directed precedence graph: ``(a, b) in edges`` means step ``a`` must finish before ``b``."""

    steps: tuple[str, ...]
    edges: frozenset[tuple[str, str]]
    duration_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_owl(cls, path: Path) -> SopGraph:
        """Load the HA-ViD precedence ontology (OWL) and keep only primitive-task nodes."""
        raise NotImplementedError("W1: OWL precedence graph conversion (spec 3.4 item 4)")

    @classmethod
    def from_json(cls, path: Path) -> SopGraph:
        """Load the committed ``sop/graph.json``."""
        raise NotImplementedError(
            "W1: sop/graph.json schema is fixed together with the OWL converter"
        )

    def to_json(self, path: Path) -> None:
        """Write ``sop/graph.json`` (nodes, edges, optional duration bounds)."""
        raise NotImplementedError(
            "W1: sop/graph.json schema is fixed together with the OWL converter"
        )

    def prerequisites(self, step: str) -> frozenset[str]:
        """All steps that must be complete before ``step`` may start."""
        raise NotImplementedError("W3: SOP state machine (spec 6.4)")

    def check_precedence(
        self, completed: Sequence[str], entering: str, t: float
    ) -> Deviation | None:
        """ORDER deviation when ``entering`` starts with an unfinished prerequisite."""
        raise NotImplementedError("W3: SOP state machine (spec 6.4)")

    def check_duration(self, step: str, elapsed_s: float, t: float) -> Deviation | None:
        """DURATION deviation when ``elapsed_s`` leaves the step's train-split bounds."""
        raise NotImplementedError("W3: duration bounds from train split (spec 6.4)")

    def check_omission(self, completed: Sequence[str], gap_s: float, t: float) -> list[Deviation]:
        """OMISSION deviations for steps still missing after the largest train-split gap."""
        raise NotImplementedError("W3: omission rule (spec 6.4)")
