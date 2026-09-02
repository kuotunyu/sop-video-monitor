"""SOP precedence graph: load HA-ViD's subject-agnostic task precedence graphs, check order.

HA-ViD ships one OWL/RDF file per plate (cylinder, gear, general) built on the HR-SAT
ontology. The parts that matter for SOP monitoring are:

- ``PrimitiveTask`` individuals (``PT1`` ...) with ``precedesPT`` edges and
  ``decomposeToAA`` edges to ``AtomicAction`` individuals (``AA3_PT7`` ...);
- ``AtomicAction`` individuals with ``precedesAA`` edges;
- ``Task`` individuals with ``decomposeToPT`` edges.

Everything is read with :mod:`xml.etree`; no ontology library is required. Identifiers are
the URI fragments after ``#``. The graphs are data from HA-ViD (CC BY-NC 4.0) and any JSON
export derived from them must keep that attribution (see ``sop/ha-vid/NOTICE.md``).
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
PRECEDENCE_RELATIONS: dict[str, str] = {"PT": "precedesPT", "AA": "precedesAA"}
DECOMPOSITION_RELATIONS: tuple[str, ...] = ("decomposeToPT", "decomposeToAA")
NODE_TYPES: dict[str, str] = {"PT": "PrimitiveTask", "AA": "AtomicAction"}


def _local(uri: str) -> str:
    """``http://x/y#PT7`` -> ``PT7``; ``{http://ns#}precedesAA`` -> ``precedesAA``.

    The brace form is checked first: ElementTree tags look like ``{namespace}tag`` and these
    namespaces themselves end in ``#``, so splitting on ``#`` first would yield ``}tag``.
    """
    if "}" in uri:
        return uri.rsplit("}", 1)[1]
    if "#" in uri:
        return uri.rsplit("#", 1)[1]
    return uri.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class TaskGraph:
    """Typed nodes plus directed edges grouped by relation name."""

    nodes: Mapping[str, str]
    edges: Mapping[str, tuple[tuple[str, str], ...]]
    source: str = ""

    def relation(self, name: str) -> tuple[tuple[str, str], ...]:
        return self.edges.get(name, ())

    def nodes_of(self, level: str) -> list[str]:
        wanted = NODE_TYPES[level]
        return sorted(node for node, kind in self.nodes.items() if kind == wanted)

    def successors(self, level: str) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {node: [] for node in self.nodes_of(level)}
        for head, tail in self.relation(PRECEDENCE_RELATIONS[level]):
            adjacency.setdefault(head, []).append(tail)
            adjacency.setdefault(tail, [])
        return {node: sorted(set(targets)) for node, targets in adjacency.items()}

    def predecessors(self, level: str) -> dict[str, list[str]]:
        reverse: dict[str, set[str]] = {node: set() for node in self.successors(level)}
        for head, tail in self.relation(PRECEDENCE_RELATIONS[level]):
            reverse.setdefault(tail, set()).add(head)
            reverse.setdefault(head, set())
        return {node: sorted(heads) for node, heads in reverse.items()}

    def topological_order(self, level: str) -> list[str]:
        """Kahn's algorithm with deterministic tie-breaking; raises on a cycle."""
        successors = self.successors(level)
        indegree = {node: 0 for node in successors}
        for targets in successors.values():
            for target in targets:
                indegree[target] += 1
        ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while ready:
            node = ready.popleft()
            order.append(node)
            for target in successors[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
            ready = deque(sorted(ready))
        if len(order) != len(successors):
            raise ValueError(f"precedence graph at level {level} contains a cycle")
        return order

    def ancestors(self, level: str) -> dict[str, frozenset[str]]:
        """Transitive predecessors per node (everything that must happen before it)."""
        predecessors = self.predecessors(level)
        closure: dict[str, frozenset[str]] = {}
        for node in self.topological_order(level):
            direct = predecessors[node]
            closure[node] = frozenset(direct).union(*(closure[p] for p in direct))
        return closure

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "nodes": dict(sorted(self.nodes.items())),
            "edges": {
                name: [list(edge) for edge in edges] for name, edges in sorted(self.edges.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TaskGraph:
        edges_raw = payload.get("edges", {})
        assert isinstance(edges_raw, Mapping)
        edges = {
            str(name): tuple((str(head), str(tail)) for head, tail in pairs)  # type: ignore[union-attr]
            for name, pairs in edges_raw.items()
        }
        nodes_raw = payload.get("nodes", {})
        assert isinstance(nodes_raw, Mapping)
        return cls(
            nodes={str(k): str(v) for k, v in nodes_raw.items()},
            edges=edges,
            source=str(payload.get("source", "")),
        )


def parse_owl(text: str, source: str = "") -> TaskGraph:
    """Parse an HR-SAT style OWL/RDF document into a :class:`TaskGraph`."""
    root = ElementTree.fromstring(text)
    nodes: dict[str, str] = {}
    edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for individual in root.iter(f"{{{OWL}}}NamedIndividual"):
        about = individual.get(f"{{{RDF}}}about")
        if not about:
            continue
        node = _local(about)
        kinds = [
            _local(child.get(f"{{{RDF}}}resource", ""))
            for child in individual
            if _local(child.tag) == "type"
        ]
        nodes[node] = kinds[0] if kinds else "Unknown"
        for child in individual:
            relation = _local(child.tag)
            target = child.get(f"{{{RDF}}}resource")
            if target and relation in (*PRECEDENCE_RELATIONS.values(), *DECOMPOSITION_RELATIONS):
                edges[relation].append((node, _local(target)))
    if not nodes:
        raise ValueError("no owl:NamedIndividual found; is this an HR-SAT OWL file?")
    return TaskGraph(
        nodes=nodes,
        edges={name: tuple(sorted(set(pairs))) for name, pairs in edges.items()},
        source=source,
    )


def load_owl(path: Path) -> TaskGraph:
    return parse_owl(path.read_text(encoding="utf-8"), source=path.name)


def load_graph(path: Path) -> TaskGraph:
    """Load either an OWL file or a JSON export written by :func:`export_json`."""
    if path.suffix.lower() == ".json":
        return TaskGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return load_owl(path)


def export_json(graph: TaskGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class OrderReport:
    """Result of checking an observed step sequence against the precedence graph."""

    level: str
    observed: list[str]
    violations: list[dict[str, object]] = field(default_factory=list)
    omissions: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    repeated: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.violations or self.omissions or self.unknown)

    def to_dict(self) -> dict[str, object]:
        return {
            "level": self.level,
            "observed": self.observed,
            "ok": self.ok,
            "violations": self.violations,
            "omissions": self.omissions,
            "unknown": self.unknown,
            "repeated": self.repeated,
        }


def check_order(graph: TaskGraph, observed: Sequence[str], level: str = "PT") -> OrderReport:
    """Flag steps performed before their transitive predecessors, omitted steps, unknown ids.

    A violation records the step, its position, and the predecessors that had not been
    observed at that point. Omissions are graph nodes never observed. Repeats are reported
    separately and are not violations by themselves (some SOPs allow re-doing a step).
    """
    ancestors = graph.ancestors(level)
    seen: set[str] = set()
    report = OrderReport(level=level, observed=list(observed))
    for index, step in enumerate(observed):
        if step not in ancestors:
            report.unknown.append(step)
            continue
        if step in seen:
            report.repeated.append(step)
        missing = sorted(ancestors[step] - seen)
        if missing:
            report.violations.append(
                {"step": step, "index": index, "missing_predecessors": missing}
            )
        seen.add(step)
    report.omissions = sorted(node for node in ancestors if node not in seen)
    return report


@dataclass(frozen=True)
class DurationBounds:
    """Per-step [low, high] duration window in seconds (from train-split quantiles, spec 6.4)."""

    low: float
    high: float


def check_durations(
    steps: Iterable[tuple[str, float]], bounds: Mapping[str, DurationBounds]
) -> list[dict[str, object]]:
    """Return one finding per step whose duration falls outside its bounds."""
    findings: list[dict[str, object]] = []
    for index, (step, duration) in enumerate(steps):
        window = bounds.get(step)
        if window is None:
            continue
        if duration < window.low:
            findings.append(
                {"step": step, "index": index, "kind": "too_short", "duration": duration}
            )
        elif duration > window.high:
            findings.append(
                {"step": step, "index": index, "kind": "too_long", "duration": duration}
            )
    return findings
