"""Learn an IndustReal step precedence graph from PSR labels and run the SOP checks on completions.

IndustReal ships no precedence graph, only ``procedure_info.json`` (step names and coarse
"expected in assembly / maintenance" flags). The graph used here is *learned from the training
recordings*: for two steps that co-occur in at least ``min_support`` training recordings of the
same kind (``assy`` / ``main``), an edge ``a -> b`` exists when ``a`` is completed before ``b`` in
every one of them (first occurrences; equal frames count as neither). Consistent orderings cannot
form a cycle, so the result is a DAG; it is transitively reduced for readability and exported in
the same JSON form as the HA-ViD graphs, so :func:`sop_monitor.sop_graph.check_order` runs on it
unchanged. Step ``k`` becomes node ``S<k>`` of type ``PrimitiveTask``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import permutations

from sop_monitor.industreal_psr import ProcedureStep
from sop_monitor.metrics.online import Completion
from sop_monitor.sop_graph import OrderReport, TaskGraph, check_order

KINDS: tuple[str, ...] = ("assy", "main")


def step_node(step: int) -> str:
    return f"S{step}"


def node_step(node: str) -> int:
    if not node.startswith("S") or not node[1:].isdigit():
        raise ValueError(f"not a step node: {node!r}")
    return int(node[1:])


def observed_order(completions: Sequence[Completion]) -> list[str]:
    """Step nodes in completion order (stable on equal frames, i.e. file/emission order)."""
    return [step_node(c.step) for c in sorted(completions, key=lambda c: c.frame)]


def _transitive_reduction(
    nodes: Sequence[str], edges: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    successors: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        successors[a].add(b)

    def reachable(start: str, skip: tuple[str, str]) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in successors[node]:
                if (node, nxt) == skip or nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        return seen

    return {(a, b) for a, b in edges if b not in reachable(a, (a, b))}


def learn_precedence(
    sequences: Sequence[Sequence[Completion]], min_support: int = 3, source: str = ""
) -> TaskGraph:
    """Precedence graph over the steps of ``sequences`` (one completion list per recording)."""
    firsts: list[dict[int, int]] = []
    for completions in sequences:
        first: dict[int, int] = {}
        for c in sorted(completions, key=lambda c: c.frame):
            first.setdefault(c.step, c.frame)
        firsts.append(first)
    steps = sorted({s for first in firsts for s in first})
    support: Counter[tuple[int, int]] = Counter()
    before: Counter[tuple[int, int]] = Counter()
    for first in firsts:
        for a, b in permutations(first, 2):
            support[a, b] += 1
            if first[a] < first[b]:
                before[a, b] += 1
    edges = {
        (step_node(a), step_node(b))
        for (a, b), n in support.items()
        if n >= min_support and before[a, b] == n
    }
    nodes = [step_node(s) for s in steps]
    reduced = _transitive_reduction(nodes, edges)
    graph = TaskGraph(
        nodes={node: "PrimitiveTask" for node in nodes},
        edges={"precedesPT": tuple(sorted(reduced))},
        source=source,
    )
    graph.topological_order("PT")  # raises on a cycle; impossible for consistent orderings
    return graph


def describe(graph: TaskGraph, steps: Sequence[ProcedureStep]) -> list[str]:
    """Human-readable edges ``a -> b`` using the reference step descriptions."""
    names = {s.id: s.description for s in steps}
    return [
        f"{names.get(node_step(a), a)} -> {names.get(node_step(b), b)}"
        for a, b in graph.relation("precedesPT")
    ]


def check_completions(graph: TaskGraph, completions: Sequence[Completion]) -> OrderReport:
    """Order / omission / unknown-step report of a completion sequence against the graph."""
    return check_order(graph, observed_order(completions), level="PT")


def sop_check_summary(
    graphs: Mapping[str, TaskGraph],
    kinds: Mapping[str, str],
    gt: Mapping[str, Sequence[Completion]],
    pred: Mapping[str, Sequence[Completion]],
) -> dict[str, object]:
    """Per-video SOP findings for ground truth and predictions plus a video-level agreement count.

    A video is "flagged" when its sequence has at least one precedence violation. Omissions are
    reported but not used for flagging: a learned graph contains every step any training
    recording of that kind performed, so partial procedures always omit something.
    """
    per_video: dict[str, dict[str, object]] = {}
    agreement: Counter[str] = Counter()
    for video in sorted(gt):
        graph = graphs[kinds[video]]
        g = check_completions(graph, gt[video])
        p = check_completions(graph, pred.get(video, []))
        gt_flag = bool(g.violations)
        pred_flag = bool(p.violations)
        key = {
            (True, True): "both_flag",
            (True, False): "only_gt_flags",
            (False, True): "only_pred_flags",
            (False, False): "neither_flags",
        }[gt_flag, pred_flag]
        agreement[key] += 1
        per_video[video] = {
            "kind": kinds[video],
            "gt_violations": g.violations,
            "gt_omissions": g.omissions,
            "gt_unknown": g.unknown,
            "pred_violations": p.violations,
            "pred_omissions": p.omissions,
            "pred_unknown": p.unknown,
            "agreement": key,
        }
    return {
        "graphs": {
            kind: {"nodes": len(g.nodes), "edges": len(g.relation("precedesPT"))}
            for kind, g in graphs.items()
        },
        "n_videos": len(per_video),
        "video_level": dict(agreement),
        "per_video": per_video,
    }
