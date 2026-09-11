"""Precedence graph learned from completion sequences and the SOP checks on completions."""

from __future__ import annotations

from pathlib import Path

import pytest

from sop_monitor.industreal_psr import PROCEDURE_INFO, load_procedure_info
from sop_monitor.industreal_sop import (
    check_completions,
    describe,
    learn_precedence,
    observed_order,
    sop_check_summary,
)
from sop_monitor.metrics.online import Completion
from sop_monitor.sop_graph import export_json, load_graph


def _seq(*pairs: tuple[int, int]) -> list[Completion]:
    return [Completion(frame, step) for frame, step in pairs]


def test_learned_graph_keeps_only_consistent_supported_orderings() -> None:
    sequences = [
        _seq((10, 3), (20, 6), (30, 9)),
        _seq((5, 3), (15, 6), (40, 9)),
        _seq((8, 3), (12, 6), (50, 9), (60, 12)),
        _seq((9, 6), (11, 3)),  # 6 before 3 once: no edge either way between 3 and 6
    ]
    graph = learn_precedence(sequences, min_support=3, source="test")
    assert graph.nodes == {
        "S3": "PrimitiveTask",
        "S6": "PrimitiveTask",
        "S9": "PrimitiveTask",
        "S12": "PrimitiveTask",
    }
    # 3 -> 9 and 6 -> 9 hold in all 3 supporting sequences; 3 <-> 6 is inconsistent; 12 lacks support.
    assert graph.relation("precedesPT") == (("S3", "S9"), ("S6", "S9"))
    # Kahn with string tie-breaking: "S12" sorts before "S3".
    assert graph.topological_order("PT") == ["S12", "S3", "S6", "S9"]


def test_transitive_reduction_drops_implied_edges() -> None:
    sequences = [_seq((1, 0), (2, 3), (3, 6))] * 3
    graph = learn_precedence(sequences, min_support=3)
    assert graph.relation("precedesPT") == (("S0", "S3"), ("S3", "S6"))  # S0 -> S6 is implied
    assert graph.ancestors("PT")["S6"] == frozenset({"S0", "S3"})


def test_checks_flag_out_of_order_and_omitted_steps(tmp_path: Path) -> None:
    graph = learn_precedence([_seq((1, 0), (2, 3), (3, 6))] * 3, source="t")
    ok = check_completions(graph, _seq((10, 0), (20, 3), (30, 6)))
    assert ok.ok and ok.omissions == []
    bad = check_completions(graph, _seq((10, 0), (20, 6)))
    assert bad.violations == [{"step": "S6", "index": 1, "missing_predecessors": ["S3"]}]
    assert bad.omissions == ["S3"]
    assert observed_order(_seq((20, 6), (10, 0))) == ["S0", "S6"]
    target = tmp_path / "g.json"
    export_json(graph, target)
    assert load_graph(target).relation("precedesPT") == graph.relation("precedesPT")


def test_descriptions_use_the_reference_step_names() -> None:
    graph = learn_precedence([_seq((1, 3), (2, 6))] * 3)
    assert describe(graph, load_procedure_info(PROCEDURE_INFO)) == [
        "Install front chassis -> Install front chassis pin"
    ]


def test_summary_counts_video_level_agreement() -> None:
    graph = learn_precedence([_seq((1, 0), (2, 3), (3, 6))] * 3)
    gt = {"a_assy": _seq((1, 0), (2, 3), (3, 6)), "b_assy": _seq((1, 0), (2, 6), (3, 3))}
    pred = {"a_assy": _seq((1, 0), (2, 6)), "b_assy": _seq((1, 0), (2, 6), (3, 3))}
    out = sop_check_summary({"assy": graph}, {"a_assy": "assy", "b_assy": "assy"}, gt, pred)
    assert out["video_level"] == {"only_pred_flags": 1, "both_flag": 1}
    assert out["per_video"]["b_assy"]["gt_violations"][0]["step"] == "S6"
    assert out["per_video"]["a_assy"]["pred_omissions"] == ["S3"]
    with pytest.raises(KeyError):
        sop_check_summary({"assy": graph}, {"a_assy": "main"}, {"a_assy": []}, {})
