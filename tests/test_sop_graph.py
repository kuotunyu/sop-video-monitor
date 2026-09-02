"""Precedence-graph loading and SOP order / omission / duration checks (spec 6.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sop_monitor.sop_graph import (
    DurationBounds,
    TaskGraph,
    check_durations,
    check_order,
    export_json,
    load_graph,
    parse_owl,
)

NS = "http://example.org/onto#"
OWL_DOC = f"""<?xml version="1.0"?>
<rdf:RDF xmlns="{NS}" xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <owl:NamedIndividual rdf:about="{NS}PT1">
    <rdf:type rdf:resource="{NS}PrimitiveTask"/>
    <decomposeToAA rdf:resource="{NS}AA1_PT1"/>
    <precedesPT rdf:resource="{NS}PT2"/>
    <precedesPT rdf:resource="{NS}PT3"/>
  </owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="{NS}PT2">
    <rdf:type rdf:resource="{NS}PrimitiveTask"/>
    <precedesPT rdf:resource="{NS}PT4"/>
  </owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="{NS}PT3">
    <rdf:type rdf:resource="{NS}PrimitiveTask"/>
    <precedesPT rdf:resource="{NS}PT4"/>
  </owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="{NS}PT4">
    <rdf:type rdf:resource="{NS}PrimitiveTask"/>
  </owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="{NS}AA1_PT1">
    <rdf:type rdf:resource="{NS}AtomicAction"/>
    <precedesAA rdf:resource="{NS}AA2_PT1"/>
  </owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="{NS}AA2_PT1">
    <rdf:type rdf:resource="{NS}AtomicAction"/>
  </owl:NamedIndividual>
  <owl:NamedIndividual rdf:about="{NS}Bolt">
    <rdf:type rdf:resource="{NS}Part"/>
  </owl:NamedIndividual>
</rdf:RDF>
"""


@pytest.fixture
def graph() -> TaskGraph:
    return parse_owl(OWL_DOC, source="test.owl")


def test_parse_reads_typed_nodes_and_relations(graph: TaskGraph) -> None:
    assert graph.nodes["PT1"] == "PrimitiveTask"
    assert graph.nodes["AA1_PT1"] == "AtomicAction"
    assert graph.nodes["Bolt"] == "Part"
    assert graph.relation("precedesPT") == (
        ("PT1", "PT2"),
        ("PT1", "PT3"),
        ("PT2", "PT4"),
        ("PT3", "PT4"),
    )
    assert graph.relation("decomposeToAA") == (("PT1", "AA1_PT1"),)
    assert graph.nodes_of("PT") == ["PT1", "PT2", "PT3", "PT4"]


def test_topological_order_and_ancestors(graph: TaskGraph) -> None:
    assert graph.topological_order("PT") == ["PT1", "PT2", "PT3", "PT4"]
    assert graph.ancestors("PT")["PT4"] == frozenset({"PT1", "PT2", "PT3"})
    assert graph.ancestors("AA")["AA2_PT1"] == frozenset({"AA1_PT1"})


def test_cycle_is_rejected() -> None:
    cyclic = OWL_DOC.replace(
        f'<owl:NamedIndividual rdf:about="{NS}PT4">\n    <rdf:type rdf:resource="{NS}PrimitiveTask"/>',
        f'<owl:NamedIndividual rdf:about="{NS}PT4">\n    <rdf:type rdf:resource="{NS}PrimitiveTask"/>\n    <precedesPT rdf:resource="{NS}PT1"/>',
    )
    with pytest.raises(ValueError, match="cycle"):
        parse_owl(cyclic).topological_order("PT")


def test_parse_rejects_documents_without_individuals() -> None:
    with pytest.raises(ValueError):
        parse_owl('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>')


def test_order_check_flags_missing_predecessors_omissions_and_unknowns(graph: TaskGraph) -> None:
    report = check_order(graph, ["PT1", "PT4", "PT2", "PT9"], level="PT")
    assert not report.ok
    assert report.violations == [
        {"step": "PT4", "index": 1, "missing_predecessors": ["PT2", "PT3"]}
    ]
    assert report.omissions == ["PT3"]
    assert report.unknown == ["PT9"]
    assert report.repeated == []


def test_order_check_passes_a_valid_sequence_and_reports_repeats(graph: TaskGraph) -> None:
    report = check_order(graph, ["PT1", "PT3", "PT2", "PT2", "PT4"], level="PT")
    assert report.ok
    assert report.repeated == ["PT2"]
    assert report.to_dict()["ok"] is True


def test_json_export_round_trips(graph: TaskGraph, tmp_path: Path) -> None:
    target = tmp_path / "sop" / "test.json"
    export_json(graph, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["source"] == "test.owl"
    restored = load_graph(target)
    assert restored.nodes == graph.nodes
    assert restored.edges == graph.edges
    assert check_order(restored, ["PT1", "PT2", "PT3", "PT4"]).ok


def test_duration_bounds_flag_only_out_of_window_steps() -> None:
    bounds = {"PT1": DurationBounds(2.0, 5.0), "PT2": DurationBounds(1.0, 3.0)}
    findings = check_durations([("PT1", 1.0), ("PT2", 2.0), ("PT2", 9.0), ("PT7", 100.0)], bounds)
    assert findings == [
        {"step": "PT1", "index": 0, "kind": "too_short", "duration": 1.0},
        {"step": "PT2", "index": 2, "kind": "too_long", "duration": 9.0},
    ]
