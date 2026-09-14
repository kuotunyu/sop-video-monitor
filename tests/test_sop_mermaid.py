"""Generated Mermaid of the learned knowledge: syntax shape, and the committed page is not stale."""

from __future__ import annotations

from pathlib import Path

from sop_monitor.sop_graph import NODE_TYPES, TaskGraph
from sop_monitor.sop_mermaid import node_id, plate_diagram, render_sop_knowledge

ROOT = Path(__file__).resolve().parents[1]


def test_plate_diagram_lists_every_node_edge_and_class() -> None:
    graph = TaskGraph(
        dict.fromkeys(("sftg", "igsft", "rgw", "odd-one"), NODE_TYPES["PT"]),
        {"precedesPT": (("sftg", "igsft"), ("igsft", "rgw"))},
    )
    text = plate_diagram("gear", graph, ["sftg", "rgw"])
    lines = text.splitlines()
    assert lines[0] == "flowchart LR"
    assert '    gear_odd_one["odd-one"]' in lines  # ids sanitised, labels kept
    assert "    gear_sftg --> gear_igsft" in lines and "    gear_igsft --> gear_rgw" in lines
    assert "    class gear_igsft,gear_odd_one step" in lines
    assert "    class gear_rgw,gear_sftg mandatory" in lines
    assert all("color:" in line for line in lines if "classDef" in line)
    assert node_id("general", "a+b c") == "general_a_b_c"


def test_committed_page_matches_the_knowledge_files() -> None:
    graphs_dir = ROOT / "sop" / "ha-vid" / "sheet"
    page = ROOT / "docs" / "sop_knowledge.md"
    assert page.read_text(encoding="utf-8") == render_sop_knowledge(graphs_dir), (
        "docs/sop_knowledge.md is stale: run `make sop-knowledge-doc`"
    )
