"""The typer entry point imports and its W0 placeholders behave as documented."""

from pathlib import Path

from typer.testing import CliRunner

from sop_monitor.cli import NOT_YET_EXIT_CODE, app

runner = CliRunner()


def test_reproduce_lite_reports_nothing_to_rebuild(tmp_path: Path) -> None:
    result = runner.invoke(app, ["reproduce-lite", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "not yet" in result.output


def test_reproduce_lite_refuses_until_regeneration_exists(tmp_path: Path) -> None:
    (tmp_path / "tas_offline.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["reproduce-lite", "--reports-dir", str(tmp_path)])
    assert result.exit_code == NOT_YET_EXIT_CODE


def test_ha_vid_split_exits_not_yet() -> None:
    result = runner.invoke(app, ["freeze-splits", "--dataset", "ha-vid"])
    assert result.exit_code == NOT_YET_EXIT_CODE
    assert "not yet" in result.output


def test_check_sop_exports_and_judges_a_sequence(tmp_path: Path) -> None:
    ns = "http://example.org/onto#"
    owl = tmp_path / "graph.owl"
    owl.write_text(
        f'<?xml version="1.0"?><rdf:RDF xmlns="{ns}" xmlns:owl="http://www.w3.org/2002/07/owl#"'
        ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<owl:NamedIndividual rdf:about="{ns}PT1"><rdf:type rdf:resource="{ns}PrimitiveTask"/>'
        f'<precedesPT rdf:resource="{ns}PT2"/></owl:NamedIndividual>'
        f'<owl:NamedIndividual rdf:about="{ns}PT2"><rdf:type rdf:resource="{ns}PrimitiveTask"/>'
        "</owl:NamedIndividual></rdf:RDF>",
        encoding="utf-8",
    )
    exported = tmp_path / "graph.json"
    result = runner.invoke(app, ["export-sop", "--owl", str(owl), "--out", str(exported)])
    assert result.exit_code == 0, result.output
    assert exported.is_file()
    good = tmp_path / "good.txt"
    good.write_text("PT1\nPT2\n", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("PT2\n", encoding="utf-8")
    assert (
        runner.invoke(app, ["check-sop", "--graph", str(exported), "--steps", str(good)]).exit_code
        == 0
    )
    result = runner.invoke(app, ["check-sop", "--graph", str(exported), "--steps", str(bad)])
    assert result.exit_code == 1
    assert '"omissions": ["PT1"]' in result.output


def test_freeze_splits_industreal_needs_labels_dir_and_freezes(tmp_path: Path) -> None:
    result = runner.invoke(app, ["freeze-splits"])
    assert result.exit_code == 1
    assert "--labels-dir" in result.output
    labels = tmp_path / "labels"
    labels.mkdir()
    for name, participant in (("train", "01"), ("val", "02"), ("test", "03")):
        (labels / f"{name}.csv").write_text(
            f"{participant}_assy_0_1,1,a,000001.jpg,000002.jpg\n", encoding="utf-8"
        )
    out = tmp_path / "splits"
    result = runner.invoke(
        app, ["freeze-splits", "--labels-dir", str(labels), "--out-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "test_sha256:" in result.output
    assert (out / "test_sha256.txt").is_file()
