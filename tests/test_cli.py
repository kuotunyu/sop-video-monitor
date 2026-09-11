"""The typer entry point: data-free commands work, data commands are honest about what is missing."""

import json
from pathlib import Path

from typer.testing import CliRunner

from sop_monitor.baseline import PredictionRow, render_tables, score_predictions, write_predictions
from sop_monitor.cli import NOT_YET_EXIT_CODE, app

runner = CliRunner()


def test_reproduce_lite_reports_nothing_to_rebuild(tmp_path: Path) -> None:
    result = runner.invoke(app, ["reproduce-lite", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "nothing to rebuild" in result.output


def _write_run(run_dir: Path, tamper: bool = False) -> None:
    rows = [
        PredictionRow("a_1", "a", f, g, {"frame": p})
        for f, (g, p) in enumerate([(1, 1), (1, 1), (2, 1), (-1, -1)])
    ] + [PredictionRow("b_1", "b", f, g, {"frame": g}) for f, g in enumerate([3, 3, -1])]
    run_dir.mkdir(parents=True)
    write_predictions(run_dir / "predictions_val.csv", rows)
    metrics = score_predictions(rows, n_boot=10, seed=1)
    if tamper:
        metrics["runs"]["frame"]["mof"]["point"] = 99.0  # type: ignore[index]
    config = {
        "selected": {},
        "runs": {"frame": "argmax"},
        "features": {"model": "x"},
        "classes": {},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "tables.md").write_text(render_tables(metrics, config), encoding="utf-8")


def test_reproduce_lite_recomputes_committed_runs_and_flags_drift(tmp_path: Path) -> None:
    _write_run(tmp_path / "good_run")
    result = runner.invoke(app, ["reproduce-lite", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "good_run: OK" in result.output
    _write_run(tmp_path / "bad_run", tamper=True)
    result = runner.invoke(app, ["reproduce-lite", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "bad_run: MISMATCH" in result.output
    assert "runs differs" in result.output


def test_score_predictions_checks_and_regenerates_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    (run_dir / "tables.md").write_text("stale", encoding="utf-8")
    result = runner.invoke(app, ["score-predictions", "--run", str(run_dir)])
    assert result.exit_code == 1
    assert "tables.md is stale" in result.output
    result = runner.invoke(app, ["score-predictions", "--run", str(run_dir), "--write"])
    assert result.exit_code == 0, result.output
    assert "metrics reproduce" in result.output
    assert (run_dir / "tables.md").read_text(encoding="utf-8") != "stale"


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


def test_reproduce_lite_handles_psr_completion_runs(tmp_path: Path) -> None:
    from sop_monitor.psr_baseline import (
        CompletionRow,
        render_psr_tables,
        score_completions,
        write_completions,
    )

    run_dir = tmp_path / "psr_run"
    run_dir.mkdir()
    rows = [
        CompletionRow("05_assy_0_1", "05", "gt", 100, 0, "05"),
        CompletionRow("05_assy_0_1", "05", "pred", 120, 0, "05"),
        CompletionRow("14_main_0_1", "14", "gt", 300, 3, "14"),
        CompletionRow("14_main_0_1", "14", "pred", 290, 3, "14"),
    ]
    write_completions(run_dir / "completions_val.csv", rows)
    metrics = score_completions(rows, n_boot=10, seed=0)
    config = {"folds": {"05": {"decoder": {"ema": 0.9, "theta_on": 0.7, "theta_off": 0.3}}}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "tables.md").write_text(render_psr_tables(metrics, config), encoding="utf-8")
    result = runner.invoke(app, ["reproduce-lite", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "psr_run: OK" in result.output
    metrics["totals"]["system_fp"] = 99  # type: ignore[index]
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    result = runner.invoke(app, ["reproduce-lite", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "totals differs" in result.output


def test_verify_splits_checks_the_committed_hash(tmp_path: Path) -> None:
    from sop_monitor.splits import hash_test_list

    result = runner.invoke(app, ["verify-splits", "--directory", "splits/industreal"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    (tmp_path / "test.csv").write_text(
        "video_id,participant,n_segments\n03_assy_0_1,03,1\n", encoding="utf-8"
    )
    (tmp_path / "test_sha256.txt").write_text(
        hash_test_list(["03_assy_0_1"]) + "\n", encoding="utf-8"
    )
    assert runner.invoke(app, ["verify-splits", "--directory", str(tmp_path)]).exit_code == 0
    (tmp_path / "test_sha256.txt").write_text("0" * 64 + "\n", encoding="utf-8")
    result = runner.invoke(app, ["verify-splits", "--directory", str(tmp_path)])
    assert result.exit_code == 1 and "MISMATCH" in result.output
