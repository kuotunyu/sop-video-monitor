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


def test_w3_placeholder_and_ha_vid_split_exit_not_yet() -> None:
    result = runner.invoke(app, ["check-sop"])
    assert result.exit_code == NOT_YET_EXIT_CODE
    assert "not yet" in result.output
    result = runner.invoke(app, ["freeze-splits", "--dataset", "ha-vid"])
    assert result.exit_code == NOT_YET_EXIT_CODE
    assert "not yet" in result.output


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
