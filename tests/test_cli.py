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


def test_w1_and_w3_placeholders_exit_nonzero() -> None:
    for command in ("freeze-splits", "check-sop"):
        result = runner.invoke(app, [command])
        assert result.exit_code == NOT_YET_EXIT_CODE, command
        assert "not yet" in result.output
