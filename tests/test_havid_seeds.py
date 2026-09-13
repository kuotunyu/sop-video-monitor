"""Multi-seed summary of HA-ViD TAS runs on fake run directories, and its reproduce-lite check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sop_monitor.cli import app
from sop_monitor.havid_seeds import summarise_seeds, write_seed_summary

METRICS = ("mof", "edit", "f1@10", "f1@25", "f1@50")


def _fake_run(root: Path, seed: int, hand: str = "lh") -> Path:
    run = root / f"run_s{seed}"
    run.mkdir(parents=True)
    runs = {
        name: {key: {"point": base + seed, "ci95": [0.0, 100.0]} for key in METRICS}
        for name, base in (("majority", 10.0), ("fusion_causal", 30.0))
    }
    (run / "metrics.json").write_text(
        json.dumps({"runs": runs, "eval_split": "val"}), encoding="utf-8"
    )
    (run / "config.json").write_text(
        json.dumps(
            {
                "spec": {"hand": hand, "mstcn": {"seed": seed, "selection_metric": "f1@10"}},
                "training": {"view0_causal": {"best_epoch": 10 + seed}},
            }
        ),
        encoding="utf-8",
    )
    return run


def test_seed_summary_means_stds_and_reproduces(tmp_path: Path) -> None:
    runs = [_fake_run(tmp_path, s) for s in (0, 1, 2)]
    payload = summarise_seeds(runs)
    fusion = payload["summary"]["fusion_causal"]["f1@10"]
    assert fusion["mean"] == pytest.approx(31.0) and fusion["std"] == pytest.approx(1.0)
    assert fusion["points"] == [30.0, 31.0, 32.0]
    assert payload["best_epochs"] == {"view0_causal": [10, 11, 12]}
    assert payload["seeds"] == [0, 1, 2] and payload["hand"] == "lh"
    with pytest.raises(ValueError, match="distinct"):
        summarise_seeds([runs[0], runs[0]])
    with pytest.raises(ValueError, match="share hand"):
        summarise_seeds([runs[0], _fake_run(tmp_path / "other", 5, hand="rh")])
    out = tmp_path / "summary"
    write_seed_summary(runs, out)
    tables = (out / "tables.md").read_text(encoding="utf-8")
    assert "| fusion_causal | 31.0 ± 1.0 (30.0–32.0)" in tables
    result = CliRunner().invoke(app, ["reproduce-lite", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "summary: OK" in result.output
    result = CliRunner().invoke(
        app,
        [
            "summarise-havid-seeds",
            "--run",
            str(runs[0]),
            "--run",
            str(runs[1]),
            "--out",
            str(tmp_path / "two"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "two" / "seed_summary.json").is_file()
