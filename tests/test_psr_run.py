"""End-to-end PSR run on a synthetic mini-dataset: leave-one-out and train->eval modes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from sop_monitor.psr_baseline import (
    PSRSpec,
    read_completions,
    read_split_ids,
    run_psr_baseline,
)

TINY = PSRSpec(
    epochs=40,
    mstcn_epochs=1,
    n_boot=5,
    emas=(0.0, 0.5),
    theta_ons=(0.6,),
    theta_offs=(0.4,),
    min_dwells=(0, 2),
)


def _make_recording(root: Path, features: Path, video_id: str, rng: np.random.Generator) -> None:
    """A 60-frame recording whose 2-d features encode the states of components 0 and 1."""
    n = 60
    states = np.zeros((n, 11), dtype=np.int8)
    states[:, 0] = 1  # base pre-installed
    states[20:, 1] = 1  # front chassis (component 1, step id 3) installed at frame 20
    x = np.zeros((n, 2), dtype=np.float32)
    x[:, 0] = states[:, 1] + 0.1 * rng.standard_normal(n)
    x[:, 1] = rng.standard_normal(n)
    np.savez(features / f"{video_id}.npz", frames=np.arange(n), features=x, n_frames=np.int64(n))
    rec = root / video_id
    rec.mkdir(parents=True)
    raw = "000000.jpg,1,0,0,0,0,0,0,0,0,0,0\n000020.jpg,1,1,0,0,0,0,0,0,0,0,0\n"
    (rec / "PSR_labels_raw.csv").write_text(raw, encoding="utf-8")
    (rec / "PSR_labels.csv").write_text("000020.jpg,3,Install front chassis\n", encoding="utf-8")
    (rec / "PSR_labels_with_errors.csv").write_text(
        "000020.jpg,3,Install front chassis\n", encoding="utf-8"
    )


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    psr = tmp_path / "psr"
    features = tmp_path / "features"
    features.mkdir()
    for video_id in ("01_assy_0_1", "02_assy_0_1", "03_assy_0_1", "04_assy_0_1"):
        _make_recording(psr, features, video_id, rng)
    return features, psr


def test_leave_one_out_run_writes_a_recomputable_directory(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    features, psr = dataset
    out = tmp_path / "run"
    result = run_psr_baseline(
        features, psr, out, TINY, heads=("linear",), decoders=("plain", "prior_dwell")
    )
    metrics = result["metrics"]
    assert set(metrics["runs"]) == {"linear_plain", "linear_prior_dwell"}
    assert metrics["n_videos"] == 4 and metrics["n_participants"] == 4
    assert (out / "completions_val.csv").is_file() and (out / "psr_audit.json").is_file()
    rows = read_completions(out / "completions_val.csv")
    assert {r.fold for r in rows} == {"01", "02", "03", "04"}
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["folds"]["01"]["prior"]["assy"]["active"][1] is True
    assert config["folds"]["01"]["prior"]["assy"]["initial_installed"][0] is True
    audit = json.loads((out / "psr_audit.json").read_text(encoding="utf-8"))
    assert all(a["raw_matches_labels"] for a in audit)
    # The clean synthetic signal is easy: every fold finds the single install.
    assert metrics["runs"]["linear_prior_dwell"]["totals"]["system_fn"] == 0


def test_train_eval_mode_fits_once_and_refuses_shared_participants(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    features, psr = dataset
    train_csv = tmp_path / "train.csv"
    eval_csv = tmp_path / "val.csv"
    train_csv.write_text(
        "video_id,participant,n_segments\n01_assy_0_1,01,1\n02_assy_0_1,02,1\n03_assy_0_1,03,1\n",
        encoding="utf-8",
    )
    eval_csv.write_text("video_id,participant,n_segments\n04_assy_0_1,04,1\n", encoding="utf-8")
    assert read_split_ids(eval_csv) == {"04_assy_0_1"}
    out = tmp_path / "split_run"
    result = run_psr_baseline(
        features,
        psr,
        out,
        TINY,
        heads=("linear",),
        decoders=("plain",),
        train_ids=read_split_ids(train_csv),
        eval_ids=read_split_ids(eval_csv),
    )
    assert result["metrics"]["n_videos"] == 1
    assert list(result["config"]["folds"]) == ["train->eval"]
    assert result["config"]["folds"]["train->eval"]["train_videos"] == 3
    assert "fit once on the 3 train-split recordings" in result["config"]["protocol"]
    with pytest.raises(ValueError, match="share a participant"):
        run_psr_baseline(
            features,
            psr,
            tmp_path / "bad",
            TINY,
            heads=("linear",),
            decoders=("plain",),
            train_ids={"01_assy_0_1", "04_assy_0_1"},
            eval_ids={"04_assy_0_1"},
        )
    with pytest.raises(ValueError, match="given together"):
        run_psr_baseline(features, psr, tmp_path / "bad2", TINY, train_ids={"01_assy_0_1"})


def test_nested_selection_and_multiple_seeds(dataset: tuple[Path, Path], tmp_path: Path) -> None:
    from sop_monitor.psr_baseline import inner_folds, load_psr_videos, seed_summary

    features, psr = dataset
    videos = load_psr_videos(features, psr)
    folds = inner_folds(videos, n_folds=4)
    assert len(folds) == 4
    held = sorted(v.participant for _, test in folds for v in test)
    assert held == ["01", "02", "03", "04"]
    assert all(len(train) == 3 and len(test) == 1 for train, test in folds)
    assert all(
        {v.participant for v in train}.isdisjoint({v.participant for v in test})
        for train, test in folds
    )
    out = tmp_path / "nested"
    result = run_psr_baseline(
        features,
        psr,
        out,
        TINY,
        heads=("linear",),
        decoders=("prior_dwell",),
        train_ids={"01_assy_0_1", "02_assy_0_1", "03_assy_0_1"},
        eval_ids={"04_assy_0_1"},
        selection="nested",
        seeds=(0, 1),
    )
    metrics = result["metrics"]
    assert set(metrics["runs"]) == {"linear_prior_dwell_s0", "linear_prior_dwell_s1"}
    assert metrics["seed_summary"]["linear_prior_dwell"]["n_seeds"] == 2
    assert "out-of-fold" in result["config"]["protocol"]
    assert result["config"]["folds"]["train->eval"]["selection"] == "nested"
    assert set(result["config"]["folds"]["train->eval"]["decoders"]) == {
        "linear_prior_dwell_s0",
        "linear_prior_dwell_s1",
    }
    rows = read_completions(out / "completions_val.csv")
    assert sum(1 for r in rows if r.source == "gt") == 1  # ground truth written once per video
    assert "Across seeds" in result["tables"]
    fake = {
        "x_s0": {
            "summary": {k: {"mean_over_videos": 1.0} for k in ("pos", "f1", "mean_delay_frames")},
            "totals": {"system_tp": 1, "system_fp": 0, "system_fn": 0},
        },
        "x_s1": {
            "summary": {k: {"mean_over_videos": 0.0} for k in ("pos", "f1", "mean_delay_frames")},
            "totals": {"system_tp": 3, "system_fp": 0, "system_fn": 0},
        },
        "y": {"summary": {}, "totals": {}},
    }
    summary = seed_summary(fake)
    assert summary["x"]["pos"] == {"mean": 0.5, "std": 0.5} and summary["x"]["system_tp"] == 2.0
    assert "y" not in summary
    with pytest.raises(ValueError, match="selection"):
        run_psr_baseline(
            features, psr, tmp_path / "bad", TINY, heads=("linear",), selection="oracle"
        )
