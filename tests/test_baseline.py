"""Development baseline pieces that need no video, torch or GPU: smoothing, label maps, scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sop_monitor.baseline import (
    PredictionRow,
    causal_smooth,
    centered_smooth,
    label_index_map,
    read_predictions,
    score_predictions,
    write_predictions,
)
from sop_monitor.metrics.offline import mof


def test_causal_smoothing_uses_only_the_present_and_the_past() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random((12, 3)).astype(np.float32)
    out = causal_smooth(probs, window=3)
    for t in range(12):
        assert out[t] == pytest.approx(probs[max(0, t - 2) : t + 1].mean(axis=0), abs=1e-6)
    altered = probs.copy()
    altered[6:] = 0.0
    assert np.allclose(causal_smooth(altered, window=3)[:6], out[:6])
    assert np.array_equal(causal_smooth(probs, window=1), probs)


def test_centered_smoothing_looks_into_the_future() -> None:
    rng = np.random.default_rng(1)
    probs = rng.random((10, 2)).astype(np.float32)
    out = centered_smooth(probs, radius=2)
    for t in range(10):
        lo, hi = max(0, t - 2), min(10, t + 3)
        assert out[t] == pytest.approx(probs[lo:hi].mean(axis=0), abs=1e-6)
    assert np.array_equal(centered_smooth(probs, radius=0), probs)
    altered = probs.copy()
    altered[5] = 0.0
    assert not np.allclose(centered_smooth(altered, radius=2)[3], out[3])


def test_smoothing_rejects_bad_windows() -> None:
    probs = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        causal_smooth(probs, window=0)
    with pytest.raises(ValueError):
        centered_smooth(probs, radius=-1)


def test_label_index_map_reserves_zero_for_background() -> None:
    mapping = label_index_map([7, 3, 3, 12])
    assert mapping == {-1: 0, 3: 1, 7: 2, 12: 3}
    with pytest.raises(ValueError):
        label_index_map([-1, 3])


def _rows() -> list[PredictionRow]:
    rows = []
    gt_a = [-1, -1, 1, 1, 1, 2, 2, -1]
    frame_a = [-1, 1, 1, 1, 2, 2, 2, -1]
    for f, (g, p) in enumerate(zip(gt_a, frame_a, strict=True)):
        rows.append(PredictionRow("a_1", "a", f, g, {"frame": p, "smooth": g}))
    gt_b = [1, 1, 2, 2]
    for f, g in enumerate(gt_b):
        rows.append(PredictionRow("b_1", "b", f, g, {"frame": -1, "smooth": g}))
    return rows


def test_prediction_table_round_trips_through_csv(tmp_path: Path) -> None:
    path = tmp_path / "predictions_val.csv"
    write_predictions(path, _rows())
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "video_id,participant,frame,gt,pred_frame,pred_smooth"
    assert read_predictions(path) == _rows()


def test_scoring_reports_every_run_with_point_and_ci() -> None:
    result = score_predictions(_rows(), n_boot=20, seed=0)
    assert result["n_videos"] == 2
    assert result["n_participants"] == 2
    assert result["n_frames"] == 12
    assert set(result["runs"]) == {"frame", "smooth"}
    smooth = result["runs"]["smooth"]
    assert smooth["mof"]["point"] == 100.0
    assert smooth["f1@50"]["point"] == 100.0
    assert smooth["mof"]["ci95"] == [100.0, 100.0]
    frame = result["runs"]["frame"]
    expected = mof(
        [
            ([-1, 1, 1, 1, 2, 2, 2, -1], [-1, -1, 1, 1, 1, 2, 2, -1]),
            ([-1] * 4, [1, 1, 2, 2]),
        ]
    )
    assert frame["mof"]["point"] == pytest.approx(expected)
    assert frame["mof"]["ci95"][0] <= frame["mof"]["point"] <= frame["mof"]["ci95"][1]
    assert result["per_video"]["frame"]["b_1"]["mof"] == 0.0
    assert result["reference_crosscheck"] == "agree"
