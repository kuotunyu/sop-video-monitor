"""PSR baseline pieces that need no features: state forward-fill, causal decoder, tables, scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sop_monitor.industreal_psr import INSTALL, REMOVE
from sop_monitor.metrics.online import Completion
from sop_monitor.psr_baseline import (
    CompletionRow,
    DecoderConfig,
    decode_completions,
    frame_states,
    read_completions,
    score_completions,
    write_completions,
)


def test_frame_states_forward_fills_between_raw_rows() -> None:
    raw = [(0, [0] * 11), (5, [1] + [0] * 10), (8, [1, -1] + [0] * 9)]
    states = frame_states(raw, n_frames=10)
    assert states.shape == (10, 11)
    assert states[:5, 0].tolist() == [0] * 5
    assert states[5:, 0].tolist() == [1] * 5
    assert states[8:, 1].tolist() == [-1, -1]
    # Rows beyond the video are ignored; frames before the first row are all zero.
    assert frame_states([(20, [1] * 11)], n_frames=10).sum() == 0


def test_decoder_emits_install_on_up_crossing_and_remove_on_down_crossing() -> None:
    probs = np.zeros((10, 2), dtype=np.float32)
    probs[2:7, 0] = 1.0  # component 0 installed for frames 2..6, then gone
    probs[:, 1] = 0.5  # component 1 never crosses either threshold
    frames = np.arange(10) * 3
    out = decode_completions(probs, frames, DecoderConfig(ema=0.0, theta_on=0.7, theta_off=0.3))
    assert out == [Completion(6, INSTALL), Completion(21, REMOVE)]
    # Smoothing delays the crossing but never emits twice for one plateau.
    smoothed = decode_completions(
        probs, frames, DecoderConfig(ema=0.5, theta_on=0.7, theta_off=0.3)
    )
    assert [c.step for c in smoothed] == [INSTALL, REMOVE]
    assert smoothed[0].frame > 6
    with pytest.raises(ValueError):
        decode_completions(probs, frames, DecoderConfig(ema=1.0))
    with pytest.raises(ValueError):
        decode_completions(probs, frames, DecoderConfig(theta_on=0.3, theta_off=0.5))


def test_decoder_is_causal() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random((30, 3)).astype(np.float32)
    frames = np.arange(30)
    config = DecoderConfig(ema=0.8, theta_on=0.6, theta_off=0.4)
    full = decode_completions(probs, frames, config)
    altered = probs.copy()
    altered[20:] = 0.0
    partial = decode_completions(altered, frames, config)
    assert [c for c in full if c.frame < 20] == [c for c in partial if c.frame < 20]


def _rows() -> list[CompletionRow]:
    rows = []
    for video, participant in (("05_assy_0_1", "05"), ("14_main_0_1", "14")):
        for frame, step in ((100, 0), (250, 3), (400, 6)):
            rows.append(CompletionRow(video, participant, "gt", frame, step, participant))
            rows.append(CompletionRow(video, participant, "pred", frame + 20, step, participant))
    rows.append(CompletionRow("14_main_0_1", "14", "pred", 900, 9, "14"))  # one spurious step
    return rows


def test_completions_round_trip_and_scoring(tmp_path: Path) -> None:
    path = tmp_path / "completions_val.csv"
    write_completions(path, _rows())
    assert read_completions(path) == _rows()
    result = score_completions(_rows(), n_boot=20, seed=0)
    assert result["n_videos"] == 2 and result["n_participants"] == 2
    per_video = result["per_video"]
    assert per_video["05_assy_0_1"]["pos"] == 1.0
    assert per_video["05_assy_0_1"]["mean_delay_frames"] == 20.0
    assert per_video["14_main_0_1"]["system_fp"] == 1
    assert per_video["14_main_0_1"]["pos"] == pytest.approx(1 - 1 / 3)
    assert result["totals"] == {
        "system_tp": 6,
        "system_fp": 1,
        "system_fn": 0,
        "n_gt": 6,
        "n_pred": 7,
    }
    assert result["summary"]["mean_delay_frames"]["mean_over_videos"] == 20.0
    assert result["summary"]["pos"]["ci95"][0] <= result["summary"]["pos"]["mean_over_videos"]
