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


def test_gt_sanity_cases_behave_as_documented() -> None:
    result = score_completions(_rows(), n_boot=5, seed=0)
    sanity = result["gt_sanity"]
    assert sanity["identity"]["pos"] == 1.0 and sanity["identity"]["mean_delay_frames"] == 0.0
    assert sanity["shift_plus_30_frames"]["mean_delay_frames"] == 30.0
    assert sanity["shift_plus_30_frames"]["pos"] == 1.0
    assert sanity["drop_last"]["system_fn"] == 2.0  # one per video
    assert sanity["swap_first_two"]["pos"] == pytest.approx(1 - 1 / 3)


def test_dwell_requires_the_crossing_to_persist() -> None:
    from sop_monitor.psr_baseline import ema_filter

    probs = np.zeros((40, 1), dtype=np.float32)
    probs[5:8, 0] = 1.0  # 3-frame blip
    probs[20:33, 0] = 1.0  # 13-frame plateau
    frames = np.arange(40)
    plain = decode_completions(probs, frames, DecoderConfig(ema=0.0, theta_on=0.7, theta_off=0.3))
    assert [(c.frame, c.step) for c in plain] == [
        (5, INSTALL),
        (8, REMOVE),
        (20, INSTALL),
        (33, REMOVE),
    ]
    dwelled = decode_completions(
        probs, frames, DecoderConfig(ema=0.0, theta_on=0.7, theta_off=0.3, min_dwell=5)
    )
    # The blip is ignored; each event is emitted 5 frames after its crossing.
    assert [(c.frame, c.step) for c in dwelled] == [(25, INSTALL), (38, REMOVE)]
    assert np.allclose(ema_filter(probs, 0.0), probs)
    smoothed = ema_filter(probs, 0.5)
    # The blip's residual (0.5**12) keeps these from being exact.
    assert smoothed[20, 0] == pytest.approx(0.5, abs=1e-3)
    assert smoothed[21, 0] == pytest.approx(0.75, abs=1e-3)
    with pytest.raises(ValueError):
        decode_completions(probs, frames, DecoderConfig(min_dwell=-1))


def test_prior_freezes_inactive_components_and_sets_the_initial_state() -> None:
    from sop_monitor.psr_baseline import ProcedurePrior

    probs = np.zeros((20, 2), dtype=np.float32)
    probs[:10, :] = 1.0  # both components look installed, then removed
    frames = np.arange(20)
    config = DecoderConfig(ema=0.0, theta_on=0.7, theta_off=0.3)
    without = decode_completions(probs, frames, config)
    assert [(c.frame, c.step) for c in without] == [
        (0, INSTALL),
        (0, 3 + INSTALL),
        (10, REMOVE),
        (10, 3 + REMOVE),
    ]
    prior = ProcedurePrior(active=np.array([True, False]), initial_installed=np.array([True, True]))
    with_prior = decode_completions(probs, frames, config, prior)
    # Component 0 starts installed, so only its removal is an event; component 1 never emits.
    assert [(c.frame, c.step) for c in with_prior] == [(10, REMOVE)]


def test_multi_run_tables_score_each_run_separately(tmp_path: Path) -> None:
    rows = [
        CompletionRow("05_assy_0_1", "05", "gt", 100, 0, "05", ""),
        CompletionRow("05_assy_0_1", "05", "pred", 120, 0, "05", "linear_plain"),
        CompletionRow("05_assy_0_1", "05", "pred", 900, 9, "05", "linear_plain"),
        CompletionRow("05_assy_0_1", "05", "pred", 110, 0, "05", "mstcn_prior_dwell"),
        CompletionRow("14_main_0_1", "14", "gt", 300, 3, "14", ""),
        CompletionRow("14_main_0_1", "14", "pred", 330, 3, "14", "linear_plain"),
        CompletionRow("14_main_0_1", "14", "pred", 305, 3, "14", "mstcn_prior_dwell"),
    ]
    path = tmp_path / "completions_val.csv"
    write_completions(path, rows)
    assert read_completions(path) == rows
    result = score_completions(rows, n_boot=10, seed=0)
    assert set(result["runs"]) == {"linear_plain", "mstcn_prior_dwell"}
    assert result["runs"]["linear_plain"]["totals"]["system_fp"] == 1
    assert result["runs"]["mstcn_prior_dwell"]["totals"] == {
        "system_tp": 2,
        "system_fp": 0,
        "system_fn": 0,
        "n_gt": 2,
        "n_pred": 2,
    }
    assert (
        result["runs"]["mstcn_prior_dwell"]["summary"]["mean_delay_frames"]["mean_over_videos"]
        == 7.5
    )
    assert "summary" not in result  # multi-run layout has no flat summary
    assert result["gt_sanity"]["identity"]["pos"] == 1.0


def test_learn_prior_uses_recordings_of_the_same_kind() -> None:
    from sop_monitor.psr_baseline import PSRVideo, learn_prior

    def video(video_id: str, initial: list[int], steps: list[int]) -> PSRVideo:
        kind = "main" if "_main_" in video_id else "assy"
        return PSRVideo(
            video_id,
            video_id[:2],
            kind,
            np.arange(1),
            np.zeros((1, 2), np.float32),
            np.zeros((1, 11), np.float32),
            np.array(initial, dtype=bool),
            [Completion(10 * i, s) for i, s in enumerate(steps)],
            False,
            1,
        )

    train = [
        video("01_assy_0_1", [1] + [0] * 10, [3, 6]),
        video("02_assy_0_1", [1] + [0] * 10, [3, 9]),
        video("01_main_0_1", [1] * 11, [32, 12]),
    ]
    assy = learn_prior(train, "assy")
    assert assy.active.tolist() == [False, True, True, True] + [False] * 7
    assert assy.initial_installed.tolist() == [True] + [False] * 10
    main = learn_prior(train, "main")
    assert main.active.tolist() == [False] * 4 + [True] + [False] * 5 + [True]
    assert main.initial_installed.all()


def test_pareto_front_and_delay_capped_choice() -> None:
    from sop_monitor.psr_baseline import GridPoint, choose_decoder, pareto_front

    def point(f1: float, delay_s: float, dwell: int = 0) -> GridPoint:
        return GridPoint(DecoderConfig(0.9, 0.7, 0.3, dwell), f1, 0.5, delay_s * 10.0)

    points = [point(0.5, 10), point(0.7, 30), point(0.6, 40), point(0.8, 60), point(0.4, 5)]
    front = pareto_front(points)
    assert [(p.f1, p.delay_frames / 10) for p in front] == [
        (0.4, 5),
        (0.5, 10),
        (0.7, 30),
        (0.8, 60),
    ]
    assert choose_decoder(points).f1 == 0.8  # best F1 regardless of delay
    assert choose_decoder(points, delay_cap_s=35).f1 == 0.7  # best within the budget
    assert choose_decoder(points, delay_cap_s=1).delay_frames == 50.0  # nothing fits: fastest
    never = [GridPoint(DecoderConfig(), 0.0, 0.0, float("inf"))]
    assert choose_decoder(never, delay_cap_s=10).delay_frames == float("inf")
    assert never[0].to_dict()["delay_s"] == -1.0
