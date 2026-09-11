"""Cross-check ``metrics.offline`` against an independent transcription of the MS-TCN eval script.

``metrics.offline`` was written from the metric definitions; ``metrics.reference_mstcn`` mirrors
the control flow of the reference ``eval.py`` (segment extraction, edit score, F1 matching).
Agreement on random and adversarial inputs is the evidence that both read the definitions alike.
"""

from __future__ import annotations

import numpy as np
import pytest

from sop_monitor.metrics.offline import combine_sequence_stats, sequence_stats, tas_metrics
from sop_monitor.metrics.reference_mstcn import evaluate as reference_evaluate

BG = (-1,)


def _random_runs(rng: np.random.Generator, n_frames: int, n_classes: int) -> list[int]:
    out: list[int] = []
    while len(out) < n_frames:
        label = int(rng.integers(-1, n_classes))
        out.extend([label] * int(rng.integers(1, 12)))
    return out[:n_frames]


@pytest.mark.parametrize("seed", range(8))
def test_offline_metrics_agree_with_the_reference_transcription(seed: int) -> None:
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(int(rng.integers(1, 5))):
        n = int(rng.integers(5, 80))
        gt = _random_runs(rng, n, 4)
        pred = _random_runs(rng, n, 4)
        # Make some predictions partially right so that TP counts are non-trivial.
        for i in range(n):
            if rng.random() < 0.5:
                pred[i] = gt[i]
        pairs.append((pred, gt))
    ours = tas_metrics(pairs, background=BG)
    theirs = reference_evaluate([p for p, _ in pairs], [g for _, g in pairs], background=BG)
    for key, value in ours.items():
        assert value == pytest.approx(theirs[key], abs=1e-9), key


def test_reference_agrees_on_edge_cases() -> None:
    cases = [
        [(list("AAAABBBB"), list("AAAABBBB"))],
        [(list("AAAAAAAB"), list("AAAABBBB"))],
        [(["bg"] * 4, list("AAAA"))],  # no predicted segments -> F1 0, edit 0
        [(list("AABA"), list("AAAA"))],  # same gt segment hit twice
    ]
    for pairs in cases:
        ours = tas_metrics(pairs, background=("bg",))
        theirs = reference_evaluate(
            [p for p, _ in pairs], [g for _, g in pairs], background=("bg",)
        )
        assert ours == pytest.approx(theirs, abs=1e-9)


def test_sequence_stats_combine_to_the_same_numbers_as_tas_metrics() -> None:
    rng = np.random.default_rng(3)
    pairs = []
    for _ in range(4):
        n = int(rng.integers(10, 60))
        gt = _random_runs(rng, n, 3)
        pred = [g if rng.random() < 0.6 else int(rng.integers(-1, 3)) for g in gt]
        pairs.append((pred, gt))
    stats = [sequence_stats(p, g, ks=(10, 25, 50), background=BG) for p, g in pairs]
    assert combine_sequence_stats(stats) == pytest.approx(tas_metrics(pairs, background=BG))
    assert combine_sequence_stats(stats[:1]) == pytest.approx(tas_metrics(pairs[:1], background=BG))
