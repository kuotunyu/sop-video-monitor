"""IndustReal-style online metrics: POS (Damerau-Levenshtein), completion F1, detection delay."""

import pytest

from sop_monitor.metrics.online import (
    completion_counts,
    completion_f1,
    damerau_levenshtein,
    mean_detection_delay,
    procedure_order_similarity,
)


def test_damerau_levenshtein_counts_an_adjacent_swap_as_one_edit() -> None:
    assert damerau_levenshtein("ABCD", "ABCD") == 0
    assert damerau_levenshtein("ABCD", "ACBD") == 1  # plain Levenshtein would say 2
    assert damerau_levenshtein("", "ABC") == 3
    assert damerau_levenshtein(["a", "b"], ["a"]) == 1
    assert damerau_levenshtein("CA", "ABC") == 3  # restricted (OSA) variant, not unrestricted DL


def test_procedure_order_similarity_hand_examples() -> None:
    gt = ["A", "B", "C", "D"]
    assert procedure_order_similarity(gt, gt) == 1.0
    assert procedure_order_similarity(["A", "C", "B", "D"], gt) == 0.75
    assert procedure_order_similarity(["A", "B"], gt) == 0.5
    assert procedure_order_similarity([], gt) == 0.0
    assert procedure_order_similarity([], []) == 1.0


def test_completion_f1_matches_by_label_as_a_multiset() -> None:
    assert completion_counts(["a", "b", "d"], ["a", "b", "c"]) == (2, 1, 1)
    assert completion_f1(["a", "b", "d"], ["a", "b", "c"]) == pytest.approx(2 / 3)
    assert completion_f1(["a", "a"], ["a"]) == pytest.approx(2 / 3)  # duplicate detection is a FP
    assert completion_f1([], ["a"]) == 0.0
    assert completion_f1(["a", "b"], ["a", "b"]) == 1.0


def test_mean_detection_delay_uses_only_detected_steps() -> None:
    detected = {"a": 12.0, "b": 20.0, "z": 99.0}
    completed = {"a": 10.0, "b": 15.0, "c": 30.0}
    assert mean_detection_delay(detected, completed) == pytest.approx(3.5)
    assert mean_detection_delay({}, completed) is None
