"""IndustReal PSR metrics, transcribed from the reference ``PSR/psr_utils.py`` behaviour.

Every expected value below was derived by hand from the reference code (costs, normalisation,
matching rules and the epsilon-smoothed F1), so this file is the sanity check spec 4.3 asks for
in the absence of the PSR label files.
"""

import pytest

from sop_monitor.metrics.online import (
    POS_COSTS,
    Completion,
    damerau_levenshtein,
    f1_reference,
    procedure_order_similarity,
    psr_performance,
)


def test_unrestricted_damerau_levenshtein_with_unit_costs() -> None:
    assert damerau_levenshtein("ABCD", "ABCD") == 0
    assert damerau_levenshtein("ABCD", "ACBD") == 1  # adjacent transposition
    assert damerau_levenshtein("", "ABC") == 3
    assert damerau_levenshtein(["a", "b"], ["a"]) == 1
    assert damerau_levenshtein("CA", "ABC") == 2  # unrestricted: transpose then insert (OSA says 3)
    assert damerau_levenshtein("kitten", "sitting") == 3


def test_reference_costs_make_a_substitution_cost_two() -> None:
    assert POS_COSTS == {"insert": 1.0, "delete": 1.0, "substitute": 2.0, "transpose": 1.0}
    assert damerau_levenshtein("ABCD", "AXCD", **POS_COSTS) == 2
    assert damerau_levenshtein("ABCD", "ACBD", **POS_COSTS) == 1
    assert damerau_levenshtein("ABCD", "ABD", **POS_COSTS) == 1


def test_procedure_order_similarity_normalises_by_gt_length_and_clips() -> None:
    gt = [0, 3, 6, 9]
    assert procedure_order_similarity(gt, gt) == 1.0
    assert procedure_order_similarity([0, 6, 3, 9], gt) == 0.75  # one transposition
    assert procedure_order_similarity([0, 42, 6, 9], gt) == 0.5  # substitution costs 2
    assert procedure_order_similarity([0, 3], gt) == 0.5  # two deletions
    assert (
        procedure_order_similarity([0, 3, 6, 9, 1, 2, 4, 5, 7], gt) == 0.0
    )  # 5 inserts > len(gt), clipped
    assert procedure_order_similarity([], gt) == 0.0
    assert procedure_order_similarity([], []) == 1.0
    assert procedure_order_similarity([1], []) == 0.0


def test_reference_f1_uses_its_epsilon_smoothing() -> None:
    assert f1_reference(fn=0, fp=0, tp=4) == pytest.approx(2 / (2 + 1e-6))
    assert f1_reference(fn=1, fp=1, tp=2) == pytest.approx(2 * (2 / 3 * 2 / 3) / (4 / 3 + 1e-6))
    assert f1_reference(fn=0, fp=0, tp=0) == pytest.approx(2e-12 / (2e-6 + 1e-6))


def _gt() -> list[Completion]:
    return [Completion(100, 0), Completion(250, 3), Completion(400, 6), Completion(700, 9)]


def test_perfect_prediction_scores_one_with_zero_delay() -> None:
    out = psr_performance(_gt(), _gt())
    assert (out["system_tp"], out["system_fp"], out["system_fn"]) == (4, 0, 0)
    assert out["pos"] == 1.0
    assert out["mean_delay_frames"] == 0.0
    assert out["f1"] == pytest.approx(f1_reference(0, 0, 4))


def test_late_detections_count_as_true_positives_with_a_delay() -> None:
    pred = [Completion(120, 0), Completion(270, 3), Completion(430, 6), Completion(710, 9)]
    out = psr_performance(_gt(), pred)
    assert (out["system_tp"], out["system_fp"], out["system_fn"]) == (4, 0, 0)
    assert out["mean_delay_frames"] == pytest.approx((20 + 20 + 30 + 10) / 4)


def test_early_observed_detection_is_a_system_false_positive() -> None:
    pred = [Completion(90, 0), Completion(270, 3), Completion(430, 6), Completion(710, 9)]
    out = psr_performance(_gt(), pred)
    assert (out["system_tp"], out["system_fp"], out["system_fn"]) == (3, 1, 0)
    assert out["mean_delay_frames"] == pytest.approx(20.0)  # the early one contributes no delay
    implied = [Completion(90, 0, observed=False), *pred[1:]]
    assert (
        psr_performance(_gt(), implied)["system_fp"] == 0
    )  # implied early step: perception-level only


def test_missing_and_surplus_completions_become_fn_and_fp() -> None:
    pred = [
        Completion(120, 0),
        Completion(270, 3),
        Completion(500, 12),
    ]  # 6 and 9 missing, 12 spurious
    out = psr_performance(_gt(), pred)
    assert (out["system_tp"], out["system_fp"], out["system_fn"]) == (2, 1, 2)
    # cheapest edit of gt [0, 3, 6, 9] into [0, 3, 12]: substitute 6 -> 12 (2) + delete 9 (1) = 3
    assert out["pos"] == pytest.approx(1 - 3 / 4)


def test_repeated_steps_are_matched_to_the_closest_later_occurrence() -> None:
    gt = [Completion(100, 0), Completion(300, 0)]
    pred = [Completion(310, 0), Completion(105, 0)]  # listed out of order on purpose
    out = psr_performance(gt, pred)
    assert (out["system_tp"], out["system_fp"], out["system_fn"]) == (2, 0, 0)
    assert out["mean_delay_frames"] == pytest.approx(7.5)
    assert out["pos"] == 1.0
    three = [*pred, Completion(600, 0)]
    assert psr_performance(gt, three)["system_fp"] == 1


def test_no_matched_completion_gives_no_delay() -> None:
    out = psr_performance(_gt(), [])
    assert out["mean_delay_frames"] is None
    assert (out["system_tp"], out["system_fp"], out["system_fn"]) == (0, 0, 4)
    assert out["pos"] == 0.0
