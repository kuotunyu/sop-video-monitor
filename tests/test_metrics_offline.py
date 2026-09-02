"""MoF / Edit / F1@k against tiny hand-computed examples, plus the subject bootstrap."""

import pytest

from sop_monitor.metrics.offline import (
    BootstrapResult,
    edit_score,
    f1_at_k,
    levenshtein,
    mof,
    segmental_counts,
    segments,
    subject_bootstrap,
    tas_metrics,
)


def test_segments_collapse_runs_and_skip_background() -> None:
    assert segments(list("AAABBC")) == [("A", 0, 3), ("B", 3, 5), ("C", 5, 6)]
    assert segments(["bg", "A", "A", "bg"], background={"bg"}) == [("A", 1, 3)]
    assert segments([]) == []


def test_levenshtein_basics() -> None:
    assert levenshtein("ABC", "ABC") == 0
    assert levenshtein("ABC", "AC") == 1
    assert levenshtein("ABC", "BAC") == 2  # no transposition in plain Levenshtein
    assert levenshtein("", "ABC") == 3


def test_mof_is_micro_averaged_over_frames() -> None:
    assert mof([([0, 1, 1, 1], [0, 0, 1, 1])]) == 75.0
    # 3/4 correct in the first sequence, 0/2 in the second: 3/6 frames, not (75 + 0) / 2.
    assert mof([([0, 1, 1, 1], [0, 0, 1, 1]), ([1, 1], [0, 0])]) == 50.0


def test_mof_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        mof([([0, 1], [0])])


def test_edit_score_hand_examples() -> None:
    gt = list("AAABBBCCC")  # segments A, B, C
    assert edit_score([(gt, gt)]) == 100.0
    # A, C vs A, B, C: one deletion over max length 3.
    assert edit_score([(list("AAACCCCCC"), gt)]) == pytest.approx(200 / 3)
    # B, A, C vs A, B, C: two substitutions over max length 3.
    assert edit_score([(list("BBBAAACCC"), gt)]) == pytest.approx(100 / 3)
    # Macro average over sequences.
    assert edit_score([(gt, gt), (list("AAACCCCCC"), gt)]) == pytest.approx((100 + 200 / 3) / 2)


def test_edit_score_ignores_background_segments() -> None:
    gt = ["bg", "A", "A", "bg", "B", "B"]
    pred = ["A", "A", "A", "A", "B", "B"]
    assert edit_score([(pred, gt)], background={"bg"}) == 100.0


def test_f1_at_k_hand_example() -> None:
    gt = list("AAAABBBB")  # A [0, 4), B [4, 8)
    pred = list("AAAAAAAB")  # A [0, 7) IoU 4/7, B [7, 8) IoU 1/4
    assert segmental_counts(pred, gt, 0.10) == (2, 0, 0)
    assert segmental_counts(pred, gt, 0.25) == (2, 0, 0)  # 1/4 >= 0.25 still counts
    assert segmental_counts(pred, gt, 0.50) == (1, 1, 1)
    assert f1_at_k([(pred, gt)], 10) == 100.0
    assert f1_at_k([(pred, gt)], 25) == 100.0
    assert f1_at_k([(pred, gt)], 50) == 50.0


def test_f1_never_hits_the_same_gt_segment_twice() -> None:
    # pred segments A [0,2), B [2,3), A [3,4) against a single gt A [0,4):
    # first A hits (IoU 0.5), B is a label mismatch, second A is a second hit -> FP.
    assert segmental_counts(list("AABA"), list("AAAA"), 0.10) == (1, 2, 0)
    assert f1_at_k([(list("AABA"), list("AAAA"))], 10) == pytest.approx(50.0)


def test_f1_is_micro_over_sequences_and_zero_without_predictions() -> None:
    gt = list("AAAABBBB")
    pairs = [(list("AAAAAAAB"), gt), (gt, gt)]
    # tp = 1 + 2, fp = 1, fn = 1 -> P = 3/4, R = 3/4.
    assert f1_at_k(pairs, 50) == 75.0
    assert f1_at_k([(["bg"] * 4, list("AAAA"))], 10, background={"bg"}) == 0.0


def test_tas_metrics_reports_all_keys() -> None:
    gt = list("AAAABBBB")
    out = tas_metrics([(gt, gt)])
    assert out == {"mof": 100.0, "edit": 100.0, "f1@10": 100.0, "f1@25": 100.0, "f1@50": 100.0}


def test_subject_bootstrap_collapses_for_identical_subjects() -> None:
    gt = list("AAAABBBB")
    pred = list("AAAAAAAB")
    per_subject = {"S1": [(pred, gt)], "S2": [(pred, gt)], "S3": [(pred, gt)]}
    res = subject_bootstrap(per_subject, lambda pairs: f1_at_k(pairs, 50), n_boot=50, seed=1)
    assert isinstance(res, BootstrapResult)
    assert res.point == res.lower == res.upper == 50.0
    assert (res.n_boot, res.seed, res.alpha) == (50, 1, 0.05)


def test_subject_bootstrap_is_seeded_and_brackets_the_point() -> None:
    gt = list("AAAABBBB")
    per_subject = {
        "S1": [(list("AAAAAAAB"), gt)],  # mof 62.5
        "S2": [(gt, gt)],  # mof 100
        "S3": [(list("BBBBAAAA"), gt)],  # mof 0
    }
    first = subject_bootstrap(per_subject, mof, n_boot=200, seed=7)
    second = subject_bootstrap(per_subject, mof, n_boot=200, seed=7)
    assert first == second
    assert first.point == pytest.approx(mof([p for ps in per_subject.values() for p in ps]))
    assert first.lower <= first.point <= first.upper
    assert first.lower < first.upper


def test_subject_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        subject_bootstrap({}, mof)
