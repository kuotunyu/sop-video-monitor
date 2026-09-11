"""Offline temporal action segmentation metrics (spec 4.2).

Definitions follow the MS-TCN reference evaluation (Farha and Gall, 2019; Lea et al., 2017):

- ``mof``        frame-wise accuracy over all frames of all sequences (micro), in percent
- ``edit_score`` segmental edit score, ``100 * (1 - Levenshtein / max(len))`` on the
                 run-length-collapsed label sequences, averaged over sequences (macro)
- ``f1_at_k``    segmental F1 with IoU >= k/100; TP/FP/FN are summed over all sequences
                 before precision/recall (micro), in percent

Background labels (if any) are excluded from segments for Edit and F1 but still count
as frames for MoF, as in the reference code. Confidence intervals resample whole test
subjects (``subject_bootstrap``; 2,000 draws proposed) so that videos of one person
never straddle the resampling unit.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

LabelSeq = Sequence[Hashable]
Pair = tuple[LabelSeq, LabelSeq]
"""``(predicted labels, ground-truth labels)`` for one sequence; equal length, one label per frame."""

DEFAULT_KS: tuple[int, ...] = (10, 25, 50)


class Segment(NamedTuple):
    label: Hashable
    start: int
    end: int  # exclusive


def segments(labels: LabelSeq, background: Collection[Hashable] = ()) -> list[Segment]:
    """Collapse a per-frame label sequence into ``(label, start, end)`` runs, skipping background."""
    out: list[Segment] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] not in background:
                out.append(Segment(labels[start], start, i))
            start = i
    return out


def levenshtein(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Plain edit distance (insert / delete / substitute, unit cost)."""
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i]
        for j, bj in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != bj)))
        prev = cur
    return prev[-1]


def _check_pair(pred: LabelSeq, gt: LabelSeq) -> None:
    if len(pred) != len(gt):
        raise ValueError(f"pred has {len(pred)} frames but gt has {len(gt)}")


def mof(pairs: Iterable[Pair]) -> float:
    """Mean-over-frames accuracy in percent, micro-averaged over all frames."""
    correct = total = 0
    for pred, gt in pairs:
        _check_pair(pred, gt)
        correct += sum(p == g for p, g in zip(pred, gt, strict=True))
        total += len(gt)
    if total == 0:
        raise ValueError("no frames")
    return 100.0 * correct / total


def edit_score(pairs: Iterable[Pair], background: Collection[Hashable] = ()) -> float:
    """Segmental edit score in percent, macro-averaged over sequences."""
    scores: list[float] = []
    for pred, gt in pairs:
        _check_pair(pred, gt)
        p = [s.label for s in segments(pred, background)]
        y = [s.label for s in segments(gt, background)]
        longest = max(len(p), len(y))
        scores.append(100.0 if longest == 0 else 100.0 * (1.0 - levenshtein(p, y) / longest))
    if not scores:
        raise ValueError("no sequences")
    return float(np.mean(scores))


def segmental_counts(
    pred: LabelSeq, gt: LabelSeq, overlap: float, background: Collection[Hashable] = ()
) -> tuple[int, int, int]:
    """``(tp, fp, fn)`` for one sequence at IoU threshold ``overlap`` in ``[0, 1]``.

    Each predicted segment is matched to the same-label ground-truth segment with the
    highest IoU; a ground-truth segment can be hit at most once.
    """
    _check_pair(pred, gt)
    p_segs = segments(pred, background)
    y_segs = segments(gt, background)
    hits = [False] * len(y_segs)
    tp = fp = 0
    for ps in p_segs:
        best_iou, best_idx = -1.0, -1
        for j, ys in enumerate(y_segs):
            if ps.label != ys.label:
                iou = 0.0
            else:
                inter = min(ps.end, ys.end) - max(ps.start, ys.start)
                union = max(ps.end, ys.end) - min(ps.start, ys.start)
                iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_idx >= 0 and best_iou >= overlap and not hits[best_idx]:
            tp += 1
            hits[best_idx] = True
        else:
            fp += 1
    fn = len(y_segs) - sum(hits)
    return tp, fp, fn


def f1_at_k(pairs: Iterable[Pair], k: int, background: Collection[Hashable] = ()) -> float:
    """Segmental F1@k in percent (``k`` is the IoU threshold in percent, e.g. 10/25/50)."""
    tp = fp = fn = 0
    for pred, gt in pairs:
        a, b, c = segmental_counts(pred, gt, k / 100.0, background)
        tp, fp, fn = tp + a, fp + b, fn + c
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 100.0 * 2 * precision * recall / (precision + recall)


def tas_metrics(
    pairs: Sequence[Pair], ks: Sequence[int] = DEFAULT_KS, background: Collection[Hashable] = ()
) -> dict[str, float]:
    """All spec 4.2 point metrics for one view or one fusion: ``mof``, ``edit``, ``f1@k``."""
    out = {"mof": mof(pairs), "edit": edit_score(pairs, background)}
    for k in ks:
        out[f"f1@{k}"] = f1_at_k(pairs, k, background)
    return out


@dataclass(frozen=True)
class SequenceStats:
    """Per-sequence sufficient statistics of ``tas_metrics``; combine any subset without re-scanning."""

    correct: int
    total: int
    edit: float
    counts: tuple[tuple[int, tuple[int, int, int]], ...]  # ((k, (tp, fp, fn)), ...)


def sequence_stats(
    pred: LabelSeq,
    gt: LabelSeq,
    ks: Sequence[int] = DEFAULT_KS,
    background: Collection[Hashable] = (),
) -> SequenceStats:
    """Everything ``tas_metrics`` needs from one ``(pred, gt)`` pair, computed once."""
    _check_pair(pred, gt)
    return SequenceStats(
        correct=int(sum(p == g for p, g in zip(pred, gt, strict=True))),
        total=len(gt),
        edit=edit_score([(pred, gt)], background),
        counts=tuple((k, segmental_counts(pred, gt, k / 100.0, background)) for k in ks),
    )


def combine_sequence_stats(stats: Sequence[SequenceStats]) -> dict[str, float]:
    """``tas_metrics`` of the union of sequences, from their :class:`SequenceStats` alone.

    MoF and F1 are micro (sums first), Edit is macro (mean of per-sequence scores), exactly as
    in :func:`tas_metrics`; this is what makes subject bootstrapping cheap.
    """
    if not stats:
        raise ValueError("no sequences")
    total = sum(s.total for s in stats)
    if total == 0:
        raise ValueError("no frames")
    out = {
        "mof": 100.0 * sum(s.correct for s in stats) / total,
        "edit": float(np.mean([s.edit for s in stats])),
    }
    for index, (k, _) in enumerate(stats[0].counts):
        tp = sum(s.counts[index][1][0] for s in stats)
        fp = sum(s.counts[index][1][1] for s in stats)
        fn = sum(s.counts[index][1][2] for s in stats)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[f"f1@{k}"] = (
            0.0
            if precision + recall == 0
            else 100.0 * 2 * precision * recall / (precision + recall)
        )
    return out


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower: float
    upper: float
    n_boot: int
    seed: int
    alpha: float


def subject_bootstrap(
    per_subject: Mapping[str, Sequence[Pair]],
    statistic: Callable[[Sequence[Pair]], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Percentile bootstrap CI of ``statistic`` with the test subject as resampling unit.

    ``point`` is the statistic on the full data (not the bootstrap mean). Subjects are
    drawn with replacement; all sequences of a drawn subject enter together.
    """
    if not per_subject:
        raise ValueError("no subjects")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    subjects = list(per_subject)
    full = [pair for s in subjects for pair in per_subject[s]]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(subjects), size=len(subjects))
        sample = [pair for i in idx for pair in per_subject[subjects[i]]]
        draws[b] = statistic(sample)
    lower, upper = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(statistic(full), float(lower), float(upper), n_boot, seed, alpha)
