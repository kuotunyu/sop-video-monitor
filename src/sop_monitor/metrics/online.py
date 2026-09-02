"""Online SOP metrics as defined by IndustReal (Schoonbeek et al., 2024; spec 4.3).

- ``procedure_order_similarity`` (POS): ``1 - DamerauLevenshtein(pred, gt) / max(len)``
  on the order in which step completions were emitted; 1.0 is a perfect order.
  The restricted (optimal string alignment) Damerau-Levenshtein variant is used, so a
  swapped adjacent pair costs 1 rather than 2.
- ``completion_f1``: F1 over detected step completions, matched by step label.
- ``mean_detection_delay``: mean ``detected_at - completed_at`` over steps that were
  both completed and detected; ``None`` when no step qualifies.

These are transcriptions of the paper's definitions; spec 4.3 schedules a one-off
sanity run on IndustReal against its published baseline to confirm they agree with
the reference implementation. Until then treat the exact matching rules as provisional.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence


def damerau_levenshtein(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Optimal-string-alignment distance: insert / delete / substitute / adjacent transpose."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[n][m]


def procedure_order_similarity(
    pred_order: Sequence[Hashable], gt_order: Sequence[Hashable]
) -> float:
    """POS in ``[0, 1]``; two empty orders are identical (1.0)."""
    longest = max(len(pred_order), len(gt_order))
    if longest == 0:
        return 1.0
    return 1.0 - damerau_levenshtein(pred_order, gt_order) / longest


def completion_counts(
    pred_completed: Iterable[Hashable], gt_completed: Iterable[Hashable]
) -> tuple[int, int, int]:
    """``(tp, fp, fn)`` of step completions matched by label (multiset intersection)."""
    pred = Counter(pred_completed)
    gt = Counter(gt_completed)
    tp = sum((pred & gt).values())
    return tp, sum(pred.values()) - tp, sum(gt.values()) - tp


def completion_f1(pred_completed: Iterable[Hashable], gt_completed: Iterable[Hashable]) -> float:
    """Completion F1 in ``[0, 1]``."""
    tp, fp, fn = completion_counts(pred_completed, gt_completed)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_detection_delay(
    detected_at: Mapping[Hashable, float], completed_at: Mapping[Hashable, float]
) -> float | None:
    """Mean signed delay (same unit as the inputs) over steps present in both mappings."""
    delays = [detected_at[s] - completed_at[s] for s in completed_at if s in detected_at]
    if not delays:
        return None
    return sum(delays) / len(delays)
