"""Online procedure-step metrics as implemented by IndustReal (Schoonbeek et al., 2024; spec 4.3).

Aligned on 2026-09-11 with the reference ``PSR/psr_utils.py`` of the IndustReal repository
(Apache-2.0), which is the code behind the published B1-B3 numbers. The definitions that matter:

- ``procedure_order_similarity`` (POS): ``1 - min(D / len(gt), 1)`` where ``D`` is the
  *unrestricted* Damerau-Levenshtein distance between the ground-truth and predicted order of
  step completions with costs insert 1, delete 1, **substitute 2**, transpose 1 (the reference
  uses ``weighted_levenshtein.dam_lev`` with exactly these costs). Normalisation is by the
  ground-truth length, not by the longer sequence, and the score is clipped at 0.
- ``psr_performance``: system-level TP/FP/FN, F1, POS and mean detection delay for one video.
  Occurrences are matched per step id; a prediction that precedes its ground-truth completion
  is a false positive, one at or after it is a true positive with delay ``pred - gt`` frames;
  surplus occurrences are FPs (or FNs); F1 uses the reference's epsilon-smoothed formula so the
  numbers are directly comparable.

Deviations from the reference, both deliberate and documented in the tests:

1. ``match_indices`` in the reference picks, for each occurrence, the counterpart with the
   smallest *positive* time difference and, when none exists, ``argmin`` over an all-``inf``
   array silently returns index 0 (an unrelated entry). Here the fallback is the closest
   earlier counterpart instead, so an early prediction becomes the FP the docstring intends.
2. A video without a single matched completion has no delay; the reference substitutes
   100 frames, this module returns ``None`` and leaves aggregation to the caller.
3. When a step id occurs more than once, the reference only matches candidates *strictly* later
   than the target, so a prediction at exactly the ground-truth frame is skipped and the ground
   truth scored against itself is not perfect. Here "at or after" is used, which is what the
   single-occurrence path (``delta >= 0`` is a TP with delay 0) already implies.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np

POS_COSTS: dict[str, float] = {"insert": 1.0, "delete": 1.0, "substitute": 2.0, "transpose": 1.0}
"""Edit costs the reference passes to ``weighted_levenshtein.dam_lev`` for POS."""


def damerau_levenshtein(
    a: Sequence[Hashable],
    b: Sequence[Hashable],
    insert: float = 1.0,
    delete: float = 1.0,
    substitute: float = 1.0,
    transpose: float = 1.0,
) -> float:
    """Unrestricted Damerau-Levenshtein distance (Lowrance-Wagner) with per-operation costs.

    Unlike the optimal-string-alignment variant, a transposed pair may be edited again, e.g.
    ``CA -> ABC`` costs 2 (transpose, insert) rather than 3.
    """
    n, m = len(a), len(b)
    inf = (n + m + 1) * max(insert, delete, substitute, transpose, 1.0) + 1.0
    d = np.full((n + 2, m + 2), inf, dtype=np.float64)
    for i in range(n + 1):
        d[i + 1, 1] = i * delete
    for j in range(m + 1):
        d[1, j + 1] = j * insert
    last_row: dict[Hashable, int] = defaultdict(int)
    for i in range(1, n + 1):
        last_col = 0
        for j in range(1, m + 1):
            k = last_row[b[j - 1]]
            col = last_col
            if a[i - 1] == b[j - 1]:
                cost = 0.0
                last_col = j
            else:
                cost = substitute
            d[i + 1, j + 1] = min(
                d[i, j] + cost,
                d[i + 1, j] + insert,
                d[i, j + 1] + delete,
                d[k, col] + (i - k - 1) * delete + transpose + (j - col - 1) * insert,
            )
        last_row[a[i - 1]] = i
    return float(d[n + 1, m + 1])


def procedure_order_similarity(
    pred_order: Sequence[Hashable], gt_order: Sequence[Hashable]
) -> float:
    """POS in ``[0, 1]`` with the reference costs and ``len(gt)`` normalisation; 1.0 is perfect."""
    if not gt_order:
        return 1.0 if not pred_order else 0.0
    distance = damerau_levenshtein(gt_order, pred_order, **POS_COSTS)
    return 1.0 - min(distance / len(gt_order), 1.0)


def f1_reference(fn: int, fp: int, tp: int) -> float:
    """The reference's epsilon-smoothed F1 (``get_f1_score``), reproduced for comparability."""
    positives = tp + fn
    predicted = tp + fp
    precision = tp / predicted if predicted else 1e-6
    recall = tp / positives if positives else 1e-6
    return 2 * (precision * recall) / (precision + recall + 1e-6)


@dataclass(frozen=True)
class Completion:
    """One step completion: frame index, step id, and whether the system *observed* it.

    ``observed=False`` marks completions a system only implied from procedure knowledge (the
    reference's ``conf == 0``); ground-truth completions are always observed.
    """

    frame: int
    step: Hashable
    observed: bool = True


def _match(
    candidates: list[int], candidate_times: np.ndarray, targets: list[int], target_times: np.ndarray
) -> list[int]:
    """For each target pick the unused candidate closest at-or-after it, else the closest before it."""
    if len(candidates) < len(targets):
        raise ValueError("need at least as many candidates as targets")
    unused = list(candidates)
    matched: list[int] = []
    for target in targets:
        t = target_times[target]
        later = [c for c in unused if candidate_times[c] >= t]
        pool = later if later else unused
        best = min(pool, key=lambda c: abs(candidate_times[c] - t))
        unused.remove(best)
        matched.append(best)
    return matched


def psr_performance(
    gt: Sequence[Completion], pred: Sequence[Completion]
) -> dict[str, float | int | None]:
    """System-level PSR metrics of one video, following the reference ``determine_performance``."""
    gt_times = np.array([c.frame for c in gt], dtype=np.int64)
    pred_times = np.array([c.frame for c in pred], dtype=np.int64)
    gt_order = [c.step for c in gt]
    pred_order = [c.step for c in pred]
    sys_fn = sys_fp = 0
    delays: list[float] = []
    for step in sorted(set(gt_order) | set(pred_order), key=repr):
        idx_gt = [i for i, s in enumerate(gt_order) if s == step]
        idx_pred = [i for i, s in enumerate(pred_order) if s == step]
        if not idx_gt:
            sys_fp += len(idx_pred)
            continue
        if not idx_pred:
            sys_fn += len(idx_gt)
            continue
        if len(idx_gt) > len(idx_pred):
            sys_fn += len(idx_gt) - len(idx_pred)
            idx_gt = _match(idx_gt, gt_times, idx_pred, pred_times)
        elif len(idx_pred) > len(idx_gt) or len(idx_pred) > 1:
            sys_fp += max(0, len(idx_pred) - len(idx_gt))
            idx_pred = _match(idx_pred, pred_times, idx_gt, gt_times)
        for g, p in zip(idx_gt, idx_pred, strict=True):
            delta = int(pred_times[p] - gt_times[g])
            if delta < 0:
                # Early completion: an *observed* one is a system FP (reference ``conf == 1``);
                # an implied one only counts at perception level, which is not tracked here.
                if pred[p].observed:
                    sys_fp += 1
            else:
                delays.append(delta)
    sys_tp = len(pred_order) - sys_fp
    return {
        "system_tp": sys_tp,
        "system_fp": sys_fp,
        "system_fn": sys_fn,
        "f1": f1_reference(sys_fn, sys_fp, sys_tp),
        "pos": procedure_order_similarity(pred_order, gt_order),
        "mean_delay_frames": float(np.mean(delays)) if delays else None,
        "n_gt": len(gt_order),
        "n_pred": len(pred_order),
    }
