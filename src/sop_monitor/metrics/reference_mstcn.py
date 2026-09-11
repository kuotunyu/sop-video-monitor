"""Independent transcription of the MS-TCN reference evaluation, used only for cross-checks.

:mod:`sop_monitor.metrics.offline` is the implementation the project reports. This module
mirrors the control flow of the reference ``eval.py`` shipped with MS-TCN (Farha and Gall,
2019; MIT) — segment extraction by label change, edit score via a full Levenshtein table,
F1 by best-IoU matching with a ``hits`` mask — so that a disagreement between the two would
point at a definition misread rather than a shared bug. ``tests/test_metrics_reference.py``
compares both on random inputs and ``score-predictions`` re-runs the comparison on every real
prediction table it scores.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Sequence

import numpy as np


def _labels_start_end(
    frame_labels: Sequence[Hashable], background: Collection[Hashable]
) -> tuple[list[Hashable], list[int], list[int]]:
    labels: list[Hashable] = []
    starts: list[int] = []
    ends: list[int] = []
    if not frame_labels:
        return labels, starts, ends
    last = frame_labels[0]
    if last not in background:
        labels.append(last)
        starts.append(0)
    i = 0
    for i, current in enumerate(frame_labels):
        if current != last:
            if current not in background:
                labels.append(current)
                starts.append(i)
            if last not in background:
                ends.append(i)
            last = current
    if last not in background:
        ends.append(i + 1)
    return labels, starts, ends


def _levenshtein_table(p: Sequence[Hashable], y: Sequence[Hashable]) -> float:
    rows, cols = len(p), len(y)
    table = np.zeros((rows + 1, cols + 1), dtype=np.float64)
    table[:, 0] = np.arange(rows + 1)
    table[0, :] = np.arange(cols + 1)
    for j in range(1, cols + 1):
        for i in range(1, rows + 1):
            if y[j - 1] == p[i - 1]:
                table[i, j] = table[i - 1, j - 1]
            else:
                table[i, j] = min(table[i - 1, j] + 1, table[i, j - 1] + 1, table[i - 1, j - 1] + 1)
    return float(table[rows, cols])


def _edit_score(
    recognized: Sequence[Hashable],
    ground_truth: Sequence[Hashable],
    background: Collection[Hashable],
) -> float:
    p, _, _ = _labels_start_end(recognized, background)
    y, _, _ = _labels_start_end(ground_truth, background)
    longest = max(len(p), len(y))
    if longest == 0:
        return 100.0
    return (1.0 - _levenshtein_table(p, y) / longest) * 100.0


def _f_score_counts(
    recognized: Sequence[Hashable],
    ground_truth: Sequence[Hashable],
    overlap: float,
    background: Collection[Hashable],
) -> tuple[float, float, float]:
    p_label, p_start, p_end = _labels_start_end(recognized, background)
    y_label, y_start, y_end = _labels_start_end(ground_truth, background)
    tp = fp = 0.0
    hits = np.zeros(len(y_label))
    y_start_arr = np.asarray(y_start, dtype=np.float64)
    y_end_arr = np.asarray(y_end, dtype=np.float64)
    for j in range(len(p_label)):
        if not y_label:
            fp += 1
            continue
        intersection = np.minimum(p_end[j], y_end_arr) - np.maximum(p_start[j], y_start_arr)
        union = np.maximum(p_end[j], y_end_arr) - np.minimum(p_start[j], y_start_arr)
        same = np.asarray([p_label[j] == y_label[x] for x in range(len(y_label))], dtype=np.float64)
        iou = (intersection / union) * same
        idx = int(np.argmax(iou))
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - float(hits.sum())
    return tp, fp, fn


def evaluate(
    predictions: Sequence[Sequence[Hashable]],
    ground_truths: Sequence[Sequence[Hashable]],
    ks: Sequence[int] = (10, 25, 50),
    background: Collection[Hashable] = (),
) -> dict[str, float]:
    """``{"mof", "edit", "f1@k"...}`` over a list of videos, following the reference main loop."""
    if len(predictions) != len(ground_truths) or not predictions:
        raise ValueError("need the same non-zero number of predictions and ground truths")
    overlaps = [k / 100.0 for k in ks]
    tp = np.zeros(len(overlaps))
    fp = np.zeros(len(overlaps))
    fn = np.zeros(len(overlaps))
    correct = total = 0
    edit = 0.0
    for recognized, gt in zip(predictions, ground_truths, strict=True):
        if len(recognized) != len(gt):
            raise ValueError("length mismatch")
        for i in range(len(gt)):
            total += 1
            if gt[i] == recognized[i]:
                correct += 1
        edit += _edit_score(recognized, gt, background)
        for s, overlap in enumerate(overlaps):
            tp1, fp1, fn1 = _f_score_counts(recognized, gt, overlap, background)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
    out = {"mof": 100.0 * correct / total, "edit": edit / len(predictions)}
    for s, k in enumerate(ks):
        precision = tp[s] / (tp[s] + fp[s]) if tp[s] + fp[s] else 0.0
        recall = tp[s] / (tp[s] + fn[s]) if tp[s] + fn[s] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[f"f1@{k}"] = float(np.nan_to_num(f1) * 100.0)
    return out
