"""IndustReal development baseline: frozen frame features -> linear head -> per-frame steps.

This is the first end-to-end slice of the pipeline (video + labels -> alignment -> per-time
prediction -> spec 4.2 metrics -> recomputable report). It is deliberately small:

- features are frozen DINOv2 frame embeddings (:mod:`sop_monitor.features`), one per sampled
  frame, cached under ``artifacts/``;
- the model is a multinomial logistic regression trained full-batch on the train split;
- ``frame`` uses one frame at a time, ``causal`` averages the posteriors of the last ``window``
  frames (past only, so it is an honest online baseline), ``offline`` averages a centered window
  and therefore looks into the future — it is *not* an online result and is labelled as such;
- ``majority`` predicts the most frequent training label everywhere (trivial floor).

All hyper-parameters are chosen on the val split only; the frozen test split is never read.
The prediction table (``predictions_<split>.csv``) is the artefact everything else is derived
from: ``score_predictions`` recomputes ``metrics.json`` from it without video, torch or GPU,
and cross-checks the numbers against :mod:`sop_monitor.metrics.reference_mstcn`.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sop_monitor.industreal import (
    BACKGROUND,
    LABEL_FILES,
    Segment,
    frame_labels,
    participant_of,
    read_labels,
    segments_by_video,
)
from sop_monitor.metrics.offline import (
    DEFAULT_KS,
    SequenceStats,
    combine_sequence_stats,
    sequence_stats,
    subject_bootstrap,
    tas_metrics,
)
from sop_monitor.metrics.reference_mstcn import evaluate as reference_evaluate

PRED_PREFIX = "pred_"

# ---------------------------------------------------------------------------------------------
# Smoothing and label bookkeeping (numpy only)


def causal_smooth(probs: np.ndarray, window: int) -> np.ndarray:
    """Mean of ``probs[t - window + 1 : t + 1]`` per row: uses the present and the past only."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1:
        return probs.copy()
    padded = np.concatenate(
        [np.zeros((1, probs.shape[1]), dtype=np.float64), np.cumsum(probs, axis=0)]
    )
    t = np.arange(probs.shape[0])
    lo = np.maximum(0, t - window + 1)
    out = (padded[t + 1] - padded[lo]) / (t + 1 - lo)[:, None]
    return out.astype(probs.dtype)


def centered_smooth(probs: np.ndarray, radius: int) -> np.ndarray:
    """Mean of ``probs[t - radius : t + radius + 1]`` per row: looks ``radius`` frames ahead."""
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if radius == 0:
        return probs.copy()
    n = probs.shape[0]
    padded = np.concatenate(
        [np.zeros((1, probs.shape[1]), dtype=np.float64), np.cumsum(probs, axis=0)]
    )
    t = np.arange(n)
    lo = np.maximum(0, t - radius)
    hi = np.minimum(n, t + radius + 1)
    out = (padded[hi] - padded[lo]) / (hi - lo)[:, None]
    return out.astype(probs.dtype)


def label_index_map(action_ids: Iterable[int]) -> dict[int, int]:
    """Dense class indices: ``BACKGROUND -> 0``, then the sorted distinct action ids from 1."""
    distinct = sorted(set(action_ids))
    if BACKGROUND in distinct:
        raise ValueError("action ids must not contain the background label")
    return {BACKGROUND: 0, **{action: index for index, action in enumerate(distinct, start=1)}}


# ---------------------------------------------------------------------------------------------
# Prediction tables


@dataclass(frozen=True)
class PredictionRow:
    """One sampled frame of one video with its ground truth and every run's prediction."""

    video_id: str
    participant: str
    frame: int
    gt: int
    preds: dict[str, int] = field(default_factory=dict)


def write_predictions(path: Path, rows: Sequence[PredictionRow]) -> None:
    if not rows:
        raise ValueError("no rows")
    runs = list(rows[0].preds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            ",".join(["video_id", "participant", "frame", "gt", *(PRED_PREFIX + r for r in runs)])
        )
        handle.write("\n")
        for row in rows:
            if list(row.preds) != runs:
                raise ValueError(
                    f"row {row.video_id}@{row.frame} has runs {list(row.preds)} != {runs}"
                )
            handle.write(
                ",".join(
                    [
                        row.video_id,
                        row.participant,
                        str(row.frame),
                        str(row.gt),
                        *map(str, row.preds.values()),
                    ]
                )
            )
            handle.write("\n")


def read_predictions(path: Path) -> list[PredictionRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[:4] != ["video_id", "participant", "frame", "gt"]:
            raise ValueError(f"unexpected prediction header: {header}")
        runs = [name.removeprefix(PRED_PREFIX) for name in header[4:]]
        if any(not name.startswith(PRED_PREFIX) for name in header[4:]):
            raise ValueError(f"prediction columns must start with {PRED_PREFIX!r}: {header[4:]}")
        return [
            PredictionRow(
                video_id=row[0],
                participant=row[1],
                frame=int(row[2]),
                gt=int(row[3]),
                preds=dict(zip(runs, map(int, row[4:]), strict=True)),
            )
            for row in reader
            if row
        ]


def _by_video(rows: Sequence[PredictionRow]) -> dict[str, list[PredictionRow]]:
    grouped: dict[str, list[PredictionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.video_id].append(row)
    return {video: sorted(group, key=lambda r: r.frame) for video, group in grouped.items()}


def score_predictions(
    rows: Sequence[PredictionRow],
    n_boot: int = 2000,
    seed: int = 0,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, object]:
    """Spec 4.2 metrics per run with participant-bootstrap CIs, from a prediction table alone.

    Point estimates come from :func:`tas_metrics` on the raw sequences; the bootstrap resamples
    participants using per-video :class:`SequenceStats` and must reproduce the same point; the
    reference transcription must agree with both. Any disagreement raises.
    """
    if not rows:
        raise ValueError("no rows")
    videos = _by_video(rows)
    runs = list(rows[0].preds)
    background = (BACKGROUND,)
    gts = {video: [r.gt for r in group] for video, group in videos.items()}
    participants = {video: group[0].participant for video, group in videos.items()}
    out: dict[str, object] = {
        "n_videos": len(videos),
        "n_participants": len(set(participants.values())),
        "n_frames": len(rows),
        "background_label": BACKGROUND,
        "bootstrap": {"unit": "participant", "n_boot": n_boot, "seed": seed, "alpha": 0.05},
        "runs": {},
        "per_video": {},
    }
    for run in runs:
        preds = {video: [r.preds[run] for r in group] for video, group in videos.items()}
        pairs = [(preds[v], gts[v]) for v in videos]
        point = tas_metrics(pairs, ks=ks, background=background)
        reference = reference_evaluate([p for p, _ in pairs], [g for _, g in pairs], ks, background)
        for key, value in point.items():
            if abs(value - reference[key]) > 1e-6:
                raise ValueError(f"{run}/{key}: offline={value} reference={reference[key]}")
        stats = {v: sequence_stats(preds[v], gts[v], ks=ks, background=background) for v in videos}
        per_subject: dict[str, list[SequenceStats]] = defaultdict(list)
        for video, stat in stats.items():
            per_subject[participants[video]].append(stat)
        summary: dict[str, dict[str, object]] = {}
        for key, value in point.items():
            boot = subject_bootstrap(
                per_subject,  # type: ignore[arg-type]
                lambda items, key=key: combine_sequence_stats(items)[key],  # type: ignore[arg-type]
                n_boot=n_boot,
                seed=seed,
            )
            if abs(boot.point - value) > 1e-6:
                raise ValueError(f"{run}/{key}: stats point {boot.point} != {value}")
            summary[key] = {"point": value, "ci95": [boot.lower, boot.upper]}
        out["runs"][run] = summary  # type: ignore[index]
        out["per_video"][run] = {  # type: ignore[index]
            video: combine_sequence_stats([stat]) for video, stat in sorted(stats.items())
        }
    out["reference_crosscheck"] = "agree"
    return out


# ---------------------------------------------------------------------------------------------
# Linear head (torch for training, numpy for inference)


@dataclass(frozen=True)
class HeadConfig:
    weight_decay: float = 1e-4
    epochs: int = 300
    lr: float = 5e-3
    seed: int = 0


@dataclass(frozen=True)
class LinearHead:
    """Standardisation statistics plus one affine layer; inference is plain numpy."""

    mean: np.ndarray
    std: np.ndarray
    weight: np.ndarray  # (n_classes, dim)
    bias: np.ndarray  # (n_classes,)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = ((x.astype(np.float32) - self.mean) / self.std) @ self.weight.T + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return (probs / probs.sum(axis=1, keepdims=True)).astype(np.float32)


def train_linear_head(
    x: np.ndarray, y: np.ndarray, n_classes: int, config: HeadConfig, device: str = "cpu"
) -> tuple[LinearHead, list[float]]:
    """Full-batch multinomial logistic regression with Adam; returns the head and the loss curve."""
    import torch

    mean = x.astype(np.float32).mean(axis=0)
    std = x.astype(np.float32).std(axis=0) + 1e-6
    torch.manual_seed(config.seed)
    xt = torch.from_numpy((x.astype(np.float32) - mean) / std).to(device)
    yt = torch.from_numpy(y.astype(np.int64)).to(device)
    layer = torch.nn.Linear(x.shape[1], n_classes).to(device)
    optimiser = torch.optim.Adam(layer.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    losses: list[float] = []
    for _ in range(config.epochs):
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(layer(xt), yt)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.item()))
    return (
        LinearHead(
            mean=mean,
            std=std,
            weight=layer.weight.detach().cpu().numpy().astype(np.float32),
            bias=layer.bias.detach().cpu().numpy().astype(np.float32),
        ),
        losses,
    )


# ---------------------------------------------------------------------------------------------
# Feature cache + labels -> aligned arrays


@dataclass(frozen=True)
class VideoFeatures:
    video_id: str
    participant: str
    frames: np.ndarray  # sampled frame indices, int64
    features: np.ndarray  # (n_sampled, dim) float16/float32
    gt: np.ndarray  # raw action ids per sampled frame, BACKGROUND where unlabelled


def load_split_features(
    features_dir: Path, labels_dir: Path, split: str
) -> tuple[list[VideoFeatures], list[Segment]]:
    """Load cached features of every labelled video of ``split`` and align the labels to them."""
    segments = read_labels(labels_dir / LABEL_FILES[split])
    loaded: list[VideoFeatures] = []
    for video_id, video_segments in sorted(segments_by_video(segments).items()):
        cache = features_dir / f"{video_id}.npz"
        if not cache.is_file():
            raise FileNotFoundError(f"no cached features for {video_id}: {cache}")
        with np.load(cache) as payload:
            frames = payload["frames"].astype(np.int64)
            features = payload["features"]
            n_frames = int(payload["n_frames"])
        labels = frame_labels(video_segments, n_frames)
        loaded.append(
            VideoFeatures(
                video_id=video_id,
                participant=participant_of(video_id),
                frames=frames,
                features=features,
                gt=labels[frames],
            )
        )
    return loaded, segments


# ---------------------------------------------------------------------------------------------
# End-to-end development run


@dataclass(frozen=True)
class BaselineSpec:
    weight_decays: tuple[float, ...] = (1e-5, 1e-4, 1e-3)
    causal_windows: tuple[int, ...] = (1, 5, 10, 20, 40)
    offline_radii: tuple[int, ...] = (0, 2, 5, 10, 20)
    epochs: int = 300
    lr: float = 5e-3
    seed: int = 0
    n_boot: int = 2000
    selection_metric: str = "mof"


def _stack(videos: Sequence[VideoFeatures]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([v.features for v in videos]).astype(np.float32),
        np.concatenate([v.gt for v in videos]),
    )


def _split_probs(videos: Sequence[VideoFeatures], probs: np.ndarray) -> list[np.ndarray]:
    out = []
    offset = 0
    for video in videos:
        out.append(probs[offset : offset + len(video.frames)])
        offset += len(video.frames)
    return out


def _val_metric(
    videos: Sequence[VideoFeatures], probs: Sequence[np.ndarray], inverse: np.ndarray, metric: str
) -> float:
    pairs = [
        (inverse[p.argmax(axis=1)].tolist(), v.gt.tolist())
        for v, p in zip(videos, probs, strict=True)
    ]
    return tas_metrics(pairs, background=(BACKGROUND,))[metric]


def run_baseline(
    features_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    spec: BaselineSpec | None = None,
    device: str = "cpu",
    train_split: str = "train",
    eval_split: str = "val",
) -> dict[str, object]:
    """Train on ``train_split``, select and report on ``eval_split``; write the run directory."""
    if eval_split == "test":
        raise ValueError("the frozen test split is not evaluated by the development baseline")
    spec = spec or BaselineSpec()
    started = time.perf_counter()
    train_videos, _ = load_split_features(features_dir, labels_dir, train_split)
    eval_videos, _ = load_split_features(features_dir, labels_dir, eval_split)
    x_train, y_train_raw = _stack(train_videos)
    x_eval, _ = _stack(eval_videos)
    index = label_index_map(int(a) for a in y_train_raw if a != BACKGROUND)
    inverse = np.array([action for action, _ in sorted(index.items(), key=lambda kv: kv[1])])
    y_train = np.array([index[int(a)] for a in y_train_raw], dtype=np.int64)
    majority = int(inverse[np.bincount(y_train).argmax()])

    # 1. weight decay on the eval split (frame-wise, no smoothing).
    selection: dict[str, object] = {
        "metric": spec.selection_metric,
        "weight_decay": {},
        "causal_window": {},
        "offline_radius": {},
    }
    heads: dict[float, tuple[LinearHead, list[float]]] = {}
    eval_probs: dict[float, np.ndarray] = {}
    for wd in spec.weight_decays:
        config = HeadConfig(weight_decay=wd, epochs=spec.epochs, lr=spec.lr, seed=spec.seed)
        t0 = time.perf_counter()
        head, losses = train_linear_head(x_train, y_train, len(index), config, device=device)
        heads[wd] = (head, losses)
        eval_probs[wd] = head.predict_proba(x_eval)
        score = _val_metric(
            eval_videos, _split_probs(eval_videos, eval_probs[wd]), inverse, spec.selection_metric
        )
        selection["weight_decay"][str(wd)] = {  # type: ignore[index]
            spec.selection_metric: score,
            "final_train_loss": losses[-1],
            "train_seconds": time.perf_counter() - t0,
        }
    best_wd = max(
        spec.weight_decays, key=lambda wd: selection["weight_decay"][str(wd)][spec.selection_metric]
    )  # type: ignore[index]
    probs = _split_probs(eval_videos, eval_probs[best_wd])

    # 2. smoothing windows, each on the selected head.
    for window in spec.causal_windows:
        smoothed = [causal_smooth(p, window) for p in probs]
        selection["causal_window"][str(window)] = _val_metric(
            eval_videos, smoothed, inverse, spec.selection_metric
        )  # type: ignore[index]
    for radius in spec.offline_radii:
        smoothed = [centered_smooth(p, radius) for p in probs]
        selection["offline_radius"][str(radius)] = _val_metric(
            eval_videos, smoothed, inverse, spec.selection_metric
        )  # type: ignore[index]
    best_window = max(spec.causal_windows, key=lambda w: selection["causal_window"][str(w)])  # type: ignore[index]
    best_radius = max(spec.offline_radii, key=lambda r: selection["offline_radius"][str(r)])  # type: ignore[index]

    # 3. final prediction table for the eval split.
    rows: list[PredictionRow] = []
    for video, p in zip(eval_videos, probs, strict=True):
        frame_pred = inverse[p.argmax(axis=1)]
        causal_pred = inverse[causal_smooth(p, best_window).argmax(axis=1)]
        offline_pred = inverse[centered_smooth(p, best_radius).argmax(axis=1)]
        for i, frame in enumerate(video.frames.tolist()):
            rows.append(
                PredictionRow(
                    video.video_id,
                    video.participant,
                    frame,
                    int(video.gt[i]),
                    {
                        "frame": int(frame_pred[i]),
                        "causal": int(causal_pred[i]),
                        "offline": int(offline_pred[i]),
                        "majority": majority,
                    },
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / f"predictions_{eval_split}.csv", rows)
    metrics = score_predictions(rows, n_boot=spec.n_boot, seed=spec.seed)
    metrics["confusions"] = top_confusions(rows, "causal")
    metrics["eval_split"] = eval_split
    metrics["train_split"] = train_split
    cache_meta = (
        json.loads((features_dir / "meta.json").read_text(encoding="utf-8"))
        if (features_dir / "meta.json").is_file()
        else {}
    )
    config = {
        "spec": asdict(spec),
        "selected": {
            "weight_decay": best_wd,
            "causal_window": best_window,
            "offline_radius": best_radius,
        },
        "selection": selection,
        "head": {
            "type": "multinomial logistic regression, full-batch Adam",
            **asdict(HeadConfig(best_wd, spec.epochs, spec.lr, spec.seed)),
        },
        "classes": {
            "n_classes": len(index),
            "background_index": 0,
            "index_to_action_id": inverse.tolist(),
            "majority_action_id": majority,
        },
        "features": cache_meta,
        "runs": {
            "frame": "argmax of the per-frame posterior (no temporal context)",
            "causal": f"argmax of the mean posterior over the last {best_window} sampled frames (past only)",
            "offline": f"argmax of the mean posterior over +-{best_radius} sampled frames (uses future frames; not an online result)",
            "majority": "constant most frequent training label",
        },
        "train_videos": len(train_videos),
        "train_frames": len(y_train),
        "eval_videos": len(eval_videos),
        "eval_frames": len(rows),
        "device": device,
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"config": config, "metrics": metrics}


def top_confusions(
    rows: Sequence[PredictionRow], run: str, limit: int = 15
) -> list[dict[str, object]]:
    """Most frequent ``(gt, pred)`` disagreements of one run, in frames (background included)."""
    counter: Counter[tuple[int, int]] = Counter()
    for row in rows:
        pred = row.preds[run]
        if pred != row.gt:
            counter[(row.gt, pred)] += 1
    return [{"gt": gt, "pred": pred, "frames": n} for (gt, pred), n in counter.most_common(limit)]


def load_metrics(run_dir: Path) -> Mapping[str, object]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


RUN_ORDER: tuple[str, ...] = ("majority", "frame", "causal", "offline")
"""Table row order: trivial floor first, then increasing temporal context; other runs follow, sorted."""


def _ordered_runs(names: Iterable[str]) -> list[str]:
    known = [run for run in RUN_ORDER if run in set(names)]
    return known + sorted(set(names) - set(RUN_ORDER))


def _fmt(entry: Mapping[str, object]) -> str:
    lo, hi = entry["ci95"]  # type: ignore[misc]
    return f"{entry['point']:.1f} [{lo:.1f}, {hi:.1f}]"


def render_tables(metrics: Mapping[str, object], config: Mapping[str, object]) -> str:
    """Markdown tables derived from ``metrics.json`` + ``config.json`` only (no data needed).

    Row order is :data:`RUN_ORDER`, never dictionary order, so the in-memory result and the
    ``sort_keys`` JSON on disk render identically.
    """
    unordered: Mapping[str, Mapping[str, Mapping[str, object]]] = metrics["runs"]  # type: ignore[assignment]
    runs = {run: unordered[run] for run in _ordered_runs(unordered)}
    descriptions: Mapping[str, str] = config.get("runs", {})  # type: ignore[assignment]
    features: Mapping[str, object] = config.get("features", {})  # type: ignore[assignment]
    boot: Mapping[str, object] = metrics.get("bootstrap", {})  # type: ignore[assignment]
    lines = [
        f"Development result on `{metrics.get('eval_split', 'val')}` "
        f"({metrics['n_videos']} videos, {metrics['n_participants']} participants, "
        f"{metrics['n_frames']} sampled frames); features `{features.get('model', '?')}` "
        f"stride {features.get('stride', '?')}; 95% CI = participant bootstrap "
        f"({boot.get('n_boot', '?')} draws, seed {boot.get('seed', '?')}).",
        "",
        "| run | MoF | Edit | F1@10 | F1@25 | F1@50 |",
        "|---|---|---|---|---|---|",
    ]
    for run, entry in runs.items():
        lines.append(
            f"| {run} | {_fmt(entry['mof'])} | {_fmt(entry['edit'])} | {_fmt(entry['f1@10'])} | "
            f"{_fmt(entry['f1@25'])} | {_fmt(entry['f1@50'])} |"
        )
    if descriptions:
        lines += ["", "| run | what it uses |", "|---|---|"]
        lines += [f"| {run} | {descriptions.get(run, '')} |" for run in runs]
    per_video: Mapping[str, Mapping[str, Mapping[str, float]]] = metrics.get("per_video", {})  # type: ignore[assignment]
    if per_video:
        run_names = _ordered_runs(per_video)
        lines += [
            "",
            "Per-video MoF:",
            "",
            "| video | " + " | ".join(run_names) + " |",
            "|---|" + "---|" * len(run_names),
        ]
        for video in sorted(next(iter(per_video.values()))):
            lines.append(
                f"| {video} | "
                + " | ".join(f"{per_video[r][video]['mof']:.1f}" for r in run_names)
                + " |"
            )
    confusions: Sequence[Mapping[str, object]] = metrics.get("confusions", [])  # type: ignore[assignment]
    if confusions:
        lines += [
            "",
            "Most frequent causal-run confusions (gt -> pred, frames; -1 = background):",
            "",
            "| gt | pred | frames |",
            "|---|---|---|",
        ]
        lines += [f"| {c['gt']} | {c['pred']} | {c['frames']} |" for c in confusions]
    return "\n".join(lines) + "\n"
