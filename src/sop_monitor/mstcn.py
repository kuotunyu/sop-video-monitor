"""MS-TCN++ temporal head (spec 6.2) on cached frame features, in a causal and a non-causal form.

Architecture follows Li et al., "MS-TCN++: Multi-Stage Temporal Convolutional Network for Action
Segmentation" (TPAMI 2020): one prediction-generation stage of dual-dilated layers followed by
refinement stages of dilated residual layers, trained with cross-entropy plus the truncated MSE
smoothing term of MS-TCN. The only change is the padding: with ``causal=True`` every kernel-3
convolution is left-padded by ``2 * dilation`` and never right-padded, so the output at frame
``t`` depends on frames ``<= t`` only (``tests/test_mstcn.py`` checks this numerically). The
non-causal twin pads symmetrically and is the offline reference that quantifies what causality
costs. Everything here is trained on the train split; the epoch is selected on val.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sop_monitor.baseline import (
    PredictionRow,
    VideoFeatures,
    label_index_map,
    load_metrics,
    load_split_features,
    render_tables,
    score_predictions,
    top_confusions,
    write_predictions,
)
from sop_monitor.industreal import BACKGROUND
from sop_monitor.metrics.offline import tas_metrics


@dataclass(frozen=True)
class MSTCNSpec:
    num_layers_pg: int = 11
    num_layers_r: int = 10
    num_refinement_stages: int = 3
    num_f_maps: int = 64
    dropout: float = 0.5
    lr: float = 5e-4
    epochs: int = 50
    eval_every: int = 5
    smoothing_weight: float = 0.15
    smoothing_clamp: float = 16.0
    seed: int = 0
    n_boot: int = 2000
    selection_metric: str = "mof"


def _build(dim: int, n_classes: int, spec: MSTCNSpec, causal: bool, activation: str = "softmax"):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class DilatedConv(nn.Module):
        """Kernel-3 dilated conv; causal = left padding only."""

        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.dilation = dilation
            self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pad = 2 * self.dilation
            x = F.pad(x, (pad, 0) if causal else (pad // 2, pad // 2))
            return self.conv(x)

    class DilatedResidualLayer(nn.Module):
        def __init__(self, dilation: int, channels: int) -> None:
            super().__init__()
            self.conv_dilated = DilatedConv(channels, dilation)
            self.conv_1x1 = nn.Conv1d(channels, channels, 1)
            self.dropout = nn.Dropout(spec.dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.relu(self.conv_dilated(x))
            out = self.dropout(self.conv_1x1(out))
            return x + out

    class DualDilatedLayer(nn.Module):
        def __init__(self, dilation_a: int, dilation_b: int, channels: int) -> None:
            super().__init__()
            self.conv_a = DilatedConv(channels, dilation_a)
            self.conv_b = DilatedConv(channels, dilation_b)
            self.conv_fusion = nn.Conv1d(2 * channels, channels, 1)
            self.dropout = nn.Dropout(spec.dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.cat([self.conv_a(x), self.conv_b(x)], dim=1)
            out = self.dropout(F.relu(self.conv_fusion(out)))
            return x + out

    class PredictionGeneration(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_in = nn.Conv1d(dim, spec.num_f_maps, 1)
            layers = spec.num_layers_pg
            self.layers = nn.ModuleList(
                [
                    DualDilatedLayer(2**i, 2 ** (layers - 1 - i), spec.num_f_maps)
                    for i in range(layers)
                ]
            )
            self.conv_out = nn.Conv1d(spec.num_f_maps, n_classes, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv_in(x)
            for layer in self.layers:
                out = layer(out)
            return self.conv_out(out)

    class Refinement(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv_in = nn.Conv1d(n_classes, spec.num_f_maps, 1)
            self.layers = nn.ModuleList(
                [DilatedResidualLayer(2**i, spec.num_f_maps) for i in range(spec.num_layers_r)]
            )
            self.conv_out = nn.Conv1d(spec.num_f_maps, n_classes, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv_in(x)
            for layer in self.layers:
                out = layer(out)
            return self.conv_out(out)

    class MSTCNpp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pg = PredictionGeneration()
            self.stages = nn.ModuleList([Refinement() for _ in range(spec.num_refinement_stages)])

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            out = self.pg(x)
            outputs = [out]
            for stage in self.stages:
                out = stage(
                    F.softmax(out, dim=1) if activation == "softmax" else torch.sigmoid(out)
                )
                outputs.append(out)
            return outputs

    return MSTCNpp()


def build_model(
    dim: int,
    n_classes: int,
    spec: MSTCNSpec | None = None,
    causal: bool = True,
    activation: str = "softmax",
):
    """Public constructor (``torch.nn.Module``); input ``(batch, dim, T)``, output list of stage logits.

    ``activation`` is what the refinement stages see: ``softmax`` for mutually exclusive classes,
    ``sigmoid`` for independent (multi-label) outputs such as component states.
    """
    if activation not in ("softmax", "sigmoid"):
        raise ValueError("activation must be 'softmax' or 'sigmoid'")
    return _build(dim, n_classes, spec or MSTCNSpec(), causal, activation)


def _loss(outputs, targets, n_classes: int, spec: MSTCNSpec):
    import torch
    from torch.nn import functional as F

    total = torch.zeros((), device=targets.device)
    for logits in outputs:
        total = total + F.cross_entropy(
            logits.transpose(1, 2).reshape(-1, n_classes), targets.reshape(-1)
        )
        log_probs = F.log_softmax(logits, dim=1)
        smooth = torch.clamp(
            F.mse_loss(log_probs[:, :, 1:], log_probs.detach()[:, :, :-1], reduction="none"),
            max=spec.smoothing_clamp,
        )
        total = total + spec.smoothing_weight * smooth.mean()
    return total


@dataclass(frozen=True)
class Standardiser:
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float32) - self.mean) / self.std).astype(np.float32)


def _predict(
    model, videos: Sequence[VideoFeatures], standardise: Standardiser, device: str
) -> list[np.ndarray]:
    """Per-video posterior of the last stage, ``(T, n_classes)`` float32."""
    import torch

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for video in videos:
            x = torch.from_numpy(standardise(video.features)).T.unsqueeze(0).to(device)
            logits = model(x)[-1]
            outputs.append(torch.softmax(logits, dim=1)[0].T.cpu().numpy())
    return outputs


def _metric(
    videos: Sequence[VideoFeatures], probs: Sequence[np.ndarray], inverse: np.ndarray, metric: str
) -> float:
    pairs = [
        (inverse[p.argmax(axis=1)].tolist(), v.gt.tolist())
        for v, p in zip(videos, probs, strict=True)
    ]
    return tas_metrics(pairs, background=(BACKGROUND,))[metric]


def train_mstcn(
    train_videos: Sequence[VideoFeatures],
    eval_videos: Sequence[VideoFeatures],
    index: dict[int, int],
    inverse: np.ndarray,
    spec: MSTCNSpec,
    causal: bool,
    device: str,
    checkpoint: Path | None = None,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Train one MS-TCN++ variant; return the eval posteriors of the best val epoch and a log.

    With ``checkpoint`` the weights of the selected epoch are saved there together with the
    feature standardiser (:func:`load_checkpoint` restores both); training is unchanged.
    """
    import torch

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x_train = np.concatenate([v.features for v in train_videos]).astype(np.float32)
    standardise = Standardiser(x_train.mean(axis=0), x_train.std(axis=0) + 1e-6)
    n_classes = len(index)
    model = build_model(x_train.shape[1], n_classes, spec, causal).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=spec.lr)
    tensors = [
        (
            torch.from_numpy(standardise(v.features)).T.unsqueeze(0).to(device),
            torch.from_numpy(np.array([index[int(a)] for a in v.gt], dtype=np.int64))
            .unsqueeze(0)
            .to(device),
        )
        for v in train_videos
    ]
    order = np.random.default_rng(spec.seed)
    curve: list[dict[str, float]] = []
    best_score, best_probs, best_epoch = -1.0, None, 0
    best_state: dict[str, object] | None = None
    started = time.perf_counter()
    for epoch in range(1, spec.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for i in order.permutation(len(tensors)):
            x, y = tensors[i]
            optimiser.zero_grad(set_to_none=True)
            loss = _loss(model(x), y, n_classes, spec)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
        if epoch % spec.eval_every == 0 or epoch == spec.epochs:
            probs = _predict(model, eval_videos, standardise, device)
            score = _metric(eval_videos, probs, inverse, spec.selection_metric)
            curve.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / len(tensors),
                    spec.selection_metric: score,
                }
            )
            if score > best_score:
                best_score, best_probs, best_epoch = score, probs, epoch
                if checkpoint is not None:
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
    assert best_probs is not None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": 1,
                "state_dict": best_state,
                "mean": torch.from_numpy(standardise.mean.astype(np.float32)),
                "std": torch.from_numpy(standardise.std.astype(np.float32)),
                "dim": int(x_train.shape[1]),
                "n_classes": n_classes,
                "causal": causal,
                "spec": asdict(spec),
                "best_epoch": best_epoch,
            },
            checkpoint,
        )
    log = {
        "causal": causal,
        "best_epoch": best_epoch,
        f"best_val_{spec.selection_metric}": best_score,
        "curve": curve,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.perf_counter() - started,
    }
    return best_probs, log


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[object, Standardiser, dict]:
    """Model (eval mode, on ``device``), standardiser and metadata saved by :func:`train_mstcn`."""
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != 1:
        raise ValueError(f"{path}: unknown checkpoint format {saved.get('format')!r}")
    raw_spec = dict(saved["spec"])
    spec_fields = {name for name in MSTCNSpec.__dataclass_fields__}
    spec = MSTCNSpec(**{k: v for k, v in raw_spec.items() if k in spec_fields})
    model = build_model(saved["dim"], saved["n_classes"], spec, saved["causal"])
    model.load_state_dict(saved["state_dict"])
    model.to(device).eval()
    standardise = Standardiser(saved["mean"].numpy(), saved["std"].numpy())
    meta = {k: saved[k] for k in ("dim", "n_classes", "causal", "best_epoch")} | {"spec": raw_spec}
    return model, standardise, meta


def predict_probs(
    model, features: np.ndarray, standardise: Standardiser, device: str
) -> np.ndarray:
    """Last-stage posterior ``(T, n_classes)`` of one feature sequence ``(T, dim)``."""
    return _predict(
        model,
        [VideoFeatures("", "", np.arange(len(features)), features, np.zeros(0))],
        standardise,
        device,
    )[0]


def run_mstcn_baseline(
    features_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    spec: MSTCNSpec | None = None,
    device: str = "cpu",
    train_split: str = "train",
    eval_split: str = "val",
) -> dict[str, object]:
    """Train the causal and the non-causal MS-TCN++ on train, select epochs on val, write the run dir."""
    if eval_split == "test":
        raise ValueError("the frozen test split is not evaluated by the development baseline")
    spec = spec or MSTCNSpec()
    started = time.perf_counter()
    train_videos, _ = load_split_features(features_dir, labels_dir, train_split)
    eval_videos, _ = load_split_features(features_dir, labels_dir, eval_split)
    y_train_raw = np.concatenate([v.gt for v in train_videos])
    index = label_index_map(int(a) for a in y_train_raw if a != BACKGROUND)
    inverse = np.array([action for action, _ in sorted(index.items(), key=lambda kv: kv[1])])
    majority = int(inverse[np.bincount([index[int(a)] for a in y_train_raw]).argmax()])

    variants: dict[str, tuple[list[np.ndarray], dict[str, object]]] = {}
    for name, causal in (("mstcn_causal", True), ("mstcn_offline", False)):
        variants[name] = train_mstcn(
            train_videos, eval_videos, index, inverse, spec, causal, device
        )

    rows: list[PredictionRow] = []
    for v_index, video in enumerate(eval_videos):
        preds = {
            name: inverse[probs[v_index].argmax(axis=1)] for name, (probs, _) in variants.items()
        }
        for i, frame in enumerate(video.frames.tolist()):
            rows.append(
                PredictionRow(
                    video.video_id,
                    video.participant,
                    frame,
                    int(video.gt[i]),
                    {"majority": majority, **{name: int(p[i]) for name, p in preds.items()}},
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / f"predictions_{eval_split}.csv", rows)
    metrics = score_predictions(rows, n_boot=spec.n_boot, seed=spec.seed)
    metrics["confusions"] = top_confusions(rows, "mstcn_causal")
    metrics["eval_split"] = eval_split
    metrics["train_split"] = train_split
    meta_path = features_dir / "meta.json"
    cache_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    config = {
        "model": "MS-TCN++ (Li et al., TPAMI 2020) on frozen frame features; epoch selected on val",
        "spec": asdict(spec),
        "training": {name: log for name, (_, log) in variants.items()},
        "classes": {
            "n_classes": len(index),
            "background_index": 0,
            "index_to_action_id": inverse.tolist(),
            "majority_action_id": majority,
        },
        "features": cache_meta,
        "runs": {
            "majority": "constant most frequent training label",
            "mstcn_causal": "MS-TCN++ with left-only padding: frame t sees frames <= t (online)",
            "mstcn_offline": "MS-TCN++ with symmetric padding: sees future frames (not an online result)",
        },
        "train_videos": len(train_videos),
        "train_frames": len(y_train_raw),
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
    tables = render_tables(
        load_metrics(out_dir), json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    )
    (out_dir / "tables.md").write_text(tables, encoding="utf-8", newline="\n")
    return {"config": config, "metrics": metrics, "tables": tables}
