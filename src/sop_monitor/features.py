"""Frozen DINOv2 frame embeddings (spec 6.1, frame branch) cached per video as ``.npz``.

The extractor is the official ``facebookresearch/dinov2`` torch.hub model with its published
weights (Apache-2.0). Each sampled frame is rescaled to ``FRAME_SIZE`` (width x height, both
multiples of the 14-pixel patch, close to the 16:9 source aspect), normalised with the ImageNet
statistics and embedded as ``[CLS ; mean of patch tokens]`` — the concatenation DINOv2 uses for
its linear evaluation. Features are stored as float16; nothing here looks at labels.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sop_monitor.video import iter_frames, probe

FRAME_SIZE: tuple[int, int] = (392, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DINOV2_DIMS: dict[str, int] = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_dinov2(name: str, device: str) -> object:
    """Load a DINOv2 backbone from torch.hub (downloads the repo and weights on first use)."""
    import torch

    if name not in DINOV2_DIMS:
        raise ValueError(f"unknown DINOv2 model {name!r}; choose from {sorted(DINOV2_DIMS)}")
    model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
    return model.eval().to(device)


def weights_digest(name: str) -> dict[str, object] | None:
    """SHA-256 and size of the cached torch.hub checkpoint, if present."""
    import torch

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / f"{name}_pretrain.pth"
    if not checkpoint.is_file():
        return None
    return {
        "file": checkpoint.name,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "bytes": checkpoint.stat().st_size,
    }


@dataclass(frozen=True)
class ExtractionStats:
    video_id: str
    n_frames: int
    n_sampled: int
    seconds: float

    @property
    def sampled_fps(self) -> float:
        return self.n_sampled / self.seconds if self.seconds else 0.0


def embed_batch(model: object, frames: np.ndarray, device: str) -> np.ndarray:
    """``frames`` is ``uint8[n, h, w, 3]`` RGB; returns ``float16[n, 2 * dim]``."""
    import torch

    x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")),
    ):
        out = model.forward_features(x)  # type: ignore[attr-defined]
        cls = out["x_norm_clstoken"]
        patches = out["x_norm_patchtokens"].mean(dim=1)
        features = torch.cat([cls, patches], dim=1)
    return features.float().cpu().numpy().astype(np.float16)


def extract_video(
    video: Path, model: object, device: str, stride: int = 1, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray, int, ExtractionStats]:
    """Decode, rescale and embed every ``stride``-th frame; returns ``(frames, features, n_frames, stats)``."""
    started = time.perf_counter()
    indices: list[int] = []
    chunks: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    pending_idx: list[int] = []
    decoded = 0
    for index, frame in iter_frames(video, stride=stride, size=FRAME_SIZE):
        decoded = index + 1
        pending.append(frame)
        pending_idx.append(index)
        if len(pending) == batch_size:
            chunks.append(embed_batch(model, np.stack(pending), device))
            indices.extend(pending_idx)
            pending, pending_idx = [], []
    if pending:
        chunks.append(embed_batch(model, np.stack(pending), device))
        indices.extend(pending_idx)
    n_frames = probe(video, count_if_missing=False).n_frames or decoded
    features = np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float16)
    stats = ExtractionStats(video.stem, n_frames, len(indices), time.perf_counter() - started)
    return np.asarray(indices, dtype=np.int64), features, n_frames, stats


def extract_videos(
    videos: Iterable[Path],
    out_dir: Path,
    model_name: str = "dinov2_vits14",
    device: str = "auto",
    stride: int = 1,
    batch_size: int = 256,
    overwrite: bool = False,
) -> list[ExtractionStats]:
    """Cache ``<out_dir>/<video_id>.npz`` per video plus ``meta.json`` describing the extractor."""
    import torch

    device = resolve_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_dinov2(model_name, device)
    meta = {
        "extractor": "facebookresearch/dinov2 via torch.hub",
        "model": model_name,
        "weights": weights_digest(model_name),
        "embedding": "concat(CLS, mean patch tokens)",
        "dim": 2 * DINOV2_DIMS[model_name],
        "frame_size_wh": list(FRAME_SIZE),
        "normalisation": "ImageNet mean/std",
        "stride": stride,
        "dtype": "float16",
        "autocast_fp16": device.startswith("cuda"),
        "torch": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "platform": platform.platform(),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    stats: list[ExtractionStats] = []
    for video in videos:
        target = out_dir / f"{video.stem}.npz"
        if target.is_file() and not overwrite:
            continue
        frames, features, n_frames, stat = extract_video(video, model, device, stride, batch_size)
        np.savez(target, frames=frames, features=features, n_frames=np.int64(n_frames))
        stats.append(stat)
    return stats
