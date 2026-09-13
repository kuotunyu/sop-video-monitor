"""Throughput / latency / backpressure curves of the streaming pipeline (spec 5.3, W4).

``run_grid`` sweeps the number of camera streams and the replay speed (1.0 = the cameras' nominal
15 fps) for one consumer and one backpressure policy, and records one :class:`PipelineReport` per
point. The consumers are:

- ``decode``: no model — measures decode, resize, buffering and batching alone;
- ``dinov2``: the frozen DINOv2 frame embedder used by the recogniser (GPU), i.e. the per-frame
  cost of the online path before the temporal head.

Measurements are properties of one machine at one moment (other processes, thermal state, power
plan); the JSON records the machine and must never be mixed with another machine's numbers.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from sop_monitor.stream.pipeline import Frame, ReplaySource, run_pipeline
from sop_monitor.stream.ring_buffer import DropPolicy

SourceFactory = Callable[[int, float], list[ReplaySource]]
"""``(n_streams, speed) -> sources``."""


def run_grid(
    make_sources: SourceFactory,
    consumer: Callable[[Sequence[Frame]], None],
    streams: Iterable[int],
    speeds: Iterable[float],
    duration_s: float,
    warmup_s: float,
    policy: DropPolicy,
    capacity: int,
    max_batch: int,
    max_wait_s: float,
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for speed in speeds:
        for n in streams:
            report = run_pipeline(
                make_sources(n, speed),
                consumer,
                duration_s=duration_s + warmup_s,
                capacity=capacity,
                policy=policy,
                max_batch=max_batch,
                max_wait_s=max_wait_s,
                warmup_s=warmup_s,
            )
            entry = report.to_dict()
            entry["speed"] = speed
            entry["offered_fps"] = n * report.target_fps_per_camera
            entry["drop_rate"] = report.skipped_frames / report.produced if report.produced else 0.0
            points.append(entry)
    return points


def video_sources(videos: Sequence[Path], fps: float, size: tuple[int, int]) -> SourceFactory:
    """Sources replaying ``videos`` round robin (stream i plays ``videos[i % len(videos)]``)."""
    from sop_monitor.video import iter_frames

    def reader(path: Path) -> Callable[[], object]:
        return lambda: iter_frames(path, size=size)

    def make(n: int, speed: float) -> list[ReplaySource]:
        count = len(videos)
        return [
            ReplaySource(f"stream{i:02d}", reader(videos[i % count]), fps=fps, speed=speed)
            for i in range(n)
        ]

    return make


def dinov2_consumer(model_name: str, device: str) -> Callable[[Sequence[Frame]], None]:
    import numpy as np

    from sop_monitor.features import embed_batch, load_dinov2

    model = load_dinov2(model_name, device)

    def consume(batch: Sequence[Frame]) -> None:
        embed_batch(model, np.stack([frame.image for frame in batch]), device)
        if device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()

    return consume


def machine() -> dict[str, object]:
    info: dict[str, object] = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    return info


def render_bench_tables(payload: Mapping[str, object]) -> str:
    config: Mapping[str, object] = payload["config"]  # type: ignore[assignment]
    lines = [
        f"Streaming benchmark, consumer `{config['consumer']}`, policy `{config['policy']}`, "
        f"buffer {config['capacity']} per stream, batch ≤ {config['max_batch']} "
        f"(wait ≤ {config['max_wait_ms']} ms), {config['duration_s']} s per point after "
        f"{config['warmup_s']} s warm-up; machine: {payload['machine']}.",
        "",
        "| speed | streams | offered fps | consumed fps | dropped | p50 latency ms | p95 latency ms | mean batch | consumer busy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in payload["points"]:  # type: ignore[union-attr]
        lines.append(
            f"| {p['speed']} | {p['cameras']} | {p['offered_fps']:.0f} | {p['throughput_fps']:.1f} "
            f"| {100 * p['drop_rate']:.1f} % | {p['latency_ms']['p50']:.0f} | {p['latency_ms']['p95']:.0f} "
            f"| {p['mean_batch']:.1f} | {100 * p['consumer_busy_fraction']:.0f} % |"
        )
    return "\n".join(lines) + "\n"


def write_bench(
    out_dir: Path, config: Mapping[str, object], points: list[dict[str, object]]
) -> dict:
    payload = {
        "config": dict(config),
        "machine": machine(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "points": points,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "tables.md").write_text(render_bench_tables(payload), encoding="utf-8", newline="\n")
    return payload
