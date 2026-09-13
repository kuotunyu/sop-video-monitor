"""Streaming benchmark grid with fake sources and a sleeping consumer."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from sop_monitor.stream import DropPolicy
from sop_monitor.stream.bench import render_bench_tables, run_grid, write_bench
from sop_monitor.stream.pipeline import Frame, ReplaySource

IMAGE = np.zeros((2, 2, 3), dtype=np.uint8)


def frames() -> Iterator[tuple[int, np.ndarray]]:
    index = 0
    while True:
        yield index, IMAGE
        index += 1


def make_sources(n: int, speed: float) -> list[ReplaySource]:
    return [ReplaySource(f"s{i}", frames, fps=40.0, speed=speed) for i in range(n)]


def test_grid_reports_offered_load_and_drops_grow_with_streams(tmp_path: Path) -> None:
    def consumer(batch: Sequence[Frame]) -> None:
        time.sleep(0.01 * len(batch))  # ≈ 100 fps capacity

    points = run_grid(
        make_sources,
        consumer,
        streams=(1, 6),
        speeds=(1.0,),
        duration_s=0.8,
        warmup_s=0.2,
        policy=DropPolicy.DROP_OLDEST,
        capacity=4,
        max_batch=4,
        max_wait_s=0.005,
    )
    assert [p["cameras"] for p in points] == [1, 6]
    assert points[0]["offered_fps"] == 40.0 and points[1]["offered_fps"] == 240.0
    assert points[0]["drop_rate"] < 0.1 < points[1]["drop_rate"]  # 240 fps offered to ≈ 100 fps
    payload = write_bench(
        tmp_path / "bench",
        {
            "consumer": "sleep",
            "policy": "drop_oldest",
            "capacity": 4,
            "max_batch": 4,
            "max_wait_ms": 5,
            "duration_s": 0.8,
            "warmup_s": 0.2,
        },
        points,
    )
    stored = json.loads((tmp_path / "bench" / "bench.json").read_text(encoding="utf-8"))
    assert stored["points"][1]["cameras"] == 6 and "platform" in stored["machine"]
    tables = (tmp_path / "bench" / "tables.md").read_text(encoding="utf-8")
    assert tables == render_bench_tables(payload) and "| 1.0 | 6 | 240 |" in tables
