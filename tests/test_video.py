"""PyAV probe and sequential decode on a synthetic clip (skipped when the `baseline` group is absent)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av")

from sop_monitor.video import iter_frames, probe, write_synthetic_video


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "clip_01.mp4"
    write_synthetic_video(path, n_frames=23, fps=10, size=(64, 48))
    return path


def test_probe_reads_fps_size_and_frame_count(clip: Path) -> None:
    info = probe(clip)
    assert info.video_id == "clip_01"
    assert (info.width, info.height, info.fps) == (64, 48, 10.0)
    assert info.n_frames == 23
    assert info.codec == "mpeg4"
    assert info.duration_s == pytest.approx(2.3, abs=0.11)


def test_iter_frames_indexes_from_zero_honours_stride_and_rescales(clip: Path) -> None:
    frames = list(iter_frames(clip, stride=5, size=(32, 24)))
    assert [index for index, _ in frames] == [0, 5, 10, 15, 20]
    for index, image in frames:
        assert image.shape == (24, 32, 3)
        assert image.dtype == np.uint8
        # Red level encodes the frame index (lossy codec: allow slack, but order must be monotone).
        assert abs(int(image[..., 0].mean()) - min(255, index * 8)) < 24
    assert len(list(iter_frames(clip))) == 23
    with pytest.raises(ValueError):
        next(iter_frames(clip, stride=0))
