"""Video probing and sequential frame decoding with PyAV (spec 5.1 camera source, offline mode).

Only the offline/file path exists so far: open a file, report its stream metadata, and yield
every ``stride``-th decoded frame as an RGB ``uint8`` array (optionally rescaled by swscale).
Frame indices are decode order starting at 0, which is what IndustReal's label frame indices
refer to. The RTSP source, watchdog and shared-memory ring buffer stay W4 work.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    path: str
    codec: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    frames_from_header: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe(path: Path, count_if_missing: bool = True) -> VideoInfo:
    """Stream metadata of ``path``; decodes the file to count frames when the header has none."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or Fraction(0))
        n_frames = int(stream.frames or 0)
        from_header = n_frames > 0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if not from_header and count_if_missing:
            stream.thread_type = "AUTO"
            n_frames = sum(1 for _ in container.decode(stream))
        return VideoInfo(
            video_id=path.stem,
            path=str(path),
            codec=stream.codec_context.name,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=fps,
            n_frames=n_frames,
            duration_s=duration if duration else (n_frames / fps if fps else 0.0),
            frames_from_header=from_header,
        )


def iter_frames(
    path: Path, stride: int = 1, size: tuple[int, int] | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, rgb_uint8[h, w, 3])`` for every ``stride``-th decoded frame.

    ``size`` is ``(width, height)``; rescaling happens in swscale before the copy to numpy.
    """
    import av

    if stride < 1:
        raise ValueError("stride must be >= 1")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index % stride:
                continue
            if size is None:
                yield index, frame.to_ndarray(format="rgb24")
            else:
                yield (
                    index,
                    frame.reformat(width=size[0], height=size[1], format="rgb24").to_ndarray(),
                )


def write_synthetic_video(
    path: Path, n_frames: int, fps: int = 10, size: tuple[int, int] = (64, 48)
) -> None:
    """Encode ``n_frames`` flat-coloured frames (frame ``i`` has red level ``i``); for tests only."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            image[..., 0] = min(255, i * 8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
