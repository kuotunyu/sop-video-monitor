"""Online recogniser: per-view causal MS-TCN++ heads run on a growing stream, fused, per hand.

This closes the loop from frames to deviations. For each hand, the three per-view causal networks
of a ``train-havid-tas --checkpoint-dir`` run receive the frame features of their camera in chunks
(:class:`StreamingHead`); their posteriors are fused per frame with the same parameter-free rule as
the offline run, mapped to HR-SAT labels and, once both hands have a label for a frame, pushed into
:class:`sop_monitor.online.OnlineSOPMonitor` (:class:`OnlineSOPPipeline`).

Correctness rests on causality: the output of a left-padded network at time ``t`` depends on
frames ``<= t`` only, so running it on the prefix seen so far gives, for the new time steps, the
same posteriors as running it on the whole recording (``tests/test_online_head.py`` checks this).
Each push recomputes the prefix — the MS-TCN++ receptive field (over 10,000 frames) exceeds every
HA-ViD recording, so no shorter window would be exact; per-layer state caching is a possible
optimisation, not needed at 15 fps. A network trained with look-ahead ``L`` names at time ``t``
the step of frame ``t - L``: labels of frame ``f`` are emitted at time ``f + L``, and at the end of
the stream the last ``L`` frames repeat the final output, exactly as ``havid_tas.advance_outputs``
does offline.

The plate is an input: an assembly station knows its work order, and the evaluation takes it from
the annotation, as the offline SOP runs do.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sop_monitor.havid_tas import FUSION_RULES, fuse_views
from sop_monitor.mstcn import Standardiser, load_checkpoint, predict_probs
from sop_monitor.online import HANDS, Deviation, OnlineSOPMonitor


class StreamingHead:
    """One causal network on a growing feature stream."""

    def __init__(self, model: object, standardise: Standardiser, device: str) -> None:
        self.model = model
        self.standardise = standardise
        self.device = device
        self._chunks: list[np.ndarray] = []
        self.n_seen = 0

    def push(self, features: np.ndarray) -> np.ndarray:
        """Append ``(k, dim)`` features; return the posteriors ``(k, n_classes)`` of those steps."""
        if features.ndim != 2:
            raise ValueError(f"expected (frames, dim), got shape {features.shape}")
        if len(features) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        self._chunks.append(features)
        prefix = np.concatenate(self._chunks) if len(self._chunks) > 1 else features
        self._chunks = [prefix]
        probs = predict_probs(self.model, prefix, self.standardise, self.device)
        new = probs[self.n_seen :]
        self.n_seen = len(prefix)
        return new


@dataclass
class HandRecogniser:
    """Per-view streaming heads of one hand, fused per frame, with the run's look-ahead."""

    heads: dict[int, StreamingHead]
    index_to_class_id: np.ndarray
    class_id_to_label: Mapping[str, str]
    lookahead: int = 0
    rule: str = ""
    n_time_steps: int = 0
    n_emitted: int = 0
    last_label: str | None = None

    def __post_init__(self) -> None:
        if self.rule not in FUSION_RULES:
            raise ValueError(f"fusion rule must be one of {FUSION_RULES}, got {self.rule!r}")
        if self.lookahead < 0:
            raise ValueError("look-ahead must be non-negative")

    @classmethod
    def from_checkpoints(cls, directory: Path, device: str, rule: str = "") -> HandRecogniser:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        heads = {}
        for view in meta["spec"]["views"]:
            model, standardise, info = load_checkpoint(directory / f"view{view}_causal.pt", device)
            if not info["causal"]:
                raise ValueError(f"{directory}: view{view}_causal.pt is not a causal network")
            heads[int(view)] = StreamingHead(model, standardise, device)
        return cls(
            heads,
            np.asarray(meta["classes"]["index_to_class_id"], dtype=np.int64),
            meta["classes"]["class_id_to_label"],
            int(meta["spec"]["lookahead"]),
            rule,
        )

    def _labels(self, fused: np.ndarray) -> list[str]:
        ids = self.index_to_class_id[fused.argmax(axis=1)]
        return [self.class_id_to_label[str(int(i))] for i in ids]

    def push(self, features: Mapping[int, np.ndarray]) -> list[str]:
        """Feed one chunk per view; return the labels of the frames that became final, in order."""
        if set(features) != set(self.heads):
            raise ValueError(f"expected features for views {sorted(self.heads)}")
        lengths = {len(chunk) for chunk in features.values()}
        if len(lengths) != 1:
            raise ValueError("every view must receive the same number of frames")
        stack = np.stack([self.heads[view].push(features[view]) for view in sorted(self.heads)])
        if stack.shape[1] == 0:
            return []
        labels = self._labels(fuse_views(stack)[self.rule])
        self.last_label = labels[-1]
        start = self.n_time_steps
        self.n_time_steps += len(labels)
        # time step t carries the label of frame t - lookahead; frames < 0 do not exist
        emitted = labels[max(0, self.lookahead - start) :]
        self.n_emitted += len(emitted)
        return emitted

    def finish(self) -> list[str]:
        """Labels of the last frames, which have no output of their own: the final output repeated."""
        if self.last_label is None:
            return []
        tail = self.n_time_steps - self.n_emitted
        self.n_emitted = self.n_time_steps
        return [self.last_label] * tail


@dataclass
class ChunkTiming:
    frames: int
    recognise_s: float
    monitor_s: float


@dataclass
class OnlineSOPPipeline:
    """Both hands' recognisers feeding one :class:`OnlineSOPMonitor`."""

    hands: dict[str, HandRecogniser]
    monitor: OnlineSOPMonitor
    labels: dict[str, list[str]] = field(default_factory=lambda: {hand: [] for hand in HANDS})
    n_pushed: int = 0
    timings: list[ChunkTiming] = field(default_factory=list)

    def _drain(self) -> list[Deviation]:
        ready = min(len(self.labels[hand]) for hand in HANDS)
        found: list[Deviation] = []
        for frame in range(self.n_pushed, ready):
            found.extend(self.monitor.push(frame, {h: self.labels[h][frame] for h in HANDS}))
        self.n_pushed = max(self.n_pushed, ready)
        return found

    def push(self, features: Mapping[int, np.ndarray]) -> list[Deviation]:
        started = time.perf_counter()
        for hand in HANDS:
            self.labels[hand].extend(self.hands[hand].push(features))
        recognised = time.perf_counter()
        found = self._drain()
        n_frames = len(next(iter(features.values())))
        self.timings.append(
            ChunkTiming(n_frames, recognised - started, time.perf_counter() - recognised)
        )
        return found

    def finish(self) -> list[Deviation]:
        for hand in HANDS:
            self.labels[hand].extend(self.hands[hand].finish())
        return self._drain() + self.monitor.finish()


def cached_feature_chunks(
    caches: Mapping[int, Path], chunk: int
) -> Iterator[dict[int, np.ndarray]]:
    """Chunks of the cached per-view features of one recording (every frame, in lockstep)."""
    arrays = {}
    for view, path in caches.items():
        with np.load(path) as payload:
            if not np.array_equal(payload["frames"], np.arange(int(payload["n_frames"]))):
                raise ValueError(f"{path}: the cache does not hold every frame")
            arrays[view] = payload["features"]
    n_frames = {len(a) for a in arrays.values()}
    if len(n_frames) != 1:
        raise ValueError(f"views differ in length: { {v: len(a) for v, a in arrays.items()} }")
    total = n_frames.pop()
    for start in range(0, total, chunk):
        yield {view: array[start : start + chunk] for view, array in arrays.items()}


def video_feature_chunks(
    videos: Mapping[int, Path],
    chunk: int,
    embed: Callable[[np.ndarray], np.ndarray],
    size: tuple[int, int],
    decode_timings: list[float] | None = None,
) -> Iterator[dict[int, np.ndarray]]:
    """Decode the per-view videos in lockstep and embed ``chunk`` frames per view at a time."""
    from sop_monitor.video import iter_frames

    readers = {view: iter_frames(path, size=size) for view, path in videos.items()}
    views = sorted(readers)
    while True:
        started = time.perf_counter()
        frames: dict[int, list[np.ndarray]] = {view: [] for view in views}
        for _ in range(chunk):
            decoded = [next(readers[view], None) for view in views]
            if any(item is None for item in decoded):
                if not all(item is None for item in decoded):
                    raise ValueError("the views' videos differ in length")
                break
            for view, (_index, image) in zip(views, decoded, strict=True):
                frames[view].append(image)
        if decode_timings is not None:
            decode_timings.append(time.perf_counter() - started)
        if not frames[views[0]]:
            return
        yield {view: embed(np.stack(frames[view])) for view in views}
        if len(frames[views[0]]) < chunk:
            return


def run_pipeline(
    pipeline: OnlineSOPPipeline, chunks: Iterable[Mapping[int, np.ndarray]]
) -> list[Deviation]:
    found: list[Deviation] = []
    for features in chunks:
        found.extend(pipeline.push(features))
    found.extend(pipeline.finish())
    return found


def latency_summary(timings: Sequence[ChunkTiming], fps: float) -> dict[str, float]:
    """Compute per chunk against the chunk's real-time budget (``frames / fps``)."""
    if not timings:
        return {}
    compute = np.array([t.recognise_s + t.monitor_s for t in timings])
    frames = np.array([t.frames for t in timings])
    return {
        "chunks": len(timings),
        "frames": int(frames.sum()),
        "compute_s_p50": float(np.percentile(compute, 50)),
        "compute_s_p95": float(np.percentile(compute, 95)),
        "compute_s_max": float(compute.max()),
        "recognise_s_total": float(sum(t.recognise_s for t in timings)),
        "monitor_s_total": float(sum(t.monitor_s for t in timings)),
        "real_time_factor": float(compute.sum() / (frames.sum() / fps)),
    }


def checkpoint_labels(directory: Path, recording: str, rule: str = "") -> list[str]:
    """Per-frame fused labels of ``recording`` from the val posteriors saved with the checkpoints."""
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    inverse = np.asarray(meta["classes"]["index_to_class_id"], dtype=np.int64)
    labels = meta["classes"]["class_id_to_label"]
    with np.load(directory / "probs_val.npz") as saved:
        probs = saved[f"fusion{rule}_causal__{recording}"].astype(np.float32)
    return [labels[str(int(i))] for i in inverse[probs.argmax(axis=1)]]


def run_online_head(
    checkpoints: Mapping[str, Path],
    chunks_of: Callable[[str], Iterable[Mapping[int, np.ndarray]]],
    recordings: Sequence[str],
    plates: Mapping[str, str],
    knowledge: tuple[Mapping, Mapping, Mapping],
    device: str,
    granularity: str = "sheet",
    min_frames: int = 1,
    rule: str = "",
    fps: float = 15.0,
) -> dict[str, object]:
    """Stream every recording through both hands' recognisers and the online monitor.

    Returns per recording the streamed labels, the deviations, the per-chunk latency summary and the
    fraction of frames whose streamed label equals the label from the checkpoint run's saved
    posteriors (1.0 up to floating-point ties when the features are the cached ones).
    """
    graphs, bounds, mandatory = knowledge
    lookahead = {
        hand: json.loads((checkpoints[hand] / "meta.json").read_text(encoding="utf-8"))["spec"][
            "lookahead"
        ]
        for hand in HANDS
    }
    out: dict[str, object] = {"lookahead": lookahead, "recordings": {}}
    for recording in recordings:
        plate = plates[recording]
        pipeline = OnlineSOPPipeline(
            {h: HandRecogniser.from_checkpoints(checkpoints[h], device, rule) for h in HANDS},
            OnlineSOPMonitor(
                graphs[plate], bounds[plate], mandatory[plate], granularity, min_frames
            ),
        )
        started = time.perf_counter()
        found = run_pipeline(pipeline, chunks_of(recording))
        wall = time.perf_counter() - started
        agreement = {}
        for hand in HANDS:
            reference = checkpoint_labels(checkpoints[hand], recording, rule)
            streamed = pipeline.labels[hand]
            if len(reference) != len(streamed):
                raise ValueError(
                    f"{recording} {hand}: streamed {len(streamed)} frames, "
                    f"the checkpoint run has {len(reference)}"
                )
            agreement[hand] = sum(a == b for a, b in zip(reference, streamed, strict=True)) / len(
                reference
            )
        out["recordings"][recording] = {  # type: ignore[index]
            "plate": plate,
            "labels": pipeline.labels,
            "deviations": [d.to_dict() for d in found],
            "latency": latency_summary(pipeline.timings, fps)
            | {"wall_s": wall, "wall_real_time_factor": wall / (len(pipeline.labels["lh"]) / fps)},
            "agreement_with_checkpoint_run": agreement,
        }
    return out
