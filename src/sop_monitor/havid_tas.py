"""HA-ViD offline temporal action segmentation on frozen frame features (spec W2, first slice).

For every camera view an MS-TCN++ (causal and non-causal, :mod:`sop_monitor.mstcn`) is trained on
that view's features of the ``train`` recordings with its epoch selected on ``val``; the views'
posteriors are then fused three parameter-free ways (:func:`fuse_views`: mean, normalised
geometric mean, confidence-weighted mean), so nothing about the fusion is tuned on val.
Prediction rows are keyed by *recording*, so the per-view and the fused runs share one table and
one subject-wise bootstrap.

Ground truth is the untrimmed temporal file expanded to one label per frame; class ids are the
indices of the official ``mapping.txt`` so that the numbers stay comparable with the paper's
protocol (``null`` is an ordinary class there too; :data:`sop_monitor.industreal.BACKGROUND` never
occurs). Feature caches are ``<video_id>.npz`` with ``frames``/``features``/``n_frames``: either the
DINOv2 cache written by ``extract-features --dataset ha-vid`` or the official I3D features exported
by :func:`export_official_features` (annotation frame ``t`` is feature column
``t - OFFICIAL_TRIM``; ``n_frames`` is restored to the annotation length).
"""

from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sop_monitor.baseline import (
    PredictionRow,
    VideoFeatures,
    label_index_map,
    load_metrics,
    render_tables,
    score_predictions,
    top_confusions,
    write_predictions,
)
from sop_monitor.havid import (
    OFFICIAL_TRIM,
    VIEW_NAMES,
    VIEWS,
    TemporalSegment,
    frame_labels,
    load_official_zip,
    load_temporal_zip,
    n_frames_of,
)
from sop_monitor.mstcn import MSTCNSpec, train_mstcn


@dataclass(frozen=True)
class HavidTASSpec:
    hand: str = "lh"
    level: str = "pt"
    views: tuple[int, ...] = VIEWS
    mstcn: MSTCNSpec = field(default_factory=MSTCNSpec)
    lookahead: int = 0
    """Frames of future context the causal networks may use: the label of frame t is read from the
    causal output at t + lookahead (a fixed output delay of lookahead / 15 s)."""
    offline: bool = True
    """Also train the non-causal (offline) twin of every view."""

    @property
    def subdataset(self) -> str:
        return f"view0_{self.hand}_{self.level}"


def read_split(split_csv: Path) -> list[dict[str, str]]:
    with split_csv.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def export_official_features(
    official_zip: Path, out_dir: Path, videos: Iterable[str] | None = None
) -> int:
    """Write the official I3D features as ``<video>.npz`` caches; return how many were written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    dim = None
    with zipfile.ZipFile(official_zip) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            posix = name.replace("\\", "/")
            if "features/" in posix and posix.endswith(".npy"):
                members[posix.rsplit("/", 1)[-1].removesuffix(".npy")] = name
        wanted = sorted(videos) if videos is not None else sorted(members)
        for vid in wanted:
            target = out_dir / f"{vid}.npz"
            if target.is_file():
                continue
            if vid not in members:
                raise FileNotFoundError(f"{official_zip}: no feature file for {vid}")
            array = np.load(io.BytesIO(archive.read(members[vid])))
            if array.ndim != 2:
                raise ValueError(f"{vid}: expected a (dim, T) array, got shape {array.shape}")
            dim, n_feature_frames = array.shape
            np.savez(
                target,
                frames=np.arange(OFFICIAL_TRIM, OFFICIAL_TRIM + n_feature_frames, dtype=np.int64),
                features=array.T.astype(np.float32),
                n_frames=np.int64(n_feature_frames + 2 * OFFICIAL_TRIM),
            )
            written += 1
    meta_path = out_dir / "meta.json"
    if written or not meta_path.is_file():
        meta = {
            "extractor": "official HA-ViD I3D features (ActionSegmentation/data/features/*.npy)",
            "model": "i3d_official",
            "dim": dim,
            "dtype": "float32 (stored as float64 by the authors)",
            "stride": 1,
            "frames": f"annotation frame t is feature column t - {OFFICIAL_TRIM}",
            "source_zip": official_zip.name,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written


def load_havid_videos(
    features_dir: Path,
    split_csv: Path,
    segments_by_recording: Mapping[str, Sequence[TemporalSegment]],
    class_ids: Mapping[str, int],
    view: int,
) -> list[VideoFeatures]:
    """Cached features of one view of every recording in ``split_csv`` with per-frame class ids."""
    videos: list[VideoFeatures] = []
    for row in read_split(split_csv):
        if int(row["view"]) != view:
            continue
        cache = features_dir / f"{row['video_id']}.npz"
        if not cache.is_file():
            raise FileNotFoundError(f"no cached features for {row['video_id']}: {cache}")
        with np.load(cache) as payload:
            frames = payload["frames"].astype(np.int64)
            features = payload["features"]
            n_frames = int(payload["n_frames"])
        segments = segments_by_recording[row["recording"]]
        if n_frames != n_frames_of(segments):
            raise ValueError(
                f"{row['video_id']}: cache covers {n_frames} frames, "
                f"annotation covers {n_frames_of(segments)}"
            )
        labels = np.array([class_ids[label] for label in frame_labels(segments)], dtype=np.int64)
        videos.append(
            VideoFeatures(
                video_id=row["recording"],
                participant=row["subject"],
                frames=frames,
                features=features,
                gt=labels[frames],
            )
        )
    return sorted(videos, key=lambda v: v.video_id)


FUSION_RULES = ("", "_geo", "_conf")
"""Suffixes of the fused runs: ``fusion_*`` (mean), ``fusion_geo_*``, ``fusion_conf_*``."""


def fuse_views(stack: np.ndarray, eps: float = 1e-8) -> dict[str, np.ndarray]:
    """Three parameter-free late-fusion rules for per-view posteriors ``stack`` of shape (V, T, C).

    ``""``: arithmetic mean. ``"_geo"``: normalised geometric mean (product of experts; a view that
    assigns ~0 to a class vetoes it). ``"_conf"``: per-frame weighted mean with each view weighted
    by its own maximum posterior, so a view that is unsure at frame *t* contributes less there.
    """
    if stack.ndim != 3:
        raise ValueError(f"expected (views, frames, classes), got shape {stack.shape}")
    mean = stack.mean(axis=0)
    geo = np.exp(np.log(stack + eps).mean(axis=0))
    geo /= geo.sum(axis=1, keepdims=True)
    weights = stack.max(axis=2, keepdims=True)
    conf = (weights * stack).sum(axis=0) / weights.sum(axis=0)
    return {"": mean, "_geo": geo, "_conf": conf}


def delay_labels(videos: Sequence[VideoFeatures], frames: int) -> list[VideoFeatures]:
    """Copies whose per-frame targets are delayed by ``frames``: target[t] = gt[t - frames].

    A causal network trained on these copies learns to name, at time t, the step of frame
    t - frames, i.e. it may use ``frames`` frames of future context for every labelled frame.
    The first ``frames`` targets repeat the first label.
    """
    if frames < 0:
        raise ValueError("look-ahead must be non-negative")
    if frames == 0:
        return list(videos)
    out: list[VideoFeatures] = []
    for v in videos:
        head = np.full(min(frames, len(v.gt)), v.gt[0], dtype=v.gt.dtype)
        out.append(
            VideoFeatures(
                v.video_id,
                v.participant,
                v.frames,
                v.features,
                np.concatenate([head, v.gt])[: len(v.gt)],
            )
        )
    return out


def advance_outputs(probs: np.ndarray, frames: int) -> np.ndarray:
    """Re-align a delayed causal output: row t becomes the output emitted at t + ``frames``;
    rows past the end repeat the last output (the end of the video gives no further frames)."""
    if frames == 0:
        return probs
    shifted = probs[frames:]
    tail = np.repeat(probs[-1:], len(probs) - len(shifted), axis=0)
    return np.concatenate([shifted, tail]) if len(shifted) else tail


def _check_aligned(
    reference: Sequence[VideoFeatures], videos: Sequence[VideoFeatures], view: int
) -> None:
    if [v.video_id for v in videos] != [v.video_id for v in reference]:
        raise ValueError(f"view {view}: eval recordings differ from view {VIEWS[0]}")
    for ref, video in zip(reference, videos, strict=True):
        if not np.array_equal(ref.frames, video.frames) or not np.array_equal(ref.gt, video.gt):
            raise ValueError(f"view {view}: {video.video_id} frames or labels differ")


def run_havid_tas(
    features_dir: Path,
    split_dir: Path,
    temporal_zip: Path,
    official_zip: Path,
    out_dir: Path,
    spec: HavidTASSpec | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Train per-view MS-TCN++ heads, fuse them, score on val and write the run directory."""
    spec = spec or HavidTASSpec()
    started = time.perf_counter()
    temporal, _ = load_temporal_zip(temporal_zip)
    mapping = load_official_zip(official_zip, [spec.subdataset])[spec.subdataset].mapping
    class_ids = {label: index for index, label in mapping.items()}
    segments = temporal[spec.subdataset]

    reference: list[VideoFeatures] | None = None
    index: dict[int, int] = {}
    inverse = np.zeros(0, dtype=np.int64)
    majority = 0
    counts: dict[str, int] = {}
    probs: dict[str, list[np.ndarray]] = {}
    logs: dict[str, object] = {}
    for view in spec.views:
        train_videos = load_havid_videos(
            features_dir, split_dir / "train.csv", segments, class_ids, view
        )
        eval_videos = load_havid_videos(
            features_dir, split_dir / "val.csv", segments, class_ids, view
        )
        if reference is None:
            reference = eval_videos
            y_train = np.concatenate([v.gt for v in train_videos])
            index = label_index_map(int(a) for a in y_train)
            inverse = np.array(
                [action for action, _ in sorted(index.items(), key=lambda kv: kv[1])]
            )
            majority = int(inverse[np.bincount([index[int(a)] for a in y_train]).argmax()])
            counts = {"train_videos": len(train_videos), "train_frames": len(y_train)}
        else:
            _check_aligned(reference, eval_videos, view)
        variants = [("causal", True)] + ([("offline", False)] if spec.offline else [])
        for name, causal in variants:
            run = f"view{view}_{name}"
            shift = spec.lookahead if causal else 0
            view_probs, logs[run] = train_mstcn(
                delay_labels(train_videos, shift),
                delay_labels(eval_videos, shift),
                index,
                inverse,
                spec.mstcn,
                causal,
                device,
            )
            probs[run] = [advance_outputs(p, shift) for p in view_probs]
    assert reference is not None
    trained_variants = ("causal", "offline") if spec.offline else ("causal",)
    for name in trained_variants:
        for rule in FUSION_RULES:
            probs[f"fusion{rule}_{name}"] = []
        for i in range(len(reference)):
            fused = fuse_views(np.stack([probs[f"view{view}_{name}"][i] for view in spec.views]))
            for rule, fused_probs in fused.items():
                probs[f"fusion{rule}_{name}"].append(fused_probs)

    rows: list[PredictionRow] = []
    for i, video in enumerate(reference):
        preds = {run: inverse[p[i].argmax(axis=1)] for run, p in probs.items()}
        for j, frame in enumerate(video.frames.tolist()):
            rows.append(
                PredictionRow(
                    video.video_id,
                    video.participant,
                    frame,
                    int(video.gt[j]),
                    {"majority": majority, **{run: int(p[j]) for run, p in preds.items()}},
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(out_dir / "predictions_val.csv", rows)
    metrics = score_predictions(rows, n_boot=spec.mstcn.n_boot, seed=spec.mstcn.seed)
    metrics["confusions"] = top_confusions(rows, "fusion_causal")
    metrics["eval_split"] = "val"
    metrics["train_split"] = "train"
    meta_path = features_dir / "meta.json"
    cache_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    descriptions = {"majority": "constant most frequent training label"}
    for view in spec.views:
        if spec.lookahead:
            descriptions[f"view{view}_causal"] = (
                f"MS-TCN++ on the {VIEW_NAMES[view]} view (view {view}), left-only padding, "
                f"output delayed by {spec.lookahead} frames: frame t is labelled from frames "
                f"<= t + {spec.lookahead} ({spec.lookahead / 15:.1f} s latency)"
            )
        else:
            descriptions[f"view{view}_causal"] = (
                f"MS-TCN++ on the {VIEW_NAMES[view]} view (view {view}), left-only padding: "
                "frame t sees frames <= t (online)"
            )
        if spec.offline:
            descriptions[f"view{view}_offline"] = (
                f"MS-TCN++ on the {VIEW_NAMES[view]} view (view {view}), symmetric padding: "
                "sees future frames (not an online result)"
            )
    tails = {"causal": ", online", "offline": ""}
    for name in trained_variants:
        tail = tails[name]
        descriptions[f"fusion_{name}"] = (
            f"mean of the per-view {name} posteriors (late fusion{tail})"
        )
        descriptions[f"fusion_geo_{name}"] = (
            f"normalised geometric mean of the per-view {name} posteriors (product of experts{tail})"
        )
        descriptions[f"fusion_conf_{name}"] = (
            f"per-frame confidence-weighted mean of the per-view {name} posteriors, "
            f"weight = each view's max posterior{tail}"
        )
    config = {
        "model": (
            "MS-TCN++ (Li et al., TPAMI 2020) per camera view on frozen frame features, epoch "
            f"selected on val {spec.mstcn.selection_metric}; late fusion = mean, normalised "
            "geometric mean and confidence-weighted mean of the per-view posteriors (parameter-free)"
        ),
        "dataset": "HA-ViD",
        "spec": {
            "hand": spec.hand,
            "level": spec.level,
            "views": list(spec.views),
            "lookahead": spec.lookahead,
            "offline": spec.offline,
            "mstcn": asdict(spec.mstcn),
        },
        "split_dir": str(split_dir),
        "training": logs,
        "classes": {
            "n_classes": len(index),
            "background_index": 0,
            "index_to_class_id": inverse.tolist(),
            "class_id_to_label": {str(i): label for i, label in sorted(mapping.items())},
            "majority_class_id": majority,
        },
        "features": cache_meta,
        "runs": descriptions,
        **counts,
        "eval_videos": len(reference),
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
