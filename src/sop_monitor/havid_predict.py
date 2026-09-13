"""Predict a HA-ViD split with the saved networks of a checkpointed ``train-havid-tas`` run.

Nothing is trained or selected here: every network is the one whose epoch was chosen on val when
the run was trained (``--checkpoint-dir``), and the fusion rules are parameter-free. The output is
a run directory in the same format as the training run (``predictions_<split>.csv``,
``metrics.json``, ``config.json``, ``tables.md``), so ``reproduce-lite`` and the SOP tools read it
unchanged. This is the only path by which the frozen test split is scored
(``docs/havid_test_protocol.md``); it refuses to write into an existing directory, so a test run
cannot be silently repeated or overwritten.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from sop_monitor.baseline import (
    PredictionRow,
    load_metrics,
    render_tables,
    score_predictions,
    top_confusions,
    write_predictions,
)
from sop_monitor.havid import load_official_zip, load_temporal_zip
from sop_monitor.havid_tas import (
    FUSION_RULES,
    HavidTASSpec,
    advance_outputs,
    fuse_views,
    load_havid_videos,
)
from sop_monitor.mstcn import load_checkpoint, predict_probs


def predict_havid_tas(
    checkpoint_dir: Path,
    features_dir: Path,
    split_csv: Path,
    temporal_zip: Path,
    official_zip: Path,
    out_dir: Path,
    device: str = "cpu",
    n_boot: int = 2000,
) -> dict[str, object]:
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; predictions are written once")
    started = time.perf_counter()
    meta = json.loads((checkpoint_dir / "meta.json").read_text(encoding="utf-8"))
    spec_meta = meta["spec"]
    spec = HavidTASSpec(hand=spec_meta["hand"], level=spec_meta["level"])
    classes = meta["classes"]
    inverse = np.asarray(classes["index_to_class_id"], dtype=np.int64)
    lookahead = int(spec_meta["lookahead"])
    temporal, _ = load_temporal_zip(temporal_zip)
    mapping = load_official_zip(official_zip, [spec.subdataset])[spec.subdataset].mapping
    class_ids = {label: index for index, label in mapping.items()}
    if {str(i): label for i, label in sorted(mapping.items())} != classes["class_id_to_label"]:
        raise ValueError("the official mapping differs from the one the networks were trained with")
    segments = temporal[spec.subdataset]
    views = [int(v) for v in spec_meta["views"]]
    variants = sorted({name.split("_", 1)[1] for name in meta["networks"]})
    probs: dict[str, list[np.ndarray]] = {}
    reference = None
    for view in views:
        videos = load_havid_videos(features_dir, split_csv, segments, class_ids, view)
        if reference is None:
            reference = videos
        elif [(v.video_id, len(v.frames)) for v in videos] != [
            (v.video_id, len(v.frames)) for v in reference
        ]:
            raise ValueError(f"view {view} covers different recordings or frames")
        for variant in variants:
            network = f"view{view}_{variant}"
            model, standardise, info = load_checkpoint(checkpoint_dir / f"{network}.pt", device)
            shift = lookahead if info["causal"] else 0
            probs[network] = [
                advance_outputs(predict_probs(model, v.features, standardise, device), shift)
                for v in videos
            ]
    assert reference is not None
    for variant in variants:
        for rule in FUSION_RULES:
            probs[f"fusion{rule}_{variant}"] = []
        for i in range(len(reference)):
            fused = fuse_views(np.stack([probs[f"view{view}_{variant}"][i] for view in views]))
            for rule, fused_probs in fused.items():
                probs[f"fusion{rule}_{variant}"].append(fused_probs)
    majority = int(classes["majority_class_id"])
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
    split = split_csv.stem
    out_dir.mkdir(parents=True)
    write_predictions(out_dir / f"predictions_{split}.csv", rows)
    metrics = score_predictions(rows, n_boot=n_boot, seed=0)
    metrics["confusions"] = top_confusions(rows, "fusion_causal")
    metrics["eval_split"] = split
    metrics["train_split"] = "train"
    training_config = json.loads(
        (Path(meta["run_dir"]) / "config.json").read_text(encoding="utf-8")
    )
    config = {
        "model": training_config["model"],
        "dataset": "HA-ViD",
        "spec": spec_meta,
        "split_csv": split_csv.as_posix(),
        "split_sha256": hashlib.sha256(split_csv.read_bytes()).hexdigest(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "training_run": meta["run_dir"],
        "training": training_config["training"],
        "classes": classes,
        "features": training_config["features"],
        "runs": training_config["runs"],
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


def compare_runs(pairs: list[tuple[Path, Path]], split: str = "val") -> dict[str, object]:
    """Reproducibility gate: frame agreement per prediction column and metric differences.

    Each pair is ``(committed run, retrained run)``; both must predict the same frames.
    """
    from sop_monitor.baseline import load_metrics, read_predictions

    out: dict[str, object] = {"split": split, "pairs": []}
    identical = True
    for committed, retrained in pairs:
        a = read_predictions(committed / f"predictions_{split}.csv")
        b = read_predictions(retrained / f"predictions_{split}.csv")
        if [(r.video_id, r.frame, r.gt) for r in a] != [(r.video_id, r.frame, r.gt) for r in b]:
            raise ValueError(f"{committed} and {retrained} cover different frames")
        columns = sorted(set(a[0].preds) & set(b[0].preds))
        agreement = {
            column: sum(x.preds[column] == y.preds[column] for x, y in zip(a, b, strict=True))
            / len(a)
            for column in columns
        }
        metrics_a = load_metrics(committed)["runs"]
        metrics_b = load_metrics(retrained)["runs"]
        deltas = {
            column: {
                key: metrics_b[column][key]["point"] - metrics_a[column][key]["point"]
                for key in ("mof", "edit", "f1@10", "f1@25", "f1@50")
            }
            for column in columns
            if column in metrics_a and column in metrics_b
        }
        same = all(value == 1.0 for value in agreement.values())
        identical = identical and same
        out["pairs"].append(  # type: ignore[union-attr]
            {
                "committed": committed.as_posix(),
                "retrained": retrained.as_posix(),
                "identical": same,
                "frame_agreement": agreement,
                "metric_deltas": deltas,
            }
        )
    out["all_identical"] = identical
    return out
