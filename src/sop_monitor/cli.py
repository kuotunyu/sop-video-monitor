"""``sop-monitor`` command line (typer).

Data-free commands (splits, SOP graph checks, scoring committed prediction tables) need only the
base install; ``audit-industreal``, ``extract-features`` and ``train-baseline`` need the
``baseline`` dependency group (PyAV, torch) and the local IndustReal copy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Multi-camera SOP sequence monitoring (development build).",
)


@app.command("freeze-splits")
def freeze_splits(
    dataset: Annotated[str, typer.Option(help="industreal | ha-vid")] = "industreal",
    labels_dir: Annotated[
        Path | None, typer.Option(help="IndustReal label CSV directory (train/val/test.csv)")
    ] = None,
    official: Annotated[
        Path | None, typer.Option(help="HA-ViD ActionSegmentation/data zip (official bundles)")
    ] = None,
    temporal: Annotated[
        Path | None, typer.Option(help="HA-ViD HAViD_temporalAnnotation.zip")
    ] = None,
    out_dir: Annotated[str, typer.Option(help="Where train/val/test.csv land")] = "splits",
    seed: Annotated[int, typer.Option()] = 0,
    n_val: Annotated[int, typer.Option(help="HA-ViD: validation subjects drawn from train")] = 6,
) -> None:
    """Rebuild the subject-wise split and write splits/*.csv + test_sha256.txt (spec 4.1)."""
    if dataset == "ha-vid":
        if official is None or temporal is None:
            typer.echo("--official and --temporal are required for ha-vid")
            raise typer.Exit(code=1)
        from sop_monitor.havid import freeze_havid_splits

        manifest = freeze_havid_splits(official, temporal, Path(out_dir) / "ha-vid", seed, n_val)
        for name, info in manifest["splits"].items():  # type: ignore[union-attr]
            typer.echo(
                f"{name}: {info['videos']} videos, {info['recordings']} recordings, "
                f"{len(info['subjects'])} subjects"
            )
        typer.echo(f"excluded recordings: {manifest['excluded_recordings']}")
        typer.echo(f"test_sha256: {manifest['test_sha256']}")
        return
    if dataset == "industreal":
        if labels_dir is None:
            typer.echo("--labels-dir is required for industreal")
            raise typer.Exit(code=1)
        from sop_monitor.industreal import freeze_industreal_splits

        manifest = freeze_industreal_splits(labels_dir, Path(out_dir))
        for name, info in manifest["splits"].items():  # type: ignore[union-attr]
            typer.echo(
                f"{name}: {info['videos']} videos, {info['rows']} segments, "
                f"{len(info['participants'])} participants"
            )
        typer.echo(f"test_sha256: {manifest['test_sha256']}")
        return
    typer.echo(f"unknown dataset {dataset!r} (industreal | ha-vid)")
    raise typer.Exit(code=1)


@app.command("verify-splits")
def verify_splits(
    directory: Annotated[
        Path, typer.Option(help="Split directory with test.csv and test_sha256.txt")
    ] = Path("splits/industreal"),
) -> None:
    """Recompute the frozen test-list hash from test.csv and compare it with test_sha256.txt (spec 4.1)."""
    import csv

    from sop_monitor.splits import hash_test_list, read_test_sha256

    with (directory / "test.csv").open(encoding="utf-8", newline="") as handle:
        video_ids = [row["video_id"] for row in csv.DictReader(handle)]
    actual = hash_test_list(video_ids)
    expected = read_test_sha256(directory / "test_sha256.txt")
    typer.echo(f"{directory}: {len(video_ids)} test videos, sha256 {actual}")
    if actual != expected:
        typer.echo(f"MISMATCH: test_sha256.txt says {expected}")
        raise typer.Exit(code=1)
    typer.echo("OK: test list hash matches test_sha256.txt")


@app.command("check-sop")
def check_sop(
    graph: Annotated[Path, typer.Option(help="OWL file or JSON export of the precedence graph")],
    steps: Annotated[
        Path | None, typer.Option(help="Observed step sequence: one step id per line")
    ] = None,
    level: Annotated[str, typer.Option(help="PT (primitive task) or AA (atomic action)")] = "PT",
) -> None:
    """Run precedence and omission checks on an observed step sequence (spec 6.4)."""
    from sop_monitor.sop_graph import check_order, load_graph

    loaded = load_graph(graph)
    if steps is None:
        typer.echo(
            json.dumps(
                {
                    "source": loaded.source,
                    "nodes": len(loaded.nodes),
                    "levels": {
                        name: {
                            "nodes": len(loaded.nodes_of(name)),
                            "order": loaded.topological_order(name),
                        }
                        for name in ("PT", "AA")
                    },
                },
                ensure_ascii=False,
            )
        )
        return
    observed = [
        line.strip() for line in steps.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    report = check_order(loaded, observed, level=level)
    typer.echo(json.dumps(report.to_dict(), ensure_ascii=False))
    raise typer.Exit(code=0 if report.ok else 1)


@app.command("export-sop")
def export_sop(
    owl: Annotated[Path, typer.Option(help="HA-ViD precedence graph (OWL)")],
    out: Annotated[Path, typer.Option(help="JSON export path")],
) -> None:
    """Export an OWL precedence graph to the repo's JSON form (spec 6.4)."""
    from sop_monitor.sop_graph import export_json, load_owl

    loaded = load_owl(owl)
    export_json(loaded, out)
    typer.echo(
        f"{owl.name}: {len(loaded.nodes_of('PT'))} PT, {len(loaded.nodes_of('AA'))} AA, "
        f"{len(loaded.relation('precedesPT'))} precedesPT, {len(loaded.relation('precedesAA'))} precedesAA -> {out}"
    )


@app.command("audit-industreal")
def audit_industreal_cmd(
    root: Annotated[Path, typer.Option(help="IndustReal root (rgb/, labels/)")] = Path(
        "data/external/industreal"
    ),
    ha_vid: Annotated[Path, typer.Option(help="HA-ViD public files directory")] = Path(
        "data/external/ha-vid-public"
    ),
    ha_vid_delivered: Annotated[
        Path, typer.Option(help="Directory with the HA-ViD archives delivered by the authors")
    ] = Path("data/external/ha-vid"),
    out: Annotated[Path, typer.Option(help="Audit JSON path")] = Path("reports/data_audit.json"),
    manifest: Annotated[
        Path | None, typer.Option(help="Also write data/manifest.json (names, sizes, SHA-256)")
    ] = None,
) -> None:
    """Probe every local video, cross-check labels against frame counts, inventory HA-ViD files."""
    from sop_monitor.audit import audit_ha_vid_public, audit_industreal, build_manifest, write_json

    payload = {"industreal": audit_industreal(root), "ha_vid_public": audit_ha_vid_public(ha_vid)}
    write_json(out, payload)
    industreal = payload["industreal"]
    typer.echo(
        f"IndustReal: {industreal['videos']['count']} videos, "  # type: ignore[index]
        f"{industreal['videos']['total_hours']:.2f} h, "  # type: ignore[index]
        f"{len(industreal['problems'])} problem(s)"  # type: ignore[index]
    )
    typer.echo(
        f"HA-ViD public: {payload['ha_vid_public']['files']} files, "  # type: ignore[index]
        f"videos={payload['ha_vid_public']['videos']}, "  # type: ignore[index]
        f"usable_for_training={payload['ha_vid_public']['usable_for_training']}"  # type: ignore[index]
    )
    if manifest is not None:
        entries = [("IndustReal", p) for p in sorted(root.glob("*.zip"))]
        entries += [
            ("IndustReal", root / "labels" / name) for name in ("train.csv", "val.csv", "test.csv")
        ]
        entries += (
            [("HA-ViD public", p) for p in sorted(ha_vid.glob("*.zip"))] if ha_vid.is_dir() else []
        )
        entries += (
            [("HA-ViD", p) for p in sorted(ha_vid_delivered.glob("*.zip"))]
            if ha_vid_delivered.is_dir()
            else []
        )
        write_json(manifest, build_manifest(entries))
        typer.echo(f"manifest: {len(entries)} files -> {manifest}")
    if industreal["problems"]:  # type: ignore[index]
        for problem in industreal["problems"]:  # type: ignore[index]
            typer.echo(f"problem: {problem}")
        raise typer.Exit(code=1)


@app.command("audit-havid")
def audit_havid_cmd(
    temporal: Annotated[Path, typer.Option(help="HAViD_temporalAnnotation.zip")] = Path(
        "data/external/ha-vid/HAViD_temporalAnnotation.zip"
    ),
    official: Annotated[
        Path | None,
        typer.Option(help="ActionSegmentation/data folder zip (splits, mapping.txt, groundTruth)"),
    ] = None,
    rgb_dir: Annotated[
        Path | None, typer.Option(help="Directory holding the extracted HA-ViD mp4s (needs PyAV)")
    ] = None,
    out: Annotated[Path, typer.Option(help="Audit JSON path")] = Path("reports/havid_audit.json"),
) -> None:
    """Cross-check the HA-ViD temporal annotations, the official split and the videos on disk."""
    from sop_monitor.audit import write_json
    from sop_monitor.havid import audit_havid

    payload = audit_havid(temporal, official, rgb_dir)
    write_json(out, payload)
    annotations = payload["annotations"]
    typer.echo(
        f"HA-ViD annotations: {annotations['recordings']} recordings, "  # type: ignore[index]
        f"{annotations['subjects']} subjects, "  # type: ignore[index]
        f"{annotations['frames']['total']} frames, "  # type: ignore[index]
        f"wrong segments={annotations['wrong']['segments']}"  # type: ignore[index]
    )
    if "official_split" in payload:
        split = payload["official_split"]
        typer.echo(
            f"official split: subject_disjoint={split['subject_disjoint']}, "  # type: ignore[index]
            f"groundTruth trim={split['ground_truth_trim_frames']}"  # type: ignore[index]
        )
    if "videos" in payload:
        videos = payload["videos"]
        typer.echo(
            f"videos: {videos['annotated_videos_found']} annotated mp4s found, "  # type: ignore[index]
            f"{videos['recordings_with_all_three_views']} recordings with all three views, "  # type: ignore[index]
            f"{videos['annotated_hours']:.2f} h"  # type: ignore[index]
        )
    for problem in payload["problems"]:  # type: ignore[union-attr]
        typer.echo(f"problem: {problem}")
    if payload["problems"]:
        raise typer.Exit(code=1)


@app.command("extract-features")
def extract_features_cmd(
    root: Annotated[Path, typer.Option(help="IndustReal root (rgb/, labels/)")] = Path(
        "data/external/industreal"
    ),
    split: Annotated[
        list[str] | None,
        typer.Option(help="Label splits whose videos to embed (default train, val)"),
    ] = None,
    out: Annotated[Path, typer.Option(help="Feature cache root")] = Path(
        "artifacts/features/industreal"
    ),
    model: Annotated[
        str, typer.Option(help="dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14")
    ] = "dinov2_vits14",
    stride: Annotated[int, typer.Option(help="Keep every stride-th frame")] = 1,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
    batch_size: Annotated[int, typer.Option()] = 256,
    overwrite: Annotated[bool, typer.Option()] = False,
    dataset: Annotated[str, typer.Option(help="industreal | ha-vid")] = "industreal",
    split_dir: Annotated[
        Path, typer.Option(help="HA-ViD: frozen split directory whose train/val videos to embed")
    ] = Path("splits/ha-vid"),
    rgb_dir: Annotated[
        Path,
        typer.Option(help="HA-ViD: directory holding the extracted mp4s (searched recursively)"),
    ] = Path("data/external/ha-vid/HAViD_rgb"),
) -> None:
    """Cache frozen DINOv2 frame features for the labelled videos of the chosen splits."""
    from sop_monitor.features import extract_videos

    videos: list[Path] = []
    if dataset == "ha-vid":
        from sop_monitor.havid_tas import read_split

        on_disk = {p.stem: p for p in rgb_dir.rglob("*.mp4")}
        for name in split or ["train", "val"]:
            for row in read_split(split_dir / f"{name}.csv"):
                if row["video_id"] not in on_disk:
                    typer.echo(f"missing mp4 for {row['video_id']} under {rgb_dir}")
                    raise typer.Exit(code=1)
                videos.append(on_disk[row["video_id"]])
    elif dataset == "industreal":
        from sop_monitor.industreal import LABEL_FILES, read_labels, segments_by_video

        for name in split or ["train", "val"]:
            for video_id in sorted(
                segments_by_video(read_labels(root / "labels" / LABEL_FILES[name]))
            ):
                videos.append(root / "rgb" / f"{video_id}.mp4")
    else:
        typer.echo(f"unknown dataset {dataset!r} (industreal | ha-vid)")
        raise typer.Exit(code=1)
    cache_dir = out / f"{model}_s{stride}"
    stats = extract_videos(videos, cache_dir, model, device, stride, batch_size, overwrite)
    total_seconds = sum(s.seconds for s in stats)
    total_sampled = sum(s.n_sampled for s in stats)
    for stat in stats:
        typer.echo(
            f"{stat.video_id}: {stat.n_sampled}/{stat.n_frames} frames in {stat.seconds:.1f}s ({stat.sampled_fps:.0f} fps)"
        )
    typer.echo(
        f"{len(stats)} video(s) embedded, {len(videos) - len(stats)} already cached; "
        f"{total_sampled} frames in {total_seconds:.1f}s -> {cache_dir}"
    )
    log = {
        "videos": [s.__dict__ for s in stats],
        "total_seconds": total_seconds,
        "total_sampled": total_sampled,
    }
    (cache_dir / "extraction_log.json").write_text(
        json.dumps(log, indent=2) + "\n", encoding="utf-8"
    )


@app.command("export-official-features")
def export_official_features_cmd(
    official: Annotated[Path, typer.Option(help="HA-ViD ActionSegmentation/data zip")] = Path(
        "data/external/ha-vid/ActionSegmentation_data.zip"
    ),
    split_dir: Annotated[
        Path | None,
        typer.Option(help="Only export the train/val videos of this split directory"),
    ] = Path("splits/ha-vid"),
    out: Annotated[Path, typer.Option(help="npz cache directory")] = Path(
        "artifacts/features/ha-vid/i3d_official"
    ),
) -> None:
    """Export the official HA-ViD I3D features into the npz layout used by the training commands."""
    from sop_monitor.havid_tas import export_official_features, read_split

    videos = None
    if split_dir is not None:
        videos = [
            row["video_id"]
            for name in ("train", "val")
            for row in read_split(split_dir / f"{name}.csv")
        ]
    written = export_official_features(official, out, videos)
    typer.echo(f"{written} feature file(s) exported -> {out}")


@app.command("train-havid-tas")
def train_havid_tas_cmd(
    features: Annotated[
        Path,
        typer.Option(help="npz cache dir, e.g. artifacts/features/ha-vid/dinov2_vitb14_s1"),
    ],
    out: Annotated[Path, typer.Option(help="Run directory for predictions/config/metrics")],
    split_dir: Annotated[Path, typer.Option()] = Path("splits/ha-vid"),
    temporal: Annotated[Path, typer.Option(help="HAViD_temporalAnnotation.zip")] = Path(
        "data/external/ha-vid/HAViD_temporalAnnotation.zip"
    ),
    official: Annotated[Path, typer.Option(help="ActionSegmentation/data zip (mapping)")] = Path(
        "data/external/ha-vid/ActionSegmentation_data.zip"
    ),
    hand: Annotated[str, typer.Option(help="lh | rh")] = "lh",
    level: Annotated[str, typer.Option(help="pt | aa")] = "pt",
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
    epochs: Annotated[int, typer.Option()] = 50,
    eval_every: Annotated[int, typer.Option()] = 5,
    n_boot: Annotated[int, typer.Option()] = 2000,
    seed: Annotated[int, typer.Option()] = 0,
    selection_metric: Annotated[
        str,
        typer.Option(
            help="Val metric that picks each network's epoch: mof | edit | f1@10 | f1@25 | f1@50"
        ),
    ] = "mof",
) -> None:
    """Per-view causal and offline MS-TCN++ plus late fusion on HA-ViD train -> val (spec W2)."""
    from sop_monitor.features import resolve_device
    from sop_monitor.havid_tas import HavidTASSpec, run_havid_tas
    from sop_monitor.mstcn import MSTCNSpec

    if hand not in ("lh", "rh") or level not in ("pt", "aa"):
        typer.echo("--hand must be lh or rh and --level pt or aa")
        raise typer.Exit(code=1)
    if selection_metric not in ("mof", "edit", "f1@10", "f1@25", "f1@50"):
        typer.echo("--selection-metric must be one of mof, edit, f1@10, f1@25, f1@50")
        raise typer.Exit(code=1)
    spec = HavidTASSpec(
        hand=hand,
        level=level,
        mstcn=MSTCNSpec(
            epochs=epochs,
            eval_every=eval_every,
            seed=seed,
            n_boot=n_boot,
            selection_metric=selection_metric,
        ),
    )
    result = run_havid_tas(
        features, split_dir, temporal, official, out, spec, device=resolve_device(device)
    )
    typer.echo(result["tables"])
    for name, log in result["config"]["training"].items():  # type: ignore[index, union-attr]
        typer.echo(
            f"{name}: best epoch {log['best_epoch']}, {log['parameters']} parameters, "
            f"{log['train_seconds']:.0f}s"
        )
    typer.echo(f"wall: {result['config']['wall_seconds']:.1f}s -> {out}")  # type: ignore[index]


@app.command("summarise-havid-seeds")
def summarise_havid_seeds_cmd(
    run: Annotated[list[Path], typer.Option(help="train-havid-tas run directories, one per seed")],
    out: Annotated[Path, typer.Option(help="Summary directory (seed_summary.json, tables.md)")],
) -> None:
    """Mean and std over seeds of every run's metrics; recomputable by reproduce-lite."""
    from sop_monitor.havid_seeds import render_seed_tables, write_seed_summary

    payload = write_seed_summary(run, out)
    typer.echo(render_seed_tables(payload))
    typer.echo(f"-> {out}")


@app.command("train-baseline")
def train_baseline_cmd(
    features: Annotated[
        Path,
        typer.Option(help="Feature cache dir, e.g. artifacts/features/industreal/dinov2_vits14_s1"),
    ],
    labels_dir: Annotated[Path, typer.Option()] = Path("data/external/industreal/labels"),
    out: Annotated[Path, typer.Option(help="Run directory for predictions/config/metrics")] = Path(
        "reports/industreal_dev_v1"
    ),
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
    epochs: Annotated[int, typer.Option()] = 300,
    n_boot: Annotated[int, typer.Option()] = 2000,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Train the linear head on train, select and report on val, write the run directory."""
    from sop_monitor.baseline import BaselineSpec, load_metrics, render_tables, run_baseline
    from sop_monitor.features import resolve_device

    spec = BaselineSpec(epochs=epochs, seed=seed, n_boot=n_boot)
    result = run_baseline(features, labels_dir, out, spec, device=resolve_device(device))
    # Render from the files just written, exactly as `score-predictions` will re-read them.
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    tables = render_tables(load_metrics(out), config)
    (out / "tables.md").write_text(tables, encoding="utf-8", newline="\n")
    typer.echo(tables)
    typer.echo(f"wall: {result['config']['wall_seconds']:.1f}s -> {out}")  # type: ignore[index]


@app.command("train-mstcn")
def train_mstcn_cmd(
    features: Annotated[
        Path,
        typer.Option(help="Feature cache dir, e.g. artifacts/features/industreal/dinov2_vits14_s1"),
    ],
    labels_dir: Annotated[Path, typer.Option()] = Path("data/external/industreal/labels"),
    out: Annotated[Path, typer.Option(help="Run directory for predictions/config/metrics")] = Path(
        "reports/industreal_dev_v2_mstcn"
    ),
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
    epochs: Annotated[int, typer.Option()] = 50,
    eval_every: Annotated[int, typer.Option()] = 5,
    n_boot: Annotated[int, typer.Option()] = 2000,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Train causal and non-causal MS-TCN++ heads on train, select epochs on val, write the run dir."""
    from sop_monitor.features import resolve_device
    from sop_monitor.mstcn import MSTCNSpec, run_mstcn_baseline

    spec = MSTCNSpec(epochs=epochs, eval_every=eval_every, seed=seed, n_boot=n_boot)
    result = run_mstcn_baseline(features, labels_dir, out, spec, device=resolve_device(device))
    typer.echo(result["tables"])
    for name, log in result["config"]["training"].items():  # type: ignore[index, union-attr]
        typer.echo(
            f"{name}: best epoch {log['best_epoch']}, {log['parameters']} parameters, "
            f"{log['train_seconds']:.0f}s"
        )
    typer.echo(f"wall: {result['config']['wall_seconds']:.1f}s -> {out}")  # type: ignore[index]


@app.command("extract-psr-labels")
def extract_psr_labels_cmd(
    archive: Annotated[
        list[Path], typer.Option(help="Recording archives, e.g. val_p1.zip val_p2.zip")
    ],
    out: Annotated[Path, typer.Option(help="Where <recording>/PSR_labels*.csv land")] = Path(
        "data/external/industreal/psr"
    ),
) -> None:
    """Extract only the PSR label CSVs from the per-split recording archives (archives untouched)."""
    from sop_monitor.industreal_psr import extract_psr_labels

    recordings = extract_psr_labels(archive, out)
    typer.echo(f"{len(recordings)} recording(s) with PSR labels -> {out}")
    for name in recordings:
        typer.echo(f"  {name}")


@app.command("train-psr")
def train_psr_cmd(
    features: Annotated[Path, typer.Option(help="Feature cache dir of the val videos")],
    psr_dir: Annotated[Path, typer.Option()] = Path("data/external/industreal/psr"),
    head: Annotated[
        list[str] | None, typer.Option(help="linear | mstcn (repeatable; default both)")
    ] = None,
    decoder: Annotated[
        list[str] | None, typer.Option(help="plain | prior_dwell (repeatable; default both)")
    ] = None,
    mstcn_epochs: Annotated[int, typer.Option()] = 40,
    mstcn_epoch_grid: Annotated[
        list[int] | None,
        typer.Option(
            help="Nested epoch selection grid for the MS-TCN++ head (repeatable; nested only)"
        ),
    ] = None,
    mstcn_epoch_criterion: Annotated[
        str, typer.Option(help="bce | decoded_f1 (how the epoch is chosen from the grid)")
    ] = "bce",
    train_split: Annotated[
        Path | None, typer.Option(help="Split CSV of training recordings (with --eval-split)")
    ] = None,
    eval_split: Annotated[
        Path | None, typer.Option(help="Split CSV of evaluated recordings (with --train-split)")
    ] = None,
    selection: Annotated[
        str,
        typer.Option(help="in_sample | nested (decoder chosen on out-of-fold train predictions)"),
    ] = "in_sample",
    delay_cap: Annotated[
        list[float] | None,
        typer.Option(help="Latency budget(s) in seconds for decoder selection (repeatable)"),
    ] = None,
    out: Annotated[Path, typer.Option()] = Path("reports/industreal_dev_v3_psr"),
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
    epochs: Annotated[int, typer.Option()] = 300,
    n_boot: Annotated[int, typer.Option()] = 2000,
    seed: Annotated[
        list[int] | None, typer.Option(help="Repeatable; several seeds become <run>_s<seed>")
    ] = None,
) -> None:
    """Step-completion baselines: leave-one-participant-out over psr-dir, or train-split -> eval-split."""
    from sop_monitor.features import resolve_device
    from sop_monitor.psr_baseline import (
        DECODERS,
        HEADS,
        PSRSpec,
        read_split_ids,
        run_psr_baseline,
    )

    spec = PSRSpec(
        epochs=epochs,
        mstcn_epochs=mstcn_epochs,
        mstcn_epoch_grid=tuple(mstcn_epoch_grid or ()),
        mstcn_epoch_criterion=mstcn_epoch_criterion,
        n_boot=n_boot,
        delay_caps_s=tuple(delay_cap or ()),
    )
    result = run_psr_baseline(
        features,
        psr_dir,
        out,
        spec,
        device=resolve_device(device),
        heads=tuple(head or HEADS),
        decoders=tuple(decoder or DECODERS),
        train_ids=read_split_ids(train_split) if train_split else None,
        eval_ids=read_split_ids(eval_split) if eval_split else None,
        selection=selection,
        seeds=tuple(seed or [0]),
    )
    typer.echo(result["tables"])
    typer.echo(f"wall: {result['config']['wall_seconds']:.1f}s -> {out}")  # type: ignore[index]


def _check_psr_run(run_dir: Path, write: bool) -> list[str]:
    """PSR runs: recompute metrics from completions_*.csv and check tables.md."""
    from sop_monitor.baseline import load_metrics
    from sop_monitor.psr_baseline import read_completions, render_psr_tables, score_completions

    committed = load_metrics(run_dir)
    boot = committed.get("bootstrap", {})
    rows = read_completions(sorted(run_dir.glob("completions_*.csv"))[0])
    run_names = (
        list(committed["runs"]) if "runs" in committed else [str(committed.get("run", "pred"))]
    )  # type: ignore[arg-type]
    recomputed = score_completions(
        rows,
        n_boot=int(boot.get("n_boot", 2000)),  # type: ignore[union-attr]
        seed=int(boot.get("seed", 0)),  # type: ignore[union-attr]
        run_names=run_names,
    )
    mismatches: list[str] = []
    scored_keys = ("summary", "per_video", "totals")
    for key in (*scored_keys, "gt_sanity", "seed_summary", "n_videos", "n_participants"):
        if key in recomputed and json.dumps(recomputed[key], sort_keys=True) != json.dumps(
            committed.get(key), sort_keys=True
        ):
            mismatches.append(f"{run_dir.name}: {key} differs from metrics.json")
    for run, scored in recomputed.get("runs", {}).items():  # type: ignore[union-attr]
        stored = committed.get("runs", {}).get(run, {})  # type: ignore[union-attr]
        for key in scored_keys:
            if json.dumps(scored[key], sort_keys=True) != json.dumps(
                stored.get(key), sort_keys=True
            ):
                mismatches.append(f"{run_dir.name}: runs/{run}/{key} differs from metrics.json")
    config_path = run_dir / "config.json"
    if config_path.is_file():
        rendered = render_psr_tables(committed, json.loads(config_path.read_text(encoding="utf-8")))
        tables_path = run_dir / "tables.md"
        if write:
            tables_path.write_text(rendered, encoding="utf-8", newline="\n")
        elif not tables_path.is_file() or tables_path.read_text(encoding="utf-8") != rendered:
            mismatches.append(
                f"{run_dir.name}: tables.md is stale (run with --write to regenerate)"
            )
    # Committed SOP checks are recomputed from the completions and the graphs they name.
    from sop_monitor.industreal_sop import sop_checks_for_run

    for checks_path in sorted(run_dir.glob("sop_checks_*.json")):
        stored = json.loads(checks_path.read_text(encoding="utf-8"))
        if "graphs_dir" not in stored:
            mismatches.append(f"{run_dir.name}: {checks_path.name} has no graphs_dir; regenerate")
            continue
        fresh = sop_checks_for_run(run_dir, str(stored["run"]), Path(str(stored["graphs_dir"])))
        for key in ("per_video", "video_level", "graphs"):
            if json.dumps(fresh[key], sort_keys=True) != json.dumps(
                stored.get(key), sort_keys=True
            ):
                mismatches.append(f"{run_dir.name}: {checks_path.name}/{key} differs")
    return mismatches


def _check_seed_summary(run_dir: Path, write: bool) -> list[str]:
    """Recompute a multi-seed summary from the listed run directories' committed metrics."""
    from sop_monitor.havid_seeds import render_seed_tables, summarise_seeds

    committed = json.loads((run_dir / "seed_summary.json").read_text(encoding="utf-8"))
    recomputed = summarise_seeds([Path(p) for p in committed["runs"]])
    mismatches: list[str] = []
    for key in ("summary", "best_epochs", "seeds"):
        if json.dumps(recomputed[key], sort_keys=True) != json.dumps(
            committed[key], sort_keys=True
        ):
            mismatches.append(f"{run_dir.name}: {key} differs from seed_summary.json")
    rendered = render_seed_tables(committed)
    tables_path = run_dir / "tables.md"
    if write:
        tables_path.write_text(rendered, encoding="utf-8", newline=chr(10))
    elif not tables_path.is_file() or tables_path.read_text(encoding="utf-8") != rendered:
        mismatches.append(f"{run_dir.name}: tables.md is stale (run with --write to regenerate)")
    return mismatches


def _check_havid_sop_run(run_dir: Path, write: bool) -> list[str]:
    """Recompute a HA-ViD SOP run's tables from its committed step CSVs and the learned JSON files."""
    from sop_monitor.havid_sop import evaluate_run, render_sop_tables

    committed = json.loads((run_dir / "sop_checks.json").read_text(encoding="utf-8"))
    config = committed["config"]
    recomputed = evaluate_run(run_dir, Path(config["graphs_dir"]), int(config["seed"]))
    mismatches: list[str] = []
    for key in ("graphs", "sources", "wrong"):
        if json.dumps(recomputed[key], sort_keys=True) != json.dumps(
            committed[key], sort_keys=True
        ):
            mismatches.append(f"{run_dir.name}: {key} differs from sop_checks.json")
    rendered = render_sop_tables(committed)
    tables_path = run_dir / "tables.md"
    if write:
        tables_path.write_text(rendered, encoding="utf-8", newline="\n")
    elif not tables_path.is_file() or tables_path.read_text(encoding="utf-8") != rendered:
        mismatches.append(f"{run_dir.name}: tables.md is stale (run with --write to regenerate)")
    return mismatches


def _check_run(run_dir: Path, write: bool) -> list[str]:
    """Recompute a run's metrics from its prediction table; return the list of mismatches."""
    from sop_monitor.baseline import (
        load_metrics,
        read_predictions,
        render_tables,
        score_predictions,
    )

    if (run_dir / "steps_val.csv").is_file():
        return _check_havid_sop_run(run_dir, write)
    if (run_dir / "seed_summary.json").is_file():
        return _check_seed_summary(run_dir, write)
    if not (run_dir / "metrics.json").is_file():
        return []
    if sorted(run_dir.glob("completions_*.csv")):
        return _check_psr_run(run_dir, write)
    tables = sorted(run_dir.glob("predictions_*.csv"))
    if not tables:
        return []
    committed = load_metrics(run_dir)
    boot = committed.get("bootstrap", {})
    recomputed = score_predictions(
        read_predictions(tables[0]),
        n_boot=int(boot.get("n_boot", 2000)),  # type: ignore[union-attr]
        seed=int(boot.get("seed", 0)),  # type: ignore[union-attr]
    )
    mismatches: list[str] = []
    for key in ("runs", "per_video", "n_frames", "n_videos", "n_participants"):
        if json.dumps(recomputed[key], sort_keys=True) != json.dumps(
            committed.get(key), sort_keys=True
        ):
            mismatches.append(f"{run_dir.name}: {key} differs from metrics.json")
    config_path = run_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rendered = render_tables(committed, config)
        tables_path = run_dir / "tables.md"
        if write:
            tables_path.write_text(rendered, encoding="utf-8", newline="\n")
        elif not tables_path.is_file() or tables_path.read_text(encoding="utf-8") != rendered:
            mismatches.append(
                f"{run_dir.name}: tables.md is stale (run with --write to regenerate)"
            )
    return mismatches


@app.command("score-predictions")
def score_predictions_cmd(
    run: Annotated[
        Path, typer.Option(help="Run directory containing predictions_*.csv and metrics.json")
    ],
    write: Annotated[
        bool, typer.Option(help="Rewrite tables.md instead of only checking it")
    ] = False,
) -> None:
    """Recompute metrics from the committed prediction table and compare with metrics.json."""
    mismatches = _check_run(run, write)
    if not (run / "metrics.json").is_file():
        typer.echo(f"{run}: no metrics.json / predictions_*.csv to score")
        raise typer.Exit(code=1)
    for line in mismatches:
        typer.echo(line)
    typer.echo(f"{run.name}: {'OK, metrics reproduce' if not mismatches else 'MISMATCH'}")
    raise typer.Exit(code=1 if mismatches else 0)


@app.command("learn-havid-sop")
def learn_havid_sop_cmd(
    temporal: Annotated[Path, typer.Option(help="HAViD_temporalAnnotation.zip")] = Path(
        "data/external/ha-vid/HAViD_temporalAnnotation.zip"
    ),
    split_dir: Annotated[Path, typer.Option()] = Path("splits/ha-vid"),
    out: Annotated[Path, typer.Option(help="Where the learned JSON files land")] = Path(
        "sop/ha-vid"
    ),
    min_support: Annotated[
        int, typer.Option(help="Recordings two steps must share for an edge")
    ] = 3,
    mandatory_fraction: Annotated[
        float,
        typer.Option(
            help="A step is mandatory when present in this fraction of a plate's recordings"
        ),
    ] = 0.9,
    min_agreement: Annotated[
        float,
        typer.Option(help="Fraction of co-occurring recordings that must agree on the order"),
    ] = 1.0,
    low_q: Annotated[float, typer.Option(help="Lower duration quantile")] = 0.05,
    high_q: Annotated[float, typer.Option(help="Upper duration quantile")] = 0.95,
    granularity: Annotated[
        str, typer.Option(help="pt (HR-SAT primitive task) | sheet (instruction-sheet step)")
    ] = "pt",
) -> None:
    """Learn per-plate precedence graphs, mandatory steps and duration bounds from the train split."""
    from sop_monitor.havid_sop import learn_havid_sop

    summary = learn_havid_sop(
        temporal,
        split_dir,
        out,
        min_support,
        mandatory_fraction,
        low_q,
        high_q,
        min_agreement=min_agreement,
        granularity=granularity,
    )
    typer.echo(
        f"{summary['recordings']} train recordings, plates {summary['plates']}, "
        f"1st percentile of step length {summary['step_frames_q01']} frames"
    )
    for plate, info in summary["graphs"].items():  # type: ignore[union-attr]
        typer.echo(
            f"{plate}: {info['nodes']} steps, {info['edges']} edges, "
            f"{info['mandatory']} mandatory, {info['bounded']} with duration bounds -> {out}"
        )


@app.command("tune-havid-sop")
def tune_havid_sop_cmd(
    out: Annotated[Path, typer.Option(help="JSON with the leave-one-subject-out grids")],
    temporal: Annotated[Path, typer.Option(help="HAViD_temporalAnnotation.zip")] = Path(
        "data/external/ha-vid/HAViD_temporalAnnotation.zip"
    ),
    split_dir: Annotated[Path, typer.Option()] = Path("splits/ha-vid"),
    granularity: Annotated[
        str, typer.Option(help="pt (HR-SAT primitive task) | sheet (instruction-sheet step)")
    ] = "pt",
) -> None:
    """Choose the graph rule and the duration window by leave-one-subject-out on the train split."""
    from sop_monitor.havid_sop import tune_havid_sop

    payload = tune_havid_sop(temporal, split_dir, out, granularity)
    typer.echo(
        "| min support | min agreement | edges | coverage | rec. false alarms | step false alarms | score |"
    )
    typer.echo("|---|---|---|---|---|---|---|")
    for e in payload["order"]:  # type: ignore[union-attr]
        typer.echo(
            f"| {e['min_support']} | {e['min_agreement']} | {e['edges_full_train']} | {100 * e['coverage']:.0f} % "
            f"| {100 * e['recording_false_alarm_rate']:.0f} % | {100 * e['step_false_alarm_rate']:.1f} % | {e['score']:.2f} |"
        )
    for e in payload["duration"]:  # type: ignore[union-attr]
        typer.echo(
            f"duration [{e['low_q']}, {e['high_q']}]: step false alarms {100 * e['step_false_alarm_rate']:.1f} %"
        )
    typer.echo(f"selected: {payload['selected']} -> {out}")


@app.command("havid-sop")
def havid_sop_cmd(
    pred_lh: Annotated[Path, typer.Option(help="Left-hand HA-ViD TAS run directory")],
    pred_rh: Annotated[Path, typer.Option(help="Right-hand HA-ViD TAS run directory")],
    out: Annotated[Path, typer.Option(help="SOP run directory")],
    run_name: Annotated[str, typer.Option(help="Prediction column to use")] = "fusion_causal",
    temporal: Annotated[Path, typer.Option(help="HAViD_temporalAnnotation.zip")] = Path(
        "data/external/ha-vid/HAViD_temporalAnnotation.zip"
    ),
    split_dir: Annotated[Path, typer.Option()] = Path("splits/ha-vid"),
    graphs: Annotated[Path, typer.Option(help="Learned JSON directory")] = Path("sop/ha-vid"),
    seed: Annotated[int, typer.Option()] = 0,
    min_segment_frames: Annotated[
        int,
        typer.Option(
            help="Absorb predicted segments shorter than this into their neighbour (0 = off)"
        ),
    ] = 0,
    granularity: Annotated[
        str, typer.Option(help="pt (HR-SAT primitive task) | sheet (instruction-sheet step)")
    ] = "pt",
) -> None:
    """Val step sequences (ground truth and predicted), SOP checks, synthetic violations, native w."""
    from sop_monitor.havid_sop import render_sop_tables, run_havid_sop

    payload = run_havid_sop(
        temporal,
        split_dir,
        graphs,
        {"lh": pred_lh, "rh": pred_rh},
        run_name,
        out,
        seed,
        min_segment_frames,
        granularity,
    )
    typer.echo(render_sop_tables(payload))
    typer.echo(f"-> {out}")


@app.command("learn-sop")
def learn_sop_cmd(
    psr_dir: Annotated[Path, typer.Option()] = Path("data/external/industreal/psr"),
    train_split: Annotated[Path, typer.Option()] = Path("splits/industreal/train.csv"),
    out: Annotated[Path, typer.Option(help="Directory for learned_precedence_<kind>.json")] = Path(
        "sop/industreal"
    ),
    min_support: Annotated[int, typer.Option()] = 3,
) -> None:
    """Learn one IndustReal step precedence graph per recording kind from the train-split PSR labels."""
    from sop_monitor.industreal_psr import PROCEDURE_INFO, load_procedure_info, load_psr_labels
    from sop_monitor.industreal_sop import KINDS, describe, learn_precedence
    from sop_monitor.psr_baseline import read_split_ids, recording_kind
    from sop_monitor.sop_graph import export_json

    steps = load_procedure_info(PROCEDURE_INFO)
    train_ids = sorted(read_split_ids(train_split))
    for kind in KINDS:
        recordings = [v for v in train_ids if recording_kind(v) == kind]
        sequences = [load_psr_labels(psr_dir / v / "PSR_labels.csv") for v in recordings]
        graph = learn_precedence(
            sequences,
            min_support=min_support,
            source=f"learned from {len(recordings)} IndustReal train-split {kind} recordings (PSR_labels.csv, min_support={min_support})",
        )
        target = out / f"learned_precedence_{kind}.json"
        export_json(graph, target)
        typer.echo(
            f"{kind}: {len(recordings)} recordings -> {len(graph.nodes)} steps, "
            f"{len(graph.relation('precedesPT'))} edges -> {target}"
        )
        for line in describe(graph, steps):
            typer.echo(f"  {line}")


@app.command("check-psr-run")
def check_psr_run_cmd(
    run: Annotated[Path, typer.Option(help="PSR run directory with completions_val.csv")],
    run_name: Annotated[
        str, typer.Option(help="Prediction run to check, e.g. mstcn_prior_dwell_s0")
    ],
    graphs: Annotated[
        Path, typer.Option(help="Directory with learned_precedence_<kind>.json")
    ] = Path("sop/industreal"),
    out: Annotated[
        str, typer.Option(help="JSON file name written into the run directory")
    ] = "sop_checks.json",
) -> None:
    """Run the precedence / omission checks on ground-truth and predicted completions of one run."""
    from sop_monitor.industreal_sop import sop_checks_for_run

    try:
        summary = sop_checks_for_run(run, run_name, graphs)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    (run / out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"graphs: {summary['graphs']}")
    typer.echo(f"video-level agreement over {summary['n_videos']} videos: {summary['video_level']}")
    typer.echo(
        "| video | kind | GT violations | GT omissions | pred violations | pred omissions | agreement |"
    )
    typer.echo("|---|---|---|---|---|---|---|")
    for video, entry in summary["per_video"].items():  # type: ignore[union-attr]
        typer.echo(
            f"| {video} | {entry['kind']} | {len(entry['gt_violations'])} | {len(entry['gt_omissions'])} | "
            f"{len(entry['pred_violations'])} | {len(entry['pred_omissions'])} | {entry['agreement']} |"
        )
    typer.echo(f"-> {run / out}")


@app.command("reproduce-lite")
def reproduce_lite(
    reports_dir: Annotated[
        str, typer.Option(help="Directory of committed run directories")
    ] = "reports",
    write: Annotated[
        bool, typer.Option(help="Rewrite tables.md files instead of only checking")
    ] = False,
) -> None:
    """Recompute every committed run's metrics from its prediction table (spec 8.3; CI path)."""
    runs = sorted(
        p
        for p in Path(reports_dir).glob("*")
        if p.is_dir()
        and (
            (p / "metrics.json").is_file()
            or (p / "steps_val.csv").is_file()
            or (p / "seed_summary.json").is_file()
        )
    )
    if not runs:
        typer.echo(f"nothing to rebuild: no run directory with metrics.json under {reports_dir}/")
        raise typer.Exit(code=0)
    mismatches: list[str] = []
    for run_dir in runs:
        found = _check_run(run_dir, write)
        mismatches.extend(found)
        typer.echo(f"{run_dir.name}: {'OK' if not found else 'MISMATCH'}")
    for line in mismatches:
        typer.echo(line)
    raise typer.Exit(code=1 if mismatches else 0)


if __name__ == "__main__":
    app()
