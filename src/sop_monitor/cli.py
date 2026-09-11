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

NOT_YET_EXIT_CODE = 2


@app.command("freeze-splits")
def freeze_splits(
    dataset: Annotated[str, typer.Option(help="industreal | ha-vid")] = "industreal",
    labels_dir: Annotated[
        Path | None, typer.Option(help="IndustReal label CSV directory (train/val/test.csv)")
    ] = None,
    out_dir: Annotated[str, typer.Option(help="Where train/val/test.csv land")] = "splits",
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Rebuild the subject-wise split and write splits/*.csv + test_sha256.txt (spec 4.1)."""
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
    typer.echo("not yet: HA-ViD freeze-splits waits for the access request (spec 3.2)")
    raise typer.Exit(code=NOT_YET_EXIT_CODE)


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
    out: Annotated[Path, typer.Option(help="Audit JSON path")] = Path(
        "reports/industreal_dev_v1/data_audit.json"
    ),
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
        write_json(manifest, build_manifest(entries))
        typer.echo(f"manifest: {len(entries)} files -> {manifest}")
    if industreal["problems"]:  # type: ignore[index]
        for problem in industreal["problems"]:  # type: ignore[index]
            typer.echo(f"problem: {problem}")
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
) -> None:
    """Cache frozen DINOv2 frame features for the labelled videos of the chosen splits."""
    from sop_monitor.features import extract_videos
    from sop_monitor.industreal import LABEL_FILES, read_labels, segments_by_video

    videos: list[Path] = []
    for name in split or ["train", "val"]:
        for video_id in sorted(segments_by_video(read_labels(root / "labels" / LABEL_FILES[name]))):
            videos.append(root / "rgb" / f"{video_id}.mp4")
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
    (out / "tables.md").write_text(tables, encoding="utf-8")
    typer.echo(tables)
    typer.echo(f"wall: {result['config']['wall_seconds']:.1f}s -> {out}")  # type: ignore[index]


def _check_run(run_dir: Path, write: bool) -> list[str]:
    """Recompute a run's metrics from its prediction table; return the list of mismatches."""
    from sop_monitor.baseline import (
        load_metrics,
        read_predictions,
        render_tables,
        score_predictions,
    )

    tables = sorted(run_dir.glob("predictions_*.csv"))
    if not tables or not (run_dir / "metrics.json").is_file():
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
            tables_path.write_text(rendered, encoding="utf-8")
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
        p for p in Path(reports_dir).glob("*") if p.is_dir() and (p / "metrics.json").is_file()
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
