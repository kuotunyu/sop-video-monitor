"""``sop-monitor`` command line (typer). W0: placeholders naming the milestone that owns each command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Multi-camera SOP sequence monitoring (W0 skeleton).",
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
    import json

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


@app.command("reproduce-lite")
def reproduce_lite(
    reports_dir: Annotated[
        str, typer.Option(help="Directory of committed reports/*.json")
    ] = "reports",
) -> None:
    """Rebuild tables and plots from committed reports/*.json without data (spec 8.3; CI path)."""
    committed = sorted(Path(reports_dir).glob("*.json"))
    if not committed:
        typer.echo(
            f"not yet: no committed JSON under {reports_dir}/ (first reports arrive in W2); "
            "nothing to rebuild"
        )
        raise typer.Exit(code=0)
    typer.echo(
        f"not yet: table/plot regeneration lands in W2 ({len(committed)} JSON file(s) found)"
    )
    raise typer.Exit(code=NOT_YET_EXIT_CODE)


if __name__ == "__main__":
    app()
