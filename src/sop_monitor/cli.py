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
    graph: Annotated[str, typer.Option(help="Path to sop/graph.json")] = "sop/graph.json",
    steps: Annotated[Path | None, typer.Option(help="Observed step sequence (CSV)")] = None,
) -> None:
    """Run precedence / duration / omission checks on an observed step sequence (W3)."""
    typer.echo(
        "not yet: check-sop lands in W3 (needs sop/graph.json and duration bounds; spec 6.4)"
    )
    raise typer.Exit(code=NOT_YET_EXIT_CODE)


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
