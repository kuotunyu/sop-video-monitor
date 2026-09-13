"""Multi-seed summary of ``train-havid-tas`` runs: mean, sample std and range per run and metric.

The runs are separate directories of the same protocol that differ only in the seed; the summary
is written to its own directory (``seed_summary.json`` + ``tables.md``) and ``reproduce-lite``
recomputes it from the listed runs' committed ``metrics.json`` / ``config.json``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

SEED_METRICS = ("mof", "edit", "f1@10", "f1@25", "f1@50")


def summarise_seeds(run_dirs: Sequence[Path]) -> dict[str, object]:
    metrics = [json.loads((d / "metrics.json").read_text(encoding="utf-8")) for d in run_dirs]
    configs = [json.loads((d / "config.json").read_text(encoding="utf-8")) for d in run_dirs]
    seeds = [int(c["spec"]["mstcn"]["seed"]) for c in configs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds are not distinct: {seeds}")
    hands = {c["spec"]["hand"] for c in configs}
    selection = {c["spec"]["mstcn"]["selection_metric"] for c in configs}
    if len(hands) != 1 or len(selection) != 1:
        raise ValueError("the runs must share hand and selection metric")
    runs = sorted(set.intersection(*(set(m["runs"]) for m in metrics)))
    summary: dict[str, dict[str, object]] = {}
    for run in runs:
        summary[run] = {}
        for key in SEED_METRICS:
            points = [float(m["runs"][run][key]["point"]) for m in metrics]
            mean = sum(points) / len(points)
            std = (
                math.sqrt(sum((p - mean) ** 2 for p in points) / (len(points) - 1))
                if len(points) > 1
                else 0.0
            )
            summary[run][key] = {
                "mean": mean,
                "std": std,
                "min": min(points),
                "max": max(points),
                "points": points,
            }
    best_epochs = {
        name: [int(c["training"][name]["best_epoch"]) for c in configs]
        for name in configs[0]["training"]
    }
    return {
        "runs": [d.as_posix() for d in run_dirs],
        "seeds": seeds,
        "hand": hands.pop(),
        "selection_metric": selection.pop(),
        "eval_split": metrics[0].get("eval_split", "val"),
        "summary": summary,
        "best_epochs": best_epochs,
    }


def _cell(entry: Mapping[str, object]) -> str:
    return (
        f"{entry['mean']:.1f} ± {entry['std']:.1f} "  # type: ignore[str-format]
        f"({entry['min']:.1f}–{entry['max']:.1f})"  # type: ignore[str-format]
    )


def render_seed_tables(payload: Mapping[str, object]) -> str:
    summary: Mapping[str, Mapping[str, Mapping[str, object]]] = payload["summary"]  # type: ignore[assignment]
    lines = [
        f"Seed summary on `{payload['eval_split']}`, hand `{payload['hand']}`, epoch selected by "
        f"val `{payload['selection_metric']}`; seeds {payload['seeds']}; cells are mean ± sample "
        "std (min–max) of the per-seed point estimates (each seed's own subject-bootstrap CI is "
        "in its run directory).",
        "",
        "| run | MoF | Edit | F1@10 | F1@25 | F1@50 |",
        "|---|---|---|---|---|---|",
    ]
    for run in sorted(summary):
        lines.append(
            f"| {run} | " + " | ".join(_cell(summary[run][k]) for k in SEED_METRICS) + " |"
        )
    epochs: Mapping[str, Sequence[int]] = payload["best_epochs"]  # type: ignore[assignment]
    lines += ["", "| network | best epoch per seed |", "|---|---|"]
    lines += [f"| {name} | {', '.join(str(e) for e in seeds)} |" for name, seeds in epochs.items()]
    return "\n".join(lines) + "\n"


def write_seed_summary(run_dirs: Sequence[Path], out_dir: Path) -> dict[str, object]:
    payload = summarise_seeds(run_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seed_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "tables.md").write_text(render_seed_tables(payload), encoding="utf-8", newline="\n")
    return payload
