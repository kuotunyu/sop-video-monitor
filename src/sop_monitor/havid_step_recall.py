"""Per-step frame recall of HA-ViD recogniser runs at instruction-sheet granularity.

The SOP checks consume steps, so a recogniser's frame metrics (MoF, F1@k) hide what matters for
them: whether each *step* of the instruction sheet is recognised at all. For every group of run
directories (for example both hands of three seeds at one look-ahead) and one prediction column,
this pools the val frames of each ground-truth sheet step and counts the frames whose predicted
label maps to the same sheet step. Steps are annotated with the plates whose learned knowledge
contains them and whether they are mandatory there. Everything is recomputed from the committed
``predictions_val.csv`` and ``config.json`` files plus the knowledge directory.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sop_monitor.baseline import read_predictions
from sop_monitor.havid import NULL, WRONG
from sop_monitor.havid_sop import PLATES, load_knowledge, sheet_step


@dataclass(frozen=True)
class RunGroup:
    name: str
    column: str
    run_dirs: tuple[Path, ...]

    @classmethod
    def parse(cls, spec: str) -> RunGroup:
        """``NAME=COLUMN@DIR[,DIR...]``."""
        name, sep, rest = spec.partition("=")
        column, at, dirs = rest.partition("@")
        paths = tuple(Path(d) for d in dirs.split(",") if d)
        if not sep or not at or not name or not column or not paths:
            raise ValueError(f"group must be NAME=COLUMN@DIR[,DIR...]: {spec!r}")
        return cls(name, column, paths)


def pooled_counts(run_dirs: Sequence[Path], column: str) -> tuple[Counter, Counter]:
    """``(gt frames, correct frames)`` per sheet step over every run directory."""
    total: Counter = Counter()
    correct: Counter = Counter()
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        labels = config["classes"]["class_id_to_label"]
        for row in read_predictions(run_dir / "predictions_val.csv"):
            if column not in row.preds:
                raise ValueError(f"{run_dir}: no prediction column {column!r}")
            truth = labels.get(str(row.gt))
            if truth is None or truth in (NULL, WRONG):
                continue
            step = sheet_step(truth)
            total[step] += 1
            predicted = labels.get(str(row.preds[column]))
            correct[step] += int(predicted is not None and sheet_step(predicted) == step)
    return total, correct


def step_recall(groups: Sequence[RunGroup], graphs_dir: Path) -> dict[str, object]:
    graphs, _bounds, mandatory = load_knowledge(graphs_dir)
    plates_of: dict[str, list[str]] = {}
    for plate in PLATES:
        for node in graphs[plate].nodes_of("PT"):
            plates_of.setdefault(node, []).append(plate)
    counts = {group.name: pooled_counts(group.run_dirs, group.column) for group in groups}
    per_dir = {
        g.name: {step: n / len(g.run_dirs) for step, n in counts[g.name][0].items()} for g in groups
    }
    reference = per_dir[groups[0].name]
    for name, frames in per_dir.items():
        if frames != reference:
            raise ValueError(f"group {name} covers different ground-truth frames per run directory")
    steps = {}
    for step in sorted(reference):
        steps[step] = {
            "plates": sorted(plates_of.get(step, [])),
            "mandatory_in": sorted(p for p in PLATES if step in mandatory[p]),
            "gt_frames_per_run_dir": reference[step],
            "recall": {
                name: correct[step] / total[step] for name, (total, correct) in counts.items()
            },
        }
    overall = {
        name: sum(correct.values()) / sum(total.values())
        for name, (total, correct) in counts.items()
    }
    mandatory_frames = {
        name: sum(correct[s] for s in total if steps[s]["mandatory_in"])
        / max(1, sum(total[s] for s in total if steps[s]["mandatory_in"]))
        for name, (total, correct) in counts.items()
    }
    return {
        "config": {
            "graphs_dir": graphs_dir.as_posix(),
            "groups": [
                {"name": g.name, "column": g.column, "run_dirs": [d.as_posix() for d in g.run_dirs]}
                for g in groups
            ],
        },
        "steps": steps,
        "overall_step_frame_recall": overall,
        "mandatory_step_frame_recall": mandatory_frames,
    }


def render_step_recall(payload: Mapping[str, object]) -> str:
    config: Mapping[str, object] = payload["config"]  # type: ignore[assignment]
    names = [g["name"] for g in config["groups"]]  # type: ignore[index, union-attr]
    steps: Mapping[str, Mapping[str, object]] = payload["steps"]  # type: ignore[assignment]
    lines = [
        "Val frame recall per instruction-sheet step (frames of the step whose predicted label maps "
        "to the same step), pooled over each group's run directories; `null` and `w` frames "
        f"excluded. Knowledge: `{config['graphs_dir']}`.",
        "",
        "| step | plates | mandatory in | gt frames per run | " + " | ".join(names) + " |",
        "|---|---|---|---|" + "---|" * len(names),
    ]
    order = sorted(
        steps,
        key=lambda s: (not steps[s]["mandatory_in"], ",".join(steps[s]["plates"]), s),  # type: ignore[arg-type]
    )
    for step in order:
        entry = steps[step]
        recall: Mapping[str, float] = entry["recall"]  # type: ignore[assignment]
        lines.append(
            f"| {step} | {', '.join(entry['plates']) or '—'} "  # type: ignore[arg-type]
            f"| {', '.join(entry['mandatory_in']) or '—'} "  # type: ignore[arg-type]
            f"| {entry['gt_frames_per_run_dir']:.0f} | "
            + " | ".join(f"{100 * recall[n]:.0f}" for n in names)
            + " |"
        )
    overall: Mapping[str, float] = payload["overall_step_frame_recall"]  # type: ignore[assignment]
    mandatory: Mapping[str, float] = payload["mandatory_step_frame_recall"]  # type: ignore[assignment]
    lines += [
        "| **all step frames** | | | | "
        + " | ".join(f"**{100 * overall[n]:.1f}**" for n in names)
        + " |",
        "| **mandatory step frames** | | | | "
        + " | ".join(f"**{100 * mandatory[n]:.1f}**" for n in names)
        + " |",
    ]
    return "\n".join(lines) + "\n"


def write_step_recall(out_dir: Path, payload: Mapping[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "step_recall.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "tables.md").write_text(render_step_recall(payload), encoding="utf-8", newline="\n")
