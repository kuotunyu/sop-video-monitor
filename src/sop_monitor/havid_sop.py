"""HA-ViD SOP layer (spec W3, first slice): step sequences, learned procedure knowledge, checks.

A *step* is one primitive-task segment of either hand (``null`` and ``w`` excluded); a segment on
one hand that overlaps a same-label segment on the other hand is one two-handed step
(:func:`step_sequence`). The plate a recording assembles is read off its label vocabulary
(:func:`plate_of`): the HA-ViD ids do not encode it and the public OWL graphs name their steps
``PT1``…, not with HR-SAT codes, so the procedure knowledge used here is *learned from the train
split* per plate, as for IndustReal:

- precedence graph: ``a -> b`` when ``a`` starts before ``b`` in every train recording of the plate
  in which both occur (:func:`sop_monitor.industreal_sop.learn_precedence`, min support);
- mandatory steps: labels present in at least ``mandatory_fraction`` of the plate's train
  recordings (only these count as omissions, because the stage-2/3 instruction variants take
  alternative routes);
- duration bounds: per-label [q_low, q_high] of the step length in seconds on train.

The checks (:func:`check_sequence`) reuse :mod:`sop_monitor.sop_graph`. The synthetic
violation table (:func:`synthetic_table`) perturbs each *val* sequence in three ways — a step moved
before one of its learned predecessors, a mandatory step deleted, a step stretched past its
duration bound — and reports how often the matching check flags exactly that step (recall) and how
often the *unperturbed* sequence is flagged (false-alarm rate). Ground-truth sequences give the
checker's own validity; predicted sequences (two per-hand runs) give the deployable number. Native
``w`` (wrong) segments are counted separately with Wilson intervals (ADR 0001 trigger B:
underpowered). Everything is recomputable from ``steps_val.csv`` / ``wrong_val.csv`` plus the
learned JSON files, without the dataset.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sop_monitor.baseline import read_predictions
from sop_monitor.havid import NULL, WRONG, TemporalSegment, load_temporal_zip
from sop_monitor.havid_tas import read_split
from sop_monitor.industreal_sop import learn_precedence
from sop_monitor.metrics.online import Completion
from sop_monitor.sop_graph import (
    DurationBounds,
    TaskGraph,
    check_durations,
    check_order,
    export_json,
    load_graph,
)

FPS = 15.0
"""HA-ViD frame rate, measured on every annotated mp4 (``reports/havid_audit.json``)."""

PLATES = ("cylinder", "gear", "general")
PLATE_SIGNATURES: dict[str, tuple[str, ...]] = {
    "cylinder": ("cb", "cc", "ck", "cs", "bs", "c1", "c2", "c3", "c4"),
    "gear": ("gl", "gs", "gw", "ft", "g1", "g2", "g3"),
    "general": (
        "bx",
        "nt",
        "ir",
        "ib",
        "lb",
        "us",
        "hq",
        "hw",
        "hd",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
    ),
}
"""Object codes (paper Figure 9) that belong to one plate; a label's substrings vote for its plate."""

KINDS = ("order", "omission", "duration")
HANDS = ("lh", "rh")


@dataclass(frozen=True)
class Step:
    label: str
    start: int
    end: int
    hands: str  # "lh", "rh" or "lh+rh"

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


def plate_of(labels: Iterable[str]) -> str:
    """The plate whose object codes dominate ``labels``; raises when no plate clearly wins."""
    labels = list(labels)
    votes = {
        plate: sum(label.count(code) for label in labels for code in codes)
        for plate, codes in PLATE_SIGNATURES.items()
    }
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    best, second = ranked[0], ranked[1]
    if best[1] == 0 or second[1] * 2 >= best[1]:
        raise ValueError(f"plate is ambiguous: {votes}")
    return best[0]


def step_sequence(lh: Sequence[TemporalSegment], rh: Sequence[TemporalSegment]) -> list[Step]:
    """Both hands' primitive tasks in start order; overlapping same-label segments become one step."""
    events = sorted(
        (
            (s.start, s.end, s.label, hand)
            for hand, segments in (("lh", lh), ("rh", rh))
            for s in segments
            if s.label not in (NULL, WRONG)
        ),
        key=lambda e: (e[0], e[1], e[3]),
    )
    steps: list[Step] = []
    for start, end, label, hand in events:
        for i in range(len(steps) - 1, -1, -1):
            other = steps[i]
            if other.end >= start and other.label == label and hand not in other.hands:
                steps[i] = Step(label, min(other.start, start), max(other.end, end), "lh+rh")
                break
        else:
            steps.append(Step(label, start, end, hand))
    return sorted(steps, key=lambda s: (s.start, s.end, s.label))


def segments_from_frames(pairs: Iterable[tuple[int, str]]) -> list[TemporalSegment]:
    """Collapse ``(frame, label)`` pairs into segments; a label change or a frame gap starts a new one."""
    segments: list[TemporalSegment] = []
    for frame, label in sorted(pairs):
        if segments and segments[-1].label == label and segments[-1].end + 1 == frame:
            segments[-1] = TemporalSegment(segments[-1].start, frame, label)
        else:
            segments.append(TemporalSegment(frame, frame, label))
    return segments


def gt_segments(
    temporal_zip: Path, recordings: Iterable[str]
) -> dict[str, dict[str, list[TemporalSegment]]]:
    """``{recording: {"lh": segments, "rh": segments}}`` at primitive-task level."""
    temporal, _ = load_temporal_zip(temporal_zip)
    return {
        rec: {hand: list(temporal[f"view0_{hand}_pt"][rec]) for hand in HANDS}
        for rec in sorted(set(recordings))
    }


def predicted_segments(
    run_dirs: Mapping[str, Path], run_name: str
) -> dict[str, dict[str, list[TemporalSegment]]]:
    """Per-hand predicted segments of one run column from two per-hand run directories."""
    per_hand: dict[str, dict[str, list[TemporalSegment]]] = {}
    for hand, run_dir in run_dirs.items():
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        labels = config["classes"]["class_id_to_label"]
        frames: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for row in read_predictions(run_dir / "predictions_val.csv"):
            frames[row.video_id].append((row.frame, labels[str(row.preds[run_name])]))
        per_hand[hand] = {rec: segments_from_frames(pairs) for rec, pairs in frames.items()}
    recordings = set(per_hand["lh"])
    if recordings != set(per_hand["rh"]):
        raise ValueError("the two per-hand runs cover different recordings")
    return {rec: {hand: per_hand[hand][rec] for hand in HANDS} for rec in sorted(recordings)}


def sequences_of(
    segments: Mapping[str, Mapping[str, Sequence[TemporalSegment]]],
) -> dict[str, list[Step]]:
    return {rec: step_sequence(hands["lh"], hands["rh"]) for rec, hands in segments.items()}


# ---- learning ---------------------------------------------------------------------------------


def learn_plate_graphs(
    sequences: Mapping[str, Sequence[Step]],
    plates: Mapping[str, str],
    min_support: int = 3,
    min_agreement: float = 1.0,
) -> dict[str, TaskGraph]:
    graphs: dict[str, TaskGraph] = {}
    for plate in PLATES:
        recs = sorted(r for r in sequences if plates[r] == plate)
        completions = [[Completion(s.start, s.label) for s in sequences[r]] for r in recs]
        graphs[plate] = learn_precedence(
            completions,
            min_support,
            source=(
                f"learned from {len(recs)} HA-ViD train recordings of the {plate} plate "
                f"(min support {min_support}, min agreement {min_agreement})"
            ),
            node=str,
            min_agreement=min_agreement,
        )
    return graphs


def mandatory_steps(
    sequences: Mapping[str, Sequence[Step]], plates: Mapping[str, str], fraction: float = 0.9
) -> dict[str, list[str]]:
    """Labels present in at least ``fraction`` of the plate's recordings."""
    out: dict[str, list[str]] = {}
    for plate in PLATES:
        recs = [r for r in sequences if plates[r] == plate]
        presence: Counter[str] = Counter()
        for r in recs:
            presence.update({s.label for s in sequences[r]})
        out[plate] = sorted(label for label, n in presence.items() if n >= fraction * len(recs))
    return out


def learn_duration_bounds(
    sequences: Mapping[str, Sequence[Step]],
    plates: Mapping[str, str],
    low_q: float = 0.05,
    high_q: float = 0.95,
    min_count: int = 5,
) -> dict[str, dict[str, DurationBounds]]:
    """Per plate and label, the [q_low, q_high] step duration in seconds (labels seen >= min_count)."""
    out: dict[str, dict[str, DurationBounds]] = {}
    for plate in PLATES:
        durations: dict[str, list[float]] = defaultdict(list)
        for r, steps in sequences.items():
            if plates[r] == plate:
                for s in steps:
                    durations[s.label].append(s.n_frames / FPS)
        out[plate] = {
            label: DurationBounds(
                float(np.quantile(values, low_q)), float(np.quantile(values, high_q))
            )
            for label, values in sorted(durations.items())
            if len(values) >= min_count
        }
    return out


def write_knowledge(
    out_dir: Path,
    graphs: Mapping[str, TaskGraph],
    bounds: Mapping[str, Mapping[str, DurationBounds]],
    mandatory: Mapping[str, Sequence[str]],
    meta: Mapping[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for plate, graph in graphs.items():
        export_json(graph, out_dir / f"learned_precedence_{plate}.json")
    (out_dir / "duration_bounds.json").write_text(
        json.dumps(
            {
                "meta": dict(meta),
                "seconds": {
                    plate: {label: [b.low, b.high] for label, b in per_plate.items()}
                    for plate, per_plate in bounds.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "mandatory_steps.json").write_text(
        json.dumps({"meta": dict(meta), "steps": dict(mandatory)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_knowledge(
    graphs_dir: Path,
) -> tuple[dict[str, TaskGraph], dict[str, dict[str, DurationBounds]], dict[str, list[str]]]:
    graphs = {
        plate: load_graph(graphs_dir / f"learned_precedence_{plate}.json") for plate in PLATES
    }
    raw = json.loads((graphs_dir / "duration_bounds.json").read_text(encoding="utf-8"))
    bounds = {
        plate: {label: DurationBounds(low, high) for label, (low, high) in per_plate.items()}
        for plate, per_plate in raw["seconds"].items()
    }
    mandatory = json.loads((graphs_dir / "mandatory_steps.json").read_text(encoding="utf-8"))
    return graphs, bounds, {plate: list(steps) for plate, steps in mandatory["steps"].items()}


# ---- checks and synthetic violations ------------------------------------------------------------


def check_sequence(
    steps: Sequence[Step],
    graph: TaskGraph,
    bounds: Mapping[str, DurationBounds],
    mandatory: Iterable[str],
) -> dict[str, object]:
    order = check_order(graph, [s.label for s in steps], level="PT")
    wanted = set(mandatory)
    return {
        "violations": order.violations,
        "omissions": [label for label in order.omissions if label in wanted],
        "unknown": order.unknown,
        "repeated": order.repeated,
        "durations": check_durations([(s.label, s.n_frames / FPS) for s in steps], bounds),
    }


def perturb_order(
    steps: Sequence[Step], graph: TaskGraph, rng: random.Random
) -> tuple[list[Step], int] | None:
    """Move one step before a learned predecessor's first occurrence; returns its new index."""
    ancestors = graph.ancestors("PT")
    labels = [s.label for s in steps]
    candidates: list[tuple[int, int]] = []
    for j, later in enumerate(labels):
        if later not in ancestors:
            continue
        for i in range(j):
            if labels[i] in ancestors[later] and labels.index(labels[i]) == i:
                candidates.append((i, j))
    if not candidates:
        return None
    i, j = rng.choice(candidates)
    moved = list(steps)
    step = moved.pop(j)
    moved.insert(i, step)
    return moved, i


def perturb_omission(
    steps: Sequence[Step], mandatory: Iterable[str], rng: random.Random
) -> tuple[list[Step], str] | None:
    """Delete one mandatory step that occurs exactly once; returns the deleted label."""
    counts = Counter(s.label for s in steps)
    candidates = sorted(
        i for i, s in enumerate(steps) if s.label in set(mandatory) and counts[s.label] == 1
    )
    if not candidates:
        return None
    i = rng.choice(candidates)
    return [s for k, s in enumerate(steps) if k != i], steps[i].label


def perturb_duration(
    steps: Sequence[Step], bounds: Mapping[str, DurationBounds], rng: random.Random
) -> tuple[list[Step], int] | None:
    """Stretch one bounded step to 1.5x its upper bound; returns its index."""
    candidates = sorted(i for i, s in enumerate(steps) if s.label in bounds)
    if not candidates:
        return None
    i = rng.choice(candidates)
    s = steps[i]
    n = round(bounds[s.label].high * FPS * 1.5) + 1
    stretched = list(steps)
    stretched[i] = Step(s.label, s.start, s.start + n - 1, s.hands)
    return stretched, i


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson 95 % interval of a proportion; ``[0, 0]`` when ``n`` is 0."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _flagged(findings: Mapping[str, object], kind: str) -> bool:
    key = {"order": "violations", "omission": "omissions", "duration": "durations"}[kind]
    return bool(findings[key])


def synthetic_table(
    sequences: Mapping[str, Sequence[Step]],
    plates: Mapping[str, str],
    graphs: Mapping[str, TaskGraph],
    bounds: Mapping[str, Mapping[str, DurationBounds]],
    mandatory: Mapping[str, Sequence[str]],
    seed: int = 0,
) -> dict[str, object]:
    """Clean-sequence findings and the detection of one synthetic violation per kind per recording."""
    rng = random.Random(seed)
    clean_rows: dict[str, dict[str, object]] = {}
    clean_flagged = {kind: 0 for kind in KINDS}
    clean_any = 0
    clean_steps = 0
    step_false_alarms = {"order": 0, "duration": 0}
    per_kind: dict[str, dict[str, object]] = {
        kind: {"applicable": 0, "detected": 0, "recordings": []} for kind in KINDS
    }
    for rec in sorted(sequences):
        steps = list(sequences[rec])
        plate = plates[rec]
        graph, plate_bounds, plate_mandatory = graphs[plate], bounds[plate], mandatory[plate]
        findings = check_sequence(steps, graph, plate_bounds, plate_mandatory)
        clean_rows[rec] = {
            "plate": plate,
            "steps": len(steps),
            "violations": len(findings["violations"]),  # type: ignore[arg-type]
            "omissions": list(findings["omissions"]),  # type: ignore[arg-type]
            "unknown": sorted(set(findings["unknown"])),  # type: ignore[arg-type]
            "durations": len(findings["durations"]),  # type: ignore[arg-type]
        }
        clean_steps += len(steps)
        step_false_alarms["order"] += len(findings["violations"])  # type: ignore[arg-type]
        step_false_alarms["duration"] += len(findings["durations"])  # type: ignore[arg-type]
        flagged_here = False
        for kind in KINDS:
            if _flagged(findings, kind):
                clean_flagged[kind] += 1
                flagged_here = True
        clean_any += int(flagged_here)
        for kind in KINDS:
            if kind == "order":
                result = perturb_order(steps, graph, rng)
            elif kind == "omission":
                result = perturb_omission(steps, plate_mandatory, rng)
            else:
                result = perturb_duration(steps, plate_bounds, rng)
            if result is None:
                continue
            perturbed, target = result
            after = check_sequence(perturbed, graph, plate_bounds, plate_mandatory)
            if kind == "order":
                detected = any(v["index"] == target for v in after["violations"])  # type: ignore[index, union-attr]
            elif kind == "omission":
                detected = target in after["omissions"]  # type: ignore[operator]
            else:
                detected = any(
                    d["index"] == target and d["kind"] == "too_long"  # type: ignore[index]
                    for d in after["durations"]  # type: ignore[union-attr]
                )
            entry = per_kind[kind]
            entry["applicable"] += 1  # type: ignore[operator]
            entry["detected"] += int(detected)  # type: ignore[operator]
            entry["recordings"].append({"recording": rec, "target": target, "detected": detected})  # type: ignore[union-attr]
    n = len(sequences)
    for kind in KINDS:
        entry = per_kind[kind]
        applicable, detected = int(entry["applicable"]), int(entry["detected"])  # type: ignore[arg-type]
        false_alarms = clean_flagged[kind]
        entry["recall"] = detected / applicable if applicable else None
        entry["recall_ci95"] = wilson(detected, applicable)
        entry["clean_flagged"] = false_alarms
        entry["false_alarm_rate"] = false_alarms / n if n else None
        entry["false_alarm_ci95"] = wilson(false_alarms, n)
        entry["precision"] = (
            detected / (detected + false_alarms) if detected + false_alarms else None
        )
        if kind in step_false_alarms:
            flagged_steps = step_false_alarms[kind]
            entry["clean_steps_flagged"] = flagged_steps
            entry["step_false_alarm_rate"] = flagged_steps / clean_steps if clean_steps else None
            entry["step_false_alarm_ci95"] = wilson(flagged_steps, clean_steps)
    return {
        "seed": seed,
        "recordings": n,
        "clean": {
            "steps": clean_steps,
            "flagged_any": clean_any,
            "flagged_by_kind": clean_flagged,
            "per_recording": clean_rows,
        },
        "synthetic": per_kind,
    }


def wrong_table(
    gt: Mapping[str, Mapping[str, Sequence[TemporalSegment]]],
    pred: Mapping[str, Mapping[str, Sequence[TemporalSegment]]],
) -> dict[str, object]:
    """Native ``w`` segments per hand and whether a predicted ``w`` segment overlaps each one."""
    n_gt = detected = n_pred = false_pred = 0
    per_recording: dict[str, dict[str, int]] = {}
    for rec in sorted(gt):
        counts = {"gt": 0, "detected": 0, "pred": 0, "false_pred": 0}
        for hand in HANDS:
            gt_w = [s for s in gt[rec][hand] if s.label == WRONG]
            pred_w = [s for s in pred.get(rec, {}).get(hand, []) if s.label == WRONG]
            counts["gt"] += len(gt_w)
            counts["pred"] += len(pred_w)
            counts["detected"] += sum(
                1 for g in gt_w if any(p.start <= g.end and p.end >= g.start for p in pred_w)
            )
            counts["false_pred"] += sum(
                1 for p in pred_w if not any(p.start <= g.end and p.end >= g.start for g in gt_w)
            )
        if counts["gt"] or counts["pred"]:
            per_recording[rec] = counts
        n_gt += counts["gt"]
        detected += counts["detected"]
        n_pred += counts["pred"]
        false_pred += counts["false_pred"]
    return {
        "gt_segments": n_gt,
        "detected": detected,
        "recall": detected / n_gt if n_gt else None,
        "recall_ci95": wilson(detected, n_gt),
        "pred_segments": n_pred,
        "false_pred_segments": false_pred,
        "precision": (n_pred - false_pred) / n_pred if n_pred else None,
        "precision_ci95": wilson(n_pred - false_pred, n_pred),
        "underpowered": n_gt < 20,
        "per_recording": per_recording,
    }


# ---- run directory ------------------------------------------------------------------------------

STEP_HEADER = ["recording", "plate", "source", "hands", "label", "start", "end"]
WRONG_HEADER = ["recording", "source", "hand", "start", "end"]


def write_steps(
    path: Path,
    sequences: Mapping[str, Mapping[str, Sequence[Step]]],
    plates: Mapping[str, str],
) -> None:
    """``sequences`` is ``{source: {recording: steps}}``."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(STEP_HEADER) + "\n")
        for source in sorted(sequences):
            for rec in sorted(sequences[source]):
                for s in sequences[source][rec]:
                    handle.write(
                        f"{rec},{plates[rec]},{source},{s.hands},{s.label},{s.start},{s.end}\n"
                    )


def read_steps(path: Path) -> tuple[dict[str, dict[str, list[Step]]], dict[str, str]]:
    sequences: dict[str, dict[str, list[Step]]] = defaultdict(lambda: defaultdict(list))
    plates: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != STEP_HEADER:
            raise ValueError(f"unexpected steps header: {reader.fieldnames}")
        for row in reader:
            plates[row["recording"]] = row["plate"]
            sequences[row["source"]][row["recording"]].append(
                Step(row["label"], int(row["start"]), int(row["end"]), row["hands"])
            )
    return {s: dict(recs) for s, recs in sequences.items()}, plates


def write_wrong(
    path: Path, segments: Mapping[str, Mapping[str, Mapping[str, Sequence[TemporalSegment]]]]
) -> None:
    """``segments`` is ``{source: {recording: {hand: segments}}}``; only ``w`` segments are kept."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(",".join(WRONG_HEADER) + "\n")
        for source in sorted(segments):
            for rec in sorted(segments[source]):
                for hand in HANDS:
                    for s in segments[source][rec][hand]:
                        if s.label == WRONG:
                            handle.write(f"{rec},{source},{hand},{s.start},{s.end}\n")


def read_wrong(
    path: Path, recordings: Iterable[str]
) -> dict[str, dict[str, dict[str, list[TemporalSegment]]]]:
    out: dict[str, dict[str, dict[str, list[TemporalSegment]]]] = {
        source: {rec: {hand: [] for hand in HANDS} for rec in recordings}
        for source in ("gt", "pred")
    }
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != WRONG_HEADER:
            raise ValueError(f"unexpected wrong header: {reader.fieldnames}")
        for row in reader:
            out[row["source"]].setdefault(row["recording"], {hand: [] for hand in HANDS})[
                row["hand"]
            ].append(TemporalSegment(int(row["start"]), int(row["end"]), WRONG))
    return out


def evaluate_run(run_dir: Path, graphs_dir: Path, seed: int) -> dict[str, object]:
    """Recompute the recomputable part of ``sop_checks.json`` from the committed CSVs."""
    graphs, bounds, mandatory = load_knowledge(graphs_dir)
    sequences, plates = read_steps(run_dir / "steps_val.csv")
    wrong = read_wrong(run_dir / "wrong_val.csv", sequences["gt"])
    return {
        "graphs": {
            plate: {
                "nodes": len(graphs[plate].nodes_of("PT")),
                "edges": len(graphs[plate].relation("precedesPT")),
                "mandatory": len(mandatory[plate]),
                "bounded": len(bounds[plate]),
            }
            for plate in PLATES
        },
        "sources": {
            source: synthetic_table(sequences[source], plates, graphs, bounds, mandatory, seed)
            for source in sorted(sequences)
        },
        "wrong": wrong_table(wrong["gt"], wrong["pred"]),
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def _ci(interval: Sequence[float]) -> str:
    return f"[{100 * interval[0]:.0f}, {100 * interval[1]:.0f}]"


def render_sop_tables(payload: Mapping[str, object]) -> str:
    graphs: Mapping[str, Mapping[str, int]] = payload["graphs"]  # type: ignore[assignment]
    sources: Mapping[str, Mapping[str, object]] = payload["sources"]  # type: ignore[assignment]
    lines = [
        "Learned procedure knowledge (train split):",
        "",
        "| plate | steps (graph nodes) | precedence edges | mandatory steps | steps with duration bounds |",
        "|---|---|---|---|---|",
    ]
    for plate in PLATES:
        g = graphs[plate]
        lines.append(
            f"| {plate} | {g['nodes']} | {g['edges']} | {g['mandatory']} | {g['bounded']} |"
        )
    for source, table in sources.items():
        clean: Mapping[str, object] = table["clean"]  # type: ignore[assignment]
        synthetic: Mapping[str, Mapping[str, object]] = table["synthetic"]  # type: ignore[assignment]
        n = table["recordings"]
        lines += [
            "",
            f"Synthetic violations on `{source}` sequences ({n} val recordings, seed {table['seed']}); "
            f"recall = perturbed step flagged by the matching check, false alarm = unperturbed "
            f"recording flagged by that check; Wilson 95 % intervals in percent:",
            "",
            "| violation | applicable | detected | recall | clean flagged | false-alarm rate | precision | per-step false alarms |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for kind in KINDS:
            e = synthetic[kind]
            if "clean_steps_flagged" in e:
                per_step = (
                    f"{e['clean_steps_flagged']} / {clean.get('steps', '?')} = "
                    f"{_pct(e['step_false_alarm_rate'])} {_ci(e['step_false_alarm_ci95'])}"  # type: ignore[arg-type]
                )
            else:
                per_step = "—"
            lines.append(
                f"| {kind} | {e['applicable']} | {e['detected']} | {_pct(e['recall'])} {_ci(e['recall_ci95'])} "  # type: ignore[arg-type]
                f"| {e['clean_flagged']} / {n} | {_pct(e['false_alarm_rate'])} {_ci(e['false_alarm_ci95'])} "  # type: ignore[arg-type]
                f"| {_pct(e['precision'])} | {per_step} |"  # type: ignore[arg-type]
            )
        lines.append(
            f"\nUnperturbed `{source}` recordings with any finding: {clean['flagged_any']} / {n}."
        )
    wrong: Mapping[str, object] = payload["wrong"]  # type: ignore[assignment]
    lines += [
        "",
        "Native `w` (wrong) segments on val, both hands, predicted `w` overlapping a ground-truth `w`"
        f" counts as detected{' — underpowered (fewer than 20 segments)' if wrong['underpowered'] else ''}:",
        "",
        "| ground-truth `w` segments | detected | recall | predicted `w` segments | false predicted | precision |",
        "|---|---|---|---|---|---|",
        f"| {wrong['gt_segments']} | {wrong['detected']} | {_pct(wrong['recall'])} {_ci(wrong['recall_ci95'])} "  # type: ignore[arg-type]
        f"| {wrong['pred_segments']} | {wrong['false_pred_segments']} | {_pct(wrong['precision'])} {_ci(wrong['precision_ci95'])} |",  # type: ignore[arg-type]
    ]
    return "\n".join(lines) + "\n"


def run_havid_sop(
    temporal_zip: Path,
    split_dir: Path,
    graphs_dir: Path,
    run_dirs: Mapping[str, Path],
    run_name: str,
    out_dir: Path,
    seed: int = 0,
) -> dict[str, object]:
    """Build the val step sequences (ground truth and predicted), evaluate, write the run directory."""
    val = sorted({row["recording"] for row in read_split(split_dir / "val.csv")})
    gt = gt_segments(temporal_zip, val)
    pred = predicted_segments(run_dirs, run_name)
    if set(pred) != set(gt):
        raise ValueError("predicted recordings differ from the val split")
    gt_sequences = sequences_of(gt)
    plates = {rec: plate_of(s.label for s in steps) for rec, steps in gt_sequences.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_steps(out_dir / "steps_val.csv", {"gt": gt_sequences, "pred": sequences_of(pred)}, plates)
    write_wrong(out_dir / "wrong_val.csv", {"gt": gt, "pred": pred})
    payload = evaluate_run(out_dir, graphs_dir, seed)
    payload["config"] = {
        "graphs_dir": graphs_dir.as_posix(),
        "prediction_runs": {hand: p.as_posix() for hand, p in run_dirs.items()},
        "run": run_name,
        "seed": seed,
        "fps": FPS,
        "val_recordings": len(val),
    }
    (out_dir / "sop_checks.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tables = render_sop_tables(payload)
    (out_dir / "tables.md").write_text(tables, encoding="utf-8", newline="\n")
    return payload


def tune_knowledge(
    sequences: Mapping[str, Sequence[Step]],
    plates: Mapping[str, str],
    subjects: Mapping[str, str],
    supports: Sequence[int] = (3, 5, 10, 20),
    agreements: Sequence[float] = (1.0, 0.95, 0.9, 0.8),
    quantiles: Sequence[tuple[float, float]] = ((0.05, 0.95), (0.02, 0.98), (0.01, 0.99)),
    min_count: int = 5,
) -> dict[str, object]:
    """Leave-one-subject-out over the train split to choose the graph rule and the duration window.

    For every (min support, min agreement) the order check's recording-level and step-level
    false-alarm rates on the held-out subject's *unperturbed* sequences and its coverage (fraction
    of held-out recordings on which a synthetic swap is possible at all); for every quantile pair
    the duration check's step-level false-alarm rate. Selection rule, fixed before val is looked
    at: order = the setting maximising coverage minus recording false-alarm rate (ties: smaller
    support); duration = the narrowest window whose step false-alarm rate is at most 5 %.
    """
    held = sorted(set(subjects.values()))

    def split(subject: str) -> tuple[dict[str, Sequence[Step]], dict[str, Sequence[Step]]]:
        train = {r: s for r, s in sequences.items() if subjects[r] != subject}
        test = {r: s for r, s in sequences.items() if subjects[r] == subject}
        return train, test

    order_grid: list[dict[str, object]] = []
    for min_support in supports:
        for min_agreement in agreements:
            n_rec = flagged_rec = n_steps = flagged_steps = applicable = 0
            for subject in held:
                train, test = split(subject)
                graphs = learn_plate_graphs(train, plates, min_support, min_agreement)
                for rec, steps in test.items():
                    graph = graphs[plates[rec]]
                    report = check_order(graph, [s.label for s in steps], level="PT")
                    n_rec += 1
                    flagged_rec += int(bool(report.violations))
                    n_steps += len(steps)
                    flagged_steps += len(report.violations)
                    applicable += int(perturb_order(steps, graph, random.Random(0)) is not None)
            full = learn_plate_graphs(sequences, plates, min_support, min_agreement)
            order_grid.append(
                {
                    "min_support": min_support,
                    "min_agreement": min_agreement,
                    "edges_full_train": sum(len(g.relation("precedesPT")) for g in full.values()),
                    "recordings": n_rec,
                    "recording_false_alarm_rate": flagged_rec / n_rec,
                    "step_false_alarm_rate": flagged_steps / n_steps,
                    "coverage": applicable / n_rec,
                    "score": applicable / n_rec - flagged_rec / n_rec,
                }
            )
    duration_grid: list[dict[str, object]] = []
    for low_q, high_q in quantiles:
        n_steps = flagged = 0
        for subject in held:
            train, test = split(subject)
            bounds = learn_duration_bounds(train, plates, low_q, high_q, min_count)
            for rec, steps in test.items():
                findings = check_durations(
                    [(s.label, s.n_frames / FPS) for s in steps], bounds[plates[rec]]
                )
                n_steps += len(steps)
                flagged += len(findings)
        duration_grid.append(
            {
                "low_q": low_q,
                "high_q": high_q,
                "steps": n_steps,
                "step_false_alarm_rate": flagged / n_steps,
            }
        )
    best_order = max(order_grid, key=lambda e: (e["score"], -int(e["min_support"])))  # type: ignore[arg-type, operator]
    acceptable = [e for e in duration_grid if e["step_false_alarm_rate"] <= 0.05]  # type: ignore[operator]
    best_duration = (
        min(acceptable, key=lambda e: float(e["high_q"]) - float(e["low_q"]))  # type: ignore[arg-type]
        if acceptable
        else duration_grid[-1]
    )
    return {
        "protocol": "leave-one-subject-out on the train split; unperturbed held-out sequences",
        "subjects": len(held),
        "order": order_grid,
        "duration": duration_grid,
        "selected": {
            "min_support": best_order["min_support"],
            "min_agreement": best_order["min_agreement"],
            "low_q": best_duration["low_q"],
            "high_q": best_duration["high_q"],
        },
    }


def tune_havid_sop(temporal_zip: Path, split_dir: Path, out_path: Path) -> dict[str, object]:
    """Run :func:`tune_knowledge` on the train split and write ``out_path`` (JSON)."""
    from sop_monitor.havid import parse_recording_id

    train = sorted({row["recording"] for row in read_split(split_dir / "train.csv")})
    sequences = sequences_of(gt_segments(temporal_zip, train))
    plates = {rec: plate_of(s.label for s in steps) for rec, steps in sequences.items()}
    subjects = {rec: parse_recording_id(rec).subject for rec in sequences}
    payload = tune_knowledge(sequences, plates, subjects)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def learn_havid_sop(
    temporal_zip: Path,
    split_dir: Path,
    out_dir: Path,
    min_support: int = 3,
    mandatory_fraction: float = 0.9,
    low_q: float = 0.05,
    high_q: float = 0.95,
    min_count: int = 5,
    min_agreement: float = 1.0,
) -> dict[str, object]:
    """Learn graphs, mandatory steps and duration bounds from the train split; write them."""
    train = sorted({row["recording"] for row in read_split(split_dir / "train.csv")})
    sequences = sequences_of(gt_segments(temporal_zip, train))
    plates = {rec: plate_of(s.label for s in steps) for rec, steps in sequences.items()}
    graphs = learn_plate_graphs(sequences, plates, min_support, min_agreement)
    mandatory = mandatory_steps(sequences, plates, mandatory_fraction)
    bounds = learn_duration_bounds(sequences, plates, low_q, high_q, min_count)
    meta = {
        "source": "HA-ViD train split (splits/ha-vid/train.csv), primitive tasks of both hands",
        "recordings": len(train),
        "min_support": min_support,
        "min_agreement": min_agreement,
        "mandatory_fraction": mandatory_fraction,
        "duration_quantiles": [low_q, high_q],
        "duration_min_count": min_count,
        "fps": FPS,
        "licence": "derived from HA-ViD annotations, CC BY-NC 4.0",
    }
    write_knowledge(out_dir, graphs, bounds, mandatory, meta)
    return {
        "recordings": len(train),
        "plates": dict(Counter(plates.values())),
        "graphs": {
            plate: {
                "nodes": len(graphs[plate].nodes_of("PT")),
                "edges": len(graphs[plate].relation("precedesPT")),
                "mandatory": len(mandatory[plate]),
                "bounded": len(bounds[plate]),
            }
            for plate in PLATES
        },
    }
