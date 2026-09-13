"""Online replay of the HA-ViD val recordings through :class:`sop_monitor.online.OnlineSOPMonitor`.

Every recording's two per-hand label streams (ground truth, or the per-frame output of causal
recogniser runs) are pushed frame by frame into a fresh monitor loaded with the plate's learned
knowledge, once per confirmation length ``min_frames`` of a grid. Every deviation is written with
the frame it was detected at (``deviations_val.csv``); ``recordings_val.csv`` holds each
recording's length, plate and whether its annotation contains ``w``. The summary
(:func:`summarise_online`) is recomputable from these two files alone:

- recordings flagged per kind, split into recordings with and without a native ``w`` (the online
  counterpart of the recording-level table of the offline SOP runs);
- alarms per recording;
- when the alarms arrive: the time into the recording of the first alarm, the lead time before
  the recording ends (order, duration and unknown alarms; omissions are only knowable at the end),
  and the latency from a step's confirmed start to its order alarm.

A prediction source carries ``output_delay_frames``: a run trained with look-ahead ``L`` emits the
label of frame ``t`` at frame ``t + L`` (``havid_tas.advance_outputs``), so every alarm time adds
``L``. At ``min_frames = 1`` the monitor's findings are compared per recording with the offline
:func:`sop_monitor.havid_sop.check_sequence` (``offline_agreement``); the known exception is two
adjacent segments with the same label, which a frame stream cannot separate.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sop_monitor.havid import WRONG, TemporalSegment
from sop_monitor.havid_sop import (
    FPS,
    HANDS,
    at_granularity,
    check_sequence,
    gt_segments,
    load_knowledge,
    plate_of,
    predicted_segments,
    step_sequence,
)
from sop_monitor.havid_tas import read_split
from sop_monitor.online import Deviation, OnlineSOPMonitor
from sop_monitor.sop_graph import DurationBounds, TaskGraph

KINDS = ("order", "omission", "duration_too_long", "duration_too_short", "unknown")
FLAG_KINDS = ("order", "omission", "duration")  # "any" = one of these, as in the offline tables
DEVIATION_HEADER = [
    "source",
    "min_frames",
    "recording",
    "kind",
    "step",
    "start_frame",
    "end_frame",
    "detected_at",
]
RECORDING_HEADER = ["recording", "plate", "n_frames", "with_wrong"]


@dataclass(frozen=True)
class PredictionSource:
    """Two per-hand TAS run directories, the prediction column, and the output delay in frames."""

    name: str
    lh: Path
    rh: Path
    run_name: str = "fusion_causal"
    output_delay_frames: int = 0

    @classmethod
    def parse(cls, spec: str) -> PredictionSource:
        """``NAME=LH_DIR,RH_DIR,RUN_NAME,DELAY_FRAMES``."""
        name, sep, rest = spec.partition("=")
        parts = rest.split(",")
        if not sep or not name or len(parts) != 4:
            raise ValueError(f"source must be NAME=LH_DIR,RH_DIR,RUN_NAME,DELAY_FRAMES: {spec!r}")
        if name == "gt":
            raise ValueError("the source name gt is reserved for the annotations")
        return cls(name, Path(parts[0]), Path(parts[1]), parts[2], int(parts[3]))


def kind_of(deviation: Deviation) -> str:
    """``order``, ``omission``, ``unknown``, ``duration_too_long`` or ``duration_too_short``."""
    if deviation.kind == "duration":
        return f"duration_{deviation.evidence['kind']}"
    return deviation.kind


def frame_labels(segments: Sequence[TemporalSegment]) -> list[str]:
    """Per-frame labels of contiguous segments starting at frame 0."""
    labels: list[str] = []
    for segment in segments:
        if segment.start != len(labels):
            raise ValueError(f"segments are not contiguous from frame 0 at {segment}")
        labels.extend([segment.label] * segment.n_frames)
    return labels


def replay(
    hands: Mapping[str, Sequence[TemporalSegment]], monitor: OnlineSOPMonitor
) -> list[Deviation]:
    """Push both hands' streams frame by frame into ``monitor``; return every deviation."""
    streams = {hand: frame_labels(hands[hand]) for hand in HANDS}
    if len(streams["lh"]) != len(streams["rh"]):
        raise ValueError("the two hands' streams differ in length")
    found: list[Deviation] = []
    for frame in range(len(streams["lh"])):
        found.extend(monitor.push(frame, {hand: streams[hand][frame] for hand in HANDS}))
    found.extend(monitor.finish())
    return found


def offline_findings(
    hands: Mapping[str, Sequence[TemporalSegment]],
    graph: TaskGraph,
    bounds: Mapping[str, DurationBounds],
    mandatory: Iterable[str],
    granularity: str,
) -> Counter:
    """The offline checker's findings as ``(kind, step)`` counts, in the online monitor's terms."""
    steps = at_granularity(step_sequence(hands["lh"], hands["rh"]), granularity)
    report = check_sequence(steps, graph, bounds, mandatory)
    found: Counter = Counter()
    found.update(("order", v["step"]) for v in report["violations"])  # type: ignore[union-attr]
    found.update(("omission", label) for label in report["omissions"])  # type: ignore[union-attr]
    found.update(("duration_" + d["kind"], d["step"]) for d in report["durations"])  # type: ignore[union-attr]
    found.update(("unknown", label) for label in set(report["unknown"]))  # type: ignore[arg-type]
    return found


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _flag_kind(kind: str) -> str | None:
    """The recording-level column a deviation kind counts towards (``None`` for unknown)."""
    if kind.startswith("duration_"):
        return "duration"
    return kind if kind in FLAG_KINDS else None


def summarise_online(
    deviations: Sequence[Mapping[str, str]],
    recordings: Sequence[Mapping[str, str]],
    delays: Mapping[str, int],
    grid: Iterable[tuple[str, int]],
) -> dict[str, object]:
    """Recording-level counts and alarm timing per source and ``min_frames``."""
    meta = {row["recording"]: row for row in recordings}
    groups = {
        "with_wrong": sorted(r for r, row in meta.items() if row["with_wrong"] == "1"),
        "without_wrong": sorted(r for r, row in meta.items() if row["with_wrong"] != "1"),
    }
    out: dict[str, object] = {}
    for source, min_frames in grid:
        delay = delays[source]
        rows = [
            r for r in deviations if r["source"] == source and int(r["min_frames"]) == min_frames
        ]
        kinds_of: dict[str, set[str]] = {rec: set() for rec in meta}
        first_frame: dict[str, int] = {}
        for row in rows:
            flag = _flag_kind(row["kind"])
            if flag is None:
                kinds_of[row["recording"]].add("unknown")
                continue
            kinds_of[row["recording"]].update((flag, "any"))
            detected = int(row["detected_at"])
            first_frame[row["recording"]] = min(
                first_frame.get(row["recording"], detected), detected
            )
        recording_level = {
            name: {"recordings": len(recs)}
            | {
                kind: sum(1 for rec in recs if kind in kinds_of[rec])
                for kind in (*FLAG_KINDS, "unknown", "any")
            }
            for name, recs in groups.items()
        }
        lead = [
            (int(meta[r["recording"]]["n_frames"]) - 1 - int(r["detected_at"]) - delay) / FPS
            for r in rows
            if r["kind"] != "omission"
        ]
        order_latency = [
            (int(r["detected_at"]) - int(r["start_frame"]) + delay) / FPS
            for r in rows
            if r["kind"] == "order"
        ]
        out[f"{source}@{min_frames}"] = {
            "source": source,
            "min_frames": min_frames,
            "output_delay_frames": delay,
            "recording_level": recording_level,
            "alarms_per_recording": {
                kind: sum(1 for r in rows if r["kind"] == kind) / len(meta) for kind in KINDS
            },
            "first_alarm_s_median": _median([(f + delay) / FPS for f in first_frame.values()]),
            "lead_before_end_s_median": _median(lead),
            "order_latency_s_median": _median(order_latency),
        }
    return out


def write_rows(path: Path, header: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary_grid(config: Mapping[str, object]) -> list[tuple[str, int]]:
    delays: Mapping[str, int] = config["delays"]  # type: ignore[assignment]
    return [(source, m) for source in delays for m in config["min_frames_grid"]]  # type: ignore[union-attr]


def _fmt(value: object, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def seed_groups(summary: Mapping[str, Mapping[str, object]]) -> dict[tuple[str, int], list[str]]:
    """Summary keys grouped by source name without its ``_s<seed>`` suffix and ``min_frames``."""
    groups: dict[tuple[str, int], list[str]] = {}
    for key, entry in summary.items():
        group = re.sub(r"_s\d+$", "", str(entry["source"]))
        groups.setdefault((group, int(entry["min_frames"])), []).append(key)  # type: ignore[call-overload]
    return groups


def _mean_std(values: Sequence[object]) -> str:
    numbers = [float(v) for v in values if v is not None]  # type: ignore[arg-type]
    if not numbers:
        return "—"
    if len(numbers) == 1:
        return f"{numbers[0]:.1f}"
    return f"{statistics.mean(numbers):.1f} ± {statistics.stdev(numbers):.1f}"


def render_online_tables(payload: Mapping[str, object]) -> str:
    config: Mapping[str, object] = payload["config"]  # type: ignore[assignment]
    summary: Mapping[str, Mapping[str, object]] = payload["summary"]  # type: ignore[assignment]
    lines = [
        f"Online SOP replay of {config['val_recordings']} val recordings at granularity "
        f"`{config['granularity']}` with the knowledge in `{config['graphs_dir']}`. Recording "
        "columns count flagged recordings as with `w` · without `w`; times include the source's "
        "output delay.",
        "",
        "| source | delay frames | min_frames | order | omission | duration | any "
        "| alarms / recording | first alarm s (median) | lead before end s (median) "
        "| order latency s (median) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in summary.values():
        level: Mapping[str, Mapping[str, int]] = entry["recording_level"]  # type: ignore[assignment]
        w, c = level["with_wrong"], level["without_wrong"]
        cells = " | ".join(
            f"{w[k]}/{w['recordings']} · {c[k]}/{c['recordings']}" for k in (*FLAG_KINDS, "any")
        )
        alarms: Mapping[str, float] = entry["alarms_per_recording"]  # type: ignore[assignment]
        lines.append(
            f"| {entry['source']} | {entry['output_delay_frames']} | {entry['min_frames']} "
            f"| {cells} | {sum(alarms.values()):.1f} | {_fmt(entry['first_alarm_s_median'])} "
            f"| {_fmt(entry['lead_before_end_s_median'])} "
            f"| {_fmt(entry['order_latency_s_median'], 2)} |"
        )
    groups = seed_groups(summary)
    if any(len(members) > 1 for members in groups.values()):
        lines += [
            "",
            "Mean ± std over the seeds of each source (`<name>_s<seed>`).",
            "",
            "| source | seeds | delay frames | min_frames | any, with `w` | any, without `w` "
            "| alarms / recording | first alarm s (median) | lead before end s (median) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for (group, min_frames), members in groups.items():
            entries = [summary[key] for key in members]
            level_of = [e["recording_level"] for e in entries]
            lines.append(
                f"| {group} | {len(entries)} | {entries[0]['output_delay_frames']} | {min_frames} "
                f"| {_mean_std([lv['with_wrong']['any'] for lv in level_of])} "  # type: ignore[index]
                f"/ {level_of[0]['with_wrong']['recordings']} "  # type: ignore[index]
                f"| {_mean_std([lv['without_wrong']['any'] for lv in level_of])} "  # type: ignore[index]
                f"/ {level_of[0]['without_wrong']['recordings']} "  # type: ignore[index]
                f"| {_mean_std([sum(e['alarms_per_recording'].values()) for e in entries])} "  # type: ignore[union-attr]
                f"| {_mean_std([e['first_alarm_s_median'] for e in entries])} "
                f"| {_mean_std([e['lead_before_end_s_median'] for e in entries])} |"
            )
    agreement: Mapping[str, Mapping[str, int]] = payload["offline_agreement"]  # type: ignore[assignment]
    lines += [
        "",
        "Offline agreement at min_frames 1: recordings whose online findings equal "
        "`check_sequence` on the same segments.",
        "",
        "| source | equal | recordings |",
        "|---|---|---|",
    ]
    lines += [f"| {s} | {e['equal']} | {e['recordings']} |" for s, e in agreement.items()]
    return "\n".join(lines) + "\n"


def run_havid_online(
    temporal_zip: Path,
    split_dir: Path,
    graphs_dir: Path,
    sources: Sequence[PredictionSource],
    min_frames_grid: Sequence[int],
    out_dir: Path,
    granularity: str = "sheet",
) -> dict[str, object]:
    """Replay the val recordings of every source through the online monitor; write the run directory."""
    knowledge_meta = json.loads((graphs_dir / "mandatory_steps.json").read_text(encoding="utf-8"))
    learned_at = knowledge_meta.get("meta", {}).get("granularity", "pt")
    if learned_at != granularity:
        raise ValueError(
            f"{graphs_dir} was learned at granularity {learned_at!r}, not {granularity!r}"
        )
    grid = list(min_frames_grid)
    if not grid or grid != sorted(set(grid)) or grid[0] < 1:
        raise ValueError("min_frames grid must be strictly increasing and start at >= 1")
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"source names must be unique: {names}")
    graphs, bounds, mandatory = load_knowledge(graphs_dir)
    val = sorted({row["recording"] for row in read_split(split_dir / "val.csv")})
    streams = {"gt": gt_segments(temporal_zip, val)}
    for source in sources:
        streams[source.name] = predicted_segments(
            {"lh": source.lh, "rh": source.rh}, source.run_name
        )
        if set(streams[source.name]) != set(val):
            raise ValueError(f"{source.name}: predicted recordings differ from the val split")
    plates = {
        rec: plate_of(s.label for s in step_sequence(hands["lh"], hands["rh"]))
        for rec, hands in streams["gt"].items()
    }
    recordings = [
        {
            "recording": rec,
            "plate": plates[rec],
            "n_frames": len(frame_labels(streams["gt"][rec]["lh"])),
            "with_wrong": int(any(s.label == WRONG for h in HANDS for s in streams["gt"][rec][h])),
        }
        for rec in val
    ]
    deviations: list[dict[str, object]] = []
    agreement: dict[str, dict[str, int]] = {}
    for name, per_rec in streams.items():
        agreement[name] = {"equal": 0, "recordings": len(val)}
        for rec in val:
            plate = plates[rec]
            for min_frames in grid:
                monitor = OnlineSOPMonitor(
                    graphs[plate], bounds[plate], mandatory[plate], granularity, min_frames
                )
                found = replay(per_rec[rec], monitor)
                deviations.extend(
                    {
                        "source": name,
                        "min_frames": min_frames,
                        "recording": rec,
                        "kind": kind_of(d),
                        "step": d.step,
                        "start_frame": "" if d.start_frame is None else d.start_frame,
                        "end_frame": "" if d.end_frame is None else d.end_frame,
                        "detected_at": d.detected_at,
                    }
                    for d in found
                )
                if min_frames == 1:
                    online = Counter((kind_of(d), d.step) for d in found)
                    offline = offline_findings(
                        per_rec[rec], graphs[plate], bounds[plate], mandatory[plate], granularity
                    )
                    agreement[name]["equal"] += int(online == offline)
    if grid[0] != 1:
        agreement = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "recordings_val.csv", RECORDING_HEADER, recordings)
    write_rows(out_dir / "deviations_val.csv", DEVIATION_HEADER, deviations)
    config: dict[str, object] = {
        "graphs_dir": graphs_dir.as_posix(),
        "granularity": granularity,
        "min_frames_grid": grid,
        "fps": FPS,
        "val_recordings": len(val),
        "sources": {
            s.name: {
                "lh": s.lh.as_posix(),
                "rh": s.rh.as_posix(),
                "run": s.run_name,
                "output_delay_frames": s.output_delay_frames,
            }
            for s in sources
        },
        "delays": {"gt": 0} | {s.name: s.output_delay_frames for s in sources},
    }
    payload: dict[str, object] = {
        "config": config,
        "offline_agreement": agreement,
        "summary": summarise_online(
            read_rows(out_dir / "deviations_val.csv"),
            read_rows(out_dir / "recordings_val.csv"),
            config["delays"],  # type: ignore[arg-type]
            summary_grid(config),
        ),
    }
    (out_dir / "online.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "tables.md").write_text(
        render_online_tables(payload), encoding="utf-8", newline="\n"
    )
    return payload
