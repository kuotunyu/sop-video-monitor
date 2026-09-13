"""HA-ViD SOP layer on synthetic sequences: steps, plates, learning, checks, synthetic violations."""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sop_monitor.cli import app
from sop_monitor.havid import CAMERA_OF_VIEW, SUBDATASETS, TemporalSegment
from sop_monitor.havid_sop import (
    FPS,
    Step,
    check_sequence,
    evaluate_run,
    learn_duration_bounds,
    learn_plate_graphs,
    mandatory_steps,
    perturb_duration,
    perturb_omission,
    perturb_order,
    plate_of,
    read_steps,
    segments_from_frames,
    step_sequence,
    synthetic_table,
    wilson,
    write_steps,
    wrong_table,
)

S = TemporalSegment


def test_plate_is_read_off_the_label_vocabulary() -> None:
    assert plate_of(["pckbx", "icbck", "sshc1", "ibscb"]) == "cylinder"
    assert plate_of(["iglft", "sspg3", "sftg1"]) == "gear"
    assert plate_of(["iibn2", "sntn5", "pbx"]) == "general"
    with pytest.raises(ValueError, match="ambiguous"):
        plate_of(["null"])


def test_two_hand_steps_merge_overlapping_same_label_segments() -> None:
    lh = [S(0, 9, "null"), S(10, 29, "icbck"), S(30, 39, "w"), S(40, 49, "ibscb")]
    rh = [S(0, 14, "null"), S(15, 34, "icbck"), S(35, 49, "pbx"), S(50, 59, "ibscb")]
    steps = step_sequence(lh, rh)
    assert [(s.label, s.start, s.end, s.hands) for s in steps] == [
        ("icbck", 10, 34, "lh+rh"),
        ("pbx", 35, 49, "rh"),
        ("ibscb", 40, 49, "lh"),
        ("ibscb", 50, 59, "rh"),
    ]
    assert segments_from_frames([(5, "a"), (6, "a"), (8, "a"), (9, "b")]) == [
        S(5, 6, "a"),
        S(8, 8, "a"),
        S(9, 9, "b"),
    ]


def _seq(labels: list[str], length: int = 30) -> list[Step]:
    return [Step(label, i * length, (i + 1) * length - 1, "lh") for i, label in enumerate(labels)]


def test_learning_checks_and_synthetic_violations(tmp_path: Path) -> None:
    order = ["pckbx", "icbck", "ibscb", "sshc1"]
    train = {f"S{n:02d}A04I01": _seq(order) for n in range(1, 6)}
    train["S06A04I01"] = _seq(["pckbx", "ibscb", "icbck", "sshc1", "sshc1"], length=60)
    plates = {rec: "cylinder" for rec in train}
    graphs = learn_plate_graphs(train, plates, min_support=3)
    graph = graphs["cylinder"]
    assert graph.nodes_of("PT") == sorted(order)
    edges = set(graph.relation("precedesPT"))
    assert ("pckbx", "icbck") in edges and ("ibscb", "sshc1") in edges
    assert ("icbck", "ibscb") not in edges  # S06 does them the other way round
    assert graphs["gear"].nodes_of("PT") == [] and graphs["general"].nodes_of("PT") == []
    mandatory = mandatory_steps(train, plates, fraction=0.9)
    assert mandatory["cylinder"] == sorted(order)
    bounds = learn_duration_bounds(train, plates, min_count=5)["cylinder"]
    assert set(bounds) == set(order)
    assert bounds["pckbx"].low == pytest.approx(30 / FPS)
    assert bounds["sshc1"].high == pytest.approx(60 / FPS, abs=0.5)

    clean = check_sequence(_seq(order), graph, bounds, mandatory["cylinder"])
    assert clean["violations"] == [] and clean["omissions"] == [] and clean["durations"] == []
    rng = random.Random(0)
    moved, index = perturb_order(_seq(order), graph, rng)  # type: ignore[misc]
    after = check_sequence(moved, graph, bounds, mandatory["cylinder"])
    assert any(v["index"] == index for v in after["violations"])  # type: ignore[index, union-attr]
    dropped, label = perturb_omission(_seq(order), mandatory["cylinder"], rng)  # type: ignore[misc]
    assert label in check_sequence(dropped, graph, bounds, mandatory["cylinder"])["omissions"]  # type: ignore[operator]
    stretched, index = perturb_duration(_seq(order), bounds, rng)  # type: ignore[misc]
    findings = check_sequence(stretched, graph, bounds, mandatory["cylinder"])["durations"]
    assert any(d["index"] == index and d["kind"] == "too_long" for d in findings)  # type: ignore[index, union-attr]
    assert perturb_order(_seq(["pckbx"]), graph, rng) is None

    val = {"S07A04I01": _seq(order), "S08A04I01": _seq(["icbck", "pckbx", "ibscb", "sshc1"])}
    val_plates = {rec: "cylinder" for rec in val}
    table = synthetic_table(val, val_plates, graphs, {"cylinder": bounds}, mandatory, seed=0)
    assert table["recordings"] == 2
    assert table["clean"]["flagged_by_kind"]["order"] == 1  # S08 violates pckbx -> icbck
    assert table["synthetic"]["order"]["applicable"] == 2
    assert table["synthetic"]["omission"]["detected"] == 2
    assert table["synthetic"]["duration"]["recall"] == 1.0
    assert table["synthetic"]["order"]["precision"] == pytest.approx(
        table["synthetic"]["order"]["detected"] / (table["synthetic"]["order"]["detected"] + 1)
    )
    again = synthetic_table(val, val_plates, graphs, {"cylinder": bounds}, mandatory, seed=0)
    assert json.dumps(again, sort_keys=True) == json.dumps(table, sort_keys=True)

    path = tmp_path / "steps_val.csv"
    write_steps(path, {"gt": val, "pred": val}, val_plates)
    sequences, read_plates = read_steps(path)
    assert sequences["gt"] == val and read_plates == val_plates


def test_wilson_and_wrong_table() -> None:
    assert wilson(0, 0) == [0.0, 0.0]
    low, high = wilson(3, 12)
    assert 0.08 < low < 0.25 < high < 0.55
    gt = {"r": {"lh": [S(0, 9, "w"), S(20, 29, "w")], "rh": []}}
    pred = {"r": {"lh": [S(5, 12, "w"), S(40, 45, "w")], "rh": [S(0, 3, "w")]}}
    table = wrong_table(gt, pred)
    assert (
        table["gt_segments"],
        table["detected"],
        table["pred_segments"],
        table["false_pred_segments"],
    ) == (2, 1, 3, 2)
    assert table["underpowered"] is True and table["precision"] == pytest.approx(1 / 3)


def _temporal_zip(
    path: Path, recordings: dict[str, tuple[list[TemporalSegment], list[TemporalSegment]]]
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for sub in SUBDATASETS:
            hand = sub[6:8]
            for rec, (lh, rh) in recordings.items():
                segments = lh if hand == "lh" else rh
                text = "".join(f"{s.start} {s.end} {s.label}\n" for s in segments)
                archive.writestr(
                    f"HAViD_temporalAnnotation/{sub}/temporal_timestamps/{rec}_{sub}_temporal.txt",
                    text,
                )
    return path


def test_cli_learn_then_evaluate_and_reproduce(tmp_path: Path) -> None:
    order = ["pckbx", "icbck", "ibscb", "sshc1"]

    def hands(
        labels: list[str], length: int = 30
    ) -> tuple[list[TemporalSegment], list[TemporalSegment]]:
        lh = [S(i * length, (i + 1) * length - 1, label) for i, label in enumerate(labels)]
        rh = [S(0, len(labels) * length - 1, "null")]
        return lh, rh

    recordings = {f"S{n:02d}A04I01": hands(order) for n in range(1, 8)}
    recordings["S08A04I01"] = hands(["icbck", "pckbx", "ibscb", "w"])
    temporal = _temporal_zip(tmp_path / "t.zip", recordings)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for name, recs in (("train", list(recordings)[:6]), ("val", list(recordings)[6:])):
        lines = ["video_id,subject,recording,view"]
        for rec in recs:
            lines += [f"{rec}{CAMERA_OF_VIEW[v]},{rec[:3]},{rec},{v}" for v in range(3)]
        (split_dir / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    graphs_dir = tmp_path / "sop"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn-havid-sop",
            "--temporal",
            str(temporal),
            "--split-dir",
            str(split_dir),
            "--out",
            str(graphs_dir),
            "--min-support",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (graphs_dir / "learned_precedence_cylinder.json").is_file()
    assert (graphs_dir / "duration_bounds.json").is_file() and (
        graphs_dir / "mandatory_steps.json"
    ).is_file()

    # per-hand "prediction" runs that copy the ground truth, with the class ids of a tiny mapping
    labels = sorted({"null", "w", *order})
    ids = {label: i for i, label in enumerate(labels)}
    run_dirs = {}
    for hand in ("lh", "rh"):
        run_dir = tmp_path / f"run_{hand}"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(
            json.dumps(
                {"classes": {"class_id_to_label": {str(i): label for label, i in ids.items()}}}
            ),
            encoding="utf-8",
        )
        rows = ["video_id,participant,frame,gt,pred_fusion_causal"]
        for rec in list(recordings)[6:]:
            segments = recordings[rec][0] if hand == "lh" else recordings[rec][1]
            for s in segments:
                for frame in range(s.start, s.end + 1):
                    rows.append(f"{rec},{rec[:3]},{frame},{ids[s.label]},{ids[s.label]}")
        (run_dir / "predictions_val.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        run_dirs[hand] = run_dir
    out = tmp_path / "sop_run"
    result = runner.invoke(
        app,
        [
            "havid-sop",
            "--pred-lh",
            str(run_dirs["lh"]),
            "--pred-rh",
            str(run_dirs["rh"]),
            "--out",
            str(out),
            "--temporal",
            str(temporal),
            "--split-dir",
            str(split_dir),
            "--graphs",
            str(graphs_dir),
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((out / "sop_checks.json").read_text(encoding="utf-8"))
    assert payload["graphs"]["cylinder"]["nodes"] == 4
    assert payload["sources"]["gt"]["clean"]["flagged_by_kind"]["order"] == 1
    assert payload["wrong"]["gt_segments"] == 1 and payload["wrong"]["detected"] == 1
    assert "| order |" in (out / "tables.md").read_text(encoding="utf-8")
    recomputed = evaluate_run(out, graphs_dir, 1)
    assert json.dumps(recomputed["sources"], sort_keys=True) == json.dumps(
        payload["sources"], sort_keys=True
    )
    result = runner.invoke(app, ["reproduce-lite", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "sop_run: OK" in result.output
