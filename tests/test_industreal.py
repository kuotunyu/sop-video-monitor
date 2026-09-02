"""IndustReal label parsing and participant-disjoint split freezing (spec 3.4 rehearsal)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sop_monitor.industreal import (
    Segment,
    freeze_industreal_splits,
    parse_label_rows,
    participant_of,
    verify_participant_disjoint,
    videos_by_participant,
)
from sop_monitor.splits import hash_test_list

ROWS = [
    ["03_assy_0_1", "8", "take_partial_model", "000027.jpg", "000039.jpg"],
    ["03_assy_0_1", "44", "put_partial_model", "000041.jpg", "000050.jpg"],
    ["05_main_1_2", "1", "align_objects", "000175.jpg", "000234.jpg"],
]


def test_parse_rows_yields_typed_segments_with_participants() -> None:
    segments = parse_label_rows(ROWS)
    assert segments[0] == Segment("03_assy_0_1", 8, "take_partial_model", 27, 39)
    assert segments[2].participant == "05"
    assert participant_of("21_assy_0_1") == "21"


@pytest.mark.parametrize(
    "bad",
    [
        [["03_assy_0_1", "8", "take", "000027.jpg"]],
        [["03_assy_0_1", "8", "take", "000040.jpg", "000039.jpg"]],
        [["03_assy_0_1", "8", "take", "abc.jpg", "000039.jpg"]],
    ],
)
def test_parse_rows_rejects_malformed_rows(bad: list[list[str]]) -> None:
    with pytest.raises(ValueError):
        parse_label_rows(bad)


def test_videos_are_grouped_by_participant() -> None:
    grouped = videos_by_participant(parse_label_rows(ROWS))
    assert grouped == {"03": ["03_assy_0_1"], "05": ["05_main_1_2"]}


def test_disjointness_check_names_the_offender() -> None:
    verify_participant_disjoint({"train": {"01"}, "val": {"02"}, "test": {"03"}})
    with pytest.raises(ValueError, match="participant 02"):
        verify_participant_disjoint({"train": {"01", "02"}, "val": {"02"}, "test": set()})


def _write(path: Path, rows: list[list[str]]) -> None:
    path.write_text("".join(",".join(row) + "\n" for row in rows), encoding="utf-8")


def test_freeze_writes_split_files_hash_and_manifest(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    _write(labels / "train.csv", [["01_assy_0_1", "1", "a", "000001.jpg", "000002.jpg"]])
    _write(labels / "val.csv", [["02_assy_0_1", "1", "a", "000001.jpg", "000002.jpg"]])
    _write(
        labels / "test.csv",
        [
            ["03_assy_0_1", "1", "a", "000001.jpg", "000002.jpg"],
            ["03_assy_1_1", "2", "b", "000001.jpg", "000002.jpg"],
            ["03_assy_1_1", "2", "b", "000003.jpg", "000004.jpg"],
        ],
    )
    out = tmp_path / "splits"
    manifest = freeze_industreal_splits(labels, out)
    assert (out / "test.csv").read_text(encoding="utf-8") == (
        "video_id,participant,n_segments\n03_assy_0_1,03,1\n03_assy_1_1,03,2\n"
    )
    expected = hash_test_list(["03_assy_0_1", "03_assy_1_1"])
    assert (out / "test_sha256.txt").read_text(encoding="utf-8") == expected + "\n"
    stored = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert stored == manifest
    assert manifest["test_sha256"] == expected
    assert manifest["action_classes"] == 2
    assert manifest["splits"]["test"] == {"rows": 3, "videos": 2, "participants": ["03"]}
    assert len(manifest["sources"]["train"]["sha256"]) == 64


def test_freeze_refuses_a_leaky_split(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    for name in ("train", "val", "test"):
        _write(labels / f"{name}.csv", [["07_assy_0_1", "1", "a", "000001.jpg", "000002.jpg"]])
    with pytest.raises(ValueError, match="participant 07"):
        freeze_industreal_splits(labels, tmp_path / "splits")
