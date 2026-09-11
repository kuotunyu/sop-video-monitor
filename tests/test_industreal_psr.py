"""IndustReal PSR label parsing and the raw-state -> step transcription."""

from __future__ import annotations

from pathlib import Path

import pytest

from sop_monitor.industreal_psr import (
    INCORRECT,
    INSTALL,
    REMOVE,
    audit_recording,
    load_procedure_info,
    load_psr_labels,
    load_psr_raw,
    raw_to_steps,
    states_to_steps,
)
from sop_monitor.metrics.online import Completion

PROCEDURE_INFO = Path(__file__).resolve().parents[1] / "sop" / "industreal" / "procedure_info.json"


def test_committed_procedure_info_has_eleven_components_times_three_steps() -> None:
    steps = load_procedure_info(PROCEDURE_INFO)
    assert len(steps) == 33
    assert [s.id % 3 for s in steps] == [INSTALL, INCORRECT, REMOVE] * 11
    assert all(s.state_idx == s.id // 3 for s in steps)
    assert steps[30].description == "Install rear wheel assy"


def test_state_transitions_follow_the_reference_rules() -> None:
    assert states_to_steps([0, 0], [1, 0], frame=10) == [Completion(10, INSTALL)]
    assert states_to_steps([0, 0], [0, -1], frame=11) == [Completion(11, 3 + INCORRECT)]
    assert states_to_steps([1, 0], [0, 0], frame=12) == [Completion(12, REMOVE)]
    assert states_to_steps([-1, 0], [1, 0], frame=13) == [Completion(13, INSTALL)]  # repaired
    assert states_to_steps([-1, 0], [0, 0], frame=14) == []  # undoing a mistake is no step
    assert states_to_steps([0, 0], [0, 0], frame=15) == []
    # Without errors, -1 reads as 0: the incorrect install vanishes, the repair is a plain install.
    assert states_to_steps([0, 0], [0, -1], frame=11, include_errors=False) == []
    assert states_to_steps([-1, 0], [1, 0], frame=13, include_errors=False) == [
        Completion(13, INSTALL)
    ]


def test_raw_rows_round_trip_to_completions(tmp_path: Path) -> None:
    (tmp_path / "PSR_labels_raw.csv").write_text(
        "000000.jpg,0,0,0\n000050.jpg,1,0,0\n000090.jpg,1,-1,0\n000120.jpg,1,1,0\n000200.jpg,0,1,0\n",
        encoding="utf-8",
    )
    raw = load_psr_raw(tmp_path / "PSR_labels_raw.csv")
    assert raw[0] == (0, (0, 0, 0))
    assert raw_to_steps(raw) == [
        Completion(50, 0),
        Completion(90, 3 + INCORRECT),
        Completion(120, 3),
        Completion(200, REMOVE),
    ]
    assert raw_to_steps(raw, include_errors=False) == [
        Completion(50, 0),
        Completion(120, 3),
        Completion(200, REMOVE),
    ]
    (tmp_path / "PSR_labels.csv").write_text(
        "000050.jpg,0,Install base\n000120.jpg,3,Install front chassis\n000200.jpg,2,Remove base\n",
        encoding="utf-8",
    )
    assert load_psr_labels(tmp_path / "PSR_labels.csv") == [
        Completion(50, 0),
        Completion(120, 3),
        Completion(200, 2),
    ]


def test_audit_cross_checks_raw_against_label_files(tmp_path: Path) -> None:
    raw = "000000.jpg," + ",".join(["0"] * 11) + "\n000050.jpg,1" + ",0" * 10 + "\n"
    (tmp_path / "PSR_labels_raw.csv").write_text(raw, encoding="utf-8")
    (tmp_path / "PSR_labels.csv").write_text("000050.jpg,0,Install base\n", encoding="utf-8")
    (tmp_path / "PSR_labels_with_errors.csv").write_text(
        "000050.jpg,0,Install base\n", encoding="utf-8"
    )
    report = audit_recording(tmp_path, n_frames=100, steps=load_procedure_info(PROCEDURE_INFO))
    assert report["raw_matches_labels"] and report["raw_matches_labels_with_errors"]
    assert report["problems"] == []
    assert audit_recording(tmp_path, n_frames=40, steps=load_procedure_info(PROCEDURE_INFO))[
        "problems"
    ] == ["label frame 50 beyond the 40-frame video"]


def test_malformed_rows_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "PSR_labels_raw.csv").write_text("000000.jpg,0,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_psr_raw(tmp_path / "PSR_labels_raw.csv")
    (tmp_path / "PSR_labels.csv").write_text("abc.jpg,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_psr_labels(tmp_path / "PSR_labels.csv")
