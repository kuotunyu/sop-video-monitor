"""HA-ViD annotation parsing, official-split cross-checks and the audit on synthetic archives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sop_monitor.cli import app
from sop_monitor.havid import (
    CAMERA_OF_VIEW,
    OFFICIAL_TRIM,
    SUBDATASETS,
    audit_havid,
    frame_labels,
    freeze_havid_splits,
    load_official_zip,
    load_temporal_zip,
    official_from_temporal,
    parse_collaboration,
    parse_recording_id,
    parse_temporal,
    parse_video_id,
)

# two recordings of 30 frames: S01 (train) and S02 (test); the left hand makes one wrong step
TEMPORAL = {
    "lh_pt": "0 9 null\n10 19 ipsft\n20 29 w\n",
    "rh_pt": "0 14 null\n15 29 ibscb\n",
    "lh_aa": "0 9 null\n10 14 gsh\n15 29 msh\n",
    "rh_aa": "0 29 null\n",
}
COLLABORATION = (
    "0 9 pause null null\n10 14 single_handed_left ipsft null\n"
    "15 19 parallel ipsft ibscb\n20 29 parallel w ibscb\n"
)
RECORDINGS = ("S01A04I01", "S02A04I01")
TEST_SUBJECTS = {"S02"}


def make_temporal_zip(path: Path, recordings: tuple[str, ...] = RECORDINGS) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for sub in SUBDATASETS:
            hand_level = sub[6:]
            for rec in recordings:
                archive.writestr(
                    f"HAViD_temporalAnnotation/{sub}/temporal_timestamps/{rec}_{sub}_temporal.txt",
                    TEMPORAL[hand_level],
                )
                archive.writestr(
                    f"HAViD_temporalAnnotation/{sub}/collaboration_timestamps/"
                    f"{rec}_{sub}_collaboration.txt",
                    COLLABORATION,
                )
    return path


def make_official_zip(
    path: Path,
    trim: int = OFFICIAL_TRIM,
    recordings: tuple[str, ...] = RECORDINGS,
    unsplit: tuple[str, ...] = (),
) -> Path:
    """Official layout: subjects in TEST_SUBJECTS go to test, ``unsplit`` recordings to neither."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("/", "")  # Dropbox folder zips carry a bare root entry
        for sub in SUBDATASETS:
            camera = CAMERA_OF_VIEW[int(sub[4])]
            segments = parse_temporal(TEMPORAL[sub[6:]])
            labels = sorted({s.label for s in segments})
            archive.writestr(
                f"{sub}/mapping.txt", "".join(f"{i} {label}\n" for i, label in enumerate(labels))
            )
            listed = [rec for rec in recordings if rec not in unsplit]
            train = [rec for rec in listed if rec[:3] not in TEST_SUBJECTS]
            test = [rec for rec in listed if rec[:3] in TEST_SUBJECTS]
            archive.writestr(
                f"{sub}/splits/train.split1.bundle", "".join(f"{r}{camera}.txt\n" for r in train)
            )
            archive.writestr(
                f"{sub}/splits/test.split1.bundle", "".join(f"{r}{camera}.txt\n" for r in test)
            )
            for rec in recordings:
                archive.writestr(
                    f"{sub}/groundTruth/{rec}{camera}.txt",
                    "".join(f"{label}\n" for label in official_from_temporal(segments, trim)),
                )
        for rec in (*recordings, "S30A04I01"):  # S30 has features but no temporal annotation
            for camera in CAMERA_OF_VIEW.values():
                archive.writestr(f"features/{rec}{camera}.npy", b"not read by the audit")
    return path


def test_ids_parse_and_reject() -> None:
    rid = parse_recording_id("S13A08I26")
    assert (rid.subject, rid.attempt, rid.stage, rid.version) == ("S13", 8, 2, 6)
    assert rid.text == "S13A08I26"
    first = parse_recording_id("S01A04I01")
    assert (first.stage, first.version, first.text) == (1, 1, "S01A04I01")
    assert parse_video_id("S01A04I01S2.txt") == ("S01A04I01", 2)
    with pytest.raises(ValueError):
        parse_recording_id("S1A04I01")
    with pytest.raises(ValueError):
        parse_video_id("S01A04I01X0")


def test_temporal_rows_are_inclusive_and_contiguous() -> None:
    segments = parse_temporal("0 26 null\n27 36 ack\n\n")
    assert [s.n_frames for s in segments] == [27, 10]
    assert frame_labels(segments)[26:28] == ["null", "ack"]
    assert len(official_from_temporal(segments)) == 37 - 2 * OFFICIAL_TRIM
    with pytest.raises(ValueError, match="expected 27"):
        parse_temporal("0 26 null\n28 36 ack\n")
    with pytest.raises(ValueError, match="expected 0"):
        parse_temporal("1 26 null\n")
    with pytest.raises(ValueError, match="unknown collaboration status"):
        parse_collaboration("0 9 duet null null\n")
    assert parse_collaboration(COLLABORATION)[3].lh == "w"


def test_audit_counts_wrong_and_checks_official_trim(tmp_path: Path) -> None:
    temporal_zip = make_temporal_zip(tmp_path / "t.zip")
    official_zip = make_official_zip(tmp_path / "o.zip")
    temporal, collaboration = load_temporal_zip(temporal_zip)
    assert set(temporal) == set(SUBDATASETS) and len(collaboration["view2_rh_aa"]) == 2
    official = load_official_zip(official_zip)
    assert official["view1_lh_pt"].test == ["S02A04I01S1"]
    assert official["view0_lh_pt"].mapping == {0: "ipsft", 1: "null", 2: "w"}

    report = audit_havid(temporal_zip, official_zip)
    assert report["problems"] == []
    ann = report["annotations"]
    assert ann["recordings"] == 2 and ann["subjects"] == 2 and ann["frames"]["total"] == 60
    assert ann["views_byte_identical"] is True
    assert ann["wrong"]["segments"] == 2 and ann["wrong"]["frames"] == 20
    assert ann["wrong"]["by_subject"] == {"S01": 1, "S02": 1}
    assert ann["wrong"]["by_hand"] == {"lh": 2, "rh": 0}
    assert ann["wrong"]["official_test_split"] == {
        "recordings": 1,
        "recordings_with_wrong": 1,
        "segments": 1,
    }
    assert ann["levels"]["lh_aa"]["verbs"] == {"grasp": 2, "move": 2}
    assert ann["recordings_per_stage"] == {"1": 2}
    assert ann["collaboration_frames"]["parallel"] == 30
    split = report["official_split"]
    assert split["subject_disjoint"] is True
    assert split["ground_truth_trim_frames"] == {str(OFFICIAL_TRIM): 24}
    assert split["subdatasets"]["view0_lh_pt"]["ground_truth_matching_trimmed_temporal"] == 2
    assert split["videos_in_no_split"] == []
    assert split["features"] == {
        "files": 9,
        "without_temporal_annotation": ["S30A04I01M0", "S30A04I01S1", "S30A04I01S2"],
        "annotated_without_features": [],
    }


def test_freeze_splits_follow_the_official_subjects(tmp_path: Path) -> None:
    recordings = ("S01A04I01", "S03A04I01", "S06A04I01", "S02A04I01", "S02A05I01")
    temporal_zip = make_temporal_zip(tmp_path / "t.zip", recordings)
    official_zip = make_official_zip(
        tmp_path / "o.zip", recordings=recordings, unsplit=("S02A05I01",)
    )
    manifest = freeze_havid_splits(official_zip, temporal_zip, tmp_path / "ha-vid", seed=0, n_val=1)
    assert manifest["excluded_recordings"] == ["S02A05I01"]
    assert manifest["splits"]["test"] == {"videos": 3, "recordings": 1, "subjects": ["S02"]}
    assert manifest["splits"]["val"]["videos"] == 3
    assert len(manifest["splits"]["train"]["subjects"]) == 2
    assert not set(manifest["splits"]["train"]["subjects"]) & set(
        manifest["splits"]["val"]["subjects"]
    )
    test_csv = (tmp_path / "ha-vid" / "test.csv").read_text(encoding="utf-8")
    assert test_csv.splitlines()[0] == "video_id,subject,recording,view"
    assert "S02A04I01S2,S02,S02A04I01,2" in test_csv
    assert "S02A05I01" not in (tmp_path / "ha-vid" / "train.csv").read_text(encoding="utf-8")
    result = CliRunner().invoke(app, ["verify-splits", "--directory", str(tmp_path / "ha-vid")])
    assert result.exit_code == 0, result.output
    again = freeze_havid_splits(official_zip, temporal_zip, tmp_path / "again", seed=0, n_val=1)
    assert again["test_sha256"] == manifest["test_sha256"]
    assert again["splits"]["val"] == manifest["splits"]["val"]
    with pytest.raises(ValueError, match="n_val=5"):
        freeze_havid_splits(official_zip, temporal_zip, tmp_path / "bad", seed=0, n_val=5)


def test_audit_reports_broken_official_files(tmp_path: Path) -> None:
    temporal_zip = make_temporal_zip(tmp_path / "t.zip")
    official_zip = make_official_zip(tmp_path / "o.zip", trim=3)
    report = audit_havid(temporal_zip, official_zip)
    assert report["official_split"]["ground_truth_trim_frames"] == {"3": 24}
    assert report["problems"] == []
    with zipfile.ZipFile(official_zip, "a") as archive:
        archive.writestr("copy/view0_lh_pt/mapping.txt", "0 null\n")
    with pytest.raises(FileNotFoundError, match="2 members"):
        load_official_zip(official_zip)


def test_cli_audit_havid_writes_json(tmp_path: Path) -> None:
    temporal_zip = make_temporal_zip(tmp_path / "t.zip")
    official_zip = make_official_zip(tmp_path / "o.zip")
    out = tmp_path / "havid_audit.json"
    result = CliRunner().invoke(
        app,
        [
            "audit-havid",
            "--temporal",
            str(temporal_zip),
            "--official",
            str(official_zip),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["annotations"]["wrong"]["segments"] == 2
    assert "videos" not in payload
    assert "wrong segments=2" in result.output
