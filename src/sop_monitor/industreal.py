"""IndustReal label parsing and split freezing (spec 3.1 metric donor; 3.4 W1 rehearsal).

IndustReal (Schoonbeek et al., WACV 2024; Apache-2.0) ships action-recognition labels as
three CSV files, one per official split, with rows of the form::

    video_id,action_id,action_name,start_frame,end_frame
    03_assy_0_1,8,take_partial_model,000027.jpg,000039.jpg

The participant id is the first underscore-separated token of ``video_id``. The official
split is participant-disjoint (12 / 5 / 10 participants); this module regenerates the split
from the raw rows, *verifies* that disjointness instead of assuming it, and freezes the
test list with the same SHA-256 contract used for HA-ViD (:mod:`sop_monitor.splits`).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from sop_monitor.splits import SPLIT_NAMES, hash_test_list

LABEL_FILES: dict[str, str] = {name: f"{name}.csv" for name in SPLIT_NAMES}


@dataclass(frozen=True)
class Segment:
    """One labelled action segment; frames are the integer indices from ``000027.jpg``."""

    video_id: str
    action_id: int
    action_name: str
    start_frame: int
    end_frame: int

    @property
    def participant(self) -> str:
        return participant_of(self.video_id)


def participant_of(video_id: str) -> str:
    """Return the participant id encoded as the first token of an IndustReal video id."""
    head, _, _ = video_id.partition("_")
    if not head:
        raise ValueError(f"cannot derive participant from video id {video_id!r}")
    return head


def _frame_index(token: str) -> int:
    stem = token.strip()
    if stem.endswith(".jpg"):
        stem = stem[: -len(".jpg")]
    if not stem.isdigit():
        raise ValueError(f"frame token {token!r} is not an integer frame index")
    return int(stem)


def parse_label_rows(rows: list[list[str]]) -> list[Segment]:
    """Parse raw CSV rows into segments, rejecting malformed rows loudly."""
    segments: list[Segment] = []
    for row in rows:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != 5:
            raise ValueError(f"expected 5 columns, got {len(row)}: {row!r}")
        video_id, action_id, action_name, start, end = (cell.strip() for cell in row)
        segment = Segment(
            video_id=video_id,
            action_id=int(action_id),
            action_name=action_name,
            start_frame=_frame_index(start),
            end_frame=_frame_index(end),
        )
        if segment.end_frame < segment.start_frame:
            raise ValueError(f"segment ends before it starts: {row!r}")
        segments.append(segment)
    return segments


def read_labels(path: Path) -> list[Segment]:
    """Read one official label CSV (no header row)."""
    with path.open(encoding="utf-8", newline="") as handle:
        return parse_label_rows(list(csv.reader(handle)))


def videos_by_participant(segments: list[Segment]) -> dict[str, list[str]]:
    """Group sorted unique video ids under their participant id."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for segment in segments:
        grouped[segment.participant].add(segment.video_id)
    return {participant: sorted(videos) for participant, videos in sorted(grouped.items())}


def verify_participant_disjoint(split_participants: dict[str, set[str]]) -> None:
    """Raise if any participant appears in more than one split."""
    seen: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        for participant in sorted(split_participants.get(split_name, set())):
            if participant in seen:
                raise ValueError(
                    f"participant {participant} is in both {seen[participant]} and {split_name}"
                )
            seen[participant] = split_name


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def freeze_industreal_splits(labels_dir: Path, out_dir: Path) -> dict[str, object]:
    """Regenerate the participant-disjoint split from the raw label CSVs and freeze it.

    Writes ``{train,val,test}.csv`` (``video_id,participant,n_segments``),
    ``test_sha256.txt`` and ``manifest.json`` (row/video/participant counts, action-class
    count, source file digests) into ``out_dir`` and returns the manifest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "dataset": "IndustReal",
        "license": "Apache-2.0",
        "split_rule": "official participant-disjoint split regenerated from the raw label CSVs",
        "sources": {},
        "splits": {},
    }
    split_participants: dict[str, set[str]] = {}
    action_classes: set[tuple[int, str]] = set()
    for split_name in SPLIT_NAMES:
        source = labels_dir / LABEL_FILES[split_name]
        segments = read_labels(source)
        grouped = videos_by_participant(segments)
        split_participants[split_name] = set(grouped)
        action_classes.update((s.action_id, s.action_name) for s in segments)
        counts: dict[str, int] = defaultdict(int)
        for segment in segments:
            counts[segment.video_id] += 1
        with (out_dir / f"{split_name}.csv").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("video_id,participant,n_segments\n")
            for participant, videos in grouped.items():
                for video_id in videos:
                    handle.write(f"{video_id},{participant},{counts[video_id]}\n")
        manifest["sources"][split_name] = {  # type: ignore[index]
            "file": source.name,
            "sha256": _sha256_file(source),
        }
        manifest["splits"][split_name] = {  # type: ignore[index]
            "rows": len(segments),
            "videos": sum(len(videos) for videos in grouped.values()),
            "participants": sorted(grouped),
        }
    verify_participant_disjoint(split_participants)
    test_videos = [
        video_id
        for videos in videos_by_participant(read_labels(labels_dir / LABEL_FILES["test"])).values()
        for video_id in videos
    ]
    digest = hash_test_list(test_videos)
    (out_dir / "test_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    manifest["action_classes"] = len(action_classes)
    manifest["test_sha256"] = digest
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
