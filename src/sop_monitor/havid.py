"""HA-ViD temporal annotations, the official action-segmentation split and the W1 data audit.

HA-ViD (Zheng, Lee and Lu, NeurIPS 2023 Datasets and Benchmarks; CC BY-NC 4.0) is delivered by
the authors as a Dropbox folder. This module reads the two archives the project uses and records
what was *measured* on the 2023-06 release (``reports/havid_audit.json``), not what the paper says.

``HAViD_temporalAnnotation.zip``
    Twelve sub-datasets ``view{0,1,2}_{lh,rh}_{pt,aa}`` (view 0 = side camera ``M0``, 1 = front
    ``S1``, 2 = top ``S2``; ``lh``/``rh`` = left/right hand; ``pt`` = primitive task, ``aa`` = atomic
    action). Each holds one ``temporal_timestamps`` file per annotated recording with rows
    ``start end label`` (frame indices, **inclusive**, contiguous from 0) and one
    ``collaboration_timestamps`` file with rows ``start end status lh_label rh_label``. The three
    views ship byte-identical files: the annotation belongs to the recording, not the camera.
    Labels are HR-SAT abbreviations, ``<verb><manipulated><target><tool>``; ``null`` is a pause and
    ``w`` is the paper's ``wrong`` (error) label (paper Figure 9). ``w`` exists only at primitive-task
    level.

``ActionSegmentation/data`` (Dropbox folder download)
    The benchmark layout used for MS-TCN/DTGRM/BCN: ``view*_*_*/{groundTruth,splits,mapping.txt}``
    plus ``features/<video>.npy`` (I3D). ``groundTruth`` files are per-frame labels **trimmed by
    five frames at both ends** relative to the temporal files (``OFFICIAL_TRIM``); the audit verifies
    this on every file. ``splits/{train,test}.split1.bundle`` list video files; the split is by
    subject (paper Supplementary 2.4) and the audit checks that the subject sets are disjoint.

Video ids are ``S<subject>A<attempt>I<stage><version><camera>``; the recording id is the same
without the camera (see ``How to read the file names.txt`` shipped with the data).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from sop_monitor.splits import SPLIT_NAMES, hash_test_list

HANDS = ("lh", "rh")
LEVELS = ("pt", "aa")
VIEWS = (0, 1, 2)
CAMERA_OF_VIEW = {0: "M0", 1: "S1", 2: "S2"}
VIEW_OF_CAMERA = {camera: view for view, camera in CAMERA_OF_VIEW.items()}
VIEW_NAMES = {0: "side", 1: "front", 2: "top"}
SUBDATASETS = tuple(f"view{v}_{h}_{lv}" for v in VIEWS for h in HANDS for lv in LEVELS)

NULL = "null"
WRONG = "w"
OFFICIAL_TRIM = 5
"""Frames dropped at each end of a temporal file to obtain the official ``groundTruth`` file."""

VERBS = {
    "a": "approach",
    "d": "disassemble",
    "g": "grasp",
    "h": "hold",
    "i": "insert",
    "l": "slide",
    "m": "move",
    "p": "place",
    "r": "rotate",
    "s": "screw",
}
"""HR-SAT action-verb abbreviations (paper Figure 9); the first character of a non-noise label."""

NOISE = {NULL: "pause (no action)", WRONG: "wrong (error)"}
COLLABORATION_STATUSES = (
    "collaboration",
    "parallel",
    "single_handed_left",
    "single_handed_right",
    "pause",
)

_RECORDING = re.compile(r"^(S\d{2})A(\d{2})I(\d)(\d)$")
_VIDEO = re.compile(r"^(S\d{2}A\d{2}I\d{2})(M0|S1|S2)$")


@dataclass(frozen=True)
class RecordingId:
    """``S01A04I01`` → subject ``S01``, attempt 4, stage 1, instruction version 1.

    The stage digit is ``0`` for stage 1 (``I01``), ``2`` for stage 2 (``I21`` to ``I26``) and ``3``
    for stage 3 (``I31``, ``I32``), per the authors' naming note; ``stage`` is the human number.
    """

    subject: str
    attempt: int
    stage: int
    version: int

    @property
    def text(self) -> str:
        digit = 0 if self.stage == 1 else self.stage
        return f"{self.subject}A{self.attempt:02d}I{digit}{self.version}"


def parse_recording_id(text: str) -> RecordingId:
    match = _RECORDING.match(text.strip())
    if match is None:
        raise ValueError(f"not a HA-ViD recording id: {text!r}")
    subject, attempt, digit, version = match.groups()
    if digit not in ("0", "2", "3"):
        raise ValueError(f"unknown stage digit in recording id {text!r}")
    return RecordingId(subject, int(attempt), 1 if digit == "0" else int(digit), int(version))


def parse_video_id(text: str) -> tuple[str, int]:
    """``S01A04I01M0`` → (``S01A04I01``, view 0); accepts a ``.txt``/``.mp4``/``.npy`` suffix."""
    stem = text.strip()
    for suffix in (".txt", ".mp4", ".npy"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    match = _VIDEO.match(stem)
    if match is None:
        raise ValueError(f"not a HA-ViD video id: {text!r}")
    recording, camera = match.groups()
    return recording, VIEW_OF_CAMERA[camera]


def video_id(recording: str, view: int) -> str:
    return f"{recording}{CAMERA_OF_VIEW[view]}"


@dataclass(frozen=True)
class TemporalSegment:
    """One labelled interval; ``start``/``end`` are inclusive 0-based frame indices."""

    start: int
    end: int
    label: str

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class CollaborationSegment:
    start: int
    end: int
    status: str
    lh: str
    rh: str


def parse_temporal(text: str, source: str = "<text>") -> list[TemporalSegment]:
    """Parse ``start end label`` rows and require a contiguous cover of ``[0, last]``."""
    segments: list[TemporalSegment] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"{source}:{line_no}: expected 'start end label', got {line!r}")
        start, end, label = int(parts[0]), int(parts[1]), parts[2]
        if end < start:
            raise ValueError(f"{source}:{line_no}: end {end} before start {start}")
        expected = segments[-1].end + 1 if segments else 0
        if start != expected:
            raise ValueError(f"{source}:{line_no}: segment starts at {start}, expected {expected}")
        segments.append(TemporalSegment(start, end, label))
    if not segments:
        raise ValueError(f"{source}: no segments")
    return segments


def parse_collaboration(text: str, source: str = "<text>") -> list[CollaborationSegment]:
    rows: list[CollaborationSegment] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"{source}:{line_no}: expected 'start end status lh rh', got {line!r}")
        if parts[2] not in COLLABORATION_STATUSES:
            raise ValueError(f"{source}:{line_no}: unknown collaboration status {parts[2]!r}")
        rows.append(
            CollaborationSegment(int(parts[0]), int(parts[1]), parts[2], parts[3], parts[4])
        )
    return rows


def n_frames_of(segments: Iterable[TemporalSegment]) -> int:
    return max(s.end for s in segments) + 1


def frame_labels(segments: Iterable[TemporalSegment]) -> list[str]:
    """Expand inclusive segments to one label per frame."""
    labels: list[str] = []
    for segment in segments:
        labels.extend([segment.label] * segment.n_frames)
    return labels


def official_from_temporal(
    segments: Iterable[TemporalSegment], trim: int = OFFICIAL_TRIM
) -> list[str]:
    """The per-frame sequence the official ``groundTruth`` file holds for the same recording."""
    labels = frame_labels(segments)
    return labels[trim : len(labels) - trim]


def wrong_segments(segments: Iterable[TemporalSegment]) -> list[TemporalSegment]:
    return [s for s in segments if s.label == WRONG]


# ---- archives -------------------------------------------------------------------------------


def _member(archive: zipfile.ZipFile, suffix: str) -> str:
    """Name of the single archive member whose path ends with ``suffix`` (folder zips vary in root)."""
    matches = [name for name in archive.namelist() if name.replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise FileNotFoundError(f"{archive.filename}: {len(matches)} members end with {suffix!r}")
    return matches[0]


def _read_text(archive: zipfile.ZipFile, name: str) -> str:
    return archive.read(name).decode("utf-8-sig")


def load_temporal_zip(
    path: Path,
) -> tuple[
    dict[str, dict[str, list[TemporalSegment]]], dict[str, dict[str, list[CollaborationSegment]]]
]:
    """``{subdataset: {recording: segments}}`` for the temporal and collaboration files."""
    temporal: dict[str, dict[str, list[TemporalSegment]]] = defaultdict(dict)
    collaboration: dict[str, dict[str, list[CollaborationSegment]]] = defaultdict(dict)
    pattern = re.compile(
        r"(view\d_[lr]h_(?:pt|aa))/(temporal|collaboration)_timestamps/"
        r"(S\d{2}A\d{2}I\d{2})_view\d_[lr]h_(?:pt|aa)_(?:temporal|collaboration)\.txt$"
    )
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            match = pattern.search(name.replace("\\", "/"))
            if match is None:
                continue
            subdataset, kind, recording = match.groups()
            text = _read_text(archive, name)
            if kind == "temporal":
                temporal[subdataset][recording] = parse_temporal(text, name)
            else:
                collaboration[subdataset][recording] = parse_collaboration(text, name)
    if set(temporal) != set(SUBDATASETS):
        raise ValueError(f"{path}: expected {SUBDATASETS}, found {sorted(temporal)}")
    return dict(temporal), dict(collaboration)


def read_bundle(text: str) -> list[str]:
    """Video ids listed in a ``*.bundle`` split file (one ``<video>.txt`` per line)."""
    videos: list[str] = []
    for line in text.splitlines():
        if line.strip():
            stem = line.strip().removesuffix(".txt")
            parse_video_id(stem)
            videos.append(stem)
    return videos


def read_mapping(text: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line in text.splitlines():
        if line.strip():
            index, label = line.split()
            mapping[int(index)] = label
    return mapping


def read_ground_truth(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


@dataclass(frozen=True)
class OfficialSubdataset:
    name: str
    mapping: dict[int, str]
    train: list[str]
    test: list[str]
    ground_truth: dict[str, list[str]]


def load_official_zip(
    path: Path, subdatasets: Iterable[str] = SUBDATASETS
) -> dict[str, OfficialSubdataset]:
    """Splits, mappings and per-frame ground truth of the benchmark folder (features are skipped)."""
    result: dict[str, OfficialSubdataset] = {}
    with zipfile.ZipFile(path) as archive:
        names = [n.replace("\\", "/") for n in archive.namelist()]
        for sub in subdatasets:
            mapping = read_mapping(_read_text(archive, _member(archive, f"{sub}/mapping.txt")))
            train = read_bundle(
                _read_text(archive, _member(archive, f"{sub}/splits/train.split1.bundle"))
            )
            test = read_bundle(
                _read_text(archive, _member(archive, f"{sub}/splits/test.split1.bundle"))
            )
            ground_truth = {}
            prefix = f"{sub}/groundTruth/"
            for name in names:
                if prefix in name and name.endswith(".txt"):
                    stem = name.rsplit("/", 1)[-1].removesuffix(".txt")
                    parse_video_id(stem)
                    ground_truth[stem] = read_ground_truth(_read_text(archive, name))
            result[sub] = OfficialSubdataset(sub, mapping, train, test, ground_truth)
    return result


def feature_videos(path: Path) -> list[str]:
    """Video ids that have an I3D feature file (``features/<video>.npy``) in the benchmark zip."""
    videos: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            name = name.replace("\\", "/")
            if "features/" in name and name.endswith(".npy"):
                stem = name.rsplit("/", 1)[-1].removesuffix(".npy")
                parse_video_id(stem)
                videos.append(stem)
    return sorted(videos)


def sha256_text(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# ---- audit ----------------------------------------------------------------------------------


def _percentiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[int(len(ordered) * 0.95)],
        "max": ordered[-1],
    }


def audit_annotations(
    temporal: Mapping[str, Mapping[str, list[TemporalSegment]]],
    collaboration: Mapping[str, Mapping[str, list[CollaborationSegment]]],
    test_recordings: set[str] | None = None,
) -> dict[str, object]:
    """Everything the annotation archive alone can tell us; never guesses frame rates."""
    problems: list[str] = []
    reference = temporal["view0_lh_pt"]
    recordings = sorted(reference)
    # 1. the twelve sub-datasets describe the same recordings with the same frame extents
    for sub in SUBDATASETS:
        if sorted(temporal[sub]) != recordings:
            problems.append(f"{sub}: recording set differs from view0_lh_pt")
    for recording in recordings:
        extents = {
            sub: n_frames_of(temporal[sub][recording])
            for sub in SUBDATASETS
            if recording in temporal[sub]
        }
        if len(set(extents.values())) != 1:
            problems.append(f"{recording}: frame extents differ across sub-datasets {extents}")
    # 2. views are byte-identical copies (annotation belongs to the recording)
    views_identical = True
    for hand in HANDS:
        for level in LEVELS:
            for recording in recordings:
                digests = {
                    sha256_text(
                        f"{s.start} {s.end} {s.label}"
                        for s in temporal[f"view{v}_{hand}_{level}"][recording]
                    )
                    for v in VIEWS
                }
                if len(digests) != 1:
                    views_identical = False
                    problems.append(f"{recording} {hand}_{level}: views differ")
    n_frames = {r: n_frames_of(reference[r]) for r in recordings}
    subjects = Counter(parse_recording_id(r).subject for r in recordings)
    stages = Counter(parse_recording_id(r).stage for r in recordings)
    # 3. label statistics per hand and level (view 0, identical to the others)
    levels: dict[str, object] = {}
    for level in LEVELS:
        for hand in HANDS:
            segs = [s for r in recordings for s in temporal[f"view0_{hand}_{level}"][r]]
            labels = Counter(s.label for s in segs)
            frames = Counter()
            for s in segs:
                frames[s.label] += s.n_frames
            verbs = Counter(VERBS.get(s.label[0], "?") for s in segs if s.label not in NOISE)
            unknown_verbs = sorted(
                {s.label for s in segs if s.label not in NOISE and s.label[0] not in VERBS}
            )
            if unknown_verbs:
                problems.append(f"{hand}_{level}: labels with unknown verb {unknown_verbs[:5]}")
            levels[f"{hand}_{level}"] = {
                "segments": len(segs),
                "classes": len(labels),
                "frames": sum(frames.values()),
                "null_frames": frames[NULL],
                "wrong_frames": frames[WRONG],
                "wrong_segments": labels[WRONG],
                "segment_frames": _percentiles([s.n_frames for s in segs]),
                "verbs": dict(sorted(verbs.items())),
            }
    # 4. the wrong label: where it is, and how much of it a subject-held-out test set would see
    wrong: dict[str, object] = {}
    by_subject: Counter[str] = Counter()
    by_stage: Counter[int] = Counter()
    by_hand: Counter[str] = Counter()
    per_recording: dict[str, int] = {}
    both_hands = 0
    for recording in recordings:
        lh = wrong_segments(temporal["view0_lh_pt"][recording])
        rh = wrong_segments(temporal["view0_rh_pt"][recording])
        if lh or rh:
            per_recording[recording] = len(lh) + len(rh)
        rid = parse_recording_id(recording)
        by_subject[rid.subject] += len(lh) + len(rh)
        by_stage[rid.stage] += len(lh) + len(rh)
        by_hand["lh"] += len(lh)
        by_hand["rh"] += len(rh)
        both_hands += sum(1 for a in lh for b in rh if (a.start, a.end) == (b.start, b.end))
    wrong_all = [
        s for r in recordings for h in HANDS for s in wrong_segments(temporal[f"view0_{h}_pt"][r])
    ]
    wrong_in_aa = any(
        s.label == WRONG for r in recordings for h in HANDS for s in temporal[f"view0_{h}_aa"][r]
    )
    wrong = {
        "level": "pt and aa" if wrong_in_aa else "pt only (no w label at atomic-action level)",
        "segments": len(wrong_all),
        "frames": sum(s.n_frames for s in wrong_all),
        "recordings_with_wrong": len(per_recording),
        "segments_identical_on_both_hands": both_hands,
        "segment_frames": _percentiles([s.n_frames for s in wrong_all]) if wrong_all else None,
        "by_subject": {k: by_subject[k] for k in sorted(subjects)},
        "by_stage": {str(k): by_stage[k] for k in sorted(stages)},
        "by_hand": dict(by_hand),
        "per_recording": per_recording,
    }
    if test_recordings is not None:
        wrong["official_test_split"] = {
            "recordings": len(test_recordings),
            "recordings_with_wrong": sum(1 for r in per_recording if r in test_recordings),
            "segments": sum(n for r, n in per_recording.items() if r in test_recordings),
        }
    # 5. collaboration status
    status_frames: Counter[str] = Counter()
    collab_ref = collaboration["view0_lh_pt"]
    for recording in recordings:
        rows = collab_ref.get(recording)
        if rows is None:
            problems.append(f"{recording}: no collaboration file")
            continue
        if rows[0].start != 0 or rows[-1].end != n_frames[recording] - 1:
            problems.append(f"{recording}: collaboration rows do not cover the recording")
        for row in rows:
            status_frames[row.status] += row.end - row.start + 1
    return {
        "recordings": len(recordings),
        "subjects": len(subjects),
        "recordings_per_subject": dict(sorted(subjects.items())),
        "recordings_per_stage": {str(k): v for k, v in sorted(stages.items())},
        "frames": {
            "total": sum(n_frames.values()),
            "per_recording": _percentiles(list(n_frames.values())),
        },
        "views_byte_identical": views_identical,
        "levels": levels,
        "wrong": wrong,
        "collaboration_frames": {k: status_frames[k] for k in COLLABORATION_STATUSES},
        "problems": problems,
    }


def audit_official(
    official: Mapping[str, OfficialSubdataset],
    temporal: Mapping[str, Mapping[str, list[TemporalSegment]]],
    features: Iterable[str] | None = None,
) -> dict[str, object]:
    """Cross-check the benchmark folder against the temporal files: split, mapping, trimming."""
    problems: list[str] = []
    trims: Counter[int] = Counter()
    per_sub: dict[str, object] = {}
    train_subjects: set[str] = set()
    test_subjects: set[str] = set()
    videos_in_no_split: set[str] = set()
    for name, sub in official.items():
        unsplit = set(sub.ground_truth) - set(sub.train) - set(sub.test)
        videos_in_no_split |= unsplit
        view = int(name[4])
        expected_camera = CAMERA_OF_VIEW[view]
        for split_name, videos in (("train", sub.train), ("test", sub.test)):
            for vid in videos:
                if not vid.endswith(expected_camera):
                    problems.append(
                        f"{name}: {split_name} lists {vid}, not a {expected_camera} video"
                    )
        train_recs = {parse_video_id(v)[0] for v in sub.train}
        test_recs = {parse_video_id(v)[0] for v in sub.test}
        if train_recs & test_recs:
            problems.append(
                f"{name}: recordings in both splits {sorted(train_recs & test_recs)[:3]}"
            )
        train_subjects |= {parse_recording_id(r).subject for r in train_recs}
        test_subjects |= {parse_recording_id(r).subject for r in test_recs}
        labels_in_files = {s.label for r in temporal[name] for s in temporal[name][r]}
        missing = sorted(labels_in_files - set(sub.mapping.values()))
        if missing:
            problems.append(f"{name}: labels absent from mapping.txt {missing[:5]}")
        matched = 0
        for vid, gt in sub.ground_truth.items():
            recording = parse_video_id(vid)[0]
            segments = temporal[name].get(recording)
            if segments is None:
                problems.append(f"{name}: groundTruth for {vid} has no temporal file")
                continue
            full = frame_labels(segments)
            trim = next(
                (
                    t
                    for t in range(0, 11)
                    if len(full) - 2 * t == len(gt) and full[t : len(full) - t] == gt
                ),
                None,
            )
            if trim is None:
                problems.append(
                    f"{name}: groundTruth {vid} is not a symmetric trim of the temporal file"
                )
            else:
                trims[trim] += 1
                matched += 1
        per_sub[name] = {
            "classes": len(sub.mapping),
            "train_videos": len(sub.train),
            "test_videos": len(sub.test),
            "ground_truth_files": len(sub.ground_truth),
            "ground_truth_matching_trimmed_temporal": matched,
            "videos_in_no_split": len(unsplit),
        }
    feature_report = None
    if features is not None:
        annotated = {video_id(r, v) for r in temporal["view0_lh_pt"] for v in VIEWS}
        available = set(features)
        feature_report = {
            "files": len(available),
            "without_temporal_annotation": sorted(available - annotated),
            "annotated_without_features": sorted(annotated - available),
        }
        if annotated - available:
            problems.append(f"{len(annotated - available)} annotated videos have no feature file")
    return {
        "subdatasets": per_sub,
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "subject_disjoint": not (train_subjects & test_subjects),
        "videos_in_no_split": sorted(videos_in_no_split),
        "ground_truth_trim_frames": {str(k): v for k, v in sorted(trims.items())},
        "features": feature_report,
        "problems": problems,
    }


def audit_videos(
    rgb_dir: Path,
    temporal: Mapping[str, Mapping[str, list[TemporalSegment]]],
) -> dict[str, object]:
    """Probe the annotated recordings' mp4s (needs PyAV) and compare frame counts with the labels."""
    from sop_monitor.video import probe

    reference = temporal["view0_lh_pt"]
    on_disk = {p.stem: p for p in rgb_dir.rglob("*.mp4")}
    problems: list[str] = []
    per_video: dict[str, object] = {}
    pairing_complete = 0
    streams: Counter[tuple[str, int, int, float]] = Counter()
    diffs: Counter[int] = Counter()
    total_seconds = 0.0
    for recording in sorted(reference):
        present = [view for view in VIEWS if video_id(recording, view) in on_disk]
        if len(present) == len(VIEWS):
            pairing_complete += 1
        else:
            problems.append(f"{recording}: views on disk {present}")
        n_labels = n_frames_of(reference[recording])
        for view in present:
            vid = video_id(recording, view)
            info = probe(on_disk[vid], count_if_missing=False)
            streams[(info.codec, info.width, info.height, round(info.fps, 3))] += 1
            diffs[info.n_frames - n_labels] += 1
            total_seconds += info.n_frames / info.fps if info.fps else 0.0
            per_video[vid] = {**info.to_dict(), "label_frames": n_labels}
            if info.n_frames < n_labels:
                problems.append(f"{vid}: {info.n_frames} frames but labels cover {n_labels}")
    return {
        "root": str(rgb_dir),
        "mp4_on_disk": len(on_disk),
        "annotated_videos_found": len(per_video),
        "recordings_with_all_three_views": pairing_complete,
        "streams": [
            {"codec": c, "width": w, "height": h, "fps": f, "count": n}
            for (c, w, h, f), n in sorted(streams.items(), key=lambda kv: -kv[1])
        ],
        "annotated_hours": total_seconds / 3600,
        "video_frames_minus_label_frames": {str(k): v for k, v in sorted(diffs.items())},
        "per_video": per_video,
        "problems": problems,
    }


def audit_havid(
    temporal_zip: Path,
    official_zip: Path | None = None,
    rgb_dir: Path | None = None,
) -> dict[str, object]:
    temporal, collaboration = load_temporal_zip(temporal_zip)
    official = load_official_zip(official_zip) if official_zip is not None else None
    test_recordings = (
        {parse_video_id(v)[0] for sub in official.values() for v in sub.test} if official else None
    )
    payload: dict[str, object] = {
        "dataset": "HA-ViD",
        "license": "CC BY-NC 4.0",
        "sources": {
            "temporal_zip": str(temporal_zip),
            "official_zip": str(official_zip) if official_zip else None,
            "rgb_dir": str(rgb_dir) if rgb_dir else None,
        },
        "frame_semantics": {
            "temporal_rows": "start end label, inclusive 0-based frame indices, contiguous from 0",
            "official_ground_truth": f"temporal per-frame labels trimmed by {OFFICIAL_TRIM} frames at each end",
            "wrong_label": f"{WRONG!r} at primitive-task level (paper: 'wrong')",
            "null_label": f"{NULL!r} = pause",
        },
        "legend": {"verbs": VERBS, "noise": NOISE, "views": VIEW_NAMES, "cameras": CAMERA_OF_VIEW},
        "annotations": audit_annotations(temporal, collaboration, test_recordings),
    }
    if official is not None:
        payload["official_split"] = audit_official(
            official,
            temporal,
            feature_videos(official_zip),  # type: ignore[arg-type]
        )
    if rgb_dir is not None:
        payload["videos"] = audit_videos(rgb_dir, temporal)
    payload["problems"] = [
        f"{section}: {problem}"
        for section in ("annotations", "official_split", "videos")
        if section in payload
        for problem in payload[section]["problems"]  # type: ignore[index]
    ]
    return payload


# ---- frozen split -----------------------------------------------------------------------------


def freeze_havid_splits(
    official_zip: Path, temporal_zip: Path, out_dir: Path, seed: int = 0, n_val: int = 6
) -> dict[str, object]:
    """Freeze ``{train,val,test}.csv``, ``test_sha256.txt`` and ``manifest.json`` (spec 4.1).

    ``test`` is the official test set: its subjects, every annotated recording of theirs, all three
    views. ``val`` holds ``n_val`` subjects drawn with ``seed`` from the official train subjects
    that have the fewest annotated recordings, so the recording-rich subjects stay in ``train``.
    Annotated recordings that appear in neither official bundle are excluded from every split and
    listed in the manifest.
    """
    official = load_official_zip(official_zip, ["view0_lh_pt"])["view0_lh_pt"]
    temporal, _ = load_temporal_zip(temporal_zip)
    recordings = sorted(temporal["view0_lh_pt"])
    subject_of = {r: parse_recording_id(r).subject for r in recordings}
    test_recordings = {parse_video_id(v)[0] for v in official.test}
    train_recordings = {parse_video_id(v)[0] for v in official.train}
    excluded = sorted(set(recordings) - test_recordings - train_recordings)
    test_subjects = {subject_of[r] for r in test_recordings}
    train_subjects = {subject_of[r] for r in train_recordings}
    if test_subjects & train_subjects:
        raise ValueError(f"official split shares subjects {sorted(test_subjects & train_subjects)}")
    per_subject = Counter(subject_of[r] for r in train_recordings)
    fewest = min(per_subject.values())
    candidates = sorted(s for s, n in per_subject.items() if n == fewest)
    if n_val > len(candidates):
        raise ValueError(
            f"n_val={n_val} but only {len(candidates)} train subjects have {fewest} recordings"
        )
    val_subjects = set(random.Random(seed).sample(candidates, n_val))
    rows: dict[str, list[tuple[str, str, str, int]]] = {name: [] for name in SPLIT_NAMES}
    for recording in recordings:
        if recording in excluded:
            continue
        subject = subject_of[recording]
        if recording in test_recordings:
            split = "test"
        elif subject in val_subjects:
            split = "val"
        else:
            split = "train"
        for view in VIEWS:
            rows[split].append((video_id(recording, view), subject, recording, view))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "dataset": "HA-ViD",
        "license": "CC BY-NC 4.0",
        "split_rule": (
            "test = official test subjects (splits/test.split1.bundle); val = n_val subjects drawn "
            "with the seed from the official train subjects with the fewest recordings; train = rest; "
            "all three views of a recording share its split"
        ),
        "seed": seed,
        "n_val": n_val,
        "sources": {"official_zip": official_zip.name, "temporal_zip": temporal_zip.name},
        "excluded_recordings": excluded,
        "splits": {},
    }
    for name in SPLIT_NAMES:
        with (out_dir / f"{name}.csv").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("video_id,subject,recording,view\n")
            for vid, subject, recording, view in rows[name]:
                handle.write(f"{vid},{subject},{recording},{view}\n")
        manifest["splits"][name] = {  # type: ignore[index]
            "videos": len(rows[name]),
            "recordings": len({row[2] for row in rows[name]}),
            "subjects": sorted({row[1] for row in rows[name]}),
        }
    digest = hash_test_list(row[0] for row in rows["test"])
    (out_dir / "test_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    manifest["test_sha256"] = digest
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
