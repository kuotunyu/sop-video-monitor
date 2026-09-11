"""Data audit for what is actually on disk (spec 3.4 W1 checklist, IndustReal + HA-ViD public files).

The audit never trusts a directory or a zip name: every video is probed, every label row is
parsed and compared with the probed frame count, and HA-ViD is reported as *absent* unless
video or annotation files exist. Output is plain JSON so the report and README quote it
instead of hand-typed numbers.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from sop_monitor.industreal import (
    LABEL_FILES,
    frame_labels,
    overlap_frame_count,
    participant_of,
    read_labels,
    segments_by_video,
)
from sop_monitor.splits import SPLIT_NAMES
from sop_monitor.video import probe

HA_VID_VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov"}
HA_VID_ANNOTATION_SUFFIXES = {".csv", ".json", ".xml", ".txt"}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def audit_industreal(root: Path) -> dict[str, object]:
    """Probe every RGB video and cross-check the three label CSVs against them."""
    rgb_dir = root / "rgb"
    videos = {
        path.stem: probe(path, count_if_missing=False) for path in sorted(rgb_dir.glob("*.mp4"))
    }
    labels: dict[str, object] = {}
    problems: list[str] = []
    labelled_videos: set[str] = set()
    for split in SPLIT_NAMES:
        source = root / "labels" / LABEL_FILES[split]
        if not source.is_file():
            problems.append(f"missing label file {source}")
            continue
        segments = read_labels(source)
        by_video = segments_by_video(segments)
        labelled_videos.update(by_video)
        overlap = labelled = covered = 0
        max_ratio = 0.0
        min_ratio = 1.0
        for video_id, video_segments in by_video.items():
            info = videos.get(video_id)
            if info is None:
                problems.append(f"{split}: {video_id} has labels but no rgb video")
                continue
            last = max(s.end_frame for s in video_segments)
            if last > info.n_frames:
                problems.append(
                    f"{split}: {video_id} labels end at {last} > {info.n_frames} frames"
                )
                continue
            ratio = last / info.n_frames
            max_ratio, min_ratio = max(max_ratio, ratio), min(min_ratio, ratio)
            aligned = frame_labels(video_segments, info.n_frames)
            labelled += int((aligned != -1).sum())
            covered += info.n_frames
            overlap += overlap_frame_count(video_segments)
        lengths = sorted(s.end_frame - s.start_frame for s in segments)
        labels[split] = {
            "file": source.name,
            "sha256": sha256_of(source),
            "rows": len(segments),
            "videos": len(by_video),
            "participants": sorted({participant_of(v) for v in by_video}),
            "action_ids": len({s.action_id for s in segments}),
            "min_start_frame": min(s.start_frame for s in segments),
            "segment_frames": {
                "min": lengths[0],
                "p50": lengths[len(lengths) // 2],
                "p95": lengths[int(len(lengths) * 0.95)],
                "max": lengths[-1],
            },
            "video_frames": covered,
            "labelled_frames": labelled,
            "labelled_fraction": labelled / covered if covered else None,
            "frames_with_two_or_more_labels": overlap,
            "last_label_over_frame_count": {"min": min_ratio, "max": max_ratio},
        }
    codecs = Counter((v.codec, v.width, v.height, round(v.fps, 3)) for v in videos.values())
    return {
        "dataset": "IndustReal",
        "root": str(root),
        "license": "Apache-2.0",
        "videos": {
            "count": len(videos),
            "streams": [
                {"codec": c, "width": w, "height": h, "fps": f, "count": n}
                for (c, w, h, f), n in codecs.items()
            ],
            "total_frames": sum(v.n_frames for v in videos.values()),
            "total_hours": sum(v.n_frames / v.fps for v in videos.values() if v.fps) / 3600,
            "frame_count_from_header": all(v.frames_from_header for v in videos.values()),
            "without_labels": sorted(set(videos) - labelled_videos),
            "per_video": {video_id: info.to_dict() for video_id, info in videos.items()},
        },
        "labels": labels,
        "frame_semantics": {
            "label_frame_index": "0-based index into the 10 fps RGB stream",
            "segment_interval": "half-open [start, end)",
            "overlap_rule": "latest onset wins (sop_monitor.industreal.frame_labels)",
        },
        "problems": problems,
    }


def audit_ha_vid_public(root: Path) -> dict[str, object]:
    """Inventory of the public HA-ViD files; videos/annotations are only counted if present."""
    files = [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
    by_suffix = Counter(p.suffix.lower() for p in files)
    videos = [p for p in files if p.suffix.lower() in HA_VID_VIDEO_SUFFIXES]
    annotations = [p for p in files if p.suffix.lower() in HA_VID_ANNOTATION_SUFFIXES]
    return {
        "dataset": "HA-ViD",
        "root": str(root),
        "license": "CC BY-NC 4.0",
        "present": root.is_dir(),
        "files": len(files),
        "by_suffix": dict(sorted(by_suffix.items())),
        "precedence_graphs": sorted(p.name for p in files if p.suffix.lower() == ".owl"),
        "instruction_pdfs": sorted(p.name for p in files if p.suffix.lower() == ".pdf"),
        "videos": len(videos),
        "annotation_files": len(annotations),
        "usable_for_training": bool(videos) and bool(annotations),
        "blocker": None
        if videos and annotations
        else "the public files hold no videos or HR-SAT annotations; "
        "the delivered archives are audited in reports/havid_audit.json",
    }


def build_manifest(entries: Iterable[tuple[str, Path]]) -> dict[str, object]:
    """``data/manifest.json`` content: file names, sizes and SHA-256 only (no links, no content)."""
    files = []
    for dataset, path in entries:
        files.append(
            {
                "dataset": dataset,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )
    return {"files": sorted(files, key=lambda f: (f["dataset"], f["file"]))}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
