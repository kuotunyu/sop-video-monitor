"""Subject-wise split freezing (spec section 4.1).

The split is rebuilt from raw HA-ViD metadata with a fixed seed (30 subjects ->
train 18 / val 4 / test 8, proposed). Every video of a subject and the three views
of one assembly always land in the same split. ``splits/{train,val,test}.csv`` and
``splits/test_sha256.txt`` are committed; CI recomputes the hash and fails on drift.

W0 implements only the hash contract. The split builder needs the dataset's
subject metadata and is stubbed until W1.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def hash_test_list(video_ids: Iterable[str]) -> str:
    """Return the SHA-256 hex digest of a frozen test list.

    Contract: identifiers are stripped of surrounding whitespace, sorted by plain
    code-point order, joined with ``"\\n"`` plus a trailing newline, and encoded as
    UTF-8. Input order therefore never changes the digest. Empty identifiers and
    duplicates are rejected because either would silently change the test set size.
    """
    ids = [video_id.strip() for video_id in video_ids]
    if not ids:
        raise ValueError("test list is empty")
    if any(not video_id for video_id in ids):
        raise ValueError("test list contains an empty identifier")
    if len(set(ids)) != len(ids):
        raise ValueError("test list contains duplicate identifiers")
    payload = "\n".join(sorted(ids)) + "\n"
    return sha256(payload.encode("utf-8")).hexdigest()


def verify_test_list(video_ids: Iterable[str], expected_sha256: str) -> bool:
    """True when ``video_ids`` hash to ``expected_sha256`` (case-insensitive hex)."""
    return hash_test_list(video_ids) == expected_sha256.strip().lower()


def write_test_sha256(path: Path, video_ids: Iterable[str]) -> str:
    """Write the digest of ``video_ids`` to ``path`` (one line) and return it."""
    digest = hash_test_list(video_ids)
    path.write_text(digest + "\n", encoding="utf-8")
    return digest


def read_test_sha256(path: Path) -> str:
    """Read a digest written by :func:`write_test_sha256`."""
    return path.read_text(encoding="utf-8").strip().lower()


@dataclass(frozen=True)
class SplitSpec:
    """Subject counts per split and the shuffle seed (spec 4.1; all values proposed)."""

    n_train: int = 18
    n_val: int = 4
    n_test: int = 8
    seed: int = 0


def build_subject_split(
    subject_to_videos: Mapping[str, Sequence[str]], spec: SplitSpec | None = None
) -> dict[str, list[str]]:
    """Assign whole subjects to train/val/test and return ``{split: sorted video ids}``.

    W1 work: needs the HR-SAT metadata to group the three views of one assembly and
    to confirm that subject ids are recoverable (spec 3.4 item 5).
    """
    raise NotImplementedError("W1: subject-wise split builder needs HA-ViD metadata (spec 4.1)")
