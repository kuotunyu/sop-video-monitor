"""The frozen test-list hash contract (spec 4.1)."""

from pathlib import Path

import pytest

from sop_monitor.splits import (
    SPLIT_NAMES,
    SplitSpec,
    build_subject_split,
    hash_test_list,
    read_test_sha256,
    verify_test_list,
    write_test_sha256,
)

TEST_IDS = ["S03_front_001", "S01_top_004", "S01_front_004"]
# sha256 of "S01_front_004\nS01_top_004\nS03_front_001\n". Changing this constant means the
# hash contract changed and every committed splits/test_sha256.txt must be regenerated.
FROZEN_DIGEST = "3062d5e0520e8c1650afe4a2199eae27ae0b3bff1722039bd096a42073dce263"


def test_hash_matches_frozen_contract() -> None:
    assert hash_test_list(TEST_IDS) == FROZEN_DIGEST


def test_hash_is_order_and_whitespace_independent() -> None:
    assert hash_test_list(reversed(TEST_IDS)) == FROZEN_DIGEST
    assert hash_test_list(f"  {video_id}\n" for video_id in TEST_IDS) == FROZEN_DIGEST


def test_hash_changes_when_the_test_set_changes() -> None:
    assert hash_test_list(TEST_IDS[:2]) != FROZEN_DIGEST
    assert hash_test_list([*TEST_IDS, "S09_left_002"]) != FROZEN_DIGEST


@pytest.mark.parametrize("bad", [[], ["a", ""], ["a", "a"], ["a", " a "]])
def test_hash_rejects_empty_and_duplicate_entries(bad: list[str]) -> None:
    with pytest.raises(ValueError):
        hash_test_list(bad)


def test_verify_accepts_uppercase_and_padded_digest() -> None:
    assert verify_test_list(TEST_IDS, FROZEN_DIGEST.upper() + "\n")
    assert not verify_test_list(TEST_IDS, "0" * 64)


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "test_sha256.txt"
    assert write_test_sha256(path, TEST_IDS) == FROZEN_DIGEST
    assert path.read_text(encoding="utf-8") == FROZEN_DIGEST + "\n"
    assert read_test_sha256(path) == FROZEN_DIGEST


def test_split_builder_keeps_subjects_whole_and_is_seed_determined() -> None:
    assert SPLIT_NAMES == ("train", "val", "test")
    subjects = {f"S{i:02d}": [f"S{i:02d}_front", f"S{i:02d}_top"] for i in range(1, 7)}
    spec = SplitSpec(n_train=3, n_val=1, n_test=2, seed=7)
    first = build_subject_split(subjects, spec)
    second = build_subject_split(dict(reversed(list(subjects.items()))), spec)
    assert first == second
    assert sorted(len(first[name]) for name in SPLIT_NAMES) == [2, 4, 6]
    all_videos = sorted(v for videos in first.values() for v in videos)
    assert all_videos == sorted(v for videos in subjects.values() for v in videos)
    for videos in first.values():
        owners = {video.split("_")[0] for video in videos}
        assert all(f"{owner}_front" in videos and f"{owner}_top" in videos for owner in owners)
    assert build_subject_split(subjects, SplitSpec(3, 1, 2, seed=8)) != first


def test_split_builder_rejects_wrong_subject_count() -> None:
    with pytest.raises(ValueError):
        build_subject_split({"S01": ["S01_front_001"]}, SplitSpec(n_train=1, n_val=1, n_test=1))
