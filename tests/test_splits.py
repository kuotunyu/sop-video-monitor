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


def test_split_builder_is_w1_work() -> None:
    assert SPLIT_NAMES == ("train", "val", "test")
    assert SplitSpec().n_train + SplitSpec().n_val + SplitSpec().n_test == 30
    with pytest.raises(NotImplementedError):
        build_subject_split({"S01": ["S01_front_001"]})
