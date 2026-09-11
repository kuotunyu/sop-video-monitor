"""IndustReal segment labels -> per-frame label arrays (half-open frames, latest onset wins)."""

from __future__ import annotations

import numpy as np
import pytest

from sop_monitor.industreal import BACKGROUND, Segment, frame_labels, overlap_frame_count


def _seg(action: int, start: int, end: int, video: str = "v") -> Segment:
    return Segment(video, action, f"a{action}", start, end)


def test_frames_are_half_open_and_unlabelled_frames_are_background() -> None:
    labels = frame_labels([_seg(1, 2, 5), _seg(3, 10, 11)], n_frames=12)
    assert labels.dtype == np.int64
    assert labels.tolist() == [-1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 3, -1]
    assert BACKGROUND == -1


def test_overlapping_segments_resolve_to_the_latest_onset() -> None:
    # 1 covers [2, 6), 2 starts at 4 while 1 is still running: frames 4 and 5 go to 2.
    labels = frame_labels([_seg(1, 2, 6), _seg(2, 4, 8)], n_frames=9)
    assert labels.tolist() == [-1, -1, 1, 1, 2, 2, 2, 2, -1]
    # Input order does not matter, only the onset.
    assert frame_labels([_seg(2, 4, 8), _seg(1, 2, 6)], n_frames=9).tolist() == labels.tolist()
    # A short segment nested in a long one wins only while it runs, then the long one resumes.
    nested = frame_labels([_seg(1, 0, 10), _seg(2, 3, 5)], n_frames=10)
    assert nested.tolist() == [1, 1, 1, 2, 2, 1, 1, 1, 1, 1]


def test_equal_onsets_let_the_later_listed_segment_win() -> None:
    assert frame_labels([_seg(1, 0, 3), _seg(2, 0, 2)], n_frames=3).tolist() == [2, 2, 1]


def test_overlap_frame_count_counts_frames_with_two_or_more_labels() -> None:
    assert overlap_frame_count([_seg(1, 2, 6), _seg(2, 4, 8)]) == 2
    assert overlap_frame_count([_seg(1, 2, 6), _seg(2, 6, 8)]) == 0
    assert overlap_frame_count([_seg(1, 0, 10), _seg(2, 3, 5), _seg(3, 4, 6)]) == 3


def test_rejects_segments_outside_the_video_or_from_another_video() -> None:
    with pytest.raises(ValueError, match="beyond"):
        frame_labels([_seg(1, 0, 13)], n_frames=12)
    with pytest.raises(ValueError, match="video"):
        frame_labels([_seg(1, 0, 2), _seg(1, 0, 2, video="w")], n_frames=12)
    assert frame_labels([], n_frames=3).tolist() == [-1, -1, -1]
