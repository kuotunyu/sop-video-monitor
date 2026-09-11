"""IndustReal procedure-step-recognition (PSR) labels: loading, state -> step conversion, audit.

File formats (from the reference ``PSR/psr_utils.py``; one folder per recording):

- ``PSR_labels.csv``: no header, ``<frame>.jpg,<step id>,<description>`` — one row per correctly
  completed step, at the frame where the annotators deemed it completed.
- ``PSR_labels_with_errors.csv``: same rows plus the wrongly executed steps.
- ``PSR_labels_raw.csv``: no header, ``<frame>.jpg,s_0,...,s_10`` — the state of each of the 11
  assembly components at that frame: ``-1`` incorrectly installed, ``0`` absent, ``1`` correct.

Step ids are ``3 * component + {0: install, 1: incorrect install, 2: remove}`` as listed in
``sop/industreal/procedure_info.json`` (Apache-2.0, copied from the reference repository).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from sop_monitor.metrics.online import Completion

N_COMPONENTS = 11
INSTALL, INCORRECT, REMOVE = 0, 1, 2
PSR_FILES = ("PSR_labels.csv", "PSR_labels_with_errors.csv", "PSR_labels_raw.csv")


@dataclass(frozen=True)
class ProcedureStep:
    id: int
    description: str
    install: bool
    state_idx: int
    expected_in_assy: bool
    expected_in_main: bool
    expected_before_subgoal: bool


def load_procedure_info(path: Path) -> list[ProcedureStep]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = [ProcedureStep(**entry) for entry in payload]
    if [s.id for s in steps] != list(range(len(steps))):
        raise ValueError("procedure_info ids must be 0..n-1 in order")
    return steps


def _frame_index(token: str) -> int:
    stem = token.strip()
    if stem.endswith(".jpg"):
        stem = stem[: -len(".jpg")]
    if not stem.isdigit():
        raise ValueError(f"frame token {token!r} is not an integer frame index")
    return int(stem)


def load_psr_labels(path: Path) -> list[Completion]:
    """``PSR_labels*.csv`` -> completions in file order (the reference's ``load_psr_labels``)."""
    out: list[Completion] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or all(not c.strip() for c in row):
                continue
            if len(row) < 2:
                raise ValueError(f"{path.name}: expected frame,id[,description], got {row!r}")
            out.append(Completion(frame=_frame_index(row[0]), step=int(row[1])))
    return out


def load_psr_raw(path: Path) -> list[tuple[int, tuple[int, ...]]]:
    """``PSR_labels_raw.csv`` -> ``[(frame, (s_0, ..., s_10)), ...]`` in file order."""
    out: list[tuple[int, tuple[int, ...]]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row or all(not c.strip() for c in row):
                continue
            states = tuple(int(c) for c in row[1:])
            if any(s not in (-1, 0, 1) for s in states):
                raise ValueError(f"{path.name}: states must be -1/0/1, got {row!r}")
            out.append((_frame_index(row[0]), states))
    return out


def states_to_steps(
    previous: Sequence[int], current: Sequence[int], frame: int, include_errors: bool = True
) -> list[Completion]:
    """Transcription of the reference ``convert_states_to_steps`` for one state transition.

    With ``include_errors=False`` every ``-1`` is first mapped to ``0`` (reference
    ``only_positive_states``), so error steps disappear and a repair from ``-1`` to ``1`` becomes a
    plain install.
    """
    if not include_errors:
        previous = [0 if s == -1 else s for s in previous]
        current = [0 if s == -1 else s for s in current]
    out: list[Completion] = []
    for k, (prev, curr) in enumerate(zip(previous, current, strict=True)):
        if prev == curr or (prev == -1 and curr == 0):
            continue  # unchanged, or undoing something wrong is not completing a step
        if curr == 1:
            step = k * 3 + INSTALL  # from 0 or from -1 (repaired)
        elif curr == -1:
            step = k * 3 + INCORRECT  # from 0, or (unexpected in the data) from 1
        else:  # curr == 0 and prev == 1
            step = k * 3 + REMOVE
        out.append(Completion(frame=frame, step=step))
    return out


def raw_to_steps(
    raw: Sequence[tuple[int, Sequence[int]]], include_errors: bool = True
) -> list[Completion]:
    """Completions implied by consecutive raw state rows (reference ``convert_all_states_to_steps``)."""
    out: list[Completion] = []
    for (_, previous), (frame, current) in pairwise(raw):
        out.extend(states_to_steps(previous, current, frame, include_errors))
    return out


def extract_psr_labels(archives: Sequence[Path], out_dir: Path) -> list[str]:
    """Copy only the ``PSR_labels*.csv`` members of the recording archives into ``out_dir/<rec>/``.

    The archives are 4-10 GB each because they also hold RGB / depth / stereo frames; nothing but
    the three small CSVs per recording is extracted, and the archives are left untouched.
    """
    import zipfile

    recordings: set[str] = set()
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if name not in PSR_FILES:
                    continue
                recording = Path(member).parent.name
                target = out_dir / recording / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                recordings.add(recording)
    return sorted(recordings)


def audit_recording(
    rec_dir: Path, n_frames: int | None, steps: Sequence[ProcedureStep]
) -> dict[str, object]:
    """Cross-check the three PSR files of one recording against each other and the frame count."""
    labels = load_psr_labels(rec_dir / "PSR_labels.csv")
    with_errors = load_psr_labels(rec_dir / "PSR_labels_with_errors.csv")
    raw = load_psr_raw(rec_dir / "PSR_labels_raw.csv")
    from_raw_clean = raw_to_steps(raw, include_errors=False)
    from_raw_errors = raw_to_steps(raw, include_errors=True)
    known = {s.id for s in steps}
    problems: list[str] = []
    if any(c.step not in known for c in with_errors):
        problems.append("step id outside procedure_info")
    if any(len(states) != N_COMPONENTS for _, states in raw):
        problems.append(f"raw rows do not have {N_COMPONENTS} states")
    frames = [c.frame for c in with_errors]
    if frames != sorted(frames):
        problems.append("PSR_labels_with_errors frames are not sorted")
    if n_frames is not None and frames and max(frames) >= n_frames:
        problems.append(f"label frame {max(frames)} beyond the {n_frames}-frame video")
    return {
        "recording": rec_dir.name,
        "n_frames": n_frames,
        "completions": len(labels),
        "completions_with_errors": len(with_errors),
        "error_steps": sum(1 for c in with_errors if c.step % 3 == INCORRECT),
        "remove_steps": sum(1 for c in with_errors if c.step % 3 == REMOVE),
        "raw_rows": len(raw),
        "raw_matches_labels": [(c.frame, c.step) for c in from_raw_clean]
        == [(c.frame, c.step) for c in labels],
        "raw_matches_labels_with_errors": [(c.frame, c.step) for c in from_raw_errors]
        == [(c.frame, c.step) for c in with_errors],
        "first_frame": frames[0] if frames else None,
        "last_frame": frames[-1] if frames else None,
        "problems": problems,
    }
