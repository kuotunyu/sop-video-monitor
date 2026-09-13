"""Repository hygiene (spec 3.3, ADR 0001): no dataset files or secrets are tracked by git.

Runs only inside a git checkout (CI and local); it inspects ``git ls-files``, never the working
tree, so the ignored local data never influences the result.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_UNDER_DATA = {"data/README.md", "data/manifest.json"}
DATA_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".zip",
    ".tar",
    ".gz",
    ".jpg",
    ".jpeg",
    ".png",
    ".h5",
}


def _tracked() -> list[str]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [p for p in result.stdout.decode("utf-8").split("\0") if p]


def test_no_dataset_or_cache_files_are_tracked() -> None:
    tracked = _tracked()
    under_data = [p for p in tracked if p.startswith("data/") and p not in ALLOWED_UNDER_DATA]
    under_artifacts = [p for p in tracked if p.startswith("artifacts/")]
    binaries = [
        p
        for p in tracked
        if Path(p).suffix.lower() in DATA_SUFFIXES and not p.startswith("docs/assets/")
    ]
    assert under_data == [], f"dataset files tracked under data/: {under_data[:5]}"
    assert under_artifacts == [], f"artifacts tracked: {under_artifacts[:5]}"
    assert binaries == [], f"binary data or weights tracked: {binaries[:5]}"


def test_no_access_links_or_env_files_are_tracked() -> None:
    tracked = _tracked()
    assert not [p for p in tracked if Path(p).name.startswith(".env") and p != ".env.example"]
    suspicious = []
    for path in tracked:
        if Path(path).suffix.lower() not in {
            ".md",
            ".py",
            ".json",
            ".yml",
            ".yaml",
            ".toml",
            ".txt",
            ".csv",
        }:
            continue
        text = (ROOT / path).read_text(encoding="utf-8", errors="ignore")
        if "dropbox.com/scl/" in text or "rlkey=" in text:
            suspicious.append(path)
    assert suspicious == [], f"files containing a Dropbox share link: {suspicious}"
