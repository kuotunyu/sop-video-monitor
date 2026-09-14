"""Repository hygiene (spec 3.3, ADR 0001): no dataset files or secrets are tracked by git, and every
commit has the single owner as its author with no co-author trailer.

Runs only inside a git checkout (CI and local); it inspects ``git ls-files``, never the working
tree, so the ignored local data never influences the result.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OWNER_EMAIL = "61350295+kuotunyu@users.noreply.github.com"
ALLOWED_UNDER_DATA = {"data/README.md", "data/manifest.json"}
SHARE_LINK_PATTERNS = ("dropbox" + ".com/scl/", "rl" + "key=")
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
    ".gif",
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
        if path == "tests/test_repository_hygiene.py":
            continue  # this file names the patterns it searches for
        text = (ROOT / path).read_text(encoding="utf-8", errors="ignore")
        if any(pattern in text for pattern in SHARE_LINK_PATTERNS):
            suspicious.append(path)
    assert suspicious == [], f"files containing a Dropbox share link: {suspicious}"


def test_every_commit_is_authored_by_the_owner_alone() -> None:
    """The repository is a single-author portfolio: GitHub must list one contributor."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        pytest.skip("pull-request checkouts contain a merge commit synthesised by GitHub")
    log = subprocess.run(
        ["git", "log", "--format=%H%x1f%ae%x1f%B%x1e"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode("utf-8")
    foreign, trailers = [], []
    for record in filter(str.strip, log.split("")):
        sha, author, body = record.strip().split("", 2)
        if author != OWNER_EMAIL:
            foreign.append(f"{sha[:7]} {author}")
        if "co-authored-by:" in body.lower():
            trailers.append(sha[:7])
    assert foreign == [], f"commits by another author: {foreign[:5]}"
    assert trailers == [], f"commits with a co-author trailer: {trailers[:5]}"


def test_every_image_referenced_by_the_docs_exists_and_is_small() -> None:
    """README and docs embed only committed files under docs/assets/, each under 2.5 MB."""
    import re

    pattern = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
    referenced: set[str] = set()
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]:
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            if target.startswith("http"):
                continue
            resolved = (path.parent / target).resolve()
            referenced.add(resolved.relative_to(ROOT).as_posix())
    assert referenced, "no images are embedded"
    for rel in sorted(referenced):
        assert rel.startswith("docs/assets/"), rel
        file = ROOT / rel
        assert file.is_file(), rel
        assert file.stat().st_size <= 2_500_000, f"{rel} is larger than 2.5 MB"
