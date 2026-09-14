"""Every Mermaid block in the public docs uses a type GitHub renders and high-contrast classes."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "MODEL_CARD.md", ROOT / "reports" / "README.md"]
GITHUB_TYPES = {
    "flowchart",
    "graph",
    "sequenceDiagram",
    "stateDiagram-v2",
    "classDiagram",
    "erDiagram",
    "gitGraph",
}
FENCE = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)


def _blocks() -> list[tuple[str, str]]:
    found = []
    for path in [*DOCS, *sorted((ROOT / "docs").rglob("*.md"))]:
        for body in FENCE.findall(path.read_text(encoding="utf-8")):
            found.append((path.relative_to(ROOT).as_posix(), body))
    return found


def test_mermaid_blocks_use_github_supported_types_and_contrast_classes() -> None:
    blocks = _blocks()
    assert blocks, "no mermaid blocks found"
    for where, body in blocks:
        first = next(line.strip() for line in body.splitlines() if line.strip())
        kind = first.split()[0]
        assert kind in GITHUB_TYPES, f"{where}: {kind!r} may not render on GitHub"
        for line in body.splitlines():
            if line.strip().startswith("classDef"):
                assert "color:" in line, f"{where}: classDef without an explicit text colour"
        assert body.count("\n") < 60, f"{where}: a diagram this long will not read on GitHub"
