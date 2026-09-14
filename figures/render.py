"""Render every figure scene with Manim and convert it to a small GIF under ``docs/assets/``.

Run through ``make figures`` (``uv run --group figures python figures/render.py``). Manim writes an
mp4 under ``artifacts/figures/`` (ignored by git); ffmpeg then builds a palette-optimised GIF at
800 px wide and 12 fps without dithering, which keeps the flat dark surface flat: Manim's own GIF
export dithers it into noise and is about a hundred times larger.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES: tuple[tuple[str, str, str], ...] = (
    ("pipeline", "PipelineScene", "pipeline"),
    ("timeline", "TimelineScene", "sop_timeline"),
    ("concepts", "CausalPaddingScene", "causal_padding"),
    ("concepts", "LookAheadScene", "look_ahead"),
    ("concepts", "RingBufferScene", "ring_buffer"),
)
RESOLUTION = "960,540"
FPS = 15
GIF_FILTER = (
    "fps=12,scale=800:-1:flags=lanczos,split[s0][s1];"
    "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
    "[s1][p]paletteuse=dither=none:diff_mode=rectangle"
)


def render(module: str, scene: str, media_dir: Path) -> Path:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "manim",
            "-r",
            RESOLUTION,
            "--fps",
            str(FPS),
            "--format",
            "mp4",
            "--media_dir",
            str(media_dir),
            "--disable_caching",
            "-o",
            scene,
            str(ROOT / "figures" / f"{module}.py"),
            scene,
        ],
        check=True,
        cwd=ROOT,
    )
    return media_dir / "videos" / module / f"540p{FPS}" / f"{scene}.mp4"


def to_gif(video: Path, gif: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is not on PATH")
    gif.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-vf", GIF_FILTER, str(gif)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="GIF names to render (default: all)")
    parser.add_argument("--media-dir", type=Path, default=ROOT / "artifacts" / "figures")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "assets")
    args = parser.parse_args()
    for module, scene, name in SCENES:
        if args.only and name not in args.only:
            continue
        video = render(module, scene, args.media_dir)
        gif = args.out_dir / f"{name}.gif"
        to_gif(video, gif)
        print(f"{gif.relative_to(ROOT)}: {gif.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
