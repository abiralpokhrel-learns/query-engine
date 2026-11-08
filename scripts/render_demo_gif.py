#!/usr/bin/env python3
"""Render the README demo GIF from a real ``qe-cli`` session.

This is intentionally not a screen recording: the script executes the commands
below, captures their combined stdout/stderr, and renders the captured text as
an animated terminal transcript. The timings in the image are real output from
the machine that generated it and vary by host.

Run from the repository root inside WSL2/Linux after building ``build/qe-cli``:

    python scripts/render_demo_gif.py

Pillow is part of the pinned environment. If the sample Parquet files are
missing, the script generates the same 200k-row dataset used by ``demo.sh``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "demo.gif"
DEFAULT_POSTER = ROOT / "docs" / "assets" / "demo-poster.png"

BACKGROUND = "#0d1117"
WINDOW = "#161b22"
TITLE_BAR = "#21262d"
BORDER = "#30363d"
TEXT = "#c9d1d9"
BRIGHT = "#e6edf3"
DIM = "#8b949e"
GREEN = "#3fb950"
BLUE = "#58a6ff"
AMBER = "#d29922"
RED = "#ff5f56"
YELLOW = "#ffbd2e"
WINDOW_GREEN = "#27c93f"

FONT_SIZE = 14
LINE_HEIGHT = 23
WIDTH = 1120
TITLE_HEIGHT = 42
FOOTER_HEIGHT = 36
PADDING_X = 36
TOP = 64


@dataclass(frozen=True)
class Scene:
    command: tuple[str, ...]
    display: tuple[str, ...]


@dataclass(frozen=True)
class ScreenState:
    lines: tuple[tuple[str, str], ...]
    cursor: int | None
    duration_ms: int


def _font_candidates(bold: bool = False) -> list[Path]:
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    return [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/truetype/liberation2")
        / name.replace("DejaVuSansMono", "LiberationMono"),
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/cour.ttf"),
    ]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _font_candidates(bold):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def ensure_sample_data() -> None:
    if (ROOT / "data" / "sales.parquet").exists():
        return
    print("sample data missing; generating data/sales.parquet", file=sys.stderr)
    subprocess.run(
        [
            sys.executable,
            "scripts/generate_data.py",
            "--out",
            "data",
            "--rows",
            "200000",
            "--customers",
            "20000",
            "--products",
            "5000",
        ],
        cwd=ROOT,
        check=True,
    )


def build_scenes(cli: str) -> list[Scene]:
    display_cli = cli if cli.startswith(("./", "/")) else f"./{cli}"
    query = (
        "SELECT region, COUNT(*) AS orders, SUM(amount) AS revenue "
        "FROM sales WHERE sale_date >= '2024-01-01' "
        "AND sale_date < '2024-02-01' "
        "GROUP BY region ORDER BY revenue DESC"
    )
    return [
        Scene(
            command=(cli, "--version"),
            display=(f"{display_cli} --version",),
        ),
        Scene(
            command=(
                cli,
                "--table",
                "sales=data/sales.parquet",
                "--query",
                query,
                "--format",
                "table",
                "--max-rows",
                "6",
            ),
            display=(
                f"{display_cli} --table sales=data/sales.parquet \\",
                '  --query "SELECT region, COUNT(*) AS orders, SUM(amount) AS revenue',
                "           FROM sales",
                "           WHERE sale_date >= '2024-01-01'",
                "             AND sale_date < '2024-02-01'",
                '           GROUP BY region ORDER BY revenue DESC" \\',
                "  --format table --max-rows 6",
            ),
        ),
    ]


def capture(scene: Scene) -> list[str]:
    completed = subprocess.run(
        scene.command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"error: {' '.join(scene.command)} failed with exit code "
            f"{completed.returncode}\n{completed.stdout}"
        )
    return completed.stdout.rstrip("\n").splitlines()


def wrap_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    return textwrap.wrap(
        line,
        width=width,
        subsequent_indent="  ",
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


def build_states(scenes: list[tuple[Scene, list[str]]], width: int) -> list[ScreenState]:
    char_width = load_font(FONT_SIZE).getlength("M")
    max_chars = max(40, int((width - 2 * PADDING_X - 12) / char_width))
    states: list[ScreenState] = []
    history: list[tuple[str, str]] = []

    states.append(ScreenState((("", "prompt"),), cursor=0, duration_ms=700))

    for scene_index, (scene, output) in enumerate(scenes):
        if scene_index:
            history.append(("", "blank"))
        command_prefix: list[tuple[str, str]] = []
        for line_index, line in enumerate(scene.display):
            style = "cmd" if line_index == 0 else "cont"
            for end in range(0, len(line) + 1, 4):
                partial = line[:end]
                visible = [*history, *command_prefix, (partial, style)]
                states.append(
                    ScreenState(
                        tuple(visible),
                        cursor=len(visible) - 1,
                        duration_ms=55,
                    )
                )
            command_prefix.append((line, style))
        history.extend(command_prefix)
        states.append(ScreenState(tuple(history), cursor=None, duration_ms=250))

        for line in output:
            for wrapped in wrap_line(line, max_chars):
                history.append((wrapped, "output"))
                states.append(ScreenState(tuple(history), cursor=None, duration_ms=105))
            states.append(ScreenState(tuple(history), cursor=None, duration_ms=35))
        states.append(ScreenState(tuple(history), cursor=None, duration_ms=750))

    history.append(("", "prompt"))
    states.append(ScreenState(tuple(history), cursor=len(history) - 1, duration_ms=1200))
    return states


def style_color(line: str, style: str) -> str:
    if style in {"cmd", "cont"}:
        return BRIGHT
    if style != "output":
        return TEXT
    if line.startswith(("timings:", "scan:")):
        return DIM
    if line.startswith("(9 rows"):
        return DIM
    if set(line.replace(" ", "")) == {"-"}:
        return DIM
    if line.startswith("cuda:"):
        if "unavailable" in line:
            return AMBER
        if "enabled" in line:
            return GREEN
    if line.startswith("qe-cli"):
        return BLUE
    if line.startswith("arrow:"):
        return DIM
    return TEXT


def render_frame(state: ScreenState, height: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    body_font = load_font(FONT_SIZE)
    small_font = load_font(11)
    title_font = load_font(12)

    draw.rounded_rectangle(
        (0, 0, WIDTH - 1, height - 1), radius=14, fill=WINDOW, outline=BORDER, width=1
    )
    draw.rounded_rectangle((0, 0, WIDTH - 1, TITLE_HEIGHT + 12), radius=14, fill=TITLE_BAR)
    draw.rectangle((0, TITLE_HEIGHT - 1, WIDTH, TITLE_HEIGHT + 12), fill=TITLE_BAR)
    draw.line((0, TITLE_HEIGHT, WIDTH, TITLE_HEIGHT), fill=BORDER, width=1)

    dot_y = TITLE_HEIGHT // 2
    for x, color in ((22, RED), (42, YELLOW), (62, WINDOW_GREEN)):
        draw.ellipse((x - 6, dot_y - 6, x + 6, dot_y + 6), fill=color)
    draw.text(
        (WIDTH / 2, dot_y),
        "qe-cli · sample workload",
        fill=DIM,
        font=title_font,
        anchor="mm",
    )
    draw.text((WIDTH - 22, dot_y), "CPU build", fill=BLUE, font=title_font, anchor="rm")

    prompt = "~/query-engine$ "
    prompt_width = body_font.getlength(prompt)
    y = TOP
    for index, (line, style) in enumerate(state.lines):
        if style == "blank":
            y += LINE_HEIGHT
            continue
        if style == "prompt":
            draw.text((PADDING_X, y), prompt, fill=GREEN, font=body_font)
            cursor_x = PADDING_X + prompt_width
        elif style == "cmd":
            draw.text((PADDING_X, y), prompt, fill=GREEN, font=body_font)
            draw.text((PADDING_X + prompt_width, y), line, fill=BRIGHT, font=body_font)
            cursor_x = PADDING_X + prompt_width + body_font.getlength(line)
        else:
            indent = PADDING_X + body_font.getlength("  ") if style == "cont" else PADDING_X
            draw.text((indent, y), line, fill=style_color(line, style), font=body_font)
            cursor_x = indent + body_font.getlength(line)
        if state.cursor == index:
            cursor_width = max(6, int(body_font.getlength("M") * 0.62))
            draw.rectangle(
                (cursor_x + 1, y + 2, cursor_x + 1 + cursor_width, y + LINE_HEIGHT - 4),
                fill=BLUE,
            )
        y += LINE_HEIGHT

    draw.line((0, height - FOOTER_HEIGHT, WIDTH, height - FOOTER_HEIGHT), fill=BORDER, width=1)
    draw.text(
        (22, height - FOOTER_HEIGHT / 2),
        "github.com/abiralpokhrel-learns/query-engine",
        fill=DIM,
        font=small_font,
        anchor="lm",
    )
    draw.text(
        (WIDTH - 22, height - FOOTER_HEIGHT / 2),
        "actual CLI output · timings vary by machine",
        fill=DIM,
        font=small_font,
        anchor="rm",
    )
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="build/qe-cli", help="path to the qe-cli executable")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output GIF path")
    parser.add_argument("--poster", type=Path, default=DEFAULT_POSTER, help="static PNG path")
    args = parser.parse_args()

    cli = args.cli
    if not (ROOT / cli).exists():
        raise SystemExit(f"error: {ROOT / cli} does not exist; run scripts/build.sh first")
    ensure_sample_data()

    scenes = build_scenes(cli)
    captured = [(scene, capture(scene)) for scene in scenes]
    states = build_states(captured, WIDTH)

    max_lines = max(len(state.lines) for state in states)
    height = max(720, TOP + max_lines * LINE_HEIGHT + FOOTER_HEIGHT + 24)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    poster = args.poster if args.poster.is_absolute() else ROOT / args.poster
    output.parent.mkdir(parents=True, exist_ok=True)
    poster.parent.mkdir(parents=True, exist_ok=True)

    frames = [
        render_frame(state, height).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        for state in states
    ]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=[state.duration_ms for state in states],
        loop=0,
        optimize=True,
        disposal=2,
    )

    poster_state = next(
        state for state in reversed(states) if state.lines and state.lines[-1][1] == "output"
    )
    render_frame(poster_state, height).save(poster, optimize=True)

    print(
        f"wrote {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output} "
        f"({output.stat().st_size / 1024:.0f} KiB)"
    )
    print(
        f"wrote {poster.relative_to(ROOT) if poster.is_relative_to(ROOT) else poster} "
        f"({poster.stat().st_size / 1024:.0f} KiB)"
    )
    print(f"frames={len(frames)} size={WIDTH}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
