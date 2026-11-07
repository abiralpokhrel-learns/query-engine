#!/usr/bin/env python3
"""Write a shields.io endpoint badge from a gcovr JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def color_for(percent: float) -> str:
    if percent >= 80:
        return "brightgreen"
    if percent >= 60:
        return "green"
    if percent >= 40:
        return "yellow"
    return "red"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="gcovr --json-summary output")
    parser.add_argument("output", type=Path, help="shields.io endpoint JSON to write")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    percent = float(summary["line_percent"])
    badge = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{percent:.1f}%",
        "color": color_for(percent),
    }
    args.output.write_text(json.dumps(badge) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
