#!/usr/bin/env python3
"""Render charts from a stored benchmark JSON file (never from hand-typed data)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHART_DIR = ROOT / "benchmarks" / "charts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        type=Path,
        nargs="?",
        default=ROOT / "benchmarks" / "results" / "latest.json",
    )
    parser.add_argument("--out", type=Path, default=CHART_DIR / "latest.png")
    args = parser.parse_args()

    results = json.loads(args.result.read_text(encoding="utf-8"))
    queries = results["queries"]
    labels = [query["id"] for query in queries]
    cpu = np.array([query["cpu"]["median_ms"] for query in queries])
    duck = np.array([query["duckdb"]["median_ms"] for query in queries])
    ratio = cpu / duck

    fig, (ax_time, ax_ratio) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.0, 1.25]}
    )
    positions = np.arange(len(labels))
    width = 0.38
    ax_time.bar(positions - width / 2, cpu, width, label="qe-cli CPU", color="#276678")
    ax_time.bar(positions + width / 2, duck, width, label="DuckDB", color="#d98b3a")
    ax_time.set_yscale("log")
    ax_time.set_ylabel("median wall time (ms, log scale)")
    ax_time.set_xticks(positions)
    ax_time.set_xticklabels(labels, rotation=28, ha="right")
    ax_time.set_title("CPU engine vs DuckDB")
    ax_time.grid(axis="y", which="both", alpha=0.25)
    ax_time.legend(frameon=False)
    for position, value in zip(positions - width / 2, cpu):
        ax_time.annotate(
            f"{value:.1f}", (position, value), textcoords="offset points",
            xytext=(0, 3), ha="center", fontsize=8
        )
    for position, value in zip(positions + width / 2, duck):
        ax_time.annotate(
            f"{value:.1f}", (position, value), textcoords="offset points",
            xytext=(0, 3), ha="center", fontsize=8
        )

    colors = ["#2f8f5b" if value < 1.0 else "#b44a4a" for value in ratio]
    ax_ratio.barh(positions, ratio, color=colors, alpha=0.9)
    ax_ratio.axvline(1.0, color="black", linewidth=1, linestyle="--")
    ax_ratio.set_yticks(positions)
    ax_ratio.set_yticklabels(labels)
    ax_ratio.invert_yaxis()
    ax_ratio.set_xlabel("CPU time / DuckDB time  (< 1 means CPU faster)")
    ax_ratio.set_title("Relative speed")
    ax_ratio.grid(axis="x", alpha=0.25)
    for position, value in zip(positions, ratio):
        ax_ratio.annotate(
            f"{value:.2f}x", (value, position), textcoords="offset points",
            xytext=(4, 0), va="center", fontsize=8
        )

    sales = results["dataset"]["tables"]["sales"]
    fig.suptitle(
        "Verified CPU vs DuckDB benchmark - "
        f"{sales['rows']:,} sales rows, {sales['row_groups']} row groups, "
        f"batch {results['config']['batch_size']:,}",
        fontsize=12,
    )
    footnote = (
        "GPU unavailable in this environment: no transfer or kernel timings are reported. "
        f"Generated {results['generated_at']}."
    )
    fig.text(0.5, 0.01, footnote, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    timestamped = args.out.with_name(f"{Path(args.result).stem}.png")
    if timestamped != args.out:
        fig.savefig(timestamped, dpi=160)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
