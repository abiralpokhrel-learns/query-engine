#!/usr/bin/env python3
"""Compare a fresh benchmark run with the committed baseline.

The comparison uses the CPU/DuckDB median ratio rather than absolute wall time,
so it remains meaningful on a different CI runner. A query is flagged when its
ratio grows by more than ``--factor`` (3x by default). Queries whose engine
median is below one millisecond are skipped because scheduler noise dominates
those measurements.

Example:
    python scripts/check_benchmark_regressions.py \
        benchmarks/results/latest.json /tmp/benchmark-before.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: cannot read {path}: {exc}") from exc


def ratio(query: dict[str, Any]) -> float | None:
    cpu = query.get("cpu") or {}
    duckdb = query.get("duckdb") or {}
    cpu_ms = cpu.get("median_ms")
    duckdb_ms = duckdb.get("median_ms")
    if not isinstance(cpu_ms, (int, float)) or not isinstance(duckdb_ms, (int, float)):
        return None
    if duckdb_ms <= 0:
        return None
    return float(cpu_ms) / float(duckdb_ms)


def sales_rows(result: dict[str, Any]) -> int | None:
    try:
        return int(result["dataset"]["tables"]["sales"]["rows"])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Committed benchmark JSON")
    parser.add_argument("current", type=Path, help="Fresh benchmark JSON")
    parser.add_argument(
        "--factor",
        type=float,
        default=3.0,
        help="Allowed ratio increase before a query is flagged (default: 3.0)",
    )
    parser.add_argument(
        "--floor-ms",
        type=float,
        default=1.0,
        help="Skip queries whose current engine median is below this value (default: 1.0)",
    )
    args = parser.parse_args()

    baseline = load(args.baseline)
    current = load(args.current)

    baseline_rows = sales_rows(baseline)
    current_rows = sales_rows(current)
    if baseline_rows != current_rows:
        print(
            f"error: dataset sizes differ (baseline={baseline_rows}, current={current_rows}); "
            "benchmark ratios are only comparable on the same dataset",
            file=sys.stderr,
        )
        return 1

    baseline_queries = {query["id"]: query for query in baseline.get("queries", [])}
    current_queries = {query["id"]: query for query in current.get("queries", [])}
    missing = sorted(set(baseline_queries) - set(current_queries))
    if missing:
        print(f"error: current benchmark is missing queries: {', '.join(missing)}", file=sys.stderr)
        return 1

    failures = 0
    print(f"{'query':<20} {'baseline':>9} {'current':>9} {'allowed':>9}  result")
    print(f"{'-' * 20} {'-' * 9} {'-' * 9} {'-' * 9}  {'-' * 16}")
    for query_id, base_query in baseline_queries.items():
        current_query = current_queries[query_id]
        if not current_query.get("verified", False):
            print(f"{query_id:<20} {'-':>9} {'-':>9} {'-':>9}  FAIL (not verified)")
            failures += 1
            continue

        base_ratio = ratio(base_query)
        current_ratio = ratio(current_query)
        cpu_ms = float((current_query.get("cpu") or {}).get("median_ms") or 0.0)
        if base_ratio is None or current_ratio is None:
            print(f"{query_id:<20} {'-':>9} {'-':>9} {'-':>9}  FAIL (missing timing)")
            failures += 1
            continue
        if cpu_ms < args.floor_ms:
            print(
                f"{query_id:<20} {base_ratio:>8.2f}x {current_ratio:>8.2f}x {'-':>9}  skip (sub-ms)"
            )
            continue

        allowed = max(base_ratio * args.factor, base_ratio + 1.0)
        if current_ratio > allowed:
            print(
                f"{query_id:<20} {base_ratio:>8.2f}x {current_ratio:>8.2f}x {allowed:>8.2f}x  "
                "FAIL (regression)"
            )
            failures += 1
        else:
            print(f"{query_id:<20} {base_ratio:>8.2f}x {current_ratio:>8.2f}x {allowed:>8.2f}x  ok")

    if failures:
        print(f"\n{failures} benchmark check(s) failed.", file=sys.stderr)
        return 1
    print("\nAll benchmark ratios are within the allowed range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
