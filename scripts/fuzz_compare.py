#!/usr/bin/env python3
"""Differential fuzzing: generate queries and compare this engine to DuckDB.

The generator stays inside the supported SQL subset, but varies filters,
aggregates, DISTINCT, inner/left joins, IN/BETWEEN, ordering, limits, and
offsets. The same generated queries are used by the pytest suite, so this file
is both a reusable generator and a larger standalone harness.

Example:
    ~/qe-env/bin/python scripts/fuzz_compare.py --queries 500 --seed 7

Pass --threads N to route every generated query through the parallel scan so
the differential sweep also covers that path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("QE_CLI", str(ROOT / "build" / "qe-cli")))
TABLE_NAMES = [
    "sales",
    "customers",
    "products",
    "edge_cases",
    "edge_left",
    "edge_right",
    "multi_left",
    "multi_right",
    "empty_table",
]

REGIONS = ["North", "South", "East", "West", "Central", "North-East", "Islands", "Remote"]
CHANNELS = ["online", "retail", "partner", "wholesale"]
STATUSES = ["completed", "returned", "pending", "cancelled"]
COUNTRIES = ["NP", "IN", "US", "GB", "DE", "JP", "AU", "BR"]
CATEGORIES = ["books", "electronics", "garden", "grocery", "sports", "toys"]
DATES = [
    "2024-01-01",
    "2024-01-15",
    "2024-02-01",
    "2024-03-01",
    "2024-06-01",
    "2024-12-31",
    "2025-01-01",
]
SALES_COLUMNS = [
    "sale_id",
    "customer_id",
    "product_id",
    "region",
    "channel",
    "status",
    "quantity",
    "unit_price",
    "discount",
    "amount",
    "sale_date",
]


def _sales_predicate(rng: random.Random, prefix: str = "") -> str:
    """Return one predicate over the sales table."""
    column = rng.choice(
        ["amount", "quantity", "region", "channel", "status", "sale_date", "customer_id"]
    )
    p = prefix
    if column == "amount":
        return rng.choice(
            [
                f"{p}amount > {rng.choice([0, 10, 50, 100, 500, 1000])}",
                f"{p}amount <= {rng.choice([-10, 0, 10, 50, 100])}",
                f"{p}amount IS NULL",
                f"{p}amount IS NOT NULL",
                f"{p}amount BETWEEN {rng.choice([0, 5, 20])} AND {rng.choice([50, 100, 500])}",
            ]
        )
    if column == "quantity":
        return rng.choice(
            [
                f"{p}quantity >= {rng.randint(0, 8)}",
                f"{p}quantity = {rng.randint(0, 10)}",
                f"{p}quantity IN ({', '.join(str(rng.randint(0, 10)) for _ in range(rng.randint(1, 3)))})",
                f"{p}quantity IS NULL",
            ]
        )
    if column == "region":
        value = rng.choice(REGIONS)
        return rng.choice(
            [
                f"{p}region = '{value}'",
                f"{p}region <> '{value}'",
                f"{p}region IN ('{value}', '{rng.choice(REGIONS)}')",
                f"{p}region IS NOT NULL",
            ]
        )
    if column == "channel":
        return f"{p}channel = '{rng.choice(CHANNELS)}'"
    if column == "status":
        return f"{p}status = '{rng.choice(STATUSES)}'"
    if column == "sale_date":
        return rng.choice(
            [
                f"{p}sale_date >= '{rng.choice(DATES)}'",
                f"{p}sale_date < '{rng.choice(DATES)}'",
                f"{p}sale_date BETWEEN '{rng.choice(DATES)}' AND '{rng.choice(DATES)}'",
                f"{p}sale_date IS NOT NULL",
            ]
        )
    return f"{p}customer_id <= {rng.randint(1, 400)}"


def _sales_where(rng: random.Random, prefix: str = "") -> str:
    count = rng.randint(0, 3)
    if count == 0:
        return ""
    predicates = [f"({_sales_predicate(rng, prefix)})" for _ in range(count)]
    joiner = "AND" if rng.random() < 0.72 else "OR"
    return " WHERE " + f" {joiner} ".join(predicates)


def _edge_where(rng: random.Random, prefix: str = "") -> str:
    kind = rng.choice(["id", "label", "category", "amount", "flag", "none"])
    p = prefix
    if kind == "none":
        return ""
    if kind == "id":
        return f" WHERE {p}id {rng.choice(['=', '<=', '>='])} {rng.randint(0, 6)}"
    if kind == "label":
        return f" WHERE {p}label {rng.choice(['=', '<>'])} '{rng.choice(['x', 'z', ''])}'"
    if kind == "category":
        return f" WHERE {p}category = '{rng.choice(['a', 'b'])}'"
    if kind == "amount":
        return f" WHERE {p}amount BETWEEN {rng.choice([-5, 0])} AND {rng.choice([7, 50, 100])}"
    return f" WHERE {p}flag = {rng.choice(['true', 'false'])}"


def _aggregate() -> str:
    return random.choice(
        [
            "COUNT(*)",
            "COUNT(amount)",
            "SUM(amount)",
            "AVG(amount)",
            "MIN(amount)",
            "MAX(amount)",
            "SUM(quantity)",
            "AVG(quantity)",
            "MIN(unit_price)",
            "MAX(unit_price)",
        ]
    )


def _make_query(rng: random.Random) -> str:
    kind = rng.choices(
        [
            "filter",
            "aggregate",
            "distinct",
            "inner_join",
            "join_aggregate",
            "left_join",
            "topk",
            "offset",
            "set_filter",
        ],
        weights=[20, 20, 12, 15, 10, 10, 8, 5, 10],
    )[0]

    if kind == "filter":
        columns = rng.sample(SALES_COLUMNS, rng.randint(1, 4))
        return (
            f"SELECT {', '.join(columns)} FROM sales{_sales_where(rng)} "
            f"ORDER BY sale_id LIMIT {rng.randint(1, 25)}"
        )

    if kind == "aggregate":
        group = rng.choice(["region", "channel", "status"])
        aggregates = [_aggregate() for _ in range(rng.randint(1, 4))]
        having = ""
        if rng.random() < 0.45:
            having = f" HAVING COUNT(*) > {rng.randint(0, 5)}"
        return (
            f"SELECT {group}, {', '.join(aggregates)} FROM sales{_sales_where(rng)} "
            f"GROUP BY {group}{having} ORDER BY {group} LIMIT {rng.randint(1, 20)}"
        )

    if kind == "distinct":
        columns = rng.sample(SALES_COLUMNS, rng.randint(1, 3))
        return (
            f"SELECT DISTINCT {', '.join(columns)} FROM sales{_sales_where(rng)} "
            f"ORDER BY {', '.join(columns)} LIMIT {rng.randint(1, 20)}"
        )

    if kind == "inner_join":
        columns = rng.sample(
            ["s.sale_id", "s.amount", "s.region", "c.country", "c.segment"], rng.randint(1, 4)
        )
        return (
            f"SELECT {', '.join(columns)} FROM sales s "
            "JOIN customers c ON s.customer_id = c.customer_id"
            f"{_sales_where(rng, 's.')} "
            f"ORDER BY s.sale_id LIMIT {rng.randint(1, 20)}"
        )

    if kind == "join_aggregate":
        aggregates = [
            _aggregate()
            .replace("amount", "s.amount")
            .replace("quantity", "s.quantity")
            .replace("unit_price", "s.unit_price")
            for _ in range(rng.randint(1, 3))
        ]
        where = _sales_where(rng, "s.")
        return (
            "SELECT c.country, "
            + ", ".join(aggregates)
            + " FROM sales s JOIN customers c ON s.customer_id = c.customer_id"
            + where
            + " GROUP BY c.country ORDER BY c.country"
        )

    if kind == "left_join":
        filter_clause = ""
        if rng.random() < 0.5:
            filter_clause = rng.choice(
                [
                    " WHERE l.id <= 3",
                    " WHERE l.id IS NOT NULL",
                    " WHERE r.right_value IS NULL",
                    " WHERE r.right_value IS NOT NULL",
                    " WHERE l.left_value >= 'l2'",
                ]
            )
        return (
            "SELECT l.id, l.left_value, r.right_value FROM edge_left l "
            "LEFT JOIN edge_right r ON l.id = r.id"
            + filter_clause
            + " ORDER BY l.id, l.left_value, r.right_value LIMIT "
            + str(rng.randint(1, 10))
        )

    if kind == "topk":
        threshold = rng.choice([0, 10, 100, 500, 1000])
        return (
            f"SELECT sale_id, amount FROM sales WHERE amount > {threshold} "
            f"ORDER BY amount DESC, sale_id LIMIT {rng.randint(1, 20)}"
        )

    if kind == "offset":
        return (
            f"SELECT sale_id, region FROM sales{_sales_where(rng)} "
            f"ORDER BY sale_id LIMIT {rng.randint(1, 10)} OFFSET {rng.randint(0, 15)}"
        )

    # IN / BETWEEN / NOT variants mixed with normal predicates.
    values = ", ".join(str(rng.randint(0, 10)) for _ in range(rng.randint(1, 3)))
    amount_low = rng.choice([0, 5, 20])
    amount_high = rng.choice([50, 100, 500])
    negate = rng.choice(["", "NOT "])
    return (
        f"SELECT sale_id, quantity, amount FROM sales WHERE "
        f"quantity {negate}IN ({values}) AND amount BETWEEN {amount_low} AND {amount_high} "
        f"ORDER BY sale_id LIMIT {rng.randint(1, 20)}"
    )


def generate_queries(seed: int = 2026, count: int = 50) -> list[str]:
    """Generate deterministic, supported SQL queries for differential testing."""
    rng = random.Random(seed)
    queries: list[str] = []
    seen: set[str] = set()
    while len(queries) < count:
        query = _make_query(rng)
        if query not in seen:
            seen.add(query)
            queries.append(query)
    return queries


def normalize(value):
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return value


def same_value(actual, expected) -> bool:
    actual = normalize(actual)
    expected = normalize(expected)
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, int) and isinstance(expected, int):
            return actual == expected
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)
    return actual == expected


def open_views(data_dir: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    for name in TABLE_NAMES:
        path = (data_dir / f"{name}.parquet").as_posix()
        conn.execute(f"CREATE VIEW \"{name}\" AS SELECT * FROM read_parquet('{path}')")
    return conn


def run_qe(data_dir: Path, query: str, threads: int = 1) -> tuple[list[list], str]:
    args = [str(BINARY)]
    for name in TABLE_NAMES:
        args += ["--table", f"{name}={data_dir / (name + '.parquet')}"]
    args += ["--query", query, "--format", "json", "--no-stats"]
    if threads > 1:
        args += ["--threads", str(threads)]
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "qe-cli failed")
    return json.loads(completed.stdout)["rows"], completed.stderr


def compare_query(conn, data_dir: Path, query: str, threads: int = 1) -> str | None:
    try:
        actual, _ = run_qe(data_dir, query, threads)
    except RuntimeError as error:
        return f"qe failed: {error}"
    expected = [normalize(row) for row in conn.execute(query).fetchall()]
    if len(actual) != len(expected):
        return f"row count qe={len(actual)} duckdb={len(expected)}"
    for row_number, (actual_row, expected_row) in enumerate(zip(actual, expected), 1):
        if len(actual_row) != len(expected_row):
            return f"column count differs in row {row_number}"
        for column_number, (actual_value, expected_value) in enumerate(
            zip(actual_row, expected_row), 1
        ):
            if not same_value(actual_value, expected_value):
                return (
                    f"value differs at row {row_number}, column {column_number}: "
                    f"qe={actual_value!r} duckdb={expected_value!r}"
                )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--rows", type=int, default=5000, help="rows when generating data")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="run every query with --threads N so the sweep covers the parallel scan",
    )
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args(argv)

    if not BINARY.exists():
        print(f"error: {BINARY} does not exist; run scripts/build.sh first", file=sys.stderr)
        return 1

    temp_dir: tempfile.TemporaryDirectory | None = None
    if args.data_dir:
        data_dir = args.data_dir
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="qe-fuzz-")
        data_dir = Path(temp_dir.name)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "generate_data.py"),
            "--out",
            str(data_dir),
            "--rows",
            str(args.rows),
            "--customers",
            "400",
            "--products",
            "80",
            "--row-group-size",
            "1024",
            "--seed",
            "424242",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

    conn = open_views(data_dir)
    queries = generate_queries(args.seed, args.queries)
    failures = 0
    attempted = 0
    try:
        for index, query in enumerate(queries, 1):
            attempted += 1
            problem = compare_query(conn, data_dir, query, args.threads)
            if problem:
                failures += 1
                print(f"FAIL [{index}/{len(queries)}] {problem}\n  {query}\n", file=sys.stderr)
                if not args.show_all and failures >= 10:
                    print("stopping after 10 failures", file=sys.stderr)
                    break
            elif args.show_all:
                print(f"ok [{index}/{len(queries)}] {query}")
    finally:
        conn.close()
        if temp_dir is not None:
            temp_dir.cleanup()

    passed = attempted - failures
    print(f"fuzz: {passed}/{attempted} queries matched DuckDB ({failures} failures)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
