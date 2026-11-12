"""End-to-end correctness tests: the engine must match DuckDB on the same data.

Run with:
    ~/qe-env/bin/python -m pytest tests -q

The tests build their own small Parquet dataset with five row groups so that
predicate pushdown and column pruning can be checked, not just assumed.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("QE_CLI", str(ROOT / "build" / "qe-cli")))


def _cuda_device_available() -> bool:
    if not BINARY.exists():
        return False
    try:
        completed = subprocess.run(
            [str(BINARY), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and "cuda: enabled" in completed.stdout


GPU_AVAILABLE = _cuda_device_available()

sys.path.insert(0, str(ROOT / "scripts"))
from fuzz_compare import generate_queries  # noqa: E402

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


@dataclass
class Run:
    returncode: int
    stdout: str
    stderr: str

    @property
    def payload(self) -> dict:
        return json.loads(self.stdout)

    @property
    def rows(self) -> list[list]:
        return self.payload["rows"]

    @property
    def columns(self) -> list[dict]:
        return self.payload["columns"]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not BINARY.exists():
        pytest.fail(f"{BINARY} does not exist; run scripts/build.sh first")
    out = tmp_path_factory.mktemp("qe-data")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_data.py"),
        "--out",
        str(out),
        "--rows",
        "5000",
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
    return out


@pytest.fixture()
def conn(data_dir: Path):
    con = duckdb.connect()
    for name in TABLE_NAMES:
        path = (data_dir / f"{name}.parquet").as_posix()
        con.execute(f"CREATE VIEW \"{name}\" AS SELECT * FROM read_parquet('{path}')")
    yield con
    con.close()


def run_qe(
    data_dir: Path,
    query: str,
    *extra: str,
    tables: Iterable[str] = TABLE_NAMES,
) -> Run:
    args = [str(BINARY)]
    for name in tables:
        args += ["--table", f"{name}={data_dir / (name + '.parquet')}"]
    args += ["--query", query, "--format", "json", *extra]
    completed = subprocess.run(args, capture_output=True, text=True)
    return Run(completed.returncode, completed.stdout, completed.stderr)


def normalize(value):
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # The JSON writer keeps output standards-compliant by emitting these as
        # strings; DuckDB's Python client returns float values.
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


def assert_matches_duckdb(data_dir: Path, conn, query: str, *extra: str) -> Run:
    run = run_qe(data_dir, query, *extra)
    assert run.returncode == 0, f"qe failed for {query!r}\n{run.stderr}"
    actual = run.rows
    expected = [normalize(row) for row in conn.execute(query).fetchall()]
    assert len(actual) == len(expected), (
        f"row count differs for {query!r}: qe={len(actual)} duckdb={len(expected)}\n"
        f"qe={actual[:5]}\nduckdb={expected[:5]}"
    )
    for row_number, (actual_row, expected_row) in enumerate(zip(actual, expected), 1):
        assert len(actual_row) == len(expected_row), f"column count differs in row {row_number}"
        for column_number, (actual_value, expected_value) in enumerate(
            zip(actual_row, expected_row), 1
        ):
            assert same_value(actual_value, expected_value), (
                f"value differs for {query!r} at row {row_number}, column {column_number}: "
                f"qe={actual_value!r} duckdb={expected_value!r}"
            )
    if expected:
        assert len(run.columns) == len(expected[0])
    return run


def assert_matches_batch_invariant(
    data_dir: Path, query: str, conn, tables: Iterable[str] = TABLE_NAMES
) -> None:
    """Runs a query at several batch sizes and compares every row to DuckDB."""
    expected = [normalize(row) for row in conn.execute(query).fetchall()]
    for batch_size in ("1", "7", "128", "8192"):
        run = run_qe(data_dir, query, "--batch-size", batch_size, "--no-stats", tables=tables)
        assert run.returncode == 0, f"batch {batch_size}: {run.stderr}"
        actual = run.rows
        assert len(actual) == len(expected), (batch_size, len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            for a, e in zip(actual_row, expected_row):
                assert same_value(a, e), (batch_size, actual_row, expected_row)


CORRECTNESS_QUERIES = [
    # Select / filter / order / limit.
    "SELECT sale_id, region, amount FROM sales WHERE amount > 1000 "
    "ORDER BY amount DESC, sale_id LIMIT 10",
    "SELECT sale_id, region FROM sales WHERE amount IS NULL ORDER BY sale_id LIMIT 20",
    "SELECT sale_id FROM sales WHERE amount != 10.5 AND region = 'North' ORDER BY sale_id LIMIT 20",
    "SELECT sale_id FROM sales WHERE NOT (amount IS NOT NULL) ORDER BY sale_id LIMIT 5",
    "SELECT region, amount FROM sales WHERE amount > 4900 ORDER BY sale_date DESC LIMIT 5",
    "SELECT ID, AMOUNT FROM EDGE_CASES ORDER BY ID",
    "SELECT id FROM edge_cases WHERE label = 'O''Brien' ORDER BY id",
    "SELECT id, label FROM edge_cases WHERE label >= 'x' ORDER BY id",
    "SELECT id FROM edge_cases WHERE flag = true ORDER BY id",
    "SELECT id FROM edge_cases WHERE flag = false ORDER BY id",
    "SELECT id FROM edge_cases WHERE id IN (1, 2) ORDER BY id",
    "SELECT id FROM edge_cases WHERE id NOT IN (1, 2) ORDER BY id",
    "SELECT id FROM edge_cases WHERE NOT (id IN (1, 2)) ORDER BY id",
    "SELECT id FROM edge_cases WHERE id IN (1, NULL, 4) ORDER BY id",
    "SELECT id FROM edge_cases WHERE id NOT IN (1, NULL) ORDER BY id",
    "SELECT id, label FROM edge_cases WHERE id BETWEEN 2 AND 4 ORDER BY id",
    "SELECT id FROM edge_cases WHERE id NOT BETWEEN 2 AND 4 ORDER BY id",
    "SELECT id FROM edge_cases WHERE amount BETWEEN 10 AND 100 ORDER BY id",
    "SELECT id FROM edge_cases WHERE id BETWEEN 2 AND 4 AND flag = true ORDER BY id",
    # LIKE: prefix, infix, single-character wildcard, empty pattern, NOT, and
    # the pattern as a projected value. All verified against DuckDB, whose LIKE
    # is case-sensitive like this implementation.
    "SELECT id FROM edge_cases WHERE label LIKE 'x%' ORDER BY id",
    "SELECT id, label FROM edge_cases WHERE label LIKE '%b%' ORDER BY id",
    "SELECT id FROM edge_cases WHERE label LIKE '_' ORDER BY id",
    "SELECT id FROM edge_cases WHERE label LIKE '' ORDER BY id",
    "SELECT id FROM edge_cases WHERE label LIKE '%' ORDER BY id",
    "SELECT id FROM edge_cases WHERE label NOT LIKE 'x%' ORDER BY id",
    "SELECT id, label LIKE 'x%' AS matches FROM edge_cases ORDER BY id",
    "SELECT id FROM edge_cases WHERE label LIKE 'O''Brien' ORDER BY id",
    "SELECT COUNT(*) FROM edge_cases WHERE category LIKE 'a%'",
    "SELECT id FROM edge_cases WHERE label LIKE '%' AND id > 2 ORDER BY id",
    "SELECT id FROM edge_cases WHERE label LIKE NULL",
    # OFFSET and top-K edge cases.
    "SELECT id FROM edge_cases ORDER BY id LIMIT 2 OFFSET 2",
    "SELECT id FROM edge_cases ORDER BY id OFFSET 4",
    "SELECT id FROM edge_cases ORDER BY id LIMIT 2 OFFSET 10",
    "SELECT id FROM edge_cases ORDER BY id LIMIT 0 OFFSET 2",
    "SELECT id, amount FROM edge_cases ORDER BY amount DESC, id LIMIT 3 OFFSET 2",
    "SELECT id FROM edge_cases ORDER BY category, id LIMIT 3",
    "SELECT id FROM edge_cases ORDER BY category DESC, id LIMIT 3",
    "SELECT DISTINCT region FROM sales ORDER BY region",
    "SELECT DISTINCT category, flag FROM edge_cases ORDER BY category, flag",
    "SELECT DISTINCT label FROM edge_cases ORDER BY label",
    "SELECT DISTINCT region FROM sales ORDER BY region LIMIT 2",
    "SELECT DISTINCT region FROM sales ORDER BY 1 DESC LIMIT 3",
    "SELECT DISTINCT amount * 2 AS doubled FROM edge_cases ORDER BY doubled",
    "SELECT DISTINCT * FROM edge_cases ORDER BY id",
    "SELECT DISTINCT region, COUNT(*) FROM sales GROUP BY region ORDER BY region",
    # Expressions and SQL arithmetic.
    "SELECT id, amount * 2 AS doubled, quantity + 3 AS q FROM edge_cases WHERE id <= 3 ORDER BY id",
    "SELECT amount / 2 AS half FROM edge_cases WHERE id = 1",
    "SELECT quantity / 2 AS half FROM edge_cases WHERE id = 1",
    "SELECT id, -quantity AS negated FROM edge_cases WHERE quantity IS NOT NULL ORDER BY id",
    "SELECT 1 + 2 * 3 AS value, 'hello' AS greeting FROM edge_cases WHERE id = 1",
    "SELECT id, amount / quantity AS ratio FROM edge_cases ORDER BY id",
    # Aggregates.
    "SELECT COUNT(*), COUNT(amount), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM sales",
    "SELECT COUNT(*), COUNT(amount), SUM(amount), AVG(amount), MIN(amount), MAX(amount) "
    "FROM edge_cases",
    "SELECT COUNT(*), COUNT(amount), SUM(amount), AVG(amount), MIN(amount), MAX(amount) "
    "FROM edge_cases WHERE id = 4",
    "SELECT COUNT(*), COUNT(value), SUM(value), AVG(value), MIN(value), MAX(value) "
    "FROM empty_table",
    "SELECT MIN(label), MAX(label), MIN(day), MAX(day), MIN(flag), MAX(flag) FROM edge_cases",
    "SELECT region, COUNT(*), SUM(quantity), AVG(quantity), MIN(unit_price), MAX(amount) "
    "FROM sales GROUP BY region ORDER BY region",
    "SELECT region, COUNT(*) FROM sales WHERE status = 'completed' GROUP BY region ORDER BY region",
    "SELECT region, COUNT(*) AS n FROM sales GROUP BY region ORDER BY n DESC, region LIMIT 3",
    "SELECT category, COUNT(*), SUM(amount), AVG(amount), MIN(quantity), MAX(amount) "
    "FROM edge_cases GROUP BY category ORDER BY category",
    "SELECT region, channel, COUNT(*), SUM(amount) FROM sales "
    "GROUP BY region, channel ORDER BY region, channel",
    "SELECT sale_date, COUNT(*), SUM(amount) FROM sales "
    "GROUP BY sale_date ORDER BY sale_date LIMIT 30",
    "SELECT category, COUNT(*) FROM edge_cases WHERE 1 = 0 GROUP BY category ORDER BY category",
    "SELECT region, COUNT(*) FROM sales GROUP BY region HAVING COUNT(*) > 500 ORDER BY region",
    "SELECT region, COUNT(*) AS n FROM sales GROUP BY region HAVING COUNT(*) > 500 ORDER BY region",
    "SELECT region FROM sales GROUP BY region HAVING SUM(amount) > 100000 ORDER BY region",
    "SELECT category, SUM(amount) FROM edge_cases GROUP BY category "
    "HAVING SUM(amount) > 100 ORDER BY category",
    "SELECT category, COUNT(*) FROM edge_cases GROUP BY category "
    "HAVING category <> 'books' AND COUNT(*) > 1 ORDER BY category",
    "SELECT COUNT(*) FROM sales HAVING COUNT(*) > 1",
    "SELECT COUNT(*) FROM sales HAVING COUNT(*) > 1000000",
    "SELECT 1 FROM sales HAVING COUNT(*) > 1",
    "SELECT 1 FROM sales HAVING 1 = 1",
    "SELECT 1 FROM sales HAVING 1 = 0",
    "SELECT COUNT(*) FROM sales HAVING 1 = 1",
    "SELECT 1 FROM empty_table HAVING COUNT(*) = 0",
    "SELECT 1 FROM empty_table HAVING COUNT(*) > 0",
    "SELECT region, COUNT(*) FROM sales GROUP BY region ORDER BY COUNT(*) DESC, region",
    "SELECT region FROM sales GROUP BY region ORDER BY COUNT(*) DESC, region LIMIT 2",
    # Grouping fast paths: dense integer keys (including NULLs), a key span
    # that is too wide for the dense table, low- and high-cardinality strings,
    # and string MIN/MAX.
    "SELECT quantity, COUNT(*), SUM(amount), MIN(amount), MAX(amount) "
    "FROM edge_cases GROUP BY quantity ORDER BY quantity",
    "SELECT category, MIN(label), MAX(label), COUNT(*) "
    "FROM edge_cases GROUP BY category ORDER BY category",
    "SELECT c.name, COUNT(*), SUM(s.amount) FROM sales s "
    "JOIN customers c ON s.customer_id = c.customer_id "
    "GROUP BY c.name ORDER BY c.name",
    # ORDER BY / null ordering.
    "SELECT id, amount FROM edge_cases ORDER BY amount ASC",
    "SELECT id, amount FROM edge_cases ORDER BY amount DESC",
    "SELECT id, category, amount FROM edge_cases ORDER BY category DESC, amount ASC",
    "SELECT id, day FROM edge_cases ORDER BY day DESC",
    "SELECT label FROM edge_cases ORDER BY label",
    "SELECT region, COUNT(*) AS n FROM sales GROUP BY region ORDER BY 2 DESC, 1 LIMIT 2",
    "SELECT * FROM edge_cases WHERE 1 = 0",
    "SELECT * FROM edge_cases ORDER BY id",
    "SELECT * FROM edge_cases LIMIT 0",
    # Dates.
    "SELECT COUNT(*), SUM(amount) FROM sales "
    "WHERE sale_date >= '2024-03-01' AND sale_date < '2024-04-01'",
    "SELECT sale_id, sale_date FROM sales WHERE sale_date < '2024-01-02' ORDER BY sale_id LIMIT 5",
    "SELECT COUNT(*) FROM sales WHERE sale_date = '2024-03-01'",
    "SELECT COUNT(*) FROM sales WHERE sale_date > '2026-12-30'",
    "SELECT id, day FROM edge_cases WHERE day = '2024-02-29' ORDER BY id",
    # Joins.
    "SELECT l.id, l.left_value, r.right_value FROM edge_left l "
    "JOIN edge_right r ON l.id = r.id ORDER BY l.id, l.left_value, r.right_value",
    "SELECT l.id, COUNT(*) FROM edge_left l JOIN edge_right r ON l.id = r.id "
    "GROUP BY l.id ORDER BY l.id",
    "SELECT COUNT(*) FROM edge_left l JOIN edge_right r ON l.id = r.id",
    "SELECT r.id, r.right_value, l.left_value FROM edge_right r "
    "JOIN edge_left l ON r.id = l.id ORDER BY r.id, r.right_value, l.left_value",
    "SELECT l.a, l.b, l.left_value, r.right_value FROM multi_left l "
    "JOIN multi_right r ON l.a = r.a AND l.b = r.b "
    "ORDER BY l.a, l.b, l.left_value, r.right_value",
    "SELECT COUNT(*) FROM multi_left l JOIN multi_right r ON l.a = r.a AND l.b = r.b",
    "SELECT COUNT(*) FROM edge_left l JOIN empty_table e ON l.id = e.id",
    "SELECT COUNT(*) FROM edge_left l JOIN edge_right r ON l.id = r.id WHERE l.id = 2",
    "SELECT l.id, r.right_value FROM edge_left l JOIN edge_right r ON l.id = r.id "
    "WHERE l.id = 2 OR r.id = 3 ORDER BY l.id, r.right_value",
    "SELECT s.sale_id, c.name, s.amount FROM sales s "
    "JOIN customers c ON s.customer_id = c.customer_id "
    "WHERE s.amount > 5000 ORDER BY s.sale_id LIMIT 5",
    "SELECT c.segment, COUNT(*), SUM(s.amount) FROM sales s "
    "JOIN customers c ON s.customer_id = c.customer_id "
    "GROUP BY c.segment ORDER BY c.segment",
    "SELECT p.category, COUNT(*), AVG(s.amount) FROM sales s "
    "JOIN products p ON s.product_id = p.product_id "
    "WHERE s.sale_date < '2024-02-01' GROUP BY p.category ORDER BY p.category",
    "SELECT l.id, r.id FROM edge_left l JOIN edge_right r ON l.id = r.id ORDER BY l.id, r.id",
    # LEFT JOIN.
    "SELECT l.id, l.left_value, r.right_value FROM edge_left l "
    "LEFT JOIN edge_right r ON l.id = r.id ORDER BY l.id, r.right_value",
    "SELECT l.id, l.left_value, r.right_value FROM edge_left l "
    "LEFT JOIN edge_right r ON l.id = r.id WHERE l.id = 2 "
    "ORDER BY l.id, r.right_value",
    "SELECT l.id, r.right_value FROM edge_left l LEFT JOIN edge_right r ON l.id = r.id "
    "WHERE r.right_value IS NULL ORDER BY l.id",
    "SELECT l.id, COUNT(r.right_value) AS matches FROM edge_left l "
    "LEFT JOIN edge_right r ON l.id = r.id GROUP BY l.id ORDER BY l.id",
    "SELECT l.id, e.id FROM edge_left l LEFT JOIN empty_table e ON l.id = e.id ORDER BY l.id",
    "SELECT l.id, r.right_value FROM edge_left l LEFT JOIN edge_right r ON l.id = r.id "
    "ORDER BY l.id, r.right_value LIMIT 3 OFFSET 1",
    "SELECT l.id, r.right_value, e.label FROM edge_left l "
    "LEFT JOIN edge_right r ON l.id = r.id "
    "LEFT JOIN edge_cases e ON l.id = e.id ORDER BY l.id, r.right_value, e.label",
    "SELECT l.a, l.b, l.left_value, r.right_value FROM multi_left l "
    "LEFT JOIN multi_right r ON l.a = r.a AND l.b = r.b "
    "ORDER BY l.a, l.b, l.left_value, r.right_value",
    "SELECT DISTINCT l.id, r.right_value FROM edge_left l "
    "LEFT JOIN edge_right r ON l.id = r.id ORDER BY l.id, r.right_value",
    "SELECT r.right_value, COUNT(l.id) FROM edge_left l "
    "LEFT JOIN edge_right r ON l.id = r.id GROUP BY r.right_value ORDER BY r.right_value",
    # Aliases and quoted identifiers.
    "SELECT amount * 2 AS double_amount FROM edge_cases WHERE id = 1",
    "SELECT region AS r, SUM(amount) AS total FROM sales GROUP BY region "
    "ORDER BY total DESC LIMIT 3",
    'SELECT e.id AS "the id", e.label AS "the label" FROM edge_cases e '
    'WHERE e.id <= 2 ORDER BY "the id"',
]

# Deterministic differential-fuzz queries. The standalone harness can run a
# much larger sweep; this keeps a representative sample in every test run.
CORRECTNESS_QUERIES.extend(generate_queries(seed=20261104, count=30))


@pytest.mark.parametrize("query", CORRECTNESS_QUERIES)
def test_matches_duckdb(data_dir: Path, conn, query: str) -> None:
    assert_matches_duckdb(data_dir, conn, query)


def parse_stats(stderr: str) -> dict[str, int]:
    match = re.search(
        r"scan: rows_scanned=(\d+) rows_after_pushdown=(\d+) "
        r"row_groups_read=(\d+) row_groups_skipped=(\d+) "
        r"row_groups_total=(\d+) batches=(\d+)",
        stderr,
    )
    assert match, stderr
    keys = [
        "rows_scanned",
        "rows_after_pushdown",
        "row_groups_read",
        "row_groups_skipped",
        "row_groups_total",
        "batches",
    ]
    return {key: int(value) for key, value in zip(keys, match.groups())}


def test_row_group_skip_and_row_level_pushdown(data_dir: Path, conn) -> None:
    query = "SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date < '2024-01-15'"
    run = assert_matches_duckdb(data_dir, conn, query)
    stats = parse_stats(run.stderr)
    assert stats["row_groups_total"] == 5
    assert stats["row_groups_skipped"] > 0, stats
    assert stats["row_groups_read"] < stats["row_groups_total"], stats
    assert stats["rows_scanned"] < 5000, stats
    assert stats["rows_after_pushdown"] < stats["rows_scanned"], stats


def test_count_star_uses_parquet_metadata(data_dir: Path, conn) -> None:
    run = assert_matches_duckdb(data_dir, conn, "SELECT COUNT(*) FROM sales")
    assert "rows_from_metadata=5000" in run.stderr, run.stderr
    assert "rows_scanned=0" in run.stderr, run.stderr
    assert "batches=0" in run.stderr, run.stderr


def test_explain_shows_metadata_only_count(data_dir: Path) -> None:
    run = run_qe(data_dir, "SELECT COUNT(*) FROM sales", "--explain", "--no-stats")
    assert run.returncode == 0, run.stderr
    scan_line = next(line for line in run.stdout.splitlines() if "Scan sales" in line)
    assert "columns=[]" in scan_line, run.stdout


def test_explain_shows_pushdown_and_column_pruning(data_dir: Path) -> None:
    query = (
        "SELECT region, SUM(amount) AS total FROM sales "
        "WHERE sale_date < '2024-01-15' GROUP BY region"
    )
    run = run_qe(data_dir, query, "--explain", "--no-stats")
    assert run.returncode == 0, run.stderr
    explain = run.stdout
    assert "Aggregate" in explain and "Scan" in explain
    assert "pushdown=[(sales.sale_date < '2024-01-15')]" in explain
    scan_line = next(line for line in explain.splitlines() if "Scan sales" in line)
    assert "columns=[region, amount, sale_date]" in scan_line
    for unused in ("sale_id", "customer_id", "product_id", "status", "quantity"):
        assert unused not in scan_line


def test_join_column_pruning(data_dir: Path) -> None:
    query = (
        "SELECT c.country, COUNT(*) FROM sales s "
        "JOIN customers c ON s.customer_id = c.customer_id "
        "WHERE s.amount > 100 GROUP BY c.country"
    )
    run = run_qe(data_dir, query, "--explain", "--no-stats")
    assert run.returncode == 0, run.stderr
    scan_lines = [line for line in run.stdout.splitlines() if "Scan " in line]
    sales_line = next(line for line in scan_lines if "Scan s " in line)
    customers_line = next(line for line in scan_lines if "Scan c " in line)
    assert "columns=[customer_id, amount]" in sales_line
    assert "columns=[customer_id, country]" in customers_line


def test_batch_sizes_do_not_change_results(data_dir: Path, conn) -> None:
    query = (
        "SELECT region, COUNT(*), SUM(amount), AVG(quantity) FROM sales "
        "GROUP BY region ORDER BY region"
    )
    expected = [normalize(row) for row in conn.execute(query).fetchall()]
    for batch_size in ("1", "7", "128", "8192"):
        run = run_qe(data_dir, query, "--batch-size", batch_size, "--no-stats")
        assert run.returncode == 0, f"batch {batch_size}: {run.stderr}"
        actual = run.rows
        assert len(actual) == len(expected)
        for actual_row, expected_row in zip(actual, expected):
            for a, e in zip(actual_row, expected_row):
                assert same_value(a, e), (batch_size, actual_row, expected_row)


def assert_matches_threads_invariant(
    data_dir: Path, query: str, conn, tables: Iterable[str] = TABLE_NAMES
) -> None:
    """Runs a query at several thread counts and batch sizes; every run must
    match DuckDB and reproduce the single-threaded rows exactly."""
    expected = [normalize(row) for row in conn.execute(query).fetchall()]
    baseline: list | None = None
    for threads, batch_size in (("1", "8192"), ("2", "8192"), ("4", "7"), ("8", "128")):
        run = run_qe(
            data_dir,
            query,
            "--threads",
            threads,
            "--batch-size",
            batch_size,
            "--no-stats",
            tables=tables,
        )
        assert run.returncode == 0, f"threads {threads}: {run.stderr}"
        actual = run.rows
        assert len(actual) == len(expected), (threads, len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            for a, e in zip(actual_row, expected_row):
                assert same_value(a, e), (threads, actual_row, expected_row)
        if baseline is None:
            baseline = actual
        else:
            assert actual == baseline, f"threads {threads} changed the result order"


THREAD_INVARIANT_QUERIES = [
    # Aggregation, top-k, pushdown, HAVING, DISTINCT, and OFFSET over five
    # row groups of sales data.
    "SELECT region, COUNT(*), SUM(amount), AVG(quantity) FROM sales "
    "GROUP BY region ORDER BY region",
    "SELECT sale_id, amount FROM sales WHERE amount > 1000 ORDER BY amount DESC, sale_id LIMIT 10",
    "SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date < '2024-01-15'",
    "SELECT region, COUNT(*) FROM sales GROUP BY region HAVING COUNT(*) > 500 ORDER BY region",
    "SELECT DISTINCT region, channel FROM sales ORDER BY region, channel",
    "SELECT sale_id FROM sales ORDER BY sale_id LIMIT 5 OFFSET 30",
    # Joins read both sides through (possibly parallel) scans.
    "SELECT c.country, SUM(s.amount) FROM sales s "
    "JOIN customers c ON s.customer_id = c.customer_id "
    "GROUP BY c.country ORDER BY c.country",
    "SELECT l.id, r.right_value FROM edge_left l "
    "LEFT JOIN edge_right r ON l.id = r.id ORDER BY l.id, r.right_value",
    # NULL-heavy grouping and an empty result.
    "SELECT quantity, COUNT(*), SUM(amount) FROM edge_cases GROUP BY quantity ORDER BY quantity",
    "SELECT region FROM sales WHERE 1 = 0 GROUP BY region",
]


@pytest.mark.parametrize("query", THREAD_INVARIANT_QUERIES)
def test_thread_counts_do_not_change_results(data_dir: Path, conn, query: str) -> None:
    assert_matches_threads_invariant(data_dir, query, conn)


def test_scan_stats_are_identical_across_threads(data_dir: Path, conn) -> None:
    query = "SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date < '2024-01-15'"
    stats_by_threads: dict[str, dict[str, int]] = {}
    for threads in ("1", "4"):
        run = assert_matches_duckdb(data_dir, conn, query, "--threads", threads)
        stats_by_threads[threads] = parse_stats(run.stderr)
    assert stats_by_threads["1"] == stats_by_threads["4"]
    stats = stats_by_threads["4"]
    assert stats["row_groups_skipped"] > 0, stats
    assert stats["rows_after_pushdown"] < stats["rows_scanned"], stats


def test_count_star_metadata_with_threads(data_dir: Path, conn) -> None:
    # The metadata-only COUNT(*) path must not change with --threads; it never
    # spawns scan workers because no payload column is decoded.
    run = assert_matches_duckdb(data_dir, conn, "SELECT COUNT(*) FROM sales", "--threads", "4")
    assert "rows_from_metadata=5000" in run.stderr, run.stderr
    assert "rows_scanned=0" in run.stderr, run.stderr
    assert "batches=0" in run.stderr, run.stderr


def test_dense_integer_groups_expand_across_batches(data_dir: Path, conn) -> None:
    path = data_dir / "dense_negative.parquet"
    # Keys start positive, then jump up and later below zero. The dense table
    # must grow in both directions without losing groups or changing results.
    conn.execute(
        f"COPY (SELECT CASE WHEN i < 8 THEN i + 10 "
        f"WHEN i < 16 THEN i + 1000 ELSE -100000 - i END AS k, "
        f"CAST(i AS DOUBLE) AS v FROM range(64) t(i) ORDER BY i) "
        f"TO '{path.as_posix()}' (FORMAT PARQUET)"
    )
    conn.execute(f"CREATE VIEW dense_negative AS SELECT * FROM read_parquet('{path.as_posix()}')")

    cases = [
        ("SELECT sale_id, COUNT(*) FROM sales GROUP BY sale_id ORDER BY sale_id", TABLE_NAMES),
        (
            "SELECT k, COUNT(*), SUM(v) FROM dense_negative GROUP BY k ORDER BY k",
            ["dense_negative"],
        ),
    ]
    for query, tables in cases:
        expected = [normalize(row) for row in conn.execute(query).fetchall()]
        for batch_size in ("1", "7", "128", "8192"):
            run = run_qe(data_dir, query, "--batch-size", batch_size, "--no-stats", tables=tables)
            assert run.returncode == 0, f"batch {batch_size}: {run.stderr}"
            actual = run.rows
            assert len(actual) == len(expected), (batch_size, len(actual), len(expected))
            for actual_row, expected_row in zip(actual, expected):
                for a, e in zip(actual_row, expected_row):
                    assert same_value(a, e), (batch_size, actual_row, expected_row)


def test_wide_integer_keys_fall_back_to_flat_grouping(data_dir: Path, conn) -> None:
    path = data_dir / "wide_keys.parquet"
    # The first batch has a compact key range, then the stream jumps to keys
    # that cannot fit in the dense table. The operator must migrate to the
    # flat hash table without losing groups created by the dense phase.
    conn.execute(
        f"COPY (SELECT CASE WHEN i < 32 THEN i + 1 ELSE 10000000 + i END AS k, "
        f"CAST(i AS DOUBLE) AS v FROM range(64) t(i) ORDER BY i) "
        f"TO '{path.as_posix()}' (FORMAT PARQUET)"
    )
    conn.execute(f"CREATE VIEW wide_keys AS SELECT * FROM read_parquet('{path.as_posix()}')")
    query = "SELECT k, COUNT(*), SUM(v) FROM wide_keys GROUP BY k ORDER BY k"
    run = run_qe(data_dir, query, "--batch-size", "8", "--no-stats", tables=["wide_keys"])
    assert run.returncode == 0, run.stderr
    expected = [normalize(row) for row in conn.execute(query).fetchall()]
    assert len(run.rows) == len(expected)
    for actual_row, expected_row in zip(run.rows, expected):
        for a, e in zip(actual_row, expected_row):
            assert same_value(a, e), (actual_row, expected_row)


def test_integer_join_typed_table_matches_duckdb(data_dir: Path, conn) -> None:
    # Duplicate keys on both sides, a NULL key on each side, an unmatched
    # probe key, and INT64 extremes: the typed single-key table must behave
    # exactly like the serialized-key path, at any batch size.
    build = data_dir / "join_build_typed.parquet"
    probe = data_dir / "join_probe_typed.parquet"
    conn.execute(
        f"COPY (SELECT CASE i WHEN 0 THEN 1 WHEN 1 THEN 1 WHEN 2 THEN 2 WHEN 3 THEN 2 "
        f"WHEN 4 THEN 2 WHEN 5 THEN NULL WHEN 6 THEN -7 "
        f"WHEN 7 THEN 9223372036854775807 ELSE 13 END AS k, i AS v "
        f"FROM range(10) t(i)) TO '{build.as_posix()}' (FORMAT PARQUET)"
    )
    conn.execute(
        f"COPY (SELECT CASE i WHEN 0 THEN 1 WHEN 1 THEN 1 WHEN 2 THEN NULL WHEN 3 THEN 2 "
        f"WHEN 4 THEN 5 WHEN 5 THEN -7 WHEN 6 THEN 999 "
        f"WHEN 7 THEN 9223372036854775807 ELSE 13 END AS k, i AS w "
        f"FROM range(9) t(i)) TO '{probe.as_posix()}' (FORMAT PARQUET)"
    )
    conn.execute(
        f"CREATE VIEW join_build_typed AS SELECT * FROM read_parquet('{build.as_posix()}')"
    )
    conn.execute(
        f"CREATE VIEW join_probe_typed AS SELECT * FROM read_parquet('{probe.as_posix()}')"
    )

    tables = ["join_build_typed", "join_probe_typed"]
    assert_matches_batch_invariant(
        data_dir,
        "SELECT p.w, b.v FROM join_probe_typed p JOIN join_build_typed b "
        "ON p.k = b.k ORDER BY p.w, b.v",
        conn,
        tables,
    )
    # The left join also exercises the unmatched-row and NULL-key path.
    assert_matches_batch_invariant(
        data_dir,
        "SELECT p.w, b.v FROM join_probe_typed p LEFT JOIN join_build_typed b "
        "ON p.k = b.k ORDER BY p.w, b.v",
        conn,
        tables,
    )


def test_string_join_typed_table_matches_duckdb(data_dir: Path, conn) -> None:
    # The empty string is a real SQL value and must match; NULL must not.
    build = data_dir / "join_build_str.parquet"
    probe = data_dir / "join_probe_str.parquet"
    conn.execute(
        f"COPY (SELECT CASE i WHEN 0 THEN 'a' WHEN 1 THEN 'a' WHEN 2 THEN 'b' "
        f"WHEN 3 THEN '' WHEN 4 THEN 'b' WHEN 5 THEN NULL ELSE 'z' END AS k, i AS v "
        f"FROM range(9) t(i)) TO '{build.as_posix()}' (FORMAT PARQUET)"
    )
    conn.execute(
        f"COPY (SELECT CASE i WHEN 0 THEN 'a' WHEN 1 THEN NULL WHEN 2 THEN 'b' "
        f"WHEN 3 THEN 'missing' WHEN 4 THEN '' ELSE 'z' END AS k, i AS w "
        f"FROM range(6) t(i)) TO '{probe.as_posix()}' (FORMAT PARQUET)"
    )
    conn.execute(f"CREATE VIEW join_build_str AS SELECT * FROM read_parquet('{build.as_posix()}')")
    conn.execute(f"CREATE VIEW join_probe_str AS SELECT * FROM read_parquet('{probe.as_posix()}')")

    tables = ["join_build_str", "join_probe_str"]
    assert_matches_batch_invariant(
        data_dir,
        "SELECT p.w, b.v FROM join_probe_str p JOIN join_build_str b "
        "ON p.k = b.k ORDER BY p.w, b.v",
        conn,
        tables,
    )
    assert_matches_batch_invariant(
        data_dir,
        "SELECT p.w, b.v FROM join_probe_str p LEFT JOIN join_build_str b "
        "ON p.k = b.k ORDER BY p.w, b.v",
        conn,
        tables,
    )


def test_json_shape_and_table_output(data_dir: Path) -> None:
    query = "SELECT COUNT(*) AS n FROM sales"
    json_run = run_qe(data_dir, query)
    assert json_run.returncode == 0
    assert json_run.columns == [{"name": "n", "type": "int64"}]
    assert json_run.rows == [[5000]]

    table_run = run_qe(data_dir, query, "--format", "table")
    assert table_run.returncode == 0
    assert "n" in table_run.stdout
    assert "5000" in table_run.stdout
    assert "timings:" in table_run.stderr
    assert "scan:" in table_run.stderr


GPU_QUERIES = [
    "SELECT COUNT(*), SUM(amount), AVG(amount) FROM sales",
    "SELECT SUM(amount) FROM sales WHERE amount > 1000",
    "SELECT SUM(amount), AVG(amount) FROM edge_cases",
    "SELECT SUM(amount) FROM sales WHERE amount IS NULL",
]


@pytest.mark.skipif(not GPU_AVAILABLE, reason="qe-cli was built without a usable CUDA device")
@pytest.mark.parametrize("query", GPU_QUERIES)
def test_gpu_aggregate_matches_duckdb(data_dir: Path, conn, query: str) -> None:
    run = assert_matches_duckdb(data_dir, conn, query, "--gpu")
    if "WHERE amount IS NULL" in query:
        assert "gpu: used=0" in run.stderr, run.stderr
        assert "reason=no-rows" in run.stderr, run.stderr
    else:
        assert "gpu: used=1" in run.stderr, run.stderr


@pytest.mark.skipif(not GPU_AVAILABLE, reason="qe-cli was built without a usable CUDA device")
def test_gpu_aggregate_handles_tiny_batches(data_dir: Path, conn) -> None:
    query = "SELECT COUNT(*), SUM(amount), AVG(amount) FROM sales"
    run = assert_matches_duckdb(data_dir, conn, query, "--gpu", "--batch-size", "7")
    assert "gpu: used=1" in run.stderr, run.stderr


@pytest.mark.skipif(not GPU_AVAILABLE, reason="qe-cli was built without a usable CUDA device")
def test_gpu_float_aggregate(data_dir: Path, conn) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = data_dir / "gpu_float.parquet"
    pq.write_table(
        pa.table({"v": pa.array([1.5, None, 2.5, -1.0, 0.0], type=pa.float32())}),
        path,
    )
    conn.execute(f"CREATE VIEW gpu_float AS SELECT * FROM read_parquet('{path.as_posix()}')")
    query = "SELECT COUNT(*), SUM(v), AVG(v) FROM gpu_float"
    run = run_qe(data_dir, query, "--gpu", tables=("gpu_float",))
    assert run.returncode == 0, run.stderr
    assert "gpu: used=1" in run.stderr, run.stderr
    expected = [[normalize(value) for value in row] for row in conn.execute(query).fetchall()]
    actual = run.rows
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        for a, e in zip(actual_row, expected_row):
            assert same_value(a, e), (actual_row, expected_row)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="qe-cli was built without a usable CUDA device")
@pytest.mark.parametrize(
    "query",
    [
        "SELECT COUNT(*), SUM(quantity) FROM sales",
        "SELECT region, SUM(amount) FROM sales GROUP BY region ORDER BY region",
        "SELECT MIN(amount), MAX(amount) FROM sales",
    ],
)
def test_gpu_falls_back_for_unsupported_aggregates(data_dir: Path, conn, query: str) -> None:
    run = assert_matches_duckdb(data_dir, conn, query, "--gpu")
    assert "gpu: used=0" in run.stderr, run.stderr


def test_positional_parquet_path_and_query(data_dir: Path) -> None:
    completed = subprocess.run(
        [
            str(BINARY),
            str(data_dir / "edge_cases.parquet"),
            "SELECT COUNT(*) FROM edge_cases",
            "--format",
            "json",
            "--no-stats",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["rows"] == [[6]]


def test_max_rows_and_repeat(data_dir: Path) -> None:
    query = "SELECT id FROM edge_cases ORDER BY id"
    run = run_qe(data_dir, query, "--max-rows", "2", "--repeat", "3")
    assert run.returncode == 0, run.stderr
    assert run.rows == [[1], [2]]
    assert "runs_ms:" in run.stderr and "runs=3" in run.stderr


def test_aggregate_after_join_and_limit(data_dir: Path, conn) -> None:
    query = (
        "SELECT c.country, SUM(s.amount) AS total FROM sales s "
        "JOIN customers c ON s.customer_id = c.customer_id "
        "GROUP BY c.country ORDER BY total DESC LIMIT 3"
    )
    assert_matches_duckdb(data_dir, conn, query)


@pytest.mark.parametrize(
    "query, fragment",
    [
        ("SELECT id FROM missing_table", "not registered"),
        ("SELECT nope FROM edge_cases", "does not exist"),
        ("SELECT id FROM edge_left l JOIN edge_right r ON l.id = r.id", "ambiguous"),
        ("SELECT * FROM edge_cases RIGHT JOIN edge_right r ON edge_cases.id = r.id", "RIGHT JOIN"),
        ("SELECT * FROM edge_cases FULL JOIN edge_right r ON edge_cases.id = r.id", "FULL JOIN"),
        ("SELECT DISTINCT region FROM sales ORDER BY amount", "select list"),
        ("SELECT id, COUNT(*) FROM edge_cases", "must appear in GROUP BY"),
        ("SELECT SUM(amount) + 1 FROM edge_cases", "combine aggregates"),
        ("SELECT * FROM edge_cases HAVING COUNT(*) > 1", "must appear in GROUP BY"),
        ("SELECT LOWER(label) FROM edge_cases", "unsupported function"),
        ("SELECT id FROM edge_cases WHERE label LIKE id", "string literal"),
        ("SELECT id FROM edge_cases WHERE id LIKE '1'", "string expression"),
        ("SELECT id FROM edge_cases WHERE label LIKE 'a\\%c'", "backslash"),
        ("SELECT id FROM edge_cases WHERE id BETWEEN 2 AND", "expected"),
        ("SELECT id FROM edge_cases WHERE id IN ()", "IN list"),
        ("SELECT * FROM edge_cases; garbage", "unexpected token"),
        ("SELECT * FROM", "expected"),
        ("SELECT id FROM edge_cases WHERE", "expected an expression"),
    ],
)
def test_rejects_unsupported_or_invalid_sql(data_dir: Path, query: str, fragment: str) -> None:
    run = run_qe(data_dir, query)
    assert run.returncode != 0
    assert fragment.lower() in run.stderr.lower(), run.stderr


def test_cli_usage_errors() -> None:
    cases = [
        ([str(BINARY)], "no Parquet input"),
        ([str(BINARY), "--table", "t=x.parquet"], "no SQL query"),
        ([str(BINARY), "--table", "bad", "--query", "SELECT 1"], "NAME=PATH"),
        (
            [str(BINARY), "--table", "sales=x.parquet", "--query", "SELECT 1", "--format", "xml"],
            "--format",
        ),
        (
            [str(BINARY), "--table", "sales=x.parquet", "--query", "SELECT 1", "--batch-size", "0"],
            "--batch-size",
        ),
        (
            [str(BINARY), "--table", "sales=x.parquet", "--query", "SELECT 1", "--threads", "0"],
            "--threads",
        ),
        (
            [str(BINARY), "--table", "sales=x.parquet", "--query", "SELECT 1", "--threads", "1025"],
            "--threads",
        ),
        (
            [str(BINARY), "--table", "sales=x.parquet", "--query", "SELECT 1", "--threads", "abc"],
            "--threads",
        ),
    ]
    for args, fragment in cases:
        completed = subprocess.run(args, capture_output=True, text=True)
        assert completed.returncode != 0
        assert fragment.lower() in completed.stderr.lower(), completed.stderr


def test_nonexistent_file_is_a_clean_error() -> None:
    completed = subprocess.run(
        [
            str(BINARY),
            str(ROOT / "no-such-file.parquet"),
            "SELECT COUNT(*) FROM no_such_file",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "error:" in completed.stderr.lower()


def test_version_and_help() -> None:
    version = subprocess.run([str(BINARY), "--version"], capture_output=True, text=True)
    assert version.returncode == 0
    assert "qe-cli" in version.stdout and "cuda:" in version.stdout
    help_run = subprocess.run([str(BINARY), "--help"], capture_output=True, text=True)
    assert help_run.returncode == 0
    assert "Usage:" in help_run.stdout
    assert "--threads" in help_run.stdout
