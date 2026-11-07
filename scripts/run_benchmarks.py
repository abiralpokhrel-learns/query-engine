#!/usr/bin/env python3
"""Run verified CPU-vs-DuckDB benchmarks and store every condition and timing.

The harness always verifies each query against DuckDB before timing it. It
records the dataset manifest, hardware, software versions, batch size, warm-up
and repetition counts, and every individual run. A GPU column is recorded as
unavailable when this build has no CUDA backend; timings are never invented.

Example:
    ~/qe-env/bin/python scripts/run_benchmarks.py --rows 1000000 --repeats 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("QE_CLI", str(ROOT / "build" / "qe-cli")))
RESULT_DIR = ROOT / "benchmarks" / "results"
CHART_DIR = ROOT / "benchmarks" / "charts"
VIEW_TABLES = ("sales", "customers")

QUERIES = [
    {
        "id": "filter_aggregate",
        "description": "One-month date range; row-group skipping plus COUNT/SUM.",
        "sql": (
            "SELECT COUNT(*), SUM(amount) FROM sales "
            "WHERE sale_date >= '2025-01-01' AND sale_date < '2025-02-01'"
        ),
    },
    {
        "id": "empty_filter",
        "description": "Date range after the data; all row groups can be skipped.",
        "sql": "SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date > '2030-01-01'",
    },
    {
        "id": "group_region",
        "description": "Low-cardinality GROUP BY with five aggregates.",
        "sql": (
            "SELECT region, COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) "
            "FROM sales GROUP BY region ORDER BY region"
        ),
    },
    {
        "id": "group_customer",
        "description": "High-cardinality GROUP BY (one group per customer).",
        "sql": (
            "SELECT customer_id, COUNT(*), SUM(amount), AVG(amount) FROM sales "
            "GROUP BY customer_id ORDER BY customer_id"
        ),
    },
    {
        "id": "topk",
        "description": "Selective filter, top-k sort, then LIMIT.",
        "sql": (
            "SELECT sale_id, amount FROM sales WHERE amount > 1000 ORDER BY amount DESC LIMIT 100"
        ),
    },
    {
        "id": "join_country",
        "description": "Hash join against customers, then aggregate by country.",
        "sql": (
            "SELECT c.country, COUNT(*), SUM(s.amount) FROM sales s "
            "JOIN customers c ON s.customer_id = c.customer_id "
            "GROUP BY c.country ORDER BY c.country"
        ),
    },
    {
        "id": "count_star",
        "description": "Full-table COUNT(*) answered from Parquet metadata by both engines.",
        "sql": "SELECT COUNT(*) FROM sales",
    },
]


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=check)


def get_engine_version() -> dict:
    completed = run_command([str(BINARY), "--version"])
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "version_text": lines,
        "name": lines[0].split()[0] if lines else "qe-cli",
        "version": lines[0].split()[1] if len(lines) > 0 and len(lines[0].split()) > 1 else "?",
        "cuda": any("cuda: enabled" in line.lower() for line in lines),
    }


def get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def get_total_memory() -> str:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return f"{kib / 1024 / 1024:.1f} GiB"
    except OSError:
        pass
    return "unknown"


def get_gpu_status() -> dict:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi not found"}
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        # nvidia-smi often prints a shutdown message after the real failure;
        # report the first line, which is the actionable one.
        return {
            "available": False,
            "reason": detail[0] if detail else "nvidia-smi failed",
        }
    gpus = []
    for line in completed.stdout.splitlines():
        name, memory, driver = [part.strip() for part in line.split(",")]
        gpus.append({"name": name, "memory": memory, "driver": driver})
    return {"available": bool(gpus), "gpus": gpus}


def get_environment() -> dict:
    try:
        uname = platform.uname()
    except Exception:
        uname = None
    return {
        "platform": platform.platform(),
        "system": uname.system if uname else platform.system(),
        "release": uname.release if uname else platform.release(),
        "machine": uname.machine if uname else platform.machine(),
        "processor": get_cpu_model(),
        "memory": get_total_memory(),
        "wsl_distro": os.environ.get("WSL_DISTRO_NAME"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "duckdb": duckdb.__version__,
    }


def ensure_data(
    data_dir: Path, rows: int, customers: int, products: int, row_group_size: int, force: bool
) -> dict:
    manifest_path = data_dir / "manifest.json"
    valid = False
    if not force and manifest_path.exists() and (data_dir / "sales.parquet").exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = (
                manifest.get("tables", {}).get("sales", {}).get("rows") == rows
                and manifest.get("tables", {}).get("customers", {}).get("rows") == customers
                and manifest.get("row_group_size") == row_group_size
            )
        except (OSError, ValueError):
            valid = False
    if not valid:
        data_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "generate_data.py"),
            "--out",
            str(data_dir),
            "--rows",
            str(rows),
            "--customers",
            str(customers),
            "--products",
            str(products),
            "--row-group-size",
            str(row_group_size),
        ]
        print("generating benchmark data:", " ".join(command))
        run_command(command)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def create_views(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    for table in VIEW_TABLES:
        path = (data_dir / f"{table}.parquet").as_posix()
        conn.execute(f"CREATE OR REPLACE VIEW \"{table}\" AS SELECT * FROM read_parquet('{path}')")


def qe_args(data_dir: Path, query: str, *extra: str) -> list[str]:
    args = [str(BINARY)]
    for table in VIEW_TABLES:
        args += ["--table", f"{table}={data_dir / (table + '.parquet')}"]
    args += ["--query", query, "--format", "json", "--no-stats", *extra]
    return args


def run_qe(data_dir: Path, query: str, *extra: str) -> subprocess.CompletedProcess:
    completed = subprocess.run(qe_args(data_dir, query, *extra), capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"qe-cli failed for query:\n{query}\n{completed.stderr}")
    return completed


def parse_timings(stderr: str) -> dict:
    timing_line = next((line for line in stderr.splitlines() if line.startswith("timings:")), None)
    if not timing_line:
        raise RuntimeError(f"no timings line in qe-cli stderr:\n{stderr}")
    timings = {}
    for field in timing_line.removeprefix("timings:").strip().split():
        key, value = field.split("=", 1)
        timings[key] = float(value)
    runs_line = next((line for line in stderr.splitlines() if line.startswith("runs_ms:")), None)
    runs = (
        [float(value) for value in runs_line.removeprefix("runs_ms:").split()] if runs_line else []
    )
    return {"timings": timings, "runs_ms": runs}


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


def verify_query(conn: duckdb.DuckDBPyConnection, data_dir: Path, query: str) -> dict:
    completed = run_qe(data_dir, query, "--max-rows", "-1")
    payload = json.loads(completed.stdout)
    actual = [[normalize(value) for value in row] for row in payload["rows"]]
    expected = [[normalize(value) for value in row] for row in conn.execute(query).fetchall()]
    if len(actual) != len(expected):
        raise AssertionError(
            f"row count mismatch for {query!r}: qe={len(actual)} duckdb={len(expected)}"
        )
    for row_index, (actual_row, expected_row) in enumerate(zip(actual, expected), 1):
        if len(actual_row) != len(expected_row):
            raise AssertionError(f"column count mismatch at row {row_index} for {query!r}")
        for column_index, (a, e) in enumerate(zip(actual_row, expected_row), 1):
            if not same_value(a, e):
                raise AssertionError(
                    f"value mismatch for {query!r} at row {row_index}, column "
                    f"{column_index}: qe={a!r} duckdb={e!r}"
                )
    return {
        "verified": True,
        "row_count": len(actual),
        "columns": payload["columns"],
    }


def measure_qe(data_dir: Path, query: str, batch_size: int, repeats: int, warmups: int) -> dict:
    for _ in range(warmups):
        run_qe(data_dir, query, "--batch-size", str(batch_size), "--max-rows", "0")
    completed = run_qe(
        data_dir,
        query,
        "--batch-size",
        str(batch_size),
        "--repeat",
        str(repeats),
        "--max-rows",
        "0",
    )
    parsed = parse_timings(completed.stderr)
    runs = parsed["runs_ms"]
    return {
        "runs_ms": runs,
        "median_ms": parsed["timings"]["execute_ms"],
        "min_ms": min(runs) if runs else parsed["timings"]["execute_ms"],
        "max_ms": max(runs) if runs else parsed["timings"]["execute_ms"],
        "parse_ms": parsed["timings"].get("parse_ms"),
        "plan_ms": parsed["timings"].get("plan_ms"),
        "catalog_ms": parsed["timings"].get("catalog_ms"),
        "batch_size": batch_size,
        "transfer_ms": None,
        "kernel_ms": None,
    }


def measure_duckdb(conn: duckdb.DuckDBPyConnection, query: str, repeats: int, warmups: int) -> dict:
    for _ in range(warmups):
        conn.execute(query).fetchall()
    runs = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        conn.execute(query).fetchall()
        runs.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "runs_ms": runs,
        "median_ms": sorted(runs)[len(runs) // 2]
        if len(runs) % 2
        else (sorted(runs)[len(runs) // 2 - 1] + sorted(runs)[len(runs) // 2]) / 2.0,
        "min_ms": min(runs),
        "max_ms": max(runs),
        "threads": conn.execute("SELECT current_setting('threads')").fetchone()[0],
    }


def summarize(queries: list[dict]) -> dict:
    cpu_wins = 0
    duckdb_wins = 0
    for query in queries:
        if query["cpu"]["median_ms"] < query["duckdb"]["median_ms"]:
            cpu_wins += 1
        elif query["cpu"]["median_ms"] > query["duckdb"]["median_ms"]:
            duckdb_wins += 1
    return {
        "cpu_wins": cpu_wins,
        "duckdb_wins": duckdb_wins,
        "ties": len(queries) - cpu_wins - duckdb_wins,
        "gpu_available": False,
        "gpu_queries": 0,
    }


def markdown_report(results: dict) -> str:
    gpu = results["gpu"]
    if gpu.get("available"):
        gpu_note = (
            "GPU timings are not reported: this build has no CUDA kernels. "
            "Transfer and kernel columns are absent rather than estimated."
        )
    else:
        reason = gpu.get("reason", "no usable GPU was detected")
        gpu_note = (
            f"GPU timings are not reported: no usable GPU was detected ({reason}). "
            "Transfer and kernel columns are absent rather than estimated."
        )
    lines = [
        "# Benchmark results",
        "",
        f"Generated: {results['generated_at']}",
        "",
        f"Dataset: **{results['dataset']['tables']['sales']['rows']:,} sales rows**, "
        f"{results['dataset']['tables']['sales']['row_groups']} row groups, "
        f"{results['dataset']['tables']['sales']['file_bytes'] / 1_000_000:.1f} MB.",
        "",
        f"Hardware: {results['environment']['processor']}, "
        f"{results['environment']['memory']}, {results['environment']['platform']}.",
        "",
        f"CPU engine batch size: {results['config']['batch_size']}; "
        f"median of {results['config']['repeats']} runs after "
        f"{results['config']['warmups']} warm-up run(s).",
        "",
        f"> {gpu_note}",
        "",
        "| Query | CPU median (ms) | DuckDB median (ms) | CPU / DuckDB | Winner |",
        "|---|---:|---:|---:|---|",
    ]
    for query in results["queries"]:
        cpu = query["cpu"]["median_ms"]
        duck = query["duckdb"]["median_ms"]
        ratio = cpu / duck if duck else float("inf")
        winner = "CPU" if cpu < duck else ("DuckDB" if duck < cpu else "tie")
        lines.append(f"| {query['id']} | {cpu:.2f} | {duck:.2f} | {ratio:.2f}x | {winner} |")
    lines += [
        "",
        f"CPU wins: **{results['summary']['cpu_wins']}**; "
        f"DuckDB wins: **{results['summary']['duckdb_wins']}**; "
        f"ties: **{results['summary']['ties']}**.",
        "",
        "A value below 1.00x means the CPU engine was faster; a value above 1.00x "
        "means DuckDB was faster. DuckDB runs with its default thread count, while "
        "the engine is single-threaded. Full timings, query text, versions, and "
        "dataset metadata are in the adjacent JSON file.",
        "",
        "## Workload",
        "",
    ]
    for query in results["queries"]:
        lines += [
            f"### {query['id']}",
            "",
            query["description"],
            "",
            "```sql",
            query["sql"],
            "```",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "bench")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--customers", type=int, default=50_000)
    parser.add_argument("--products", type=int, default=5_000)
    parser.add_argument("--row-group-size", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if not BINARY.exists():
        parser.error(f"{BINARY} does not exist; run scripts/build.sh first")
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be >= 1 and --warmups >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    data_dir = args.data.resolve()
    manifest = ensure_data(
        data_dir, args.rows, args.customers, args.products, args.row_group_size, args.force_data
    )
    gpu = get_gpu_status()
    engine = get_engine_version()

    conn = duckdb.connect()
    create_views(conn, data_dir)
    duckdb_threads = conn.execute("SELECT current_setting('threads')").fetchone()[0]

    print("verifying every benchmark query against DuckDB...")
    query_results = []
    for spec in QUERIES:
        verification = (
            {"verified": False, "row_count": None, "columns": None}
            if args.skip_verify
            else verify_query(conn, data_dir, spec["sql"])
        )
        print(f"  {spec['id']}: verified={verification['verified']}")
        query_results.append(
            {
                "id": spec["id"],
                "description": spec["description"],
                "sql": spec["sql"],
                **verification,
            }
        )

    print("benchmarking...")
    for query in query_results:
        query["cpu"] = measure_qe(
            data_dir, query["sql"], args.batch_size, args.repeats, args.warmups
        )
        query["duckdb"] = measure_duckdb(conn, query["sql"], args.repeats, args.warmups)
        query["gpu"] = None
        query["gpu_reason"] = (
            "No CUDA kernel in this build; no GPU timings are fabricated."
            if not engine["cuda"]
            else "CUDA build present but no GPU execution backend was configured."
        )
        print(
            f"  {query['id']}: CPU {query['cpu']['median_ms']:.2f} ms, "
            f"DuckDB {query['duckdb']['median_ms']:.2f} ms"
        )
    conn.close()

    results = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed_cpu_only" if not engine["cuda"] else "completed",
        "engine": engine,
        "gpu": gpu,
        "environment": get_environment(),
        "dataset": manifest,
        "config": {
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "duckdb_threads": duckdb_threads,
            "cpu_threads": 1,
            "timer": "CLI internal execute_ms (scan open through result materialization)",
            "duckdb_timer": "Python client execute + fetchall in the same process",
            "data_dir": str(data_dir),
        },
        "queries": query_results,
    }
    results["summary"] = summarize(query_results)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = RESULT_DIR / f"bench_{stamp}.json"
    latest_path = RESULT_DIR / "latest.json"
    report_path = RESULT_DIR / f"bench_{stamp}.md"
    latest_md = RESULT_DIR / "latest.md"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report = markdown_report(results)
    report_path.write_text(report, encoding="utf-8")
    latest_md.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {result_path}")
    print(f"wrote {latest_path}")
    print(f"wrote {latest_md}")

    plot = ROOT / "scripts" / "plot_benchmarks.py"
    if plot.exists():
        completed = subprocess.run(
            [sys.executable, str(plot), str(result_path)],
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            print(completed.stdout.strip())
        else:
            print("warning: chart generation failed:\n" + completed.stderr, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
