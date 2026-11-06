# qe — a columnar analytical query engine

[![CI](https://github.com/abiralpokhrel-learns/query-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/abiralpokhrel-learns/query-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small SQL query engine built from first principles in C++20. It reads Apache
Parquet, compiles a defined SQL subset into a vectorized operator pipeline, and
returns results that are tested against DuckDB. A Python harness benchmarks the
engine on the same data and queries as DuckDB and stores every timing, version,
and dataset detail needed to reproduce the numbers.

**Status: CPU engine 0.2.0.** Version 1 is complete, and selected version 2
features are now implemented: `DISTINCT`, `HAVING`, `LEFT JOIN`, `IN` /
`BETWEEN`, `OFFSET`, metadata-only `COUNT(*)`, and a top-k sort for
`ORDER BY ... LIMIT`. All 166 correctness tests pass against DuckDB on generated
edge-case datasets, the C++ parser suite passes, and the differential fuzzer
matches DuckDB on multi-seed sweeps. Every benchmark query is verified against
DuckDB on the 1M-row dataset. The latest results are in
[`benchmarks/results/latest.md`](benchmarks/results/latest.md).

> **GPU note:** this build is CPU-only. The current machine has no usable CUDA
> device: `nvidia-smi` cannot initialize NVML, and a minimal CUDA probe compiled
> with the installed toolkit fails at `cudaGetDevice` because the driver is
> unavailable. No GPU kernel was written or measured, and the benchmark output
> reports no fabricated GPU numbers. The CUDA design remains version 2 work in
> the [engineering write-up](docs/engineering-writeup.md).

## Demo in one command

From WSL2 / Linux with the environment already built:

```bash
scripts/demo.sh
```

The demo registers the included sample tables, prints the engine version, then
runs representative filter, aggregate, join, top-k, `DISTINCT`, `LEFT JOIN`, and
`HAVING` queries with timings and scan statistics. It rebuilds the binary or regenerates the sample data if either
is missing.

## Quick start from a clean checkout

```bash
# 1. Create the pinned C++ / Python environment (conda-forge).
micromamba env create -f environment.yml
micromamba activate qe

# 2. Build the engine.
scripts/build.sh

# 3. Run the test suites.
ctest --test-dir build --output-on-failure
python -m pytest tests -q

# 4. Run the demo.
scripts/demo.sh
```

`scripts/build.sh` honours `QE_PREFIX`, `CONDA_PREFIX`, `CMAKE_BIN`,
`BUILD_DIR`, and `CMAKE_BUILD_TYPE`. If no environment is active it falls back
to `~/qe-env`, the layout used during development.

## What the engine does

### Storage

- Reads flat Parquet files through Apache Arrow / Parquet C++.
- Projects only the columns a query references (column pruning).
- Skips row groups whose Parquet min/max statistics prove that a predicate
  cannot match.
- Applies residual filters while scanning, so rows are discarded before they
  reach the operators.
- Answers predicate-free `COUNT(*)` from row-group metadata without decoding a
  payload column.
- Emits fixed-size columnar `RecordBatch`es.

### SQL subset

```sql
SELECT [DISTINCT] [* | expr [AS alias], ...]
FROM table [alias]
[JOIN | LEFT [OUTER] JOIN table [alias] ON left.key = right.key ...]
[WHERE predicate]
[GROUP BY expr, ...]
[HAVING predicate]
[ORDER BY expr | ordinal [ASC|DESC], ...]
[LIMIT n] [OFFSET n];
```

Supported expressions: column references, integer / double / string / boolean
/ `NULL` literals, `+ - * /`, comparisons, `AND`, `OR`, `NOT`, `IS NULL`,
`IS NOT NULL`, `[NOT] IN (...)`, `[NOT] BETWEEN ... AND ...`, and `COUNT(*)`,
`COUNT(col)`, `SUM`, `AVG`, `MIN`, `MAX`. Inner and left outer joins support one
or more equality conditions. Identifiers are case-insensitive, quoted
identifiers and `''`-escaped strings are accepted, and dates are compared as ISO
`YYYY-MM-DD` literals.

`DISTINCT` removes duplicate output rows; `HAVING` filters groups after
aggregation; `OFFSET` skips rows after ordering; `COUNT(*)` with no predicate is
answered from Parquet metadata instead of decoding a column. `ORDER BY ...
LIMIT` uses a bounded top-k sort rather than sorting every matching row.

Unsupported SQL is rejected with a clear error rather than being silently
misinterpreted. Still out of scope: `RIGHT`/`FULL`/`CROSS`/`NATURAL` joins,
`JOIN ... USING`, `LIKE`, `NULLS FIRST/LAST`, `UNION`/`EXCEPT`/`INTERSECT`,
subqueries, CTEs, window functions, `DISTINCT` inside aggregates, aggregate
arguments that are expressions, `GROUP BY` expressions, and non-aggregate
columns mixed with aggregates.

### Execution

Operators process batches of 8,192 rows by default (`--batch-size` changes it):

```
Scan -> Filter -> Join -> Aggregate -> HAVING -> Project -> Distinct -> Sort -> Limit
```

The exact order is chosen by the planner: sort runs before projection when an
`ORDER BY` key is not selected, and distinct runs before sort because ordering
applies to the deduplicated rows. Left joins materialize the preserved side and
extend unmatched rows with typed `NULL`s. Filters are pushed into the scan during
planning. The planner also prunes columns, resolves aliases and ordinals, selects
the smaller join input as the build side, and inserts projections only where the
output schema requires them. `--explain` prints the optimized plan, including
pushed predicates and the columns read from each file.

## Test suite

The Python suite compares every result with DuckDB on the same generated
Parquet files, including:

- filters with `NULL`, boolean, string, and date semantics;
- `IN` / `NOT IN` / `BETWEEN` / `NOT BETWEEN`, including `NULL` lists;
- empty inputs, empty groups, `NULL` groups, and single-row groups;
- integer and floating-point arithmetic, division, and negation;
- multi-key hash joins, duplicate keys, and `NULL` join keys;
- left joins with unmatched rows, `NULL` filters, and chained joins;
- `DISTINCT`, `HAVING`, `OFFSET`, aliases, ordinals, and `LIMIT 0`;
- top-k ordering and aggregate-after-join plans;
- unsupported SQL and CLI usage errors;
- row-group skipping and column pruning asserted from `--explain` and scan stats;
- metadata-only `COUNT(*)` and the empty plan it produces;
- identical results across batch sizes 1, 7, 128, and 8,192.

The C++ suite covers lexing, parsing, precedence, new clauses, and rejection of
out-of-scope SQL. A deterministic differential fuzzer
([`scripts/fuzz_compare.py`](scripts/fuzz_compare.py)) generates queries inside
the supported subset, runs each against this engine and DuckDB, and compares
every value; a 30-query sample is part of every pytest run.

```bash
python -m pytest tests -q                     # 166 tests
ctest --test-dir build --output-on-failure    # parser/unit tests
python scripts/fuzz_compare.py --queries 1000 --seed 2026
```

## Benchmarks

The harness verifies every query against DuckDB before timing it, then records
warm-ups, repeats, raw runs, medians, query text, hardware, library versions,
and the dataset manifest. Charts are rendered from the stored JSON, never from
hand-typed numbers.

```bash
python scripts/run_benchmarks.py --rows 1000000 --repeats 5 --warmups 2
```

Latest run: 1,000,000 sales rows, 16 row groups, 22.5 MB Parquet, batch size
8,192, AMD Ryzen 7 170 (16 logical CPUs), 7.4 GiB RAM in WSL2, Ubuntu 24.04.
DuckDB used its default 16 threads; the engine is single-threaded.

| Query | qe CPU (ms) | DuckDB (ms) | CPU / DuckDB | Winner |
|---|---:|---:|---:|---|
| filter_aggregate | 9.22 | 27.98 | 0.33x | qe |
| empty_filter | 0.27 | 13.89 | 0.02x | qe |
| group_region | 244.75 | 55.29 | 4.43x | DuckDB |
| group_customer | 691.02 | 138.22 | 5.00x | DuckDB |
| topk | 164.46 | 81.66 | 2.01x | DuckDB |
| join_country | 773.26 | 112.36 | 6.88x | DuckDB |
| count_star | 0.01 | 13.36 | 0.00x | qe |

Wins and losses are both reported. The engine wins on selective scans because
Parquet row-group statistics let it skip data and because its scan pipeline has
little overhead; it also wins metadata-only `COUNT(*)` because it decodes no
payload column. It loses on large hash aggregations, hash joins, and top-k
queries: the aggregate and join loops are row-wise inside each batch, while
DuckDB uses vectorized, multi-threaded operators. See the
[engineering write-up](docs/engineering-writeup.md) for the analysis and for the
optimization that cut the low-cardinality `GROUP BY` from 5.1 s to around 0.25 s.

> GPU transfer and kernel timings are deliberately absent. No CUDA build exists
> on this machine, and the harness never invents measurements.

## Repository layout

```
src/
  cli/        table and JSON output
  engine/     expressions, keys, operators, planner, executor
  sql/        lexer, recursive-descent parser, AST
  storage/    Parquet catalog, schema checks, pushdown
tests/
  sql_test.cpp        C++ parser/unit tests
  test_engine.py      end-to-end correctness vs DuckDB
scripts/
  build.sh            CMake + Ninja build
  generate_data.py    deterministic Parquet generator
  run_benchmarks.py   verified benchmark harness
  plot_benchmarks.py  chart renderer (stored JSON only)
  fuzz_compare.py     differential SQL fuzzer against DuckDB
  demo.sh             one-command demo
benchmarks/
  results/            latest + timestamped JSON/Markdown reports
  charts/             charts generated from the JSON
data/                 generated sample data (200k sales rows, created by demo)
docs/
  engineering-writeup.md
```

The sample Parquet files are deterministic but are not committed; `scripts/demo.sh`
creates them on first run with `scripts/generate_data.py`.

## Verified environment

| Component | Version |
|---|---|
| OS | WSL2 Ubuntu 24.04.4 LTS, kernel 6.18.33.2 |
| Compiler | g++ 13.3.0 |
| Build system | CMake 4.4.5, Ninja 1.13.2 |
| Arrow / Parquet | 13.0.0 |
| DuckDB | 2.0.0 |
| Python | 3.12.16 |
| PyArrow / NumPy | 13.0.0 / 1.26.4 |
| CUDA toolkit | 12.9.86 (compiles; no usable device) |
| GPU | none exposed to this environment |

## Known limits (0.2.0)

- Read-only, single-process, single-threaded engine.
- Flat schemas only; nested Parquet types are rejected.
- Inner and left joins only, with equality keys that must have exactly the same
  Arrow type on both sides. No right, full, cross, semi, or anti joins.
- No `LIKE`, `NULLS FIRST/LAST`, set operations, subqueries, CTEs, or window
  functions.
- No `DISTINCT` inside aggregates; aggregate arguments must be plain columns;
  `GROUP BY` accepts plain columns only.
- Sort and aggregation are in-memory and assume current data sizes; no spilling.
- Aggregation and join probing still contain row-wise inner loops; they are not
  yet vectorized or parallel.
- No CUDA path in this build because no usable CUDA device is exposed. The
  intended design and transfer accounting are documented in the write-up.

Remaining version 2 work is listed in the
[write-up](docs/engineering-writeup.md#8-limits-and-version-2-ideas): vectorized hash
aggregation and join probing, multi-threading, CUDA filter/aggregation on a
machine with a usable GPU, the remaining join types and SQL clauses, join-key
type coercion, spilling, and nested-type support. These are listed rather than
half-implemented.

## Author

Built by [Abiral Pokhrel](https://github.com/abiralpokhrel-learns).

## License

MIT. See [LICENSE](LICENSE).
