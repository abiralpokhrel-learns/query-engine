# Engineering write-up

## 1. What was built

`qe` is a columnar analytical query engine written in C++20. It reads Apache
Parquet files, parses a bounded SQL subset, plans a vectorized operator pipeline,
executes it in fixed-size columnar batches, and writes table or JSON output from
a CLI. DuckDB is used as the correctness oracle: every test query and every
benchmark query is executed against DuckDB on identical Parquet files and the
results are compared exactly (with an explicit tolerance for floating point).

The project is deliberately small. Version 1 is a finished CPU engine with a
verified benchmark harness, not a half-built attempt at a full database. On top
of that base, version 2 has added `DISTINCT`, `HAVING`, `LEFT JOIN`, `IN` /
`BETWEEN`, `OFFSET`, metadata-only `COUNT(*)`, a top-k sort, and an opt-in CUDA
global-aggregate path. The GPU implementation, verification, and measured
results are documented in Section 7.

At the time of writing:

- 175 tests pass against DuckDB on a CUDA build (166 CPU tests plus 9 GPU-gated
  tests); the GPU tests skip on CPU-only builds.
- The C++ parser/unit suite passes, and the CUDA build adds a real reduction
  self-test.
- A deterministic differential fuzzer has matched DuckDB on multi-seed sweeps
  of generated queries; a fixed sample runs inside pytest.
- Eight benchmark queries are verified against DuckDB, timed after warm-ups, and
  stored with full environment and dataset metadata.
- Wins and losses are both reported; the latest table is in Section 5, and the
  GPU comparison is in Section 7.3.

## 2. Architecture

The code is split into layers with narrow boundaries:

| Layer | Location | Responsibility |
|---|---|---|
| Storage | `src/storage/` | Open Parquet, validate the schema, prune columns, use row-group statistics, apply residual filters, emit batches |
| SQL | `src/sql/` | Hand-written lexer and recursive-descent parser; AST; planner-facing statement types |
| Engine | `src/engine/` | Expressions, hash keys, scan/filter/project/aggregate/join/sort/limit operators, planner, executor |
| CLI | `src/cli/`, `src/main.cpp` | Option parsing, table/JSON rendering, timings, scan statistics, explain output |

A query flows through four stages:

```
SQL text
  -> lexer/parser (AST)
  -> planner (logical operator tree + pushdown + column pruning)
  -> executor (build operators, run batch pipeline)
  -> result table (Arrow) -> table or JSON
```

### 2.1 Storage and pushdown

`ParquetSource` wraps `parquet::arrow::FileReader` and exposes the file schema
and estimated row count. The scan operator:

1. asks the catalog for the table's schema;
2. maps only the referenced columns to file column indices (column pruning);
3. converts simple predicates (`column op literal`) into row-group statistics
   filters;
4. reads surviving row groups through a `RecordBatchReader`;
5. evaluates any remaining predicates on each batch and compacts the batch with
   an Arrow filter kernel.

Row-group statistics are the main reason the engine wins the two selective
queries in the benchmark. For example, a one-month date range on a file whose
dates are sorted within each row group can skip most groups outright. The scan
statistics printed to stderr (`row_groups_read`, `row_groups_skipped`,
`rows_scanned`, `rows_after_pushdown`, `batches`) make this observable rather
than assumed.

### 2.2 SQL parser and planner

The parser is a hand-written recursive-descent parser. It supports:

- `SELECT [DISTINCT]` lists with `*`, qualified stars, expressions, and aliases;
- `FROM` with aliases, repeated inner joins, and left outer joins;
- equality join conditions;
- `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY` (expressions or ordinals), `LIMIT`,
  and `OFFSET`;
- arithmetic, comparisons, `AND`/`OR`/`NOT`, `IS [NOT] NULL`;
- `[NOT] IN (...)`, `[NOT] BETWEEN ... AND ...`;
- `COUNT(*)`, `COUNT(col)`, `SUM`, `AVG`, `MIN`, `MAX`.

`IN` and `BETWEEN` are rewritten during parsing into comparisons joined with
`OR`/`AND`. The rewrites preserve SQL three-valued logic, so `NOT IN` with a
`NULL` in the list still evaluates to `NULL` rather than a spurious match.

Everything outside that contract is rejected with an error that names the
construct (`RIGHT`/`FULL`/`CROSS` joins, `JOIN ... USING`, `LIKE`, `NULLS
FIRST/LAST`, set operations, subqueries, unsupported functions, trailing tokens,
and so on). The parser never guesses. Keyword expectations check the exact
keyword, not merely that *some* keyword was found.

The planner performs two high-value optimizations:

- **Predicate pushdown**: predicates that only reference a base table are moved
  into that table's scan. A second copy remains above joins when needed.
- **Column pruning**: each scan reads only the columns used by projections,
  filters, join keys, grouping keys, and sort keys.

It also resolves aliases and `ORDER BY` ordinals, checks aggregate/group-by
legality, rejects ambiguous columns, selects the smaller join input as the hash
build side, and inserts projections only when the output schema differs from the
child schema. Grouped queries are rewritten so `HAVING` becomes a filter over
aggregate outputs. `SELECT DISTINCT` projects before deduplication and requires
`ORDER BY` expressions to appear in the select list. A sort that feeds a
`LIMIT` receives a `limit_hint`, which enables the bounded top-k path.
`--explain` prints the resulting tree.

### 2.3 Execution

Operators pull batches from their children. The default batch size is 8,192
rows and is configurable with `--batch-size`. The core operators are:

- **Scan**: Parquet reader + pushed filters + residual filtering. A scan with no
  required columns and no predicate counts rows directly from Parquet metadata.
- **Filter**: evaluates a predicate array and compacts the batch.
- **Project**: evaluates expressions with Arrow compute kernels.
- **Aggregate**: hash grouping with `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.
- **Join**: in-memory hash join over one or more equality keys; duplicate keys
  probe multiple build rows, and `NULL` keys never match. A left join
  materializes the preserved side, appends a typed `NULL` row to the build side,
  and points unmatched rows at that row.
- **Distinct**: materializes its input, hashes a type-tagged key per row, and
  keeps the first occurrence of each key.
- **Sort**: materializes, computes a stable index permutation, and applies it
  with `Arrow::compute::Take`. When a `LIMIT` follows, `nth_element` partitions
  the input down to the first K rows before sorting them.
- **Limit**: skips `OFFSET` rows, slices the first `n` remaining rows, and stops
  pulling.

`AVG` is computed as a double sum plus a non-null count, so it matches DuckDB's
semantics without accumulating per-row values. Integer `SUM` uses a 128-bit
accumulator and reports overflow instead of wrapping. `MIN`/`MAX` preserve the
input type and handle dates, strings, booleans, and numerics. `NULL` ordering
matches DuckDB: ascending puts `NULL`s last, descending puts them first.

## 3. Correctness strategy

Correctness is defined as "same rows, same values, same order as DuckDB on the
same file". The Python suite generates a deterministic dataset with five row
groups for the main sales table, then creates DuckDB views over exactly the same
Parquet files. For each query it runs `qe-cli --format json`, normalizes dates
and non-finite floats, and compares every value. It also asserts:

- row counts and column counts;
- `NULL` handling in filters, joins, aggregates, and sort keys;
- `IN`/`NOT IN`/`BETWEEN`/`NOT BETWEEN`, including lists containing `NULL`;
- `DISTINCT`, `HAVING`, `OFFSET`, aliases, ordinals, and empty results;
- empty tables, empty groups, and a table where every aggregate input is
  `NULL`;
- duplicate join keys and `NULL` join keys;
- left joins with unmatched rows, `NULL`-side filters, and chained joins;
- batch-size invariance (1, 7, 128, 8,192 rows);
- that `--explain` shows the expected pushed predicate and the exact pruned
  column list;
- that row-group skipping actually happens (`row_groups_skipped > 0` and
  `rows_after_pushdown < rows_scanned`);
- that predicate-free `COUNT(*)` reports `rows_from_metadata` and scans zero
  rows;
- that unsupported SQL and malformed CLI invocations fail with useful messages.

The standalone fuzzer extends this beyond hand-written cases. It generates
deterministic filters, aggregates, `DISTINCT`, inner and left joins,
`IN`/`BETWEEN`, `HAVING`, ordering, limits, and offsets, then compares every
value with DuckDB. The generator stays inside the supported subset, so a
mismatch is an engine bug rather than an unsupported query.

CI adds several independent quality gates: `clang-format` and `ruff` checks,
warnings-as-errors builds, a separate AddressSanitizer / UndefinedBehaviorSanitizer
job, and a coverage job that enforces a line threshold and publishes line and
branch results as a build artifact and repository-hosted badge. These do not
replace the DuckDB oracle; they catch a different class of defects, such as
formatting drift, undefined behavior, and untested branches.

The generated data deliberately contains nulls, skew, duplicate keys, quoted
strings (`O'Brien`), leap-day and pre-epoch dates, empty strings, and empty
tables. These are the cases that tend to break naive implementations.

## 4. Benchmark method

`scripts/run_benchmarks.py` is the only source of performance numbers. It:

1. loads or generates the configured dataset and records its manifest;
2. opens DuckDB views over the same files;
3. **verifies every query against DuckDB before timing it**;
4. runs warm-ups, then N measured repetitions for the engine and DuckDB;
5. records every individual run, the median, min, max, batch size, thread
   counts, query text, engine version, library versions, CPU, memory, and OS;
   with `--gpu`, it also verifies the GPU output, parses the measured `gpu:`
   line, and records end-to-end, transfer, and kernel times;
6. writes timestamped JSON/Markdown plus `latest.json` / `latest.md`;
7. renders charts from the stored JSON with `scripts/plot_benchmarks.py`.

The `global_aggregate` query is verified and reported, but the pull-request check
does not gate its CPU/DuckDB ratio: DuckDB's thread scaling makes that scan-bound
ratio unstable across runners. The other benchmark queries remain gated at 3x.

The engine timing is its internal `execute_ms`: opening the scan through result
materialization, excluding process start, catalog load, parsing, and planning.
DuckDB is timed from Python around `execute(...).fetchall()`, which includes its
own planning and result materialization. Both run on the same warmed files in
the same WSL2 instance. The engine is single-threaded; DuckDB uses its default
16 threads. That asymmetry is recorded in the result file and repeated in the
report rather than hidden.

The benchmark intentionally includes a metadata-only case (`count_star`), a
query where row-group pruning dominates (`empty_filter`), and queries the engine
loses. It does not select only favorable cases.

Pull requests run the same verified harness on a fresh runner. The check
compares each query's CPU/DuckDB ratio with the committed baseline and fails on
a ratio increase of more than 3x; sub-millisecond queries are skipped because
scheduler noise dominates them. Raw PR results are uploaded as CI artifacts, so
a reviewer can inspect the numbers rather than trusting a summary.

## 5. Results

Latest run: 1,000,000 sales rows, 16 row groups, 22.5 MB, batch 8,192, AMD
Ryzen 7 170 (16 logical CPUs), 7.4 GiB RAM in WSL2, Ubuntu 24.04, NVIDIA
GeForce RTX 3050 Laptop GPU (4 GB, compute 8.6). The table below is CPU vs
DuckDB; the GPU runs are in Section 7.3.

| Query | qe CPU (ms) | DuckDB (ms) | Ratio | Winner |
|---|---:|---:|---:|---|
| filter_aggregate | 5.04 | 15.98 | 0.32x | qe |
| global_aggregate | 56.85 | 28.39 | 2.00x | DuckDB |
| empty_filter | 0.22 | 10.14 | 0.02x | qe |
| group_region | 172.14 | 34.95 | 4.93x | DuckDB |
| group_customer | 555.51 | 81.84 | 6.79x | DuckDB |
| topk | 93.55 | 42.11 | 2.22x | DuckDB |
| join_country | 435.34 | 60.64 | 7.18x | DuckDB |
| count_star | 0.01 | 8.40 | 0.00x | qe |

### Why the engine wins three queries

Two wins are selective scans. `empty_filter` asks for a date range after the
end of the data; Parquet statistics prove that no row group can match, so almost
nothing is read. `filter_aggregate` reads a one-month slice and aggregates it.
The engine's scan path has very little per-query overhead, while DuckDB still
pays for its optimizer and execution framework.

The third win is metadata-only `COUNT(*)`. Both engines avoid decoding the
table, but the engine's total is a few row-group metadata lookups and a
zero-column count batch, so `execute_ms` rounds to 0.01 ms. That number measures
low query overhead rather than scan throughput; it is reported because it is
real, not because it generalizes.

### Why the engine loses five queries

- **`global_aggregate`**: the full-table `SUM`/`AVG` still runs through the
  row-wise aggregate loop on the CPU. DuckDB's vectorized, multi-threaded
  aggregate is about 2x faster. The optional GPU path is compared in Section
  7.3; it narrows the gap but does not beat DuckDB or reliably beat the CPU.
- **`group_region` / `group_customer`**: the aggregate consumes a batch row by
  row, hashing a serialized key for each row. DuckDB uses vectorized,
  multi-threaded hash aggregation. The engine is roughly 4.9-6.8x slower even
  after the optimization described below.
- **`join_country`**: the hash join builds and probes row by row, then feeds a
  row-wise aggregate. DuckDB's join and aggregate pipelines are vectorized and
  parallel; the gap is about 7.2x.
- **`topk`**: the engine now partitions to the top K before sorting, but it still
  scans and filters every batch with one thread and hashes no work away. DuckDB
  combines a parallel scan with its own top-k path; the gap is about 2.2x.

The honest conclusion is the one in the brief: a student-built engine should
not be expected to beat a mature vectorized engine overall. The value is in the
three cases where the design choices pay off, and in being able to explain the
five cases where they do not.

## 6. A performance bug found by the benchmark

The first benchmark run exposed a pathological result: `group_region` took
**5,120 ms** while `group_customer` took 873 ms. A low-cardinality group-by
should be the easier query, so the ordering was a signal that something was
wrong, not merely "DuckDB is faster".

Three problems were found in `AggregateOperator::Consume`:

1. **A scalar per group-key value per row.** Every input row called
   `Array::GetScalar` for the group key and serialized the resulting
   `shared_ptr<Scalar>`. For string keys this allocated a `StringScalar` and a
   buffer copy for every row. The fix added `AppendArrayKey`, which writes the
   same type-tagged, length-prefixed key directly from the typed Arrow array.
   Key values are now materialized only when a new group is created.
2. **Arrow compute dispatch per MIN/MAX row.** `MIN` and `MAX` called
   `arrow::compute::CallFunction("less", ...)` for every row to compare the
   current value with the running extreme. That is a full function lookup and
   scalar-boxing path per comparison. The fix compares directly against the
   typed array value and only materializes a new scalar when the extreme
   actually changes.
3. **A scalar per aggregate argument per row.** `SUM`, `AVG`, and the initial
   `MIN`/`MAX` value also called `GetScalar` for every input row. Typed
   array readers for integers, doubles, dates, booleans, and strings now feed
   the aggregate states directly.

The effect on the same query and data:

| Stage | `group_region` median |
|---|---:|
| Before | 5,120 ms |
| After direct array key encoding | 4,559 ms |
| After direct MIN/MAX comparison | 259 ms |
| After direct aggregate argument reads (latest run) | 244.75 ms |

The key-encoding change alone barely moved this query because MIN/MAX
dominated it; it matters more for string-keyed group-bys without extrema. The
MIN/MAX change removed the real bottleneck. This is exactly why the benchmark
harness exists: the bug was invisible in correctness tests and only appeared
when a low-cardinality query was timed next to a high-cardinality one.

The remaining row-wise loops are a known limitation. Replacing them with
vectorized hash aggregation and a vectorized join probe is the largest
performance item on the version 2 list.

### 6.1 Two version 2 optimizations

Two smaller fixes removed work that the benchmark made visible:

- **Metadata-only `COUNT(*)`.** Column pruning can now produce a scan with no
  columns. When that scan has no predicate, the operator sums row counts from
  Parquet row-group metadata and emits a zero-column batch carrying the count.
  The aggregate's count-only path consumes it without touching a payload column.
  The scan statistics report `rows_from_metadata` and `rows_scanned=0`, so the
  optimization is observable rather than implied.
- **Bounded top-k.** A sort that feeds `LIMIT` now receives `limit_hint =
  limit + offset`. It partitions the input with `nth_element` and sorts only the
  K survivors instead of sorting every matching row. The subsequent limit
  operator applies the offset, so the result is unchanged.

Neither optimization changes SQL semantics, and both are covered by the same
DuckDB comparison tests as the unoptimized paths.

## 7. GPU status and design

The original plan called for CUDA kernels for filter and/or aggregation on an
RTX 3050, with explicit host-to-device transfer accounting and a CPU fallback.
The RTX 3050 Laptop GPU is now visible in both Windows and WSL2, and a CUDA
12.9.86 build can execute and measure a real reduction. The backend is still
deliberately narrow: it is an opt-in global-aggregate path, not a general GPU
execution engine.

### 7.1 What is implemented

- `QE_ENABLE_CUDA=ON` compiles `src/gpu/cuda_runtime.cu`; the default build
  compiles a stub and never links CUDA.
- `qe-cli --gpu` accepts no-`GROUP BY` queries that combine `COUNT` with
  floating-point `SUM`/`AVG`. The GPU computes the floating-point reductions;
  `COUNT`, the Parquet scan, filters, and all output formatting stay on the host.
- The operator buffers only valid values in bounded host chunks (1,048,576
  doubles per aggregate), then performs one host-to-device copy, one reduction
  kernel, and one device-to-host copy per chunk. A reusable `gpu::Workspace`
  keeps device buffers and CUDA events alive across chunks, avoiding
  `cudaMalloc` and event creation on every batch.
- The reduction kernel uses 256-thread blocks, up to 256 blocks, per-block
  partial sums in shared memory, and `atomicAdd` for the final partials. The
  self-test also exercises the validity-bitmap path.
- `--gpu` checks that CUDA was compiled in and that a device is available, then
  runs a real self-test before timing. The self-test loads the CUDA module so
  module initialization is not charged to the first timed query.
- Unsupported shapes (`GROUP BY`, integer sums, `MIN`/`MAX`, joins, non-aggregate
  queries) fall back to the CPU path and report `gpu: used=0`. A CUDA runtime
  failure during an active GPU query is returned as an error rather than
  silently replaying data on the CPU.
- `ExecutionStats` records whether a kernel ran, the number of GPU reductions,
  transfer milliseconds, and kernel milliseconds. The CLI prints the measured
  values on stderr even with `--no-stats`.

### 7.2 Verification

- `tests/cuda_test.cpp` queries the device, prints its name, compute capability,
  and memory, then runs a 2,048-value reduction with nulls and verifies the
  count and sum on the host. CTest treats a missing device as a skip.
- `tests/test_engine.py` compares GPU `COUNT`/`SUM`/`AVG` results with DuckDB,
  including a null-heavy edge-case table and a 7-row batch size. The GPU tests
  skip unless `qe-cli --version` reports `cuda: enabled`.
- The benchmark harness has an opt-in `--gpu` mode. It verifies both CPU and GPU
  results against DuckDB, parses the measured `gpu:` line, and records
  transfer/kernel timings. A CUDA build with no `--gpu` request records no GPU
  timings.

### 7.3 Measured result

The verified run used the RTX 3050 Laptop GPU (4 GB, compute 8.6, CUDA 12.9.86,
driver 591.84) on 1,000,000 sales rows. GPU end-to-end times are medians of
three runs after a warm-up; the CLI self-test ran before timing. Transfer and
kernel columns are the separately measured CUDA components.

| Query | CPU (ms) | GPU end-to-end (ms) | Transfer (ms) | Kernel (ms) | GPU / CPU |
|---|---:|---:|---:|---:|---:|
| `filter_aggregate` | 5.04 | 5.23 | 0.329 | 0.106 | 1.04x |
| `global_aggregate` | 56.85 | 66.36 | 2.682 | 0.180 | 1.17x |

The result is reported without a speedup claim. On `filter_aggregate`, the
kernel and transfer are tiny, but workspace setup and host buffering cost
slightly more than the 5.04 ms CPU query, so the GPU path is essentially tied
and 1.04x slower end to end. On `global_aggregate`, moving `SUM` and `AVG` over
one million rows to the GPU does not offset the host-side buffering: 66.36 ms
versus 56.85 ms on the CPU, a 1.17x GPU/CPU ratio. Repeated local runs of the
full-table sum straddle 1.0x, so the honest conclusion is roughly tied at best.
DuckDB remains faster on both queries because it vectorizes and multi-threads the
scan and aggregation; at this data size the Parquet scan and row-wise host work
dominate. The GPU is not claimed to accelerate queries that spend most of their
time decoding Parquet.

### 7.4 Remaining GPU work

The next useful steps are broader device coverage, not larger claims:

1. Grouped aggregation and GPU filter kernels, starting with low-cardinality
   group keys and row-group-sized batches.
2. A GPU scan/decode path or fused filter to avoid paying host decoding cost
   before a transfer; without it, global aggregates remain scan-bound.
3. Async, pinned transfers and overlap with host decoding instead of the current
   chunk-level copy/kernel/copy sequence.
4. Integer aggregate types, `MIN`/`MAX`, and joins.
5. A shared buffer for `SUM` and `AVG` over the same argument, which would
   remove the duplicate host pass visible in `global_aggregate`.

Those paths are intentionally not claimed until they can be measured on the
same verified device.

## 8. Limits and version 2 ideas

Known 0.2.0 limits:

- single-threaded, single-process, read-only;
- flat schemas only; nested Parquet types are rejected at catalog load;
- inner and left joins only, with equality keys whose Arrow types must match
  exactly; no right, full, cross, semi, or anti joins;
- no `LIKE`, `NULLS FIRST/LAST`, `UNION`/`EXCEPT`/`INTERSECT`, subqueries, CTEs,
  or window functions;
- no `DISTINCT` inside aggregates, no aggregate arguments that are expressions,
  and no `GROUP BY` expressions;
- in-memory sort and aggregation, with no spilling;
- row-wise inner loops for aggregation and join probing;
- the CUDA path is optional and limited to global floating-point `SUM`/`AVG`
  plus host-side `COUNT`; grouped aggregates, filters, joins, and integer sums
  remain CPU-only.

Version 2 work still open, in priority order:

1. Vectorized hash aggregation (batch-level key encoding and columnar state
   updates) and a vectorized join probe.
2. Multi-threaded scans and operators.
3. CUDA grouped aggregation, filter kernels, and joins, with separate
   transfer/kernel timers and CPU fallback; the global floating-point path is
   implemented and measured.
4. Remaining join types and join-key type coercion.
5. `LIKE`, set operations, subqueries, CTEs, and window functions.
6. Aggregate expressions, `DISTINCT` aggregates, and `GROUP BY` expressions.
7. Memory accounting and spilling for sorts and aggregations.
8. Additional Parquet encodings and nested-type support.

## 9. Lessons

- **Correctness before speed.** The engine was made to match DuckDB before any
  performance work. When the aggregate was optimized, the correctness suite caught
  nothing because the changes preserved semantics; without that suite, the
  optimization would have been a gamble.
- **Benchmarks find bugs that tests cannot.** The 5.1-second low-cardinality
  group-by was not a correctness failure. It was a design smell visible only in
  comparison with a neighboring query.
- **Negative tests can be too weak.** While adding tests for the new clauses,
  `BETWEEN 1 OR 2` was found to parse successfully: the parser's `Expect`
  checked that the next token was *a* keyword, not the required keyword. The
  parser now compares keyword text exactly, and the rejection suite covers it.
- **Honest losses are more useful than selective wins.** Reporting four losses
  forces a concrete explanation of where the engine's row-wise loops and
  single-threaded execution fall short. That explanation is more valuable than
  another favorable microbenchmark.
- **Scope discipline is the finish line.** The GPU path was implemented only
  after the RTX 3050 became visible and a real reduction could be verified.
  Keeping it opt-in preserved the CPU engine's default behavior, and the
  benchmark reports the honest outcome: in the committed run both GPU paths are
  slower end to end, though the global aggregate is close; scan and host
  setup costs dominate.

## 10. Reproducing everything

```bash
micromamba env create -f environment.yml
micromamba activate qe
scripts/build.sh
ctest --test-dir build --output-on-failure
python -m pytest tests -q
python scripts/fuzz_compare.py --queries 1000 --seed 2026
python scripts/run_benchmarks.py --rows 1000000 --repeats 5 --warmups 2
scripts/demo.sh
```

The contributor workflow, formatting commands, sanitizer build, and SQL-feature
checklist are in [`CONTRIBUTING.md`](../CONTRIBUTING.md). Release history is in
[`CHANGELOG.md`](../CHANGELOG.md).

The benchmark command regenerates `benchmarks/results/bench_<UTC>.json` and
`.md`, updates `latest.*`, and renders `benchmarks/charts/latest.png` from the
stored JSON. Hardware, versions, seed, row-group size, query text, and every
individual run are in the JSON. The dataset can be regenerated deterministically
with `scripts/generate_data.py`; the manifest records the seed and shape.
