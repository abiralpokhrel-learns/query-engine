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
`BETWEEN`, `OFFSET`, metadata-only `COUNT(*)`, a top-k sort, an opt-in CUDA
global-aggregate path (Section 7), batch-vectorized hash aggregation
(Section 6.2), batch-vectorized single-key hash joins (Section 6.3), and an
opt-in parallel row-group scan (`--threads`, Section 6.4).

At the time of writing:

- 207 tests pass against DuckDB on a CUDA build (198 CPU tests plus 9 GPU-gated
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
- `[NOT] IN (...)`, `[NOT] BETWEEN ... AND ...`, `[NOT] LIKE` with `%` and `_`;
- `COUNT(*)`, `COUNT(col)`, `SUM`, `AVG`, `MIN`, `MAX`.

`IN` and `BETWEEN` are rewritten during parsing into comparisons joined with
`OR`/`AND`. The rewrites preserve SQL three-valued logic, so `NOT IN` with a
`NULL` in the list still evaluates to `NULL` rather than a spurious match.
`LIKE` is evaluated with Arrow's regex kernel: the pattern is escaped, `%`
becomes `.*`, `_` becomes `.`, and the result is anchored, so a LIKE pattern can
only mean what SQL LIKE means. The pattern must be a string literal, and a
backslash is rejected because the two engines disagree about its meaning.

Everything outside that contract is rejected with an error that names the
construct (`RIGHT`/`FULL`/`CROSS` joins, `JOIN ... USING`, `NULLS FIRST/LAST`,
set operations, subqueries, unsupported functions, trailing tokens, and so on). The parser never guesses. Keyword expectations check the exact
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
  With `--threads N > 1` it decodes row groups concurrently: worker `w` owns the
  row groups at positions `w, w+N, ...`, each worker opens its own
  `parquet::arrow::FileReader`, and a decoded, filtered row group is buffered as
  one "unit" until the consumer reaches it. Units are consumed strictly in
  row-group order — the order the serial reader produces — and a fixed budget of
  `2N` decode tokens bounds how many row groups sit in memory at once. Slicing
  into `batch_size` rows, applying pushed filters, and skipping empty slices
  mirror the serial loop exactly, so the emitted batch sequence, results, and
  scan statistics are identical at any thread count. A worker error is captured
  under the coordinator's mutex and returned from the consumer's next pull, and
  the destructor sets a shutdown flag before joining.
- **Filter**: evaluates a predicate array and compacts the batch.
- **Project**: evaluates expressions with Arrow compute kernels.
- **Aggregate**: hash grouping with `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.
- **Join**: in-memory hash join over one or more equality keys. With a single
  integer or string key, the build side inserts into an open-addressed typed
  table and probes do inline matching; multi-key joins hash a type-tagged
  serialized key per row. Duplicate keys probe multiple build rows, and `NULL`
  keys never match. A left join materializes the preserved side, appends a typed
  `NULL` row to the build side, and points unmatched rows at that row.
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
- thread-count invariance (`--threads` 1, 2, 4, 8, including threads × batch
  size combinations), equal scan statistics between the serial and parallel
  scans, and an unchanged metadata-only `COUNT(*)` path under threads;
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

Latest run: 2026-11-12, 1,000,000 sales rows, 16 row groups, 22.5 MB, batch
8,192, AMD Ryzen 7 170 (16 logical CPUs), 7.4 GiB RAM in WSL2, Ubuntu 24.04,
CPU-only build. Five timed repetitions after two warm-ups at each scan thread
count; DuckDB uses its default 16 threads. Every thread count is verified
against DuckDB before timing. The GPU runs are in Section 7.3 and come from the
2026-11-09 run, because this one has no CUDA build.

| Query | qe 1 thread | qe 4 threads | qe 8 threads | DuckDB | 1-thread ratio | 8-thread ratio |
|---|---:|---:|---:|---:|---:|---:|
| filter_aggregate | 4.76 | 5.15 | 4.54 | 19.90 | 0.24x | 0.23x |
| global_aggregate | 52.27 | 26.61 | 29.64 | 33.58 | 1.56x | 0.88x |
| empty_filter | 0.09 | 0.07 | 0.11 | 9.40 | 0.01x | 0.01x |
| group_region | 116.50 | 54.78 | 63.42 | 41.56 | 2.80x | 1.53x |
| group_customer | 202.56 | 137.65 | 138.89 | 104.21 | 1.94x | 1.33x |
| topk | 119.14 | 53.38 | 58.54 | 61.41 | 1.94x | 0.95x |
| join_country | 297.83 | 203.53 | 204.69 | 75.87 | 3.93x | 2.70x |
| count_star | 0.01 | 0.01 | 0.01 | 7.25 | 0.00x | 0.00x |

Wins against DuckDB: 3/8 at one thread, 5/8 at four, 5/8 at eight. The parallel
scan is what moves the story: `global_aggregate` and `topk` cross below 1.0x
once decode overlaps the single-threaded operator behind it, and the three
remaining losses are operator-bound rather than scan-bound.

Four threads matched or beat eight on every scan-bound query. That is the
expected shape for this dataset: 16 row groups distributed across `N` workers
leaves two units per worker at eight threads and four at four, while the
consumer stays serial, so the fifth through eighth readers mostly add memory
residency and core contention. The scaling limit is the operator pipeline, not
the decoder, which is the argument for the parallel-operator work in Section 8.

Absolute times drift between sessions: every query here moved against the
previous committed run, including DuckDB itself, which is why cross-session
ratios are context rather than attribution. The alternating A/B runs in
Sections 6.2, 6.3, and 6.4 remain the attribution measurements for the
individual optimizations.

### Why the engine wins five queries at four threads

Two wins are selective scans. `empty_filter` asks for a date range after the
end of the data; Parquet statistics prove that no row group can match, so almost
nothing is read. `filter_aggregate` reads a one-month slice and aggregates it.
The engine's scan path has very little per-query overhead, while DuckDB still
pays for its optimizer and execution framework.

Two more are scan-parallel wins rather than operator wins. `topk` (0.87x) and
`global_aggregate` (0.79x) at four threads each spend most of their single
threaded time waiting on Parquet decode; overlapping that decode with the sort
or aggregate behind it is enough to pass DuckDB even though DuckDB's operators
are vectorized and parallel. This is a statement about this dataset and this
machine, not about the operators: neither query's operator code changed.

The third win is metadata-only `COUNT(*)`. Both engines avoid decoding the
table, but the engine's total is a few row-group metadata lookups and a
zero-column count batch, so `execute_ms` rounds to 0.01 ms. That number measures
low query overhead rather than scan throughput; it is reported because it is
real, not because it generalizes.

### Why the engine still loses three queries

- **`group_region` / `group_customer`**: group IDs are built for the whole
  batch and the aggregates are updated in a fused pass, but grouping, result
  materialization, and the final sort all remain single-threaded. At four scan
  threads the gaps are 1.32x and 1.32x; at eight they are 1.53x and 1.33x. The
  scan is no longer the explanation, so further progress needs a parallel
  operator rather than more decode concurrency.
- **`join_country`**: the probe is batch-vectorized for single integer and
  string keys, but the probe loop, the per-column `Take` that materializes
  joined rows, and the following aggregate all run on one thread, while DuckDB's
  join pipeline is vectorized and parallel. The gap is 2.68x at four threads and
  2.70x at eight, down from 7.3x before the typed table and probe landed
  (attribution in Section 6.3). It is the largest remaining loss.

The honest conclusion is the one in the brief: an engine with a parallel scan
in front of a single-threaded operator pipeline should not be expected to beat
a mature vectorized engine overall. The value is in the five cases where the
design choices pay off, and in being able to explain the three cases where they
do not.

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

The aggregate hot path was later replaced by the batch-vectorized design in
Section 6.2, and the hash join probe followed in Section 6.3. The largest
remaining row-wise costs are the single-threaded scan and the sort/output path.

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

### 6.2 Batch-vectorized hash aggregation

The old aggregate called `UpdateGroup` once per input row. For every row it
serialized the grouping key, looked it up, and then looped over every aggregate,
converting each argument with a per-row type switch. MIN/MAX retained an Arrow
scalar and could allocate a new one whenever the extreme changed. That design
was simple and correct, but it paid key-construction, dispatch, and allocation
costs one million times per million-row query.

The current implementation separates grouping into two batch-level phases:

1. **Build group IDs for the whole batch.** Integer keys use an open-addressed
   flat table, or a direct-index table when the observed key span is at most
   4,194,304 values (16 MiB of `int32` slots). Compact keys therefore need one
   array lookup per row instead of a hash probe. If a later batch widens the
   range too far, the operator rebuilds the flat table and continues, so
   correctness does not depend on the first batch being representative.
   Single-string keys with at most 32 groups stay in a small linear list that
   avoids constructing and hashing a key string; the list migrates to the
   generic hash map when cardinality grows. Multiple keys continue to use the
   serialized-key map.
2. **Accumulate in one fused pass.** Argument values and validity bitmaps are
   resolved once per batch. A single row loop then updates all aggregates for
   that row's group in adjacent typed state. COUNT/SUM/AVG and floating-point
   MIN/MAX are fused; integer, boolean, date, and string MIN/MAX use the typed
   fallback pass. Group state is allocated in blocks rather than one small
   vector per group, and string extrema are allocated only for string MIN/MAX.

The GPU global-aggregate path is unchanged: grouping remains CPU-only and the
CUDA path still handles only no-`GROUP BY` floating-point reductions.

The committed benchmark improved `group_customer` from 555.51 ms to 230.04 ms
(2.41x) and `group_region` from 172.14 ms to 120.10 ms (1.43x). Those two runs
are from different sessions and include the absolute-time drift described
above, so the causal comparison is a separate alternating A/B run against the
previous commit: five rounds of minimum `execute_ms` on the same machine
measured 490 -> 203 ms for `group_customer` (2.42x), 200 -> 117 ms for
`group_region` (1.71x), 68 -> 53 ms for `global_aggregate` (1.27x), and
essentially no change for `topk` or `join_country`, where the join dominates.
The engine is still single-threaded, so this narrows the DuckDB gap rather
than closing it.

Tests cover dense-key expansion across batches in both directions, fallback when
a key span is too wide for the direct table, NULL group keys, low- and
high-cardinality string groups, typed integer, boolean, date, and string
MIN/MAX, and result invariance at batch sizes 1, 7, 128, and 8,192.

### 6.3 Batch-vectorized hash join

The join already shared two properties with the aggregate: the build side is
materialized once, and the probe ran one serialized-key lookup per probe row.
Unlike grouping, a join's build side never has to migrate between structures
because it is complete before the first probe batch arrives; the fast path is
therefore simpler than the aggregate's dense-to-flat migration:

1. **Build a typed table once.** With exactly one equality condition, integer
   keys (`int8`..`uint64`, `date32`) go into an open-addressed table sized to a
   0.5 load factor, and single string keys store a `std::string_view` pointing
   into the retained build-key buffer plus its hash. Build rows with `NULL` keys
   are skipped, matching SQL equality. Each slot carries a chain (`head` and
   `last` row indices plus a per-row `next` index) that is appended in ascending
   build-row order, so duplicate-key matches come out in the same order as the
   old `std::unordered_map<std::string, std::vector<int64_t>>` produced them.
   Multi-key conditions and float keys keep the serialized-key path and its
   exact bitwise key semantics.
2. **Probe in one fused pass per batch.** The probe key array is evaluated once
   per batch; one loop then computes the lookup inline (integer value or
   string-view hash plus content compare on a linear-probe collision), walks the
   chain, and appends `(probe_row, build_row)` index pairs directly into the
   `Take` builders. No key string is constructed and no map is touched for
   common single-key shapes. The left-outer path reuses the same loop with
   unmatched rows appended at the sentinel row, preserving the previous output
   order exactly.

The same committed run reports `join_country` at 304.51 ms versus 548.12 ms in
the previous session's run, but those are cross-session numbers and this query
drifts heavily; the causal comparison is an alternating A/B against the
previous commit: 14 interleaved rounds, minimum `execute_ms` on the same
machine, measured 536.5 -> 300.2 ms for `join_country` (1.79x) and 370.6 ->
226.5 ms for a bare filter-project-join query with no aggregate (1.64x).
Medians tell the same story (591 -> 324 ms and 407 -> 239 ms).
The remaining time is the single-threaded scan of 1.05 M rows, output
materialization, and the following aggregate.

Tests add a dedicated build/probe pair with duplicate keys on both sides, a
`NULL` key on each side, unmatched probe keys, `INT64` extremes, and an
empty-string key, run as inner and left joins at batch sizes 1, 7, 128, and
8,192 against DuckDB; the differential fuzzer (400 queries, two seeds) also
exercises the new path.

### 6.4 Parallel row-group scan

The next bottleneck is the serial scan itself: every benchmark query pays one
thread for decoding 1.05 M rows before any operator sees them. The response is
an opt-in parallel scan (`--threads N`) rather than a parallel executor — the
planner and operators stay untouched, and the default build keeps the
single-threaded behavior.

The design trades decode parallelism for a strict ordering guarantee. Row
groups are distributed round-robin to `N` workers, each owning one independent
`parquet::arrow::FileReader` (the Arrow reader is not safe to share across
threads). A worker decodes its row group completely, applies the pushed filters
per `batch_size` slice, and publishes the unit; the consumer hands out batches
only from the row group the serial reader would have reached next. A fixed
budget of `2N` decode tokens bounds the buffer, so peak memory stays close to
the serial scan and `--threads 1024` cannot turn into an unbounded prefetch.
Because the batch sequence is bit-identical to the serial scan, downstream
`COUNT`/`SUM`/`AVG` accumulation, sort stability, and output order cannot
depend on the thread count.

Writing the thread-invariance tests caught a latent aggregate bug before the
feature shipped. The vectorized fast path read `raw_values()` and indexed it
with an additional `offset + row`, but Arrow's `raw_values()` already includes
the slice offset, so any value array with a nonzero offset was read shifted.
Serial scans only ever emit offset-zero batches — the reader builds fresh
arrays and filters compact them — which is why the batch-size sweep never
noticed. The parallel scan slices each decoded row group, and the first
`--threads 4, --batch-size 7` test run returned a `SUM` of `-9.94e+298` with a
correct `COUNT` and `AVG` (the `AVG` argument was cast to double first, giving
a fresh offset-zero array; `COUNT(*)` reads no values at all). The fix resolves
offset-exclusive pointers for values and keeps the explicit offset only for the
validity bitmap, where it belongs; the same bug hid in five more accumulation
loops (integer `SUM`, integer and double `MIN`/`MAX`).

An alternating same-session A/B on the 1M-row dataset (`--threads 8` vs
`--threads 1`, 7 interleaved rounds, 5 repetitions per invocation, minimum
`execute_ms`; the same Ryzen 7 170 machine as the committed run, dataset read
from the ext4 home directory rather than `/mnt/c`) measured:

| Query | threads=1 | threads=8 | speedup |
|---|---:|---:|---:|
| `topk` (sort over 1M rows) | 33.7 ms | 11.8 ms | 2.9x |
| `group_region` (`GROUP BY` 1M rows) | 56.9 ms | 29.2 ms | 1.95x |
| global `COUNT`/`SUM`/`AVG` | 18.8 ms | 10.5 ms | 1.8x |
| `join_country`-shaped aggregate | 233.1 ms | 196.6 ms | 1.2x |

Medians tell the same story (2.7x, 2.1x, 1.7x, 1.1x). The scan-bound queries
gain most because decode overlaps with the single-threaded aggregate and sort;
the join query gains least because its serial hash probe and key serialization
dominate. The committed benchmark JSON now records this scaling directly:
`--scan-threads 1,4,8` times the same verified queries at each worker count, so
Section 5's table is produced by the harness rather than by comparing the A/B
run against the default build.

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
| `filter_aggregate` | 7.18 | 10.11 | 0.217 | 0.050 | 1.41x |
| `global_aggregate` | 63.49 | 80.57 | 3.179 | 0.629 | 1.27x |

The result is reported without a speedup claim, and the CPU won both queries in
this run. On `filter_aggregate`, the kernel and transfer are tiny; the 10.11 ms
end-to-end result versus 7.18 ms on the CPU is within the noise this few-ms
query shows between local runs, and earlier committed runs had the GPU slightly
ahead of the CPU on it, so the honest conclusion is a tie at best rather than a
speedup. On `global_aggregate`, moving `SUM` and `AVG` over one million rows to
the GPU does not offset the host-side buffering: 80.57 ms versus 63.49 ms on the
CPU, a 1.27x GPU/CPU ratio. Repeated local runs of the full-table sum straddle
1.0x, so the honest conclusion is roughly tied at best. DuckDB remains faster on
both queries because it vectorizes and multi-threads the scan and aggregation;
at this data size the Parquet scan and host-side work dominate. The GPU is not
claimed to accelerate queries that spend most of their time decoding Parquet.

### 7.4 Scaling to ten million rows

The open question was whether the GPU path becomes worthwhile once transfers are
amortized over more rows. On this device it does not. The same binary
(`build-cuda/qe-cli`, `qe` 0.3.0) measured one and ten million sales rows in one
session, three repetitions after one warm-up, with every result verified against
DuckDB before timing. The artifacts are stored in
`benchmarks/results/gpu_1m_20261112.json` and
`benchmarks/results/gpu_10m_20261112.json`.

| Rows | Query | CPU (ms) | GPU end-to-end (ms) | Transfer (ms) | Kernel (ms) | GPU / CPU |
|---|---:|---:|---:|---:|---:|---:|
| 1M | `filter_aggregate` | 6.10 | 9.10 | 0.326 | 0.288 | 1.49x |
| 1M | `global_aggregate` | 54.58 | 97.81 | 3.966 | 0.602 | 1.79x |
| 10M | `filter_aggregate` | 35.56 | 34.29 | 0.532 | 0.256 | 0.96x |
| 10M | `global_aggregate` | 525.70 | 752.87 | 59.265 | 27.872 | 1.43x |

Three things follow from those numbers.

1. **Transfer is not what limits the path.** At ten million rows the transfer and
   kernel together are 87.1 ms of a 752.87 ms query. The other 665.7 ms is
   host-side work, so making the data larger cannot fix the ratio.
2. **The GPU components do not scale linearly.** Transfer grows 14.9x for 10x the
   data and the kernel grows 46x. That is per-flush and sustained-load clock
   behaviour on a laptop GPU rather than bandwidth, and it needs its own
   measurement before any grouped-kernel work is justified.
3. **The one GPU win is not a speedup claim.** `filter_aggregate` at ten million
   rows is 34.29 ms against 35.56 ms on the CPU, and DuckDB does the same query
   in 28.13 ms. A 1.3 ms difference is reported as a measurement, not a win.

This reorders the remaining GPU work. Grouped aggregation would need a
GPU-resident hash table *and* the removal of the duplicate host pass, and even
then the CPU scan in front of it costs more than DuckDB's entire query at this
size: 525.70 ms of the 752.87 ms GPU `global_aggregate` on the host, against
161.87 ms for all of DuckDB's work. The honest order is to remove the host pass,
measure kernel scaling in isolation, and only then consider device-side grouping.

### 7.5 Remaining GPU work

The next useful steps are broader device coverage, not larger claims. Section
7.4 changes their order: the host-side copy and the kernel's scaling come first,
because they bound what any wider device path could win.

1. Remove the duplicate host pass: share one buffer between `SUM` and `AVG` over
   the same argument, and push Arrow buffers to the device instead of copying
   them into a `std::vector<double>` first.
2. Measure a kernel-only reduction at several sizes to explain the 46x kernel
   growth between one and ten million rows.
3. Grouped aggregation and GPU filter kernels, starting with low-cardinality
   group keys and row-group-sized batches, after items 1 and 2.
4. A GPU scan/decode path or fused filter to avoid paying host decoding cost
   before a transfer; without it, global aggregates remain scan-bound.
5. Async, pinned transfers and overlap with host decoding instead of the current
   chunk-level copy/kernel/copy sequence.
6. Integer aggregate types, `MIN`/`MAX`, and joins.

Those paths are intentionally not claimed until they can be measured on the
same verified device.

## 8. Limits and version 2 ideas

Known 0.3.0 limits:

- read-only, single-process; operators other than the scan are
  single-threaded, and the scan decodes row groups in parallel only with
  opt-in `--threads`;
- flat schemas only; nested Parquet types are rejected at catalog load;
- inner and left joins only, with equality keys whose Arrow types must match
  exactly; no right, full, cross, semi, or anti joins;
- no `NULLS FIRST/LAST`, `UNION`/`EXCEPT`/`INTERSECT`, subqueries, CTEs, or
  window functions; `LIKE` supports `%` and `_` and rejects patterns containing
  a backslash, and a pattern must be a string literal;
- no `DISTINCT` inside aggregates, no aggregate arguments that are expressions,
  and no `GROUP BY` expressions;
- in-memory sort and aggregation, with no spilling;
- scalar accumulation loops within vectorized aggregation batches, row-wise
  key serialization for multi-key joins, and no parallel operators beyond the
  scan;
- the CUDA path is optional and limited to global floating-point `SUM`/`AVG`
  plus host-side `COUNT`; grouped aggregates, filters, joins, and integer sums
  remain CPU-only.

Version 2 work still open, in priority order:

1. Parallel operators beyond the scan: join build/probe, aggregation, sort,
   and the executor loop itself. The 0.3.0 scan-thread measurements make this
   the priority and also pick the target: at four scan threads the three
   remaining losses are operator-bound (1.32x, 1.32x, 2.68x) and the join is the
   largest, so parallel aggregation alone would not move the worst case.
2. CUDA grouped aggregation, filter kernels, and joins, with separate
   transfer/kernel timers and CPU fallback; the global floating-point path is
   implemented and measured, and Section 7.4 argues the host-side copy has to
   go first.
3. Remaining join types and join-key type coercion.
4. Set operations, subqueries, CTEs, and window functions, then broader `LIKE`
   (a column pattern or an `ESCAPE` clause).
5. Aggregate expressions, `DISTINCT` aggregates, and `GROUP BY` expressions.
6. Memory accounting and spilling for sorts and aggregations.
7. Additional Parquet encodings and nested-type support.

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
