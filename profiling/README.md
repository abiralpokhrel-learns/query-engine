# Profiling qe

`EXPLAIN ANALYZE` reports wall time, self time, rows, and batches for every
physical operator. Use it first: it answers "which operator is slow" without a
profiler.

This directory records the lower-level profile used to confirm where time goes
inside the operators that remain expensive.

## Environment

- WSL2, Linux 6.18.33.2-microsoft-standard-WSL2.
- `perf` is not installed, and `kernel.perf_event_paranoid` is `2`. WSL2 does
  not expose a usable hardware PMU, so the reproducible path is callgrind.
- Valgrind 3.22.0 from the Ubuntu noble package. The machine has no passwordless
  sudo, so the `.deb` was unpacked under `~/valgrind-local` and run with
  `VALGRIND_LIB` pointed at its libexec directory.

## Reproduce

```bash
bash scripts/profile.sh
```

The script builds `build-profile` with `RelWithDebInfo`, tries `perf record`
when perf exists, and falls back to callgrind. The command is:

```bash
valgrind --tool=callgrind --callgrind-out-file=profiling/callgrind.out \
  --collect-atstart=no "--toggle-collect=qe::ExecutePlan*" \
  -- build-profile/qe-cli data/sales.parquet \
  "SELECT region, COUNT(*), SUM(amount) FROM sales WHERE amount > 100 \
   GROUP BY region ORDER BY region" --no-stats
callgrind_annotate --inclusive=no --threshold=95 --auto=no \
  profiling/callgrind.out
```

`--toggle-collect` starts collection only inside `qe::ExecutePlan`, so dynamic
loader and process-startup instruction counts do not swamp the query.

## What the profile showed

The query scans 200,000 rows, reads `region` and `amount`, pushes
`amount > 100` into the scan, and produces nine groups. Callgrind recorded
132,680,989 instructions inside query execution. The dominant self costs were:

- Parquet decoding, dictionary expansion, Snappy decompression, and Arrow
  builders: about 52% of instructions.
- Arrow `Filter`/`Take` kernels plus `qe::ApplyFilter`: about 12%.
- `AggregateOperator::AccumulateBatch` and `BuildGroupIds`: about 9%.

That matches the `EXPLAIN ANALYZE` split: the scan is the largest inclusive
cost, the filter is next, and the aggregate's own work is a smaller share. The
callgrind report is committed as `callgrind-report.txt`; the raw
`callgrind.out` is ignored because it is large and machine-specific.

## Why there is no flame graph

`perf` is absent and this WSL2 kernel does not expose the hardware PMU needed
for sampling, so a perf flame graph could not be produced here. The callgrind
report is the requested fallback. On a native Linux host, `scripts/profile.sh`
uses `perf record` and, when `stackcollapse-perf.pl` and `flamegraph.pl` are on
`PATH`, writes `profiling/flamegraph.svg`.
