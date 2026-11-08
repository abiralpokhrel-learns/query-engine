# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Contributor guide, pull-request checklist, and issue templates.
- `clang-format` and `ruff` configuration, with format checks in CI.
- ASan/UBSan CI job, warnings-as-errors builds, coverage reporting, and a
  repository-hosted coverage badge.
- Pull-request benchmark verification with a ratio-based regression check.
- Mermaid architecture diagram and a real CLI help transcript in the README.
- README terminal demo (GIF and static poster) rendered from actual CLI output,
  an embedded chart generated from the committed benchmark JSON, release/CUDA
  badges, and a short engineering-highlights section.
- Community health files: private vulnerability-reporting policy, code of
  conduct, and cross-editor defaults.
- Branch protection requiring pull requests and green CI before merge.
- Optional CUDA global-aggregate path (`QE_ENABLE_CUDA`, `--gpu`) for
  no-`GROUP BY` queries that combine `COUNT` with floating-point `SUM`/`AVG`;
  the GPU computes the reductions while `COUNT`, filtering, and the scan stay
  on the host. The CPU path remains the default and unsupported queries fall
  back to CPU. The binary reports device availability and per-query GPU
  transfer/kernel timing.
- `--gpu` mode in the benchmark harness: GPU results are verified against
  DuckDB before timing, and the report records the measured GPU end-to-end,
  transfer, and kernel times. The committed run reports both GPU paths as
  roughly tied or slower than the CPU rather than making a blanket speedup
  claim.

### Changed

- CI now uses the current `actions/checkout` and `setup-micromamba` releases.

## [0.2.0] - 2026-11-06

First public release of `qe`, a columnar SQL query engine built from first
principles in C++20.

### Added

- Parquet storage with column pruning, predicate pushdown, row-group skipping,
  and metadata-only `COUNT(*)`.
- SQL subset with `DISTINCT`, inner and left joins, `WHERE`, `GROUP BY`,
  `HAVING`, `ORDER BY`, `LIMIT`, `OFFSET`, `IN`, and `BETWEEN`.
- Aggregates: `COUNT(*)`, `COUNT(column)`, `SUM`, `AVG`, `MIN`, and `MAX`.
- Vectorized batch pipeline and bounded top-k sorting for `ORDER BY ... LIMIT`.
- 166 end-to-end correctness tests against DuckDB, a C++ parser/unit suite, and
  a deterministic differential fuzzer.
- Verified CPU-vs-DuckDB benchmarks on one million rows, with stored JSON,
  Markdown reports, and generated charts.
- Engineering write-up covering architecture, correctness strategy, benchmark
  methodology, honest wins and losses, and the deferred CUDA design.

### Fixed

- Parser keyword expectations now require the exact keyword rather than any
  keyword token, rejecting malformed clauses such as `BETWEEN 1 OR 2`.
- Fuzz reporting now shows the correct pass count when a run stops early.

[Unreleased]: https://github.com/abiralpokhrel-learns/query-engine/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/abiralpokhrel-learns/query-engine/releases/tag/v0.2.0
