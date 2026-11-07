# Contributing to qe

Thanks for taking an interest in `qe`. The project is a small, deliberately
scoped columnar SQL engine. Contributions are most useful when they preserve the
three rules that define it:

1. **Correctness is measured, not assumed.** SQL behavior is compared with
   DuckDB on the same Parquet files.
2. **Benchmarks are honest.** Performance numbers come from
   `scripts/run_benchmarks.py`, and losses are reported alongside wins.
3. **Unsupported work is refused clearly.** The parser rejects SQL outside the
   documented subset instead of guessing.

## Development workflow

`main` is protected. Every change must go through a pull request with green CI.
Do not push directly to `main`.

```bash
git switch -c feature/short-description
# ... make one focused change ...
scripts/build.sh
ctest --test-dir build --output-on-failure
python -m pytest tests -q
git push -u origin feature/short-description
gh pr create --fill
```

Keep commits small and independently understandable. Prefer commit messages
such as `engine: vectorize hash join probe` or `sql: reject trailing tokens`
over `fix stuff` or `updates`.

## Environment

The pinned environment is in `environment.yml` and uses conda-forge:

```bash
micromamba env create -f environment.yml
micromamba activate qe
```

`scripts/build.sh` honours `QE_PREFIX`, `CONDA_PREFIX`, `CMAKE_BIN`,
`BUILD_DIR`, `CMAKE_BUILD_TYPE`, and `QE_WARNINGS_AS_ERRORS=ON`. It falls back
to `~/qe-env` when no environment is active.

## Build and test

```bash
scripts/build.sh
ctest --test-dir build --output-on-failure
python -m pytest tests -q
python scripts/fuzz_compare.py --queries 1000 --seed 2026
scripts/demo.sh
```

The Python suite is the main correctness gate: it generates deterministic
Parquet data, runs each query through `qe-cli` and DuckDB, and compares every
value. It also checks batch-size invariance, row-group pruning, column pruning,
and metadata-only `COUNT(*)`.

For a sanitizer build:

```bash
cmake -S . -B build-sanitize -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DQE_WARNINGS_AS_ERRORS=ON \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-sanitize -j 4
QE_CLI=build-sanitize/qe-cli python -m pytest tests -q
```

## Code style

- C++ is formatted with `clang-format` (see `.clang-format`).
- Python is formatted and linted with `ruff` (see `pyproject.toml`).
- CI builds with `-Wall -Wextra -Wpedantic -Werror`.

Run the same checks locally before pushing:

```bash
clang-format -i $(git ls-files '*.cpp' '*.h')
ruff format .
ruff check .
```

## Adding a SQL feature

A SQL feature is not complete when the parser accepts it. Work through the
whole path:

1. **Parser and AST** — add tokens or expression nodes in `src/sql/`, then update
   the recursive-descent parser. Unsupported forms must produce a useful error.
2. **Planner** — wire the feature into `src/engine/plan.cpp`, including alias
   resolution, column pruning, and any pushdown or limit hints it affects.
3. **Execution** — implement or extend the operator in `src/engine/operators.cpp`
   using Arrow columnar kernels where possible. Preserve SQL `NULL` semantics.
4. **Parser tests** — add positive and negative cases to `tests/sql_test.cpp`.
5. **DuckDB comparisons** — add queries to `tests/test_engine.py`, including
   `NULL`, empty-input, and multi-batch cases.
6. **Fuzzing** — extend `scripts/fuzz_compare.py` if the feature belongs in its
   generated query space.
7. **Docs** — update the SQL subset and known limits in `README.md`, and add a
   `CHANGELOG.md` entry under `Unreleased`.
8. **Benchmarks** — if the feature is performance-sensitive, run the harness and
   report the result honestly, including regressions.

Use `--explain` and the scan statistics to confirm that pruning and pushdown
behave as expected rather than assuming they do.

## Benchmarks and GPU work

Never invent a timing, a speedup, or a GPU result. The benchmark harness
verifies every query against DuckDB before timing it and records the hardware,
versions, dataset manifest, and every raw run. GPU fields stay `null` when no
usable CUDA device is present.

Run benchmarks with:

```bash
python scripts/run_benchmarks.py --rows 1000000 --repeats 5 --warmups 2
```

Pull requests that touch scan, aggregation, join, or sort code run a smaller
benchmark verification job and compare CPU/DuckDB ratios with the committed
baseline. Catastrophic regressions fail the check; ordinary noise does not.

## Pull requests

Use the pull-request template. A good PR includes:

- one focused change and a clear summary;
- tests that fail before the change and pass after it;
- updated documentation and changelog for user-visible behavior;
- benchmark evidence when performance is relevant;
- no fabricated or hand-typed measurements.

CI must be green before merge. The required checks are the main build/test job,
the ASan/UBSan job, and the coverage job.
