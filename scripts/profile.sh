#!/usr/bin/env bash
# Profile a qe query with perf when available, otherwise callgrind.
#
# WSL2 commonly exposes no hardware PMU and may not ship perf at all. The
# callgrind fallback only needs Valgrind and produces a function-level report
# that is committed next to this script's output.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="${BUILD_DIR:-build-profile}"
BINARY="$BUILD_DIR/qe-cli"
DATA="${DATA:-data/sales.parquet}"
QUERY="${1:-SELECT region, COUNT(*), SUM(amount) FROM sales WHERE amount > 100 GROUP BY region ORDER BY region}"
OUT_DIR="${OUT_DIR:-profiling}"
mkdir -p "$OUT_DIR"

if [[ ! -x "$BINARY" ]]; then
  echo "profile: building $BINARY (RelWithDebInfo)" >&2
  cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_PREFIX_PATH="${QE_PREFIX:-$HOME/qe-env}"
  cmake --build "$BUILD_DIR" -j "${JOBS:-4}"
fi

PERF_OUT="$OUT_DIR/perf.data"
if command -v perf >/dev/null 2>&1; then
  echo "profile: perf record" >&2
  if perf record -o "$PERF_OUT" -F 999 -g --call-graph dwarf -- \
      "$BINARY" "$DATA" "$QUERY" --no-stats; then
    perf report -i "$PERF_OUT" --stdio --percent-limit 1 \
      | tee "$OUT_DIR/perf-report.txt"
    if command -v stackcollapse-perf.pl >/dev/null 2>&1 &&
       command -v flamegraph.pl >/dev/null 2>&1; then
      perf script -i "$PERF_OUT" \
        | stackcollapse-perf.pl \
        | flamegraph.pl >"$OUT_DIR/flamegraph.svg"
      echo "profile: wrote $OUT_DIR/flamegraph.svg" >&2
    fi
    exit 0
  fi
  echo "profile: perf failed (WSL often blocks counters); using callgrind" >&2
fi

VALGRIND_BIN="${VALGRIND_BIN:-$(command -v valgrind || true)}"
if [[ -z "$VALGRIND_BIN" && -x "$HOME/valgrind-local/usr/bin/valgrind" ]]; then
  VALGRIND_BIN="$HOME/valgrind-local/usr/bin/valgrind"
  export VALGRIND_LIB="${VALGRIND_LIB:-$HOME/valgrind-local/usr/libexec/valgrind}"
fi
if [[ -z "$VALGRIND_BIN" ]]; then
  echo "profile: neither perf nor valgrind found" >&2
  echo "profile: install perf, or unpack the valgrind .deb under ~/valgrind-local" >&2
  exit 2
fi

CALLGRIND_OUT="$OUT_DIR/callgrind.out"
CALLGRIND_ANNOTATE="$(dirname "$VALGRIND_BIN")/callgrind_annotate"
# Instrument only the query execution, not process startup and library
# relocation. Without this, the dynamic loader dominates the instruction count.
"$VALGRIND_BIN" --tool=callgrind --callgrind-out-file="$CALLGRIND_OUT" \
  --collect-atstart=no "--toggle-collect=qe::ExecutePlan*" \
  -- "$BINARY" "$DATA" "$QUERY" --no-stats
"$CALLGRIND_ANNOTATE" --inclusive=no --threshold=95 --auto=no "$CALLGRIND_OUT" \
  | tee "$OUT_DIR/callgrind-report.txt"
echo "profile: wrote $OUT_DIR/callgrind-report.txt" >&2
