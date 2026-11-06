#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CMAKE_BIN="${CMAKE_BIN:-$(command -v cmake || true)}"
CMAKE_BIN="${CMAKE_BIN:-$HOME/qe-env/bin/cmake}"
if [[ -n "${QE_PREFIX:-}" ]]; then
  PREFIX="$QE_PREFIX"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  PREFIX="$CONDA_PREFIX"
else
  PREFIX="$HOME/qe-env"
fi
BUILD_DIR="${BUILD_DIR:-build}"

"$CMAKE_BIN" -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  -DCMAKE_PREFIX_PATH="$PREFIX"
"$CMAKE_BIN" --build "$BUILD_DIR" -j "${JOBS:-4}"
