#!/usr/bin/env bash
# Profile the optimized reduction kernel with Nsight Compute.
#
# The conda package installs ncu under nsight-compute/<version>/ncu, while the
# wrapper in bin/ expects nsight-compute-<version>/; invoke the real binary.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/build-cuda-microbench/reduce_bench}"
OUT="${OUT:-$ROOT/cuda-microbench/results/ncu.txt}"
SIZE="${SIZE:-16777216}"

if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN; build it with:" >&2
  echo "  cmake -S cuda-microbench -B build-cuda-microbench -G Ninja \\" >&2
  echo "    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86" >&2
  echo "  cmake --build build-cuda-microbench -j4" >&2
  exit 1
fi

NCU="${NCU:-$(command -v ncu || true)}"
if [[ -z "$NCU" || "$NCU" == "$HOME/micromamba/envs/ncu/bin/ncu" ]]; then
  for candidate in "$HOME"/micromamba/envs/ncu/nsight-compute/*/ncu; do
    [[ -x "$candidate" ]] && NCU="$candidate" && break
  done
fi
if [[ -z "$NCU" || ! -x "$NCU" ]]; then
  cat >&2 <<'EOF'
ncu not found. Install it in its own environment:
  micromamba create -y -n ncu -c nvidia nsight-compute
Then invoke the real binary under ~/micromamba/envs/ncu/nsight-compute/*/ncu.
EOF
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
echo "profiling $BIN with $NCU" | tee "$OUT"
set +e
"$NCU" \
  --target-processes all \
  --kernel-name regex:reduce_warp_kernel \
  --launch-count 1 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
  "$BIN" --sizes "$SIZE" --iters 2 --no-transfer --no-cub 2>&1 | tee -a "$OUT"
status=${PIPESTATUS[0]}
set -e

if grep -q "ERR_NVGPUCTRPERM" "$OUT"; then
  cat >&2 <<'EOF'

Nsight Compute is installed, but this driver denied performance counters.
On Windows, enable NVIDIA Control Panel -> Developer -> Manage GPU Performance
Counters -> allow access, then restart WSL. Native Linux hosts usually need the
driver's perf-counter permission configured by an administrator. The benchmark
still measures achieved bandwidth with CUDA events, so the numbers in
results/latest.md remain valid; only the hardware-counter breakdown is blocked.
EOF
  exit 2
fi
exit "$status"
