# CUDA reduction microbenchmark

This is a standalone CUDA project, separate from the query engine. It answers a
narrow question that the engine's GPU path could not answer on its own: how
close is a hand-written reduction to the memory bandwidth of this GPU, and what
does the host/device transfer cost?

The engine's CUDA path only handles a global `SUM`/`AVG` over a floating-point
column. This project isolates that operation, compares four implementations,
adds a filter-plus-aggregate variant, and records the transfer paths separately.
The goal is engineering evidence, not a headline speedup claim.

## What is measured

| Path | Description |
|---|---|
| `atomic` | Up to one element per thread, one `atomicAdd(double)` per thread. The naive shape most people write first; the grid is capped, so larger inputs loop but still issue one atomic per thread. |
| `block` | Scalar loads, shared-memory tree reduction per block, then a small final reduction. |
| `warp` | `double2` loads, warp-shuffle reduction, then a small final reduction. |
| `cub` | `cub::DeviceReduce::Sum` as the library baseline. |
| `filter` | `double2` loads with a `value > 0` predicate; accumulates `COUNT` and `SUM` for matching values with one atomic pair per block. |
| `pageable` | `cudaMemcpy` from pageable host memory, warp reduction, result copy back. |
| `pinned` | `cudaMemcpyAsync` from pinned host memory on one stream. |
| `four_streams` | The input is split into four chunks on four streams, with copy and kernel work issued per chunk. |

Every result is checked against a long-double host reference before the row is
reported. The harness warms every path once, then reports the median of 20
kernel iterations and 5 transfer iterations. The generated report is
[`results/latest.md`](results/latest.md); the raw values are in
[`results/latest.json`](results/latest.json).

## Results

Measured 2026-11-18 in WSL2 Ubuntu 24.04 on a **NVIDIA GeForce RTX 3050 Laptop
GPU** (compute 8.6, 16 SMs, 4 GiB), with CUDA 12.9.86. The device reports a
192.0 GB/s theoretical DRAM bandwidth. The input is random doubles in
`[-1, 1]`, so the filter keeps about half of them.

| Elements | Size | Atomic | Block | Warp | CUB | Filter | Warp GB/s | Filter GB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16,777,216 | 128 MiB | 27.57 ms | 0.73 ms | 0.72 ms | 0.73 ms | 0.75 ms | 185.7 | 178.4 |
| 67,108,864 | 512 MiB | 27.56 ms | 2.84 ms | 2.84 ms | 2.85 ms | 2.89 ms | 189.4 | 185.5 |
| 134,217,728 | 1 GiB | 27.68 ms | 5.64 ms | 5.64 ms | 5.66 ms | 5.71 ms | 190.4 | 188.0 |

All rows verified.

### Kernel bandwidth

The warp-shuffle and CUB paths reach **185-190 GB/s**, or **96-99%** of the
device's theoretical bandwidth. CUB is within about 1% of the hand-written
kernel at every size, which is the expected result when both are memory-bound:
there is little headroom left for a smarter reduction.

The naive atomic path is the outlier. One atomic per thread serializes on the
same address, so it stays around **27.6 ms** regardless of input size: about
**38x slower** than the warp reduction at 128 MiB and still **4.9x slower** at
1 GiB. The block reduction is competitive once the data is large, but its
scalar loads cost about 1.4% at 128 MiB (0.73 ms vs 0.72 ms); the difference
disappears at 512 MiB and 1 GiB.

The filter variant adds a predicate and a `COUNT`/`SUM` pair to the same pass.
It still reaches **178-188 GB/s**, within about 4% of the plain reduction, so
on this GPU the predicate is effectively free next to the memory traffic.

### Transfer dominates the end-to-end query

| Elements | Pageable | Pinned / 1 stream | Pinned / 4 streams | Pageable GB/s | Pinned GB/s | Streams GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16,777,216 | 18.54 ms | 10.96 ms | 11.27 ms | 14.5 | 24.5 | 23.8 |
| 67,108,864 | 72.99 ms | 41.55 ms | 41.62 ms | 14.7 | 25.8 | 25.8 |
| 134,217,728 | 194.71 ms | 88.74 ms | 81.30 ms | 11.0 | 24.2 | 26.4 |

`GB/s` counts both directions, though the result copy is only eight bytes.

Pinned memory is **1.7-2.2x faster** than pageable memory end to end. Four
streams help only at 1 GiB (81.3 ms vs 88.7 ms, about 8%); at 128 and 512 MiB
the difference is inside run-to-run variation. On this WSL setup the copy
engine is effectively serialized, so the expected overlap does not materialize.

The important number is the ratio: at 1 GiB the kernel takes **5.64 ms** while
the pinned end-to-end path takes **88.74 ms**. Over **93%** of the query time is
host/device movement, not arithmetic. That is why the engine's GPU path reports
transfer and kernel time separately, and why widening GPU coverage without
first fixing the data path would not pay off.

## Nsight Compute

The profiling script is [`scripts/profile_ncu.sh`](scripts/profile_ncu.sh).
Run it from the repository root with
`bash cuda-microbench/scripts/profile_ncu.sh`; set `NCU=/path/to/ncu` if the
binary is somewhere else. Nsight Compute 2024.1.1 was installed in a separate
environment and did attach to the process, but this WSL driver denied hardware
performance counters:

```text
ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

On Windows, enable **NVIDIA Control Panel -> Developer -> Manage GPU Performance
Counters -> allow access**, then restart WSL. Native Linux hosts need the
driver's perf-counter permission configured by an administrator. The script
prints that guidance and exits with status 2 when it hits the permission wall.

The bandwidth numbers above do not depend on Nsight Compute: they are computed
from CUDA event timings and the known input size. What is missing is the
hardware-counter breakdown (DRAM read bytes, achieved occupancy, warp stall
reasons), so treat the roofline conclusion as "the measured time is consistent
with the theoretical bandwidth", not as counter-verified proof.

## Build and run

From WSL with the project environment, from the repository root:

```bash
export PATH="$HOME/qe-env/bin:$PATH"
cmake -S cuda-microbench -B build-cuda-microbench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda-microbench -j4

./build-cuda-microbench/reduce_bench \
  --iters 20 --transfer-iters 5 \
  --json cuda-microbench/results/latest.json \
  --markdown cuda-microbench/results/latest.md
```

Options: `--sizes N,N,...` (default 16M, 64M, 128M doubles),
`--iters`, `--transfer-iters`, `--no-transfer`, `--no-cub`, `--json`,
`--markdown`, `--help`.

## Caveats

- These are WSL2 numbers on a 4 GiB laptop GPU. Native Linux transfer
  bandwidth will usually be higher, and the four-stream result may change.
- The kernel is double precision because the engine aggregates `double`. A
  float kernel would move half the bytes and could go faster on the same bus.
- Summation order differs between paths and within the atomic path across
  runs. Every row is verified with a `1e-6` relative tolerance rather than
  bit-for-bit equality.
- This project does not widen the engine's GPU coverage. It measures the
  layer underneath it. A grouped aggregate over low-cardinality keys and a
  device-side filter that materializes rows are the next experiments; both
  need a plan for the transfer cost shown above.
