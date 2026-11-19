# CUDA reduction microbenchmark

Generated 2026-11-18T02:01:41Z on **NVIDIA GeForce RTX 3050 Laptop GPU** (compute 8.6, 16 SMs, 4.00 GiB). Theoretical DRAM bandwidth: 192.0 GB/s. 20 kernel iterations and 5 transfer iterations after warm-up; median values are shown.

| Elements | Size | Atomic (ms) | Block (ms) | Warp (ms) | CUB (ms) | Filter (ms) | Warp GB/s | Filter GB/s | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16777216 | 128.0 MiB | 27.57 | 0.73 | 0.72 | 0.73 | 0.75 | 185.7 | 178.4 | verified |
| 67108864 | 512.0 MiB | 27.56 | 2.84 | 2.84 | 2.85 | 2.89 | 189.4 | 185.5 | verified |
| 134217728 | 1.00 GiB | 27.68 | 5.64 | 5.64 | 5.66 | 5.71 | 190.4 | 188.0 | verified |

The `Filter` column runs the same vectorized pass with a `value > 0` predicate and accumulates `COUNT` plus `SUM` for matching values. It shows whether predicate evaluation costs anything beyond the memory traffic.

## End-to-end transfer

The transfer rows move the input to the GPU, run the warp-shuffle reduction, and copy one double back. `GB/s` counts both directions.

| Elements | Pageable (ms) | Pinned (ms) | Four streams (ms) | Pageable GB/s | Pinned GB/s | Streams GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16777216 | 18.54 | 10.96 | 11.27 | 14.5 | 24.5 | 23.8 |
| 67108864 | 72.99 | 41.55 | 41.62 | 14.7 | 25.8 | 25.8 |
| 134217728 | 194.71 | 88.74 | 81.30 | 11.0 | 24.2 | 26.4 |

All reductions and the filter aggregate were checked against a long-double host reference. The pageable path includes the driver's staging copy; the pinned and four-stream paths use `cudaMemcpyAsync`.
