# Benchmark results

Generated: 2026-11-10T07:55:38.315959+00:00

Dataset: **1,000,000 sales rows**, 16 row groups, 22.5 MB.

Hardware: AMD Ryzen 7 170 with Radeon Graphics, 7.4 GiB, Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39; GPU: NVIDIA GeForce RTX 3050 Laptop GPU.

CPU engine batch size: 8192; median of 3 runs after 1 warm-up run(s).

> GPU measurements use the optional CUDA global-aggregate path (`--gpu`). 2 of 8 queries ran a CUDA kernel; the rest fell back to CPU or were not eligible. GPU median is the end-to-end `execute_ms` including the CPU scan; transfer and kernel times are measured separately below.

| Query | CPU median (ms) | GPU median (ms) | DuckDB median (ms) | CPU / DuckDB | GPU / CPU | GPU path |
|---|---:|---:|---:|---:|---:|---|
| filter_aggregate | 7.18 | 10.11 | 21.79 | 0.33x | 1.41x | CUDA global aggregate |
| global_aggregate | 63.49 | 80.57 | 47.34 | 1.34x | 1.27x | CUDA global aggregate |
| empty_filter | 0.39 | - | 10.09 | 0.04x | - | CPU fallback (no-supported-gpu-operator) |
| group_region | 125.89 | - | 45.91 | 2.74x | - | CPU fallback (no-supported-gpu-operator) |
| group_customer | 260.41 | - | 114.23 | 2.28x | - | CPU fallback (no-supported-gpu-operator) |
| topk | 118.86 | - | 64.30 | 1.85x | - | CPU fallback (no-supported-gpu-operator) |
| join_country | 304.51 | - | 83.27 | 3.66x | - | CPU fallback (no-supported-gpu-operator) |
| count_star | 0.02 | - | 9.13 | 0.00x | - | CPU fallback (no-supported-gpu-operator) |

CPU wins vs DuckDB: **3**; DuckDB wins: **5**; ties: **0**.

GPU used for **2** queries; GPU faster than CPU: **0**, CPU faster than GPU: **2**, ties: **0**.

A value below 1.00x means the first engine was faster; a value above 1.00x means the second was faster. DuckDB runs with its default thread count, while the engine is single-threaded. Full timings, per-run values, query text, versions, and dataset metadata are in the adjacent JSON file.

### GPU kernel components

Transfer and kernel times are from the last timed run reported by the CLI; the end-to-end median uses the same `--repeat` runs.

| Query | GPU end-to-end (ms) | Transfer (ms) | Kernel (ms) | GPU / CPU |
|---|---:|---:|---:|---:|
| filter_aggregate | 10.11 | 0.217 | 0.050 | 1.41x |
| global_aggregate | 80.57 | 3.179 | 0.629 | 1.27x |

## Workload

### filter_aggregate

One-month date range; row-group skipping plus COUNT/SUM.

```sql
SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date >= '2025-01-01' AND sale_date < '2025-02-01'
```

GPU: **used**, 1 reduction(s), transfer 0.217 ms, kernel 0.050 ms.

### global_aggregate

Full-table global COUNT/SUM/AVG; exercises the optional CUDA aggregate path.

```sql
SELECT COUNT(*), SUM(amount), AVG(amount) FROM sales
```

GPU: **used**, 2 reduction(s), transfer 3.179 ms, kernel 0.629 ms.

### empty_filter

Date range after the data; all row groups can be skipped.

```sql
SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date > '2030-01-01'
```

GPU: **CPU fallback** (no-supported-gpu-operator).

### group_region

Low-cardinality GROUP BY with five aggregates.

```sql
SELECT region, COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM sales GROUP BY region ORDER BY region
```

GPU: **CPU fallback** (no-supported-gpu-operator).

### group_customer

High-cardinality GROUP BY (one group per customer).

```sql
SELECT customer_id, COUNT(*), SUM(amount), AVG(amount) FROM sales GROUP BY customer_id ORDER BY customer_id
```

GPU: **CPU fallback** (no-supported-gpu-operator).

### topk

Selective filter, top-k sort, then LIMIT.

```sql
SELECT sale_id, amount FROM sales WHERE amount > 1000 ORDER BY amount DESC LIMIT 100
```

GPU: **CPU fallback** (no-supported-gpu-operator).

### join_country

Hash join against customers, then aggregate by country.

```sql
SELECT c.country, COUNT(*), SUM(s.amount) FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY c.country ORDER BY c.country
```

GPU: **CPU fallback** (no-supported-gpu-operator).

### count_star

Full-table COUNT(*) answered from Parquet metadata by both engines.

```sql
SELECT COUNT(*) FROM sales
```

GPU: **CPU fallback** (no-supported-gpu-operator).
