# Benchmark results

Generated: 2026-11-08T07:07:28.142414+00:00

Dataset: **1,000,000 sales rows**, 16 row groups, 22.5 MB.

Hardware: AMD Ryzen 7 170 with Radeon Graphics, 7.4 GiB, Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39; GPU: NVIDIA GeForce RTX 3050 Laptop GPU.

CPU engine batch size: 8192; median of 3 runs after 1 warm-up run(s).

> GPU measurements use the optional CUDA global-aggregate path (`--gpu`). 2 of 8 queries ran a CUDA kernel; the rest fell back to CPU or were not eligible. GPU median is the end-to-end `execute_ms` including the CPU scan; transfer and kernel times are measured separately below.

| Query | CPU median (ms) | GPU median (ms) | DuckDB median (ms) | CPU / DuckDB | GPU / CPU | GPU path |
|---|---:|---:|---:|---:|---:|---|
| filter_aggregate | 4.25 | 6.12 | 19.75 | 0.22x | 1.44x | CUDA global aggregate |
| global_aggregate | 64.12 | 59.03 | 32.71 | 1.96x | 0.92x | CUDA global aggregate |
| empty_filter | 0.09 | - | 8.10 | 0.01x | - | CPU fallback (no-supported-gpu-operator) |
| group_region | 183.48 | - | 35.06 | 5.23x | - | CPU fallback (no-supported-gpu-operator) |
| group_customer | 404.86 | - | 85.99 | 4.71x | - | CPU fallback (no-supported-gpu-operator) |
| topk | 108.34 | - | 51.62 | 2.10x | - | CPU fallback (no-supported-gpu-operator) |
| join_country | 416.32 | - | 72.36 | 5.75x | - | CPU fallback (no-supported-gpu-operator) |
| count_star | 0.02 | - | 7.09 | 0.00x | - | CPU fallback (no-supported-gpu-operator) |

CPU wins vs DuckDB: **3**; DuckDB wins: **5**; ties: **0**.

GPU used for **2** queries; GPU faster than CPU: **1**, CPU faster than GPU: **1**, ties: **0**.

A value below 1.00x means the first engine was faster; a value above 1.00x means the second was faster. DuckDB runs with its default thread count, while the engine is single-threaded. Full timings, per-run values, query text, versions, and dataset metadata are in the adjacent JSON file.

### GPU kernel components

Transfer and kernel times are from the last timed run reported by the CLI; the end-to-end median uses the same `--repeat` runs.

| Query | GPU end-to-end (ms) | Transfer (ms) | Kernel (ms) | GPU / CPU |
|---|---:|---:|---:|---:|
| filter_aggregate | 6.12 | 0.205 | 0.060 | 1.44x |
| global_aggregate | 59.03 | 2.272 | 0.184 | 0.92x |

## Workload

### filter_aggregate

One-month date range; row-group skipping plus COUNT/SUM.

```sql
SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date >= '2025-01-01' AND sale_date < '2025-02-01'
```

GPU: **used**, 1 reduction(s), transfer 0.205 ms, kernel 0.060 ms.

### global_aggregate

Full-table global COUNT/SUM/AVG; exercises the optional CUDA aggregate path.

```sql
SELECT COUNT(*), SUM(amount), AVG(amount) FROM sales
```

GPU: **used**, 2 reduction(s), transfer 2.272 ms, kernel 0.184 ms.

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
