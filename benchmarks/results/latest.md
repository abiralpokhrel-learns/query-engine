# Benchmark results

Generated: 2026-11-12T15:00:01.186481+00:00

Dataset: **1,000,000 sales rows**, 16 row groups, 22.5 MB.

Hardware: AMD Ryzen 7 170 with Radeon Graphics, 7.4 GiB, Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39; GPU: NVIDIA GeForce RTX 3050 Laptop GPU.

CPU engine batch size: 8192; median of 5 runs after 2 warm-up run(s).

Scan thread counts measured: 1, 4, 8.

> CPU-only run: qe-cli was built without CUDA; rebuild with QE_ENABLE_CUDA=ON to measure the GPU path. No GPU timings are reported.

| Query | CPU median (ms) | GPU median (ms) | DuckDB median (ms) | CPU / DuckDB | GPU / CPU | GPU path |
|---|---:|---:|---:|---:|---:|---|
| filter_aggregate | 4.76 | - | 19.90 | 0.24x | - | not requested |
| global_aggregate | 52.27 | - | 33.58 | 1.56x | - | not requested |
| empty_filter | 0.09 | - | 9.40 | 0.01x | - | not requested |
| group_region | 116.50 | - | 41.56 | 2.80x | - | not requested |
| group_customer | 202.56 | - | 104.21 | 1.94x | - | not requested |
| topk | 119.14 | - | 61.41 | 1.94x | - | not requested |
| join_country | 297.83 | - | 75.87 | 3.93x | - | not requested |
| count_star | 0.01 | - | 7.25 | 0.00x | - | not requested |
### Scan thread scaling

`--threads N` parallelizes Parquet row-group decoding only; every other
operator stays single-threaded. Values are `execute_ms` medians, with the
CPU/DuckDB ratio in parentheses. Each thread count is verified against
DuckDB before it is timed, so a faster number cannot come from changed rows.

| Query | DuckDB (ms) | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| filter_aggregate | 19.90 | 4.76 (0.24x) | 5.15 (0.26x) | 4.54 (0.23x) |
| global_aggregate | 33.58 | 52.27 (1.56x) | 26.61 (0.79x) | 29.64 (0.88x) |
| empty_filter | 9.40 | 0.09 (0.01x) | 0.07 (0.01x) | 0.11 (0.01x) |
| group_region | 41.56 | 116.50 (2.80x) | 54.78 (1.32x) | 63.42 (1.53x) |
| group_customer | 104.21 | 202.56 (1.94x) | 137.65 (1.32x) | 138.89 (1.33x) |
| topk | 61.41 | 119.14 (1.94x) | 53.38 (0.87x) | 58.54 (0.95x) |
| join_country | 75.87 | 297.83 (3.93x) | 203.53 (2.68x) | 204.69 (2.70x) |
| count_star | 7.25 | 0.01 (0.00x) | 0.01 (0.00x) | 0.01 (0.00x) |

Wins vs DuckDB by scan thread count — 1 thread(s): **3/8**; 4 thread(s): **5/8**; 8 thread(s): **5/8**.


CPU wins vs DuckDB at the primary scan thread count (1): **3**; DuckDB wins: **5**; ties: **0**.

A value below 1.00x means the first engine was faster; a value above 1.00x means the second was faster. DuckDB runs with its default thread count. The engine's default scan is single-threaded; the scan-thread table above reports the opt-in parallel decode over the same verified results. Full timings, per-run values, query text, versions, and dataset metadata are in the adjacent JSON file.

## Workload

### filter_aggregate

One-month date range; row-group skipping plus COUNT/SUM.

```sql
SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date >= '2025-01-01' AND sale_date < '2025-02-01'
```

### global_aggregate

Full-table global COUNT/SUM/AVG; exercises the optional CUDA aggregate path.

```sql
SELECT COUNT(*), SUM(amount), AVG(amount) FROM sales
```

### empty_filter

Date range after the data; all row groups can be skipped.

```sql
SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date > '2030-01-01'
```

### group_region

Low-cardinality GROUP BY with five aggregates.

```sql
SELECT region, COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM sales GROUP BY region ORDER BY region
```

### group_customer

High-cardinality GROUP BY (one group per customer).

```sql
SELECT customer_id, COUNT(*), SUM(amount), AVG(amount) FROM sales GROUP BY customer_id ORDER BY customer_id
```

### topk

Selective filter, top-k sort, then LIMIT.

```sql
SELECT sale_id, amount FROM sales WHERE amount > 1000 ORDER BY amount DESC LIMIT 100
```

### join_country

Hash join against customers, then aggregate by country.

```sql
SELECT c.country, COUNT(*), SUM(s.amount) FROM sales s JOIN customers c ON s.customer_id = c.customer_id GROUP BY c.country ORDER BY c.country
```

### count_star

Full-table COUNT(*) answered from Parquet metadata by both engines.

```sql
SELECT COUNT(*) FROM sales
```
