# Benchmark results

Generated: 2026-11-18T03:09:48.295779+00:00

Dataset: **1,000,000 sales rows**, 16 row groups, 22.5 MB.

Hardware: AMD Ryzen 7 170 with Radeon Graphics, 7.4 GiB, Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39; GPU: NVIDIA GeForce RTX 3050 Laptop GPU.

CPU engine batch size: 8192; median of 5 runs after 2 warm-up run(s).

Scan thread counts measured: 1, 4, 8.

> CPU-only run: qe-cli was built without CUDA; rebuild with QE_ENABLE_CUDA=ON to measure the GPU path. No GPU timings are reported.

| Query | CPU median (ms) | GPU median (ms) | DuckDB median (ms) | CPU / DuckDB | GPU / CPU | GPU path |
|---|---:|---:|---:|---:|---:|---|
| filter_aggregate | 4.09 | - | 16.20 | 0.25x | - | not requested |
| global_aggregate | 43.81 | - | 27.58 | 1.59x | - | not requested |
| empty_filter | 0.07 | - | 11.88 | 0.01x | - | not requested |
| group_region | 92.14 | - | 32.10 | 2.87x | - | not requested |
| group_customer | 190.61 | - | 86.59 | 2.20x | - | not requested |
| topk | 88.81 | - | 46.00 | 1.93x | - | not requested |
| join_country | 293.26 | - | 68.32 | 4.29x | - | not requested |
| count_star | 0.01 | - | 8.70 | 0.00x | - | not requested |
### Scan thread scaling

`--threads N` parallelizes Parquet row-group decoding only; every other
operator stays single-threaded. Values are `execute_ms` medians, with the
CPU/DuckDB ratio in parentheses. Each thread count is verified against
DuckDB before it is timed, so a faster number cannot come from changed rows.

| Query | DuckDB (ms) | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| filter_aggregate | 16.20 | 4.09 (0.25x) | 4.08 (0.25x) | 5.94 (0.37x) |
| global_aggregate | 27.58 | 43.81 (1.59x) | 20.42 (0.74x) | 28.17 (1.02x) |
| empty_filter | 11.88 | 0.07 (0.01x) | 0.07 (0.01x) | 0.07 (0.01x) |
| group_region | 32.10 | 92.14 (2.87x) | 39.99 (1.25x) | 51.34 (1.60x) |
| group_customer | 86.59 | 190.61 (2.20x) | 95.59 (1.10x) | 120.63 (1.39x) |
| topk | 46.00 | 88.81 (1.93x) | 39.11 (0.85x) | 38.88 (0.85x) |
| join_country | 68.32 | 293.26 (4.29x) | 197.88 (2.90x) | 161.31 (2.36x) |
| count_star | 8.70 | 0.01 (0.00x) | 0.01 (0.00x) | 0.01 (0.00x) |

Wins vs DuckDB by scan thread count — 1 thread(s): **3/8**; 4 thread(s): **5/8**; 8 thread(s): **4/8**.


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
SELECT sale_id, amount FROM sales WHERE amount > 1000 ORDER BY amount DESC, sale_id LIMIT 100
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
