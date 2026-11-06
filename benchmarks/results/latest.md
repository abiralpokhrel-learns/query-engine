# Benchmark results

Generated: 2026-11-06T16:06:19.874094+00:00

Dataset: **1,000,000 sales rows**, 16 row groups, 22.5 MB.

Hardware: AMD Ryzen 7 170 with Radeon Graphics, 7.4 GiB, Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39.

CPU engine batch size: 8192; median of 5 runs after 2 warm-up run(s).

> GPU timings are not reported: no usable GPU was detected (Failed to initialize NVML: N/A). Transfer and kernel columns are absent rather than estimated.

| Query | CPU median (ms) | DuckDB median (ms) | CPU / DuckDB | Winner |
|---|---:|---:|---:|---|
| filter_aggregate | 9.22 | 27.98 | 0.33x | CPU |
| empty_filter | 0.27 | 13.89 | 0.02x | CPU |
| group_region | 244.75 | 55.29 | 4.43x | DuckDB |
| group_customer | 691.02 | 138.22 | 5.00x | DuckDB |
| topk | 164.46 | 81.66 | 2.01x | DuckDB |
| join_country | 773.26 | 112.36 | 6.88x | DuckDB |
| count_star | 0.01 | 13.36 | 0.00x | CPU |

CPU wins: **3**; DuckDB wins: **4**; ties: **0**.

A value below 1.00x means the CPU engine was faster; a value above 1.00x means DuckDB was faster. DuckDB runs with its default thread count, while the engine is single-threaded. Full timings, query text, versions, and dataset metadata are in the adjacent JSON file.

## Workload

### filter_aggregate

One-month date range; row-group skipping plus COUNT/SUM.

```sql
SELECT COUNT(*), SUM(amount) FROM sales WHERE sale_date >= '2025-01-01' AND sale_date < '2025-02-01'
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
