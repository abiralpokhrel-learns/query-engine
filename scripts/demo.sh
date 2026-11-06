#!/usr/bin/env bash
# One-command demo. Builds the engine and generates sample data if needed,
# then runs representative queries against the included Parquet files.
set -euo pipefail
cd "$(dirname "$0")/.."

BINARY="build/qe-cli"
PYTHON="${PYTHON:-python3}"

if [[ ! -x "$BINARY" ]]; then
  echo "==> building qe-cli"
  scripts/build.sh
fi

if [[ ! -f data/sales.parquet || ! -f data/customers.parquet || ! -f data/products.parquet ]]; then
  echo "==> generating sample data"
  "$PYTHON" scripts/generate_data.py --out data --rows 200000 --customers 20000 --products 5000
fi

run_query() {
  local title="$1"
  local sql="$2"
  echo
  echo "==> $title"
  "$BINARY" \
    --table sales=data/sales.parquet \
    --table customers=data/customers.parquet \
    --table products=data/products.parquet \
    --query "$sql" \
    --format table --max-rows 12
}

echo "==> engine"
"$BINARY" --version

run_query "Selective date filter + GROUP BY (exercises Parquet row-group skipping)" \
  "SELECT region, COUNT(*), SUM(amount) AS total FROM sales \
   WHERE sale_date >= '2024-01-01' AND sale_date < '2024-02-01' \
   GROUP BY region ORDER BY total DESC"

run_query "Hash join + filter + sort" \
  "SELECT s.sale_id, c.name, s.amount FROM sales s \
   JOIN customers c ON s.customer_id = c.customer_id \
   WHERE s.amount > 5000 ORDER BY s.amount DESC, s.sale_id LIMIT 8"

run_query "Join + GROUP BY + aggregate" \
  "SELECT p.category, COUNT(*) AS sales, AVG(s.amount) AS avg_amount FROM sales s \
   JOIN products p ON s.product_id = p.product_id \
   GROUP BY p.category ORDER BY avg_amount DESC"

run_query "LEFT JOIN + HAVING (version 2 clauses)" \
  "SELECT c.customer_id, c.name, c.country, COUNT(s.sale_id) AS sales FROM customers c \
   LEFT JOIN sales s ON c.customer_id = s.customer_id \
   GROUP BY c.customer_id, c.name, c.country HAVING COUNT(s.sale_id) = 0 \
   ORDER BY c.customer_id"

run_query "DISTINCT (version 2 clause)" \
  "SELECT DISTINCT country FROM customers ORDER BY country"

run_query "Top-k by amount" \
  "SELECT sale_id, amount FROM sales WHERE amount > 1000 \
   ORDER BY amount DESC, sale_id LIMIT 5"

echo
echo "==> done"
echo "Timings and scan statistics are printed to stderr; results are on stdout."
echo "Run scripts/run_benchmarks.py to reproduce the full comparison with DuckDB."
