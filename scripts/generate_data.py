#!/usr/bin/env python3
"""Generate deterministic Parquet data for the query engine demo and tests.

The files are intentionally varied: nulls, skewed groups, duplicate join keys,
multiple row groups, and dates sorted within the file so row-group statistics
can actually prove that a date range is absent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import duckdb
except ImportError:  # DuckDB is only required for the oracle and benchmarks.
    duckdb = None

REGIONS = np.array(["North", "South", "East", "West", "Central", "North-East", "Islands", "Remote"])
REGION_WEIGHTS = np.array([0.28, 0.22, 0.16, 0.12, 0.10, 0.07, 0.03, 0.02])
CHANNELS = np.array(["online", "retail", "partner", "wholesale"])
CHANNEL_WEIGHTS = np.array([0.52, 0.27, 0.14, 0.07])
SEGMENTS = np.array(["consumer", "smb", "enterprise", "public-sector"])
SEGMENT_WEIGHTS = np.array([0.58, 0.24, 0.13, 0.05])
COUNTRIES = np.array(["NP", "IN", "US", "GB", "DE", "JP", "AU", "BR"])
CATEGORIES = np.array(["books", "electronics", "garden", "grocery", "sports", "toys"])
STATUSES = np.array(["completed", "returned", "pending", "cancelled"])
STATUS_WEIGHTS = np.array([0.83, 0.08, 0.06, 0.03])
EPOCH = dt.date(1970, 1, 1)


def nullable(values, null_mask=None, type_=None) -> pa.Array:
    """Build an Arrow array, optionally marking True entries in null_mask as null."""
    if null_mask is None:
        return pa.array(values, type=type_)
    return pa.array(values, type=type_, mask=np.asarray(null_mask, dtype=bool))


def choose(rng: np.random.Generator, values: np.ndarray, size: int, weights=None) -> np.ndarray:
    return rng.choice(values, size=size, p=weights)


def make_customers(rng: np.random.Generator, rows: int) -> pa.Table:
    ids = np.arange(1, rows + 1, dtype=np.int64)
    names = np.array([f"Customer {i:07d}" for i in ids], dtype=object)
    name_null = rng.random(rows) < 0.002
    for index in np.flatnonzero(name_null)[:10]:
        names[index] = "O'Brien, Pat"

    segments = choose(rng, SEGMENTS, rows, SEGMENT_WEIGHTS).astype(object)
    segment_null = rng.random(rows) < 0.04
    countries = choose(rng, COUNTRIES, rows).astype(object)
    signup = rng.integers(0, 2600, size=rows, dtype=np.int32)
    signup_null = rng.random(rows) < 0.01

    return pa.table(
        {
            "customer_id": nullable(ids, type_=pa.int64()),
            "name": nullable(names, name_null, pa.string()),
            "segment": nullable(segments, segment_null, pa.string()),
            "country": nullable(countries, type_=pa.string()),
            "signup_date": nullable(signup, signup_null, pa.date32()),
        }
    )


def make_products(rng: np.random.Generator, rows: int) -> pa.Table:
    ids = np.arange(1, rows + 1, dtype=np.int64)
    categories = choose(rng, CATEGORIES, rows).astype(object)
    category_null = rng.random(rows) < 0.015
    prices = np.round(rng.gamma(2.2, 18.0, size=rows) + 0.99, 2)
    prices_null = rng.random(rows) < 0.005
    active = rng.random(rows) < 0.88
    return pa.table(
        {
            "product_id": nullable(ids, type_=pa.int64()),
            "category": nullable(categories, category_null, pa.string()),
            "unit_price": nullable(prices, prices_null, pa.float64()),
            "active": nullable(active, type_=pa.bool_()),
        }
    )


def make_sales(
    rng: np.random.Generator,
    rows: int,
    customers: int,
    products: int,
    start_date: dt.date,
    days: int,
) -> pa.Table:
    sale_ids = np.arange(1, rows + 1, dtype=np.int64)
    customer_ids = rng.integers(1, customers + 1, size=rows, dtype=np.int64)
    customer_null = rng.random(rows) < 0.01
    product_ids = rng.integers(1, products + 1, size=rows, dtype=np.int64)
    product_null = rng.random(rows) < 0.002

    regions = choose(rng, REGIONS, rows, REGION_WEIGHTS).astype(object)
    region_null = rng.random(rows) < 0.02
    channels = choose(rng, CHANNELS, rows, CHANNEL_WEIGHTS).astype(object)
    statuses = choose(rng, STATUSES, rows, STATUS_WEIGHTS).astype(object)

    quantities = rng.integers(1, 11, size=rows, dtype=np.int64)
    quantity_null = rng.random(rows) < 0.012
    unit_prices = np.round(rng.gamma(2.0, 16.0, size=rows) + 1.0, 2)
    unit_price_null = rng.random(rows) < 0.004
    discounts = np.round(rng.beta(1.3, 7.0, size=rows), 4)
    discount_null = rng.random(rows) < 0.02

    amounts = quantities * unit_prices * (1.0 - discounts)
    amounts = np.round(amounts, 2)
    amount_null = (rng.random(rows) < 0.006) | quantity_null | unit_price_null | discount_null
    # A few deliberately large orders create a skewed SUM.
    amounts[rng.random(rows) < 0.0005] *= 250.0

    offsets = np.sort(rng.integers(0, days, size=rows, dtype=np.int32))
    date_series = (start_date - EPOCH).days + offsets
    date_null = rng.random(rows) < 0.002

    return pa.table(
        {
            "sale_id": nullable(sale_ids, type_=pa.int64()),
            "customer_id": nullable(customer_ids, customer_null, pa.int64()),
            "product_id": nullable(product_ids, product_null, pa.int64()),
            "region": nullable(regions, region_null, pa.string()),
            "channel": nullable(channels, type_=pa.string()),
            "status": nullable(statuses, type_=pa.string()),
            "quantity": nullable(quantities, quantity_null, pa.int64()),
            "unit_price": nullable(unit_prices, unit_price_null, pa.float64()),
            "discount": nullable(discounts, discount_null, pa.float64()),
            "amount": nullable(amounts, amount_null, pa.float64()),
            "sale_date": nullable(date_series, date_null, pa.date32()),
        }
    )


def make_edge_cases() -> dict[str, pa.Table]:
    days = np.array(
        [
            (dt.date(2024, 2, 29) - EPOCH).days,
            (dt.date(2024, 1, 1) - EPOCH).days,
            (dt.date(1970, 1, 1) - EPOCH).days,
            (dt.date(1969, 12, 31) - EPOCH).days,
            (dt.date(2025, 12, 31) - EPOCH).days,
            (dt.date(2023, 6, 15) - EPOCH).days,
        ],
        dtype=np.int32,
    )
    edge = pa.table(
        {
            "id": nullable(np.array([1, 2, 3, 4, 5, 6], dtype=np.int64), type_=pa.int64()),
            "category": nullable(
                np.array(["a", "b", "a", None, "b", ""], dtype=object),
                np.array([False, False, False, True, False, False]),
                pa.string(),
            ),
            "amount": nullable(
                np.array([10.5, -2.0, 0.0, 0.0, 7.25, 100.0], dtype=np.float64),
                np.array([False, False, False, True, False, False]),
                pa.float64(),
            ),
            "quantity": nullable(
                np.array([1, 2, 4, 0, 0, 5], dtype=np.int64),
                np.array([False, False, False, True, False, False]),
                pa.int64(),
            ),
            "flag": nullable(
                np.array([True, False, True, False, False, True], dtype=np.bool_),
                np.array([False, False, False, True, False, False]),
                pa.bool_(),
            ),
            "day": nullable(
                days,
                np.array([False, False, False, False, False, True]),
                pa.date32(),
            ),
            "label": nullable(
                np.array(["x", "O'Brien", "x", None, "", "z"], dtype=object),
                np.array([False, False, False, True, False, False]),
                pa.string(),
            ),
        }
    )
    # Duplicate keys and NULL keys are deliberate: the inner join emits four
    # rows for key 2 and must not match either NULL key.
    edge_left = pa.table(
        {
            "id": nullable(
                np.array([1, 2, 2, 3, 0, 5], dtype=np.int64),
                np.array([False, False, False, False, True, False]),
                pa.int64(),
            ),
            "left_value": pa.array(["l1", "l2a", "l2b", "l3", "lnull", "l5"]),
        }
    )
    edge_right = pa.table(
        {
            "id": nullable(
                np.array([2, 2, 3, 4, 0], dtype=np.int64),
                np.array([False, False, False, False, True]),
                pa.int64(),
            ),
            "right_value": pa.array(["r2a", "r2b", "r3", "r4", "rnull"]),
        }
    )
    multi_left = pa.table(
        {
            "a": nullable(
                np.array([1, 1, 2, 2, 0, 3], dtype=np.int64),
                np.array([False, False, False, False, True, False]),
                pa.int64(),
            ),
            "b": nullable(
                np.array([10, 20, 10, 10, 10, 0], dtype=np.int64),
                np.array([False, False, False, False, False, True]),
                pa.int64(),
            ),
            "left_value": pa.array(["l1", "l2", "l3a", "l3b", "lnull", "lnull2"]),
        }
    )
    multi_right = pa.table(
        {
            "a": nullable(
                np.array([1, 2, 2, 2, 3, 0], dtype=np.int64),
                np.array([False, False, False, False, False, True]),
                pa.int64(),
            ),
            "b": nullable(
                np.array([10, 10, 10, 20, 0, 0], dtype=np.int64),
                np.array([False, False, False, False, True, True]),
                pa.int64(),
            ),
            "right_value": pa.array(["r1", "r2a", "r2b", "r3", "rnull", "rnull2"]),
        }
    )
    empty = pa.table(
        {
            "id": pa.array([], type=pa.int64()),
            "value": pa.array([], type=pa.float64()),
        }
    )
    return {
        "edge_cases": edge,
        "edge_left": edge_left,
        "edge_right": edge_right,
        "multi_left": multi_left,
        "multi_right": multi_right,
        "empty_table": empty,
    }


def write_table(table: pa.Table, path: Path, row_group_size: int, compression: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        row_group_size=row_group_size,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "rows": table.num_rows,
        "columns": [{"name": field.name, "type": str(field.type)} for field in table.schema],
        "row_groups": parquet.num_row_groups,
        "file_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--rows", type=int, default=200_000, help="sales rows")
    parser.add_argument("--customers", type=int, default=20_000, help="customer rows")
    parser.add_argument("--products", type=int, default=5_000, help="product rows")
    parser.add_argument("--row-group-size", type=int, default=65_536)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--days", type=int, default=1095)
    args = parser.parse_args()

    if args.rows < 1 or args.customers < 1 or args.products < 1:
        parser.error("rows, customers, and products must be positive")
    if args.row_group_size < 1:
        parser.error("--row-group-size must be positive")

    rng = np.random.default_rng(args.seed)
    start_date = dt.date.fromisoformat(args.start_date)

    customers = make_customers(rng, args.customers)
    products = make_products(rng, args.products)
    sales = make_sales(rng, args.rows, args.customers, args.products, start_date, args.days)
    edges = make_edge_cases()

    tables = {
        "customers": customers,
        "products": products,
        "sales": sales,
        **edges,
    }
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seed": args.seed,
        "row_group_size": args.row_group_size,
        "compression": args.compression,
        "start_date": args.start_date,
        "days": args.days,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pyarrow": pa.__version__,
        "numpy": np.__version__,
        "duckdb": getattr(duckdb, "__version__", None),
        "tables": {},
    }

    for name, table in tables.items():
        info = write_table(
            table, args.out / f"{name}.parquet", args.row_group_size, args.compression
        )
        manifest["tables"][name] = info
        print(
            f"wrote {args.out / (name + '.parquet')}: "
            f"{info['rows']:,} rows, {info['row_groups']} row groups, "
            f"{info['file_bytes'] / 1_000_000:.2f} MB"
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
