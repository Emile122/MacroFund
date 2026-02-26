# Data Module (ATHENA DataPack)

This module implements a partitioned local data lake with three layers:
- `datastore/bronze`: immutable ingestion snapshots per run date
- `datastore/silver`: normalized partitioned datasets
- `datastore/gold`: strategy-ready partitioned datasets

## CLI

Run via Python module:

```bash
python -m data update_prices --last-n-days 7 --person emile
python -m data update_macro --vendor fred --person emile
python -m data update_cot --person emile
python -m data update_policy --input-file path/to/curve.csv --person emile
python -m data build_all --smart
python -m data backfill --start 2024-01-01 --end 2024-12-31 --person emile
```

## Incremental Rules

- Price partition changes at `silver.market_prices/dt=X` trigger `gold.returns_daily/dt in {X, X+1}`.
- Macro release partition changes trigger affected month rebuilds in `gold.macro_asof_monthly`.
- New COT week triggers `gold.cot_z_weekly` recompute for last `K` weeks.
- Policy curve changes at `dt=X` trigger `gold.policy_features/dt=X`.

## Metadata and Audit

Every job writes:
- `datastore/meta/dataset_versions.json`
- `datastore/meta/runs/dt=YYYY-MM-DD/run_manifest.json`

Each partition entry tracks fingerprint (`sha256`), row count, write time, and path.

## Legacy Compatibility

Use `data.store.dataset.legacy_single_table_view(...)` to read partitioned data as a single table scan for legacy consumers.
