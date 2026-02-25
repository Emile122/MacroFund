# MacroFund Data Architecture

The system now uses a Parquet data lake with strict layer separation:

- `datastore/bronze/<dataset>/ingest_date=YYYY-MM-DD/part-00000.parquet`
- `datastore/silver/<dataset>/.../part-00000.parquet`
- `datastore/gold/<dataset>/.../part-00000.parquet`

Modeling reads only from `gold` via `src/athena_regime/data/lake.py`.
No CSV input is required by the feature/model pipeline.

## Core Commands

Update one dataset incrementally:

```powershell
athena data update --dataset returns_daily
athena data update --dataset cot_z_weekly
athena data update --dataset policy_features
athena data update --dataset macro_asof_monthly
```

Provider selection is environment-driven:
- `ATHENA_DATA_PROVIDER=mock` (default test stub)
- `ATHENA_DATA_PROVIDER=<module.path:ClassName>` for custom API providers

Prune old partitions (example: bronze retention):

```powershell
athena data prune --dataset returns_daily --level bronze --older-than 180d
```

Run model pipeline modes:

```powershell
athena run --config configs/base.yaml infer
athena run --config configs/base.yaml backtest
athena run --config configs/base.yaml stress --scenario risk_off
athena run --config configs/base.yaml visualize --target-run-id <RUN_ID>
athena run --config configs/research.yaml sweep --max-runs 12
```

Config profiles:
- `configs/dev.yaml`: faster debugging iterations
- `configs/research.yaml`: higher model robustness + exploration
- `configs/production.yaml`: tighter risk/cost assumptions

Visualization outputs:
- Return evolution + NAV + events: `runs/<run_id>/artifacts/returns_nav_events.png`
- Risk diagnostics (underwater + rolling vol/sharpe): `runs/<run_id>/artifacts/risk_diagnostics.png`
- Regime diagnostics (probabilities + state path): `runs/<run_id>/artifacts/regime_diagnostics.png`
- Regime transition matrix: `runs/<run_id>/artifacts/regime_transition_matrix.png`
- Weight diagnostics (gross/net/heatmap/turnover): `runs/<run_id>/artifacts/weights_diagnostics.png`
- Rebalance diagnostics (distance/triggers/confidence): `runs/<run_id>/artifacts/rebalance_diagnostics.png`
- Return distributions (overall + by regime): `runs/<run_id>/artifacts/return_distribution.png`

## Design Rules

- Ingestion: incremental, idempotent partition writes.
- Data access: all reads/writes go through `DataLake`.
- Features/models: storage-agnostic, no file-path coupling.
