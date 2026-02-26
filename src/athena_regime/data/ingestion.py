from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.athena_regime.data.lake import DATASET_SPECS, DataLake, DatasetSpec
from src.athena_regime.data.providers import Provider


DEFAULT_UPDATE_DATASETS = (
    "returns_daily",
    "cot_z_weekly",
    "policy_features",
    "macro_asof_monthly",
)


def update_dataset(
    lake: DataLake,
    dataset: str,
    provider: Provider,
    start: str | None = None,
    end: str | None = None,
    default_window_days: int = 7,
) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    end_dt = pd.Timestamp(end or datetime.now(timezone.utc).date().isoformat()).normalize()
    start_dt = pd.Timestamp(start).normalize() if start else _next_start_from_latest(lake, dataset, spec, end_dt, default_window_days)

    if start_dt > end_dt:
        return {
            "dataset": dataset,
            "status": "up_to_date",
            "start": start_dt.date().isoformat(),
            "end": end_dt.date().isoformat(),
            "rows_fetched": 0,
            "partitions_written": [],
        }

    raw = provider.fetch(dataset, start_dt.date().isoformat(), end_dt.date().isoformat())
    if raw is None or raw.empty:
        return {
            "dataset": dataset,
            "status": "no_data",
            "start": start_dt.date().isoformat(),
            "end": end_dt.date().isoformat(),
            "rows_fetched": 0,
            "partitions_written": [],
        }

    frame = raw.copy()
    frame = _ensure_partition_column(frame, spec)
    frame = _normalize_partition_values(frame, spec.partition_key)

    today = datetime.now(timezone.utc).date().isoformat()
    lake.write_partition(
        dataset=dataset,
        partition_key=f"ingest_date={today}",
        df=frame,
        level="bronze",
        validate_schema=False,
    )

    written: list[str] = []
    for part_value, part_df in frame.groupby(spec.partition_key):
        part_key = f"{spec.partition_key}={part_value}"
        cleaned = part_df.copy()
        lake.write_partition(dataset=dataset, partition_key=part_key, df=cleaned, level="silver", validate_schema=True)
        lake.write_partition(dataset=dataset, partition_key=part_key, df=cleaned, level="gold", validate_schema=True)
        written.append(part_key)

    return {
        "dataset": dataset,
        "status": "updated",
        "start": start_dt.date().isoformat(),
        "end": end_dt.date().isoformat(),
        "rows_fetched": int(len(frame)),
        "partitions_written": sorted(written),
    }


def prune_dataset(
    lake: DataLake,
    dataset: str,
    level: str,
    older_than: str | None = None,
    keep_last_n: int | None = None,
) -> dict[str, Any]:
    deleted = lake.prune(
        dataset=dataset,
        level=level,
        older_than=older_than,
        keep_last_n=keep_last_n,
    )
    return {
        "dataset": dataset,
        "level": level,
        "deleted_partitions": deleted,
        "deleted_count": len(deleted),
    }


def _next_start_from_latest(
    lake: DataLake,
    dataset: str,
    spec: DatasetSpec,
    end_dt: pd.Timestamp,
    default_window_days: int,
) -> pd.Timestamp:
    latest = lake.latest_partition(dataset, level="gold")
    if not latest:
        return (end_dt - pd.Timedelta(days=default_window_days - 1)).normalize()

    _, value = latest.split("=", 1)
    if spec.partition_key == "dt":
        return pd.Timestamp(value) + pd.Timedelta(days=1)
    if spec.partition_key == "week":
        week_start = _week_to_timestamp(value)
        return week_start + pd.Timedelta(days=7)
    if spec.partition_key == "month":
        month_start = pd.Timestamp(f"{value}-01")
        return (month_start + pd.offsets.MonthBegin(1)).normalize()
    return (end_dt - pd.Timedelta(days=default_window_days - 1)).normalize()


def _ensure_partition_column(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    out = df.copy()
    part_col = spec.partition_key
    if part_col in out.columns:
        return out

    temporal = spec.temporal_column
    if temporal is None or temporal not in out.columns:
        raise ValueError(
            f"Dataset '{spec}' missing partition column '{part_col}' and temporal column '{temporal}'."
        )
    ts = pd.to_datetime(out[temporal], errors="raise")
    if part_col == "dt":
        out[part_col] = ts.dt.date.astype(str)
    elif part_col == "week":
        iso = ts.dt.isocalendar()
        out[part_col] = iso["year"].astype(str).str.zfill(4) + "-" + iso["week"].astype(str).str.zfill(2)
    elif part_col == "month":
        out[part_col] = ts.dt.to_period("M").astype(str)
    else:
        out[part_col] = ts.astype(str)
    return out


def _normalize_partition_values(df: pd.DataFrame, part_col: str) -> pd.DataFrame:
    out = df.copy()
    if part_col == "dt":
        out[part_col] = pd.to_datetime(out[part_col], errors="raise").dt.date.astype(str)
    elif part_col == "week":
        out[part_col] = out[part_col].astype(str)
    elif part_col == "month":
        out[part_col] = pd.to_datetime(out[part_col], errors="raise").dt.to_period("M").astype(str)
    return out


def _week_to_timestamp(week_value: str) -> pd.Timestamp:
    year, week = week_value.split("-")
    return pd.Timestamp(datetime.fromisocalendar(int(year), int(week), 1))
