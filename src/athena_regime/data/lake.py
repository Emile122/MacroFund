from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
import shutil
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


_PART_RE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.+)$")


@dataclass(frozen=True)
class DatasetSpec:
    level: str
    partition_key: str
    required_columns: tuple[str, ...]
    dtypes: dict[str, str]
    primary_key: tuple[str, ...]
    temporal_column: str | None
    allow_extra_columns: bool = False


class DataLakeError(ValueError):
    pass


DATASET_SPECS: dict[str, DatasetSpec] = {
    # Gold
    "returns_daily": DatasetSpec(
        level="gold",
        partition_key="dt",
        required_columns=("dt", "instrument", "return", "px_t", "px_t_1"),
        dtypes={
            "dt": "date",
            "instrument": "string",
            "return": "float64",
            "px_t": "float64",
            "px_t_1": "float64",
        },
        primary_key=("dt", "instrument"),
        temporal_column="dt",
    ),
    "cot_z_weekly": DatasetSpec(
        level="gold",
        partition_key="week",
        required_columns=("week", "contract", "group", "net_pos", "zscore"),
        dtypes={
            "week": "string",
            "contract": "string",
            "group": "string",
            "net_pos": "int64",
            "zscore": "float64",
        },
        primary_key=("week", "contract", "group"),
        temporal_column="week",
    ),
    "policy_features": DatasetSpec(
        level="gold",
        partition_key="dt",
        required_columns=(
            "dt",
            "implied_cut_prob_3m",
            "implied_cut_prob_6m",
            "term_front_slope",
            "terminal_shift_30d",
            "source",
        ),
        dtypes={
            "dt": "date",
            "implied_cut_prob_3m": "float64",
            "implied_cut_prob_6m": "float64",
            "term_front_slope": "float64",
            "terminal_shift_30d": "float64",
            "source": "string",
        },
        primary_key=("dt",),
        temporal_column="dt",
    ),
    "macro_asof_monthly": DatasetSpec(
        level="gold",
        partition_key="month",
        required_columns=("month_end",),
        dtypes={"month_end": "date"},
        primary_key=("month_end",),
        temporal_column="month_end",
        allow_extra_columns=True,
    ),
    # Silver
    "market_prices": DatasetSpec(
        level="silver",
        partition_key="dt",
        required_columns=("vendor", "dt", "instrument", "px_last", "currency", "source_id", "fetch_ts"),
        dtypes={
            "vendor": "string",
            "dt": "date",
            "instrument": "string",
            "px_last": "float64",
            "currency": "string",
            "source_id": "string",
            "fetch_ts": "timestamp",
        },
        primary_key=("dt", "instrument", "vendor"),
        temporal_column="dt",
    ),
    "cot_weekly": DatasetSpec(
        level="silver",
        partition_key="week",
        required_columns=("dt_ingest", "report_date", "contract", "group", "net_pos", "long", "short", "open_interest"),
        dtypes={
            "dt_ingest": "date",
            "report_date": "date",
            "contract": "string",
            "group": "string",
            "net_pos": "int64",
            "long": "int64",
            "short": "int64",
            "open_interest": "int64",
        },
        primary_key=("report_date", "contract", "group"),
        temporal_column="report_date",
    ),
    "policy_curve": DatasetSpec(
        level="silver",
        partition_key="dt",
        required_columns=("dt", "tenor_days", "implied_rate", "source"),
        dtypes={
            "dt": "date",
            "tenor_days": "int64",
            "implied_rate": "float64",
            "source": "string",
        },
        primary_key=("dt", "tenor_days"),
        temporal_column="dt",
    ),
    "macro_releases": DatasetSpec(
        level="silver",
        partition_key="dt",
        required_columns=("dt_ingest", "series_id", "period", "release_date", "value", "unit", "source", "vintage_id", "notes"),
        dtypes={
            "dt_ingest": "date",
            "series_id": "string",
            "period": "date",
            "release_date": "date",
            "value": "float64",
            "unit": "string",
            "source": "string",
            "vintage_id": "string",
            "notes": "string",
        },
        primary_key=("dt_ingest", "series_id", "period", "vintage_id"),
        temporal_column="dt_ingest",
    ),
}


class DataLake:
    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        dataset: str,
        start: str | date | datetime | None = None,
        end: str | date | datetime | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        spec = self._spec(dataset)
        dataset_root = self._dataset_root(dataset, spec.level)
        if not dataset_root.exists():
            return pd.DataFrame(columns=columns or list(spec.required_columns))

        table = ds.dataset(str(dataset_root), format="parquet", partitioning="hive").to_table(columns=columns)
        df = table.to_pandas()
        if df.empty:
            return df

        df = self._coerce_types(df, spec, fail_on_extra=False)
        df = self._filter_temporal(df, spec, start=start, end=end)
        sort_cols = [c for c in (spec.temporal_column, *spec.primary_key) if c in df.columns]
        if sort_cols:
            df = df.sort_values(list(dict.fromkeys(sort_cols)), kind="mergesort").reset_index(drop=True)
        return df

    def write_partition(
        self,
        dataset: str,
        partition_key: str,
        df: pd.DataFrame,
        level: str = "gold",
        validate_schema: bool = True,
    ) -> None:
        spec = self._spec(dataset)
        part_name, part_value = self._split_partition_key(partition_key)
        expected_part = spec.partition_key if validate_schema else part_name
        if part_name != expected_part:
            raise DataLakeError(
                f"Partition key mismatch for dataset={dataset}: expected {expected_part}=..., got {partition_key}"
            )

        frame = df.copy()
        if validate_schema:
            frame = self._coerce_types(frame, spec, fail_on_extra=True)
            self._validate_schema_columns(frame, spec)

        partition_col = spec.partition_key if validate_schema else part_name
        frame = self._ensure_partition_value(frame, partition_col, part_value, spec=spec)
        self._validate_primary_key(frame, spec.primary_key)

        partition_dir = self._dataset_root(dataset, level) / partition_key
        tmp_dir = partition_dir.parent / f"{partition_dir.name}.tmp"
        filename = "part-00000.parquet"

        existing = self._read_partition_if_exists(partition_dir, partition_col=partition_col, part_value=part_value)
        if not existing.empty:
            combined = pd.concat([existing, frame], ignore_index=True)
            combined = self._coerce_types(combined, spec, fail_on_extra=False)
            if spec.primary_key:
                combined = combined.drop_duplicates(list(spec.primary_key), keep="last")
            frame = combined
            self._validate_primary_key(frame, spec.primary_key)

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        to_write = frame.drop(columns=[partition_col], errors="ignore")
        to_write.to_parquet(tmp_dir / filename, index=False)

        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        tmp_dir.rename(partition_dir)

    def latest_partition(self, dataset: str, level: str = "gold") -> str | None:
        parts = self.list_partitions(dataset, level=level)
        if not parts:
            return None
        spec = self._spec(dataset)
        ranked = sorted(parts, key=lambda p: self._partition_sort_key(spec.partition_key, p))
        return ranked[-1]

    def list_partitions(self, dataset: str, level: str = "gold") -> list[str]:
        self._spec(dataset)
        root = self._dataset_root(dataset, level)
        if not root.exists():
            return []
        out: list[str] = []
        for item in root.iterdir():
            if item.is_dir() and _PART_RE.match(item.name):
                out.append(item.name)
        return sorted(out)

    def prune(
        self,
        dataset: str,
        level: str,
        older_than: str | None = None,
        keep_last_n: int | None = None,
    ) -> list[str]:
        spec = self._spec(dataset)
        partitions = self.list_partitions(dataset, level=level)
        if not partitions:
            return []

        parsed: list[tuple[str, datetime]] = [
            (part, self._partition_sort_key(spec.partition_key, part)) for part in partitions
        ]
        parsed.sort(key=lambda x: x[1])

        cutoff = self._parse_older_than(older_than) if older_than else None
        keep = keep_last_n or 0
        keep_set = {name for name, _ in parsed[-keep:]} if keep > 0 else set()

        to_delete: list[str] = []
        for name, key_dt in parsed:
            if name in keep_set:
                continue
            if cutoff is not None and key_dt >= cutoff:
                continue
            to_delete.append(name)

        root = self._dataset_root(dataset, level)
        for part in to_delete:
            shutil.rmtree(root / part, ignore_errors=True)
        return to_delete

    def _spec(self, dataset: str) -> DatasetSpec:
        if dataset not in DATASET_SPECS:
            raise DataLakeError(f"Unsupported dataset '{dataset}'.")
        return DATASET_SPECS[dataset]

    def _dataset_root(self, dataset: str, level: str) -> Path:
        return self.root_path / level / dataset

    @staticmethod
    def _split_partition_key(partition_key: str) -> tuple[str, str]:
        match = _PART_RE.match(partition_key)
        if not match:
            raise DataLakeError(f"Invalid partition key: '{partition_key}'. Expected key=value.")
        return match.group("key"), match.group("value")

    def _coerce_types(self, df: pd.DataFrame, spec: DatasetSpec, fail_on_extra: bool) -> pd.DataFrame:
        frame = df.copy()
        self._validate_schema_columns(frame, spec, fail_on_extra=fail_on_extra)

        for col, dtype in spec.dtypes.items():
            if col not in frame.columns:
                continue
            frame[col] = self._coerce_series(frame[col], dtype)
        return frame

    @staticmethod
    def _coerce_series(series: pd.Series, dtype: str) -> pd.Series:
        if dtype == "string":
            return series.astype("string")
        if dtype == "float64":
            return pd.to_numeric(series, errors="raise").astype("float64")
        if dtype == "int64":
            return pd.to_numeric(series, errors="raise").astype("int64")
        if dtype == "date":
            return pd.to_datetime(series, errors="raise").dt.tz_localize(None).dt.normalize()
        if dtype == "timestamp":
            return pd.to_datetime(series, errors="raise", utc=True)
        raise DataLakeError(f"Unsupported dtype '{dtype}' in schema.")

    @staticmethod
    def _validate_schema_columns(df: pd.DataFrame, spec: DatasetSpec, fail_on_extra: bool = True) -> None:
        missing = [c for c in spec.required_columns if c not in df.columns]
        if missing:
            raise DataLakeError(f"Missing required columns: {missing}")
        if fail_on_extra and not spec.allow_extra_columns:
            expected = set(spec.required_columns)
            extras = sorted(c for c in df.columns if c not in expected)
            if extras:
                raise DataLakeError(f"Unexpected columns for schema: {extras}")

    @staticmethod
    def _validate_primary_key(df: pd.DataFrame, primary_key: tuple[str, ...]) -> None:
        if not primary_key:
            return
        dupes = df.duplicated(list(primary_key), keep=False)
        if dupes.any():
            raise DataLakeError(f"Primary key violation for columns {list(primary_key)}.")

    def _read_partition_if_exists(
        self,
        partition_dir: Path,
        partition_col: str,
        part_value: str,
    ) -> pd.DataFrame:
        if not partition_dir.exists():
            return pd.DataFrame()
        files = sorted(partition_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if partition_col not in frame.columns:
            frame[partition_col] = part_value
        return frame

    def _ensure_partition_value(
        self,
        df: pd.DataFrame,
        part_col: str,
        part_value: str,
        spec: DatasetSpec | None = None,
    ) -> pd.DataFrame:
        frame = df.copy()
        target_value: Any = part_value
        if spec is not None and part_col in spec.dtypes and spec.dtypes[part_col] == "date":
            target_value = pd.Timestamp(part_value).normalize()
        if part_col not in frame.columns:
            frame[part_col] = target_value
            return frame

        values = frame[part_col].dropna().map(lambda v: self._stringify_partition_value(part_col, v)).unique().tolist()
        if len(values) > 1:
            raise DataLakeError(f"Partition column '{part_col}' contains multiple values: {values}")
        if values and values[0] != part_value:
            raise DataLakeError(f"Partition value mismatch for '{part_col}': frame has {values[0]} but key is {part_value}.")
        frame[part_col] = target_value
        return frame

    @staticmethod
    def _stringify_partition_value(part_col: str, value: Any) -> str:
        if part_col in {"dt", "ingest_date"}:
            return pd.Timestamp(value).date().isoformat()
        if part_col == "month":
            ts = pd.Timestamp(value)
            return f"{ts.year:04d}-{ts.month:02d}"
        if part_col == "week":
            text = str(value)
            if re.match(r"^\d{4}-\d{2}$", text):
                return text
            ts = pd.Timestamp(value)
            iso = ts.isocalendar()
            return f"{iso.year:04d}-{iso.week:02d}"
        return str(value)

    def _filter_temporal(
        self,
        df: pd.DataFrame,
        spec: DatasetSpec,
        start: str | date | datetime | None,
        end: str | date | datetime | None,
    ) -> pd.DataFrame:
        if spec.temporal_column is None or spec.temporal_column not in df.columns:
            return df

        if start is None and end is None:
            return df

        out = df.copy()
        col = spec.temporal_column
        if col == "week":
            values = out[col].map(self._week_to_datetime)
            if start is not None:
                out = out[values >= self._coerce_boundary(start)]
                values = values.loc[out.index]
            if end is not None:
                out = out[values <= self._coerce_boundary(end)]
            return out

        if col == "month":
            values = out[col].map(self._month_to_datetime)
            if start is not None:
                out = out[values >= self._coerce_boundary(start)]
                values = values.loc[out.index]
            if end is not None:
                out = out[values <= self._coerce_boundary(end)]
            return out

        values = pd.to_datetime(out[col], errors="coerce")
        if start is not None:
            out = out[values >= self._coerce_boundary(start)]
            values = values.loc[out.index]
        if end is not None:
            out = out[values <= self._coerce_boundary(end)]
        return out

    def _partition_sort_key(self, partition_col: str, partition_key: str) -> datetime:
        _, value = self._split_partition_key(partition_key)
        if partition_col in {"dt", "ingest_date"}:
            return pd.Timestamp(value).to_pydatetime()
        if partition_col == "week":
            return self._week_to_datetime(value)
        if partition_col == "month":
            return self._month_to_datetime(value)
        return pd.Timestamp(value).to_pydatetime()

    @staticmethod
    def _week_to_datetime(value: Any) -> datetime:
        text = str(value)
        if text.startswith("week="):
            text = text.split("=", 1)[1]
        year, week = text.split("-")
        return datetime.fromisocalendar(int(year), int(week), 1)

    @staticmethod
    def _month_to_datetime(value: Any) -> datetime:
        text = str(value)
        if text.startswith("month="):
            text = text.split("=", 1)[1]
        return pd.Timestamp(f"{text}-01").to_pydatetime()

    @staticmethod
    def _coerce_boundary(value: str | date | datetime) -> datetime:
        return pd.Timestamp(value).to_pydatetime()

    @staticmethod
    def _parse_older_than(value: str) -> datetime:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        text = value.strip().lower()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
            return pd.Timestamp(text).to_pydatetime()

        match = re.match(r"^(?P<n>\d+)(?P<u>[dwmy])$", text)
        if not match:
            raise DataLakeError(f"Unsupported older_than value '{value}'. Use e.g. 180d, 12w, 6m, 2y.")
        n = int(match.group("n"))
        unit = match.group("u")
        if unit == "d":
            return now - timedelta(days=n)
        if unit == "w":
            return now - timedelta(weeks=n)
        if unit == "m":
            return now - timedelta(days=n * 30)
        return now - timedelta(days=n * 365)
