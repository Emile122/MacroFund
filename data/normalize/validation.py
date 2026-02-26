from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _check_numeric(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return True
    return pd.to_numeric(non_null, errors="coerce").notna().all()


def _check_date(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return True
    return pd.to_datetime(non_null, errors="coerce").notna().all()


def _check_timestamp(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return True
    return pd.to_datetime(non_null, errors="coerce", utc=True).notna().all()


_TYPE_CHECKERS = {
    "string": lambda s: True,
    "float64": _check_numeric,
    "int64": _check_numeric,
    "date": _check_date,
    "timestamp": _check_timestamp,
}


class ValidationError(ValueError):
    pass


def load_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(Path(schema_path).read_text(encoding="utf-8"))


def ensure_monotonic(df: pd.DataFrame, col: str) -> None:
    values = pd.to_datetime(df[col], errors="coerce")
    if not values.is_monotonic_increasing:
        raise ValidationError(f"Column {col} is not monotonic increasing")


def ensure_no_duplicates(df: pd.DataFrame, cols: list[str]) -> None:
    dupes = df.duplicated(cols, keep=False)
    if dupes.any():
        raise ValidationError(f"Duplicate keys found for columns {cols}")


def validate_frame(
    df: pd.DataFrame,
    schema_path: Path,
    critical_series: list[str] | None = None,
    missing_threshold: int = 0,
) -> dict[str, Any]:
    schema = load_schema(schema_path)
    required = schema.get("required", [])
    cols = schema.get("columns", {})

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    for col, expected_type in cols.items():
        if col not in df.columns:
            continue
        checker = _TYPE_CHECKERS.get(expected_type)
        if checker and not checker(df[col]):
            raise ValidationError(f"Column {col} failed type check {expected_type}")

    if "dt" in df.columns:
        # Check on the actual input order; sorting before the check would
        # always pass, masking callers that provide unsorted data.
        ensure_monotonic(df, "dt")

    if critical_series and "series_id" in df.columns:
        missing_count = len(set(critical_series) - set(df["series_id"].dropna().astype(str).unique()))
        if missing_count > missing_threshold:
            raise ValidationError(
                f"Missing critical series threshold breached: missing={missing_count} threshold={missing_threshold}"
            )

    return {"sort_keys": schema.get("sort_keys", [])}


def validate_bloomberg_excel_template(df: pd.DataFrame, first_col_expected: str = "Date") -> None:
    if not len(df.columns):
        raise ValidationError("Excel export is empty")
    if df.columns[0] != first_col_expected:
        raise ValidationError("First column must be Date")
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce", format="%Y-%m-%d")
    if dates.isna().any():
        raise ValidationError("Date column must be ISO YYYY-MM-DD")
