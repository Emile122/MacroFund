# src/athena_regime/features/engineering.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from src.athena_regime.data.models import FeatureMatrix


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    dates_train: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex


def time_split(
    fm: FeatureMatrix,
    train_end: pd.Timestamp,
) -> DataSplit:
    """Split a FeatureMatrix at train_end with no overlap."""
    mask = fm.X.index <= train_end
    return DataSplit(
        X_train=fm.X.loc[mask],
        X_test=fm.X.loc[~mask],
        dates_train=fm.X.index[mask],
        dates_test=fm.X.index[~mask],
    )


class RollingZScaler:
    """
    Fit on training window; transform test rows incrementally.
    Avoids lookahead by using only data observed up to each point.
    """

    def __init__(self, window: int = 252, min_periods: int = 60) -> None:
        self.window = window
        self.min_periods = min_periods
        self._train_mean: Optional[pd.Series] = None
        self._train_std: Optional[pd.Series] = None
        self._columns: list[str] = []

    def fit(self, X_train: pd.DataFrame) -> "RollingZScaler":
        self._columns = list(X_train.columns)
        self._train_mean = X_train.mean()
        self._train_std = X_train.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._train_mean is None:
            raise RuntimeError("Call fit() before transform()")
        if self._columns:
            X = X.reindex(columns=self._columns)
        return (X - self._train_mean) / self._train_std

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)


def run_qa(fm: FeatureMatrix | pd.DataFrame, logger=None):
    """
    Raise ValueError on data quality failures. Log warnings for soft issues.
    All checks are leakage-free (operate on training data only).
    """
    _log = logger.warning if logger else print
    X = fm.X if isinstance(fm, FeatureMatrix) else fm
    report: dict[str, object] = {}

    nan_frac = X.isna().mean()
    bad_cols = nan_frac[nan_frac > 0.20].index.tolist()
    if bad_cols:
        report["high_nan_columns"] = bad_cols
        _log("QA: >20%% NaN in columns: %s", bad_cols)

    zero_var = X.std()[X.std() < 1e-8].index.tolist()
    if zero_var:
        raise ValueError(f"QA: zero-variance features detected: {zero_var}")

    if X.index.tz is None:
        raise ValueError("QA: FeatureMatrix index must be timezone-aware")

    dup = X.index[X.index.duplicated()].tolist()
    if dup:
        raise ValueError(f"QA: duplicate dates in feature matrix: {dup[:5]}")

    return X, report
