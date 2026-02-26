# src/athena_regime/data/models.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class FeatureMatrix:
    X: pd.DataFrame              # shape (n_days, n_features); DatetimeIndex
    feature_names: list[str]
    metadata: dict               # build params, source vintage ids


@dataclass
class RegimeResult:
    """Output of RegimeInferenceEngine.infer()."""
    posteriors: pd.DataFrame     # (n_days, n_states); columns = state labels
    labels: pd.Series            # dominant label per day
    model_id: str                # identifies the fitted model version
    diagnostics: dict


# BacktestResult lives in src.athena_regime.backtest.engine — import from there.
