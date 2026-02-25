# tests/test_engineering.py
"""
Unit tests for RollingZScaler (extracted from features/engineering.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.athena_regime.features.engineering import RollingZScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_X(
    n_rows: int = 200,
    n_cols: int = 3,
    start: str = "2020-01-01",
    freq: str = "B",
    tz: str = "UTC",
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_rows, freq=freq, tz=tz)
    data = rng.normal(size=(n_rows, n_cols))
    cols = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(data, index=idx, columns=cols)


def _assert_frame_close(a: pd.DataFrame, b: pd.DataFrame, tol: float = 1e-12) -> None:
    assert list(a.columns) == list(b.columns), "Column mismatch"
    assert a.shape == b.shape, "Shape mismatch"
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    nan_mask = np.isnan(av) & np.isnan(bv)
    diff = np.abs(av - bv)
    diff[nan_mask] = 0.0
    assert np.nanmax(diff) <= tol, f"DataFrame values differ (max diff={np.nanmax(diff)})"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_transform_raises_if_not_fit() -> None:
    X = _make_X()
    scaler = RollingZScaler()
    with pytest.raises(RuntimeError, match="Call fit\\(\\) before transform\\(\\)"):
        _ = scaler.transform(X)


def test_fit_sets_mean_and_std_shapes() -> None:
    X_train = _make_X(n_rows=100, n_cols=5)
    scaler = RollingZScaler().fit(X_train)

    assert scaler._train_mean is not None
    assert scaler._train_std is not None
    assert list(scaler._train_mean.index) == list(X_train.columns)
    assert list(scaler._train_std.index) == list(X_train.columns)


def test_fit_transform_equals_fit_then_transform() -> None:
    X_train = _make_X(n_rows=120, n_cols=4, seed=1)
    scaler_a = RollingZScaler()
    out_a = scaler_a.fit_transform(X_train)

    scaler_b = RollingZScaler()
    scaler_b.fit(X_train)
    out_b = scaler_b.transform(X_train)

    pd.testing.assert_frame_equal(out_a, out_b)


def test_transform_preserves_index_and_columns() -> None:
    X_train = _make_X(n_rows=80, n_cols=3, seed=2)
    X_test = _make_X(n_rows=20, n_cols=3, start="2020-05-01", seed=3)

    scaler = RollingZScaler().fit(X_train)
    out = scaler.transform(X_test)

    assert out.index.equals(X_test.index)
    assert list(out.columns) == list(X_test.columns)


def test_scaling_uses_training_stats_only() -> None:
    X_train = _make_X(n_rows=200, n_cols=2, seed=4)
    X_test = _make_X(n_rows=50, n_cols=2, start="2020-10-01", seed=5) + 100.0

    scaler = RollingZScaler().fit(X_train)
    out = scaler.transform(X_test)

    train_scaled = scaler.transform(X_train)
    assert np.allclose(train_scaled.mean().values, 0.0, atol=1e-10)
    assert np.allclose(train_scaled.std(ddof=1).values, 1.0, atol=1e-10)
    assert (out.mean().abs() > 1.0).all()


def test_zero_std_columns_are_handled_by_replacing_with_1() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="B", tz="UTC")
    X_train = pd.DataFrame(
        {"const": np.ones(len(idx)), "vary": np.arange(len(idx), dtype=float)},
        index=idx,
    )

    scaler = RollingZScaler().fit(X_train)
    assert scaler._train_std is not None
    assert scaler._train_std["const"] == 1.0  # replaced from 0 to 1

    out = scaler.transform(X_train)
    assert np.allclose(out["const"].values, 0.0)


def test_output_is_finite_for_regular_inputs() -> None:
    X_train = _make_X(n_rows=150, n_cols=3, seed=6)
    X_test = _make_X(n_rows=40, n_cols=3, start="2020-08-01", seed=7)

    scaler = RollingZScaler().fit(X_train)
    out = scaler.transform(X_test)

    assert np.isfinite(out.to_numpy()).all()


def test_basic_correctness() -> None:
    idx_train = pd.date_range("2020-01-01", periods=5, freq="D")
    X_train = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 12.0, 14.0, 16.0, 18.0]},
        index=idx_train,
    )
    idx_test = pd.date_range("2020-01-06", periods=3, freq="D")
    X_test = pd.DataFrame(
        {"a": [6.0, 7.0, 8.0], "b": [20.0, 22.0, 24.0]},
        index=idx_test,
    )

    scaler = RollingZScaler()
    scaler.fit(X_train)

    mean = X_train.mean()
    std = X_train.std().replace(0, 1.0)
    expected_train = (X_train - mean) / std
    expected_test = (X_test - mean) / std

    _assert_frame_close(scaler.transform(X_train), expected_train)
    _assert_frame_close(scaler.transform(X_test), expected_test)

    scaler2 = RollingZScaler()
    _assert_frame_close(scaler2.fit_transform(X_train), expected_train)


def test_nan_preserved_in_output() -> None:
    X_train_nan = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]},
        index=pd.date_range("2022-01-01", periods=3, freq="D"),
    )
    scaler = RollingZScaler().fit(X_train_nan)
    out_nan = scaler.transform(X_train_nan)
    assert np.isnan(out_nan.loc[X_train_nan["a"].isna(), "a"]).all()
