from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.athena_regime.data.lake import DataLake

logger = logging.getLogger(__name__)

COT_FFILL_LIMIT = 7
SURPRISE_FFILL_DAYS = 5


def as_of_align(
    series: pd.Series | pd.DataFrame,
    target_idx: pd.DatetimeIndex,
    ffill_limit: int | None = None,
) -> pd.DataFrame:
    if isinstance(series, pd.Series):
        df = series.to_frame()
    else:
        df = series.copy()

    df.index = pd.to_datetime(df.index)
    target = pd.DataFrame(index=pd.to_datetime(target_idx))
    combined = df.reindex(target.index.union(df.index)).sort_index()
    combined = combined.ffill(limit=ffill_limit)
    return combined.reindex(target.index)


def build_feature_matrix(
    lake: DataLake,
    start_date: str | None = None,
    end_date: str | None = None,
    cot_z_window: int = 52,  # kept for API compatibility
) -> pd.DataFrame:
    del cot_z_window
    returns_long = lake.load("returns_daily", start=start_date, end=end_date)
    cot_long = lake.load("cot_z_weekly", start=start_date, end=end_date)
    policy_wide = lake.load("policy_features", start=start_date, end=end_date)
    macro_wide = lake.load("macro_asof_monthly", start=start_date, end=end_date)

    if returns_long.empty or "dt" not in returns_long.columns:
        raise ValueError("Gold dataset returns_daily is empty or missing 'dt'. Run `athena data update --dataset returns_daily`.")

    returns_long = returns_long.copy()
    returns_long["dt"] = pd.to_datetime(returns_long["dt"])
    dates = returns_long["dt"].drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("No dates available in returns_daily.")

    target_idx = pd.DatetimeIndex(dates)
    if target_idx.tz is None:
        target_idx = target_idx.tz_localize("UTC")

    logger.info(
        "Building feature matrix for %d dates (%s -> %s)",
        len(target_idx),
        target_idx[0].date(),
        target_idx[-1].date(),
    )

    ret_feats = _build_returns_features(returns_long, target_idx)
    cot_feats = _build_cot_features(cot_long, target_idx)
    policy_feats = _build_policy_features(policy_wide, target_idx)
    macro_feats = _build_macro_surprise_proxy(macro_wide, target_idx)

    parts = [frame for frame in [ret_feats, cot_feats, policy_feats, macro_feats] if not frame.empty]
    X = pd.concat(parts, axis=1)
    X.index.name = "date"
    logger.info("Feature matrix shape: %s | columns: %d", X.shape, len(X.columns))
    return X


def build_prices_matrix(
    lake: DataLake,
    start_date: str | None = None,
    end_date: str | None = None,
    target_idx: pd.DatetimeIndex | None = None,
) -> pd.DataFrame | None:
    returns_long = lake.load("returns_daily", start=start_date, end=end_date)
    if returns_long.empty:
        return None
    required = {"dt", "instrument", "px_t"}
    if not required.issubset(returns_long.columns):
        return None

    frame = returns_long.copy()
    frame["dt"] = pd.to_datetime(frame["dt"])
    prices = frame.pivot_table(index="dt", columns="instrument", values="px_t", aggfunc="last").sort_index()
    if target_idx is not None:
        target = pd.DatetimeIndex(target_idx)
        idx = target.tz_convert(None) if target.tz is not None else target
        prices = prices.reindex(idx).ffill(limit=3)
    if prices.index.tz is None:
        prices.index = prices.index.tz_localize("UTC")
    return prices


def _build_returns_features(returns_long: pd.DataFrame, target_idx: pd.DatetimeIndex) -> pd.DataFrame:
    if returns_long.empty:
        return pd.DataFrame(index=target_idx)

    frame = returns_long.copy()
    frame["dt"] = pd.to_datetime(frame["dt"]).dt.tz_localize(None)
    frame["tag"] = frame["instrument"].astype(str).str.replace("_price", "", regex=False)

    ret_wide = frame.pivot_table(index="dt", columns="tag", values="return", aggfunc="last").sort_index()
    ret_wide.columns = [f"ret_{c}" for c in ret_wide.columns]

    tgt = target_idx.tz_convert(None) if target_idx.tz is not None else target_idx
    ret_aligned = as_of_align(ret_wide, tgt, ffill_limit=3)
    ret_aligned.index = target_idx

    features = pd.DataFrame(index=target_idx)
    for col in ret_aligned.columns:
        tag = col.replace("ret_", "")
        features[col] = ret_aligned[col]
        features[f"ma5_{tag}"] = ret_aligned[col].rolling(5, min_periods=3).mean()
        features[f"ma20_{tag}"] = ret_aligned[col].rolling(20, min_periods=10).mean()

    if "ret_SP500" in features and "ret_US10Y" in features:
        features["stocks_vs_bonds"] = features["ret_SP500"] - features["ret_US10Y"]
    elif "ret_SPX_US" in features and "ret_US10Y_US" in features:
        features["stocks_vs_bonds"] = features["ret_SPX_US"] - features["ret_US10Y_US"]

    if "ret_OIL" in features and "ret_SP500" in features:
        features["oil_vs_stocks"] = features["ret_OIL"] - features["ret_SP500"]
    if "ret_GOLD" in features and "ret_USD_index" in features:
        features["gold_vs_usd"] = features["ret_GOLD"] - features["ret_USD_index"]
    if "ret_US10Y" in features and "ret_US2Y" in features:
        features["curve_slope"] = features["ret_US10Y"] - features["ret_US2Y"]
    elif "ret_US10Y_US" in features and "ret_US2Y_US" in features:
        features["curve_slope"] = features["ret_US10Y_US"] - features["ret_US2Y_US"]

    return features


def _build_cot_features(cot_long: pd.DataFrame, target_idx: pd.DatetimeIndex) -> pd.DataFrame:
    if cot_long.empty:
        return pd.DataFrame(index=target_idx)
    if "week" not in cot_long.columns:
        return pd.DataFrame(index=target_idx)

    frame = cot_long.copy()
    frame["pub_date"] = pd.to_datetime(
        frame["week"].astype(str).str.replace(r"^week=", "", regex=True) + "-4",
        format="%Y-%W-%w",
        errors="coerce",
    )
    frame = frame.dropna(subset=["pub_date"])
    if frame.empty:
        return pd.DataFrame(index=target_idx)

    cot_wide = frame.pivot_table(index="pub_date", columns="contract", values="zscore", aggfunc="mean").sort_index()
    cot_wide.columns = [f"cot_z_{str(c).replace('_net', '').replace(' ', '_')}" for c in cot_wide.columns]

    tgt = target_idx.tz_convert(None) if target_idx.tz is not None else target_idx
    cot_daily = as_of_align(cot_wide, tgt, ffill_limit=COT_FFILL_LIMIT)
    cot_daily.index = target_idx
    return cot_daily


def _build_policy_features(policy_wide: pd.DataFrame, target_idx: pd.DatetimeIndex) -> pd.DataFrame:
    if policy_wide.empty:
        return pd.DataFrame(index=target_idx)
    if "dt" not in policy_wide.columns:
        return pd.DataFrame(index=target_idx)

    frame = policy_wide.copy()
    frame["dt"] = pd.to_datetime(frame["dt"]).dt.tz_localize(None)
    frame = frame.set_index("dt").sort_index()
    frame = frame.drop(columns=["source"], errors="ignore")
    frame = frame.rename(
        columns={
            "implied_cut_prob_3m": "fw_prob_cut_3m",
            "implied_cut_prob_6m": "fw_prob_cut_6m",
            "term_front_slope": "fw_term_slope",
            "terminal_shift_30d": "fw_terminal_shift",
        }
    )

    tgt = target_idx.tz_convert(None) if target_idx.tz is not None else target_idx
    aligned = as_of_align(frame, tgt, ffill_limit=None)
    aligned.index = target_idx

    if "fw_prob_cut_3m" in aligned.columns:
        aligned["fw_net_cut_bias"] = aligned["fw_prob_cut_3m"] - (1.0 - aligned["fw_prob_cut_3m"])
        aligned["fw_policy_uncertainty"] = 1.0 - aligned["fw_prob_cut_3m"].abs()
    return aligned


def _build_macro_surprise_proxy(macro_wide: pd.DataFrame, target_idx: pd.DatetimeIndex) -> pd.DataFrame:
    if macro_wide.empty:
        return pd.DataFrame({"surprise_score": 0.0}, index=target_idx)
    if "month_end" not in macro_wide.columns:
        return pd.DataFrame({"surprise_score": 0.0}, index=target_idx)

    frame = macro_wide.copy()
    frame["month_end"] = pd.to_datetime(frame["month_end"]).dt.tz_localize(None)
    frame = frame.set_index("month_end").sort_index()

    numeric_cols = [c for c in frame.columns if not c.endswith("__vintage_id")]
    frame = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if frame.empty or len(frame) < 2:
        return pd.DataFrame({"surprise_score": 0.0}, index=target_idx)

    changes = frame.diff()
    roll_std = changes.rolling(12, min_periods=3).std().replace(0, np.nan)
    surprises = (changes / roll_std).mean(axis=1).fillna(0.0).clip(-3, 3)

    tgt = target_idx.tz_convert(None) if target_idx.tz is not None else target_idx
    aligned = as_of_align(surprises.to_frame("surprise_score"), tgt, ffill_limit=SURPRISE_FFILL_DAYS)
    aligned.index = target_idx
    aligned["surprise_score"] = aligned["surprise_score"].fillna(0.0)
    return aligned
