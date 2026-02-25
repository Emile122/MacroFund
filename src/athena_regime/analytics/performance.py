# src/athena_regime/analytics/performance.py
from __future__ import annotations
import numpy as np
import pandas as pd
from src.athena_regime.backtest.engine import BacktestResult


def compute_metrics(result: BacktestResult, risk_free_rate: float = 0.05) -> dict:
    r = result.daily_returns.dropna()
    ann = 252

    cagr = (1 + r).prod() ** (ann / len(r)) - 1
    vol = r.std() * ann ** 0.5
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else np.nan
    downside = r[r < 0].std() * ann ** 0.5
    sortino = (cagr - risk_free_rate) / downside if downside > 0 else np.nan

    cum = (1 + r).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    turnover = (
        result.weights_history.diff().abs().sum(axis=1).mean()
        if result.weights_history is not None else np.nan
    )

    return {
        "cagr": round(cagr, 4),
        "ann_vol": round(vol, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "avg_daily_turnover": round(turnover, 4),
        "n_days": len(r),
    }
