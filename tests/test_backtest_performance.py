from __future__ import annotations

import numpy as np
import pandas as pd

from src.athena_regime.backtest.engine import compute_performance


def test_compute_performance_aligns_labels_and_emits_tail_metrics() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    ret = pd.Series([0.01, -0.02, 0.015, 0.0, -0.005, 0.02], index=idx)
    rebal = pd.DataFrame(
        {
            "do_rebalance": [True, False, False, True, False, False],
            "distance": [0.2, 0.0, 0.0, 0.3, 0.0, 0.0],
            "surprise_triggered": [False, False, False, True, False, False],
            "max_prob": [0.8, 0.7, 0.6, 0.9, 0.75, 0.65],
        },
        index=idx,
    )
    labels = pd.Series(["A", "A", "B", "B"], index=idx[:4])  # intentionally shorter index

    perf = compute_performance(ret, rebal, labels, risk_free_rate=0.0, freq=252)
    assert perf["n_trading_days"] == 6
    assert "var_95" in perf
    assert "cvar_95" in perf
    assert "tail_ratio" in perf
    assert set(perf["conditional_returns"].keys()) == {"A", "B"}
    assert np.isfinite(perf["avg_jsd_distance"])
