from __future__ import annotations

import pandas as pd


def canonicalize_policy(raw: pd.DataFrame, dt: str, source: str = "ois") -> pd.DataFrame:
    frame = raw.copy()
    frame["dt"] = pd.to_datetime(dt).date()
    frame["source"] = source
    required = ["dt", "tenor_days", "implied_rate", "source"]
    for col in required:
        if col not in frame.columns:
            raise ValueError(f"Missing required policy column: {col}")
    return frame[required].copy()
