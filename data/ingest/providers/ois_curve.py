from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_curve(path: Path, dt: str | None = None) -> pd.DataFrame:
    """Load front-end futures/OIS curve data from local CSV."""
    frame = pd.read_csv(path)
    required = {"tenor_days", "implied_rate"}
    if required.issubset(frame.columns):
        return frame

    fedwatch_cols = {"date", "prob_cut_1m", "prob_cut_3m", "prob_hike_1m", "implied_rate_chg_3m"}
    if fedwatch_cols.issubset(frame.columns):
        return _fedwatch_row_to_curve(frame, dt)

    missing = required - set(frame.columns)
    raise ValueError(
        "Policy curve input missing columns: "
        f"{sorted(missing)}. Accepted formats are "
        "['tenor_days','implied_rate'] or FedWatch columns "
        "['date','prob_cut_1m','prob_cut_3m','prob_hike_1m','implied_rate_chg_3m']."
    )


def _fedwatch_row_to_curve(frame: pd.DataFrame, dt: str | None) -> pd.DataFrame:
    rows = frame.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows = rows.dropna(subset=["date"]).sort_values("date")
    if rows.empty:
        raise ValueError("FedWatch input has no valid date rows")

    if dt:
        target = pd.to_datetime(dt, errors="coerce")
        if pd.isna(target):
            raise ValueError(f"Invalid dt for policy conversion: {dt}")
        eligible = rows.loc[rows["date"] <= target]
        row = eligible.iloc[-1] if not eligible.empty else rows.iloc[-1]
    else:
        row = rows.iloc[-1]

    prob_cut_1m = float(row["prob_cut_1m"])
    prob_hike_1m = float(row["prob_hike_1m"])
    prob_cut_3m = float(row["prob_cut_3m"])
    implied_rate_chg_3m = float(row["implied_rate_chg_3m"])

    short_bias = prob_hike_1m - prob_cut_1m
    r30 = short_bias
    r90 = implied_rate_chg_3m
    r180 = implied_rate_chg_3m + 0.5 * (prob_cut_3m - short_bias)
    r365 = implied_rate_chg_3m + (prob_cut_3m - short_bias)

    return pd.DataFrame(
        {
            "tenor_days": [30, 90, 180, 365],
            "implied_rate": [r30, r90, r180, r365],
            "source": ["fedwatch"] * 4,
        }
    )
