from __future__ import annotations

import pandas as pd

from data.normalize.renaming import vendor_to_internal


def canonicalize_prices(
    raw: pd.DataFrame,
    vendor: str,
    universe_path,
    fetch_ts,
    currency: str = "USD",
) -> pd.DataFrame:
    frame = raw.copy()
    if frame.empty and len(frame.columns) == 0:
        return pd.DataFrame(
            columns=["vendor", "dt", "instrument", "px_last", "currency", "source_id", "fetch_ts"]
        )
    if set(["dt", "ticker", "px_last"]).issubset(frame.columns):
        df = frame
    else:
        # Assume Date + multiple ticker columns.
        if len(frame.columns) == 0:
            return pd.DataFrame(
                columns=["vendor", "dt", "instrument", "px_last", "currency", "source_id", "fetch_ts"]
            )
        date_col = frame.columns[0]
        df = frame.melt(id_vars=[date_col], var_name="ticker", value_name="px_last")
        df = df.rename(columns={date_col: "dt"})
    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    df["instrument"] = df["ticker"].apply(lambda t: vendor_to_internal(vendor, str(t), universe_path))
    df["vendor"] = vendor
    df["currency"] = currency
    df["source_id"] = df["ticker"].astype(str)
    df["fetch_ts"] = pd.to_datetime(fetch_ts, utc=True)
    out = df[["vendor", "dt", "instrument", "px_last", "currency", "source_id", "fetch_ts"]].copy()
    out["px_last"] = pd.to_numeric(out["px_last"], errors="coerce")
    out = out.dropna(subset=["dt", "instrument", "px_last"])
    return out
