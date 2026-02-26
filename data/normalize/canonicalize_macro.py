from __future__ import annotations

import pandas as pd

from data.normalize.renaming import vendor_to_internal


def canonicalize_macro(raw: pd.DataFrame, vendor: str, universe_path, dt_ingest: str, vintage_id: str) -> pd.DataFrame:
    frame = raw.copy()
    expected = {"period", "series_id", "value"}
    if not expected.issubset(frame.columns):
        date_col = frame.columns[0]
        frame = frame.melt(id_vars=[date_col], var_name="series_id", value_name="value")
        frame = frame.rename(columns={date_col: "period"})
    frame["series_id"] = frame["series_id"].astype(str).apply(lambda s: vendor_to_internal(vendor, s, universe_path))
    frame["period"] = pd.to_datetime(frame["period"]).dt.to_period("M").dt.to_timestamp("M").dt.date
    if "release_date" in frame.columns:
        frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce").dt.date
    else:
        frame["release_date"] = pd.NaT
    frame["dt_ingest"] = pd.to_datetime(dt_ingest).date()
    frame["unit"] = frame.get("unit", "index")
    frame["source"] = vendor
    frame["vintage_id"] = vintage_id
    frame["notes"] = frame.get("notes", "")
    out = frame[["dt_ingest", "series_id", "period", "release_date", "value", "unit", "source", "vintage_id", "notes"]]
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["period", "series_id", "value"])
