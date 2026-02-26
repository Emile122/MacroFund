from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from data.normalize.validation import validate_frame
from data.store.metadata import MetadataStore
from data.store.upsert import replace_partition


def _lag_months(value: str) -> int:
    # ISO period strings like P1M
    if value.startswith("P") and value.endswith("M"):
        return int(value[1:-1])
    return 1


def build_macro_asof(
    datastore_root: Path,
    months_to_rebuild: list[str],
    lags_path: Path,
    schema_path: Path,
) -> dict[str, str]:
    canonical_root = Path(datastore_root) / "silver" / "macro_releases"
    published_root = Path(datastore_root) / "gold"
    metadata = MetadataStore(Path(datastore_root) / "meta")

    if not canonical_root.exists() or not months_to_rebuild:
        return {}

    rows = []
    for file in canonical_root.glob("dt=*/part-00000.parquet"):
        rows.append(pd.read_parquet(file))
    if not rows:
        return {}
    releases = pd.concat(rows, ignore_index=True)
    releases["period"] = pd.to_datetime(releases["period"])
    releases["dt_ingest"] = pd.to_datetime(releases["dt_ingest"])
    releases["release_date"] = pd.to_datetime(releases["release_date"], errors="coerce")

    lags = yaml.safe_load(Path(lags_path).read_text(encoding="utf-8"))
    lag_map = lags.get("series_lags", {})
    default_lag = _lag_months(lags.get("default_release_lag", "P1M"))

    outputs: dict[str, str] = {}
    for month in sorted(set(months_to_rebuild)):
        month_end = pd.Timestamp(f"{month}-01") + pd.offsets.MonthEnd(0)
        picks = []
        for series_id, grp in releases.groupby("series_id"):
            lag = _lag_months(lag_map.get(series_id, f"P{default_lag}M"))
            candidate = grp.copy()
            candidate["effective_release"] = candidate["release_date"]
            missing = candidate["effective_release"].isna()
            candidate.loc[missing, "effective_release"] = candidate.loc[missing, "period"] + pd.offsets.MonthEnd(lag)
            candidate = candidate[candidate["effective_release"] <= month_end]
            if candidate.empty:
                continue
            candidate = candidate.sort_values(["period", "dt_ingest", "vintage_id"])
            row = candidate.iloc[-1]
            picks.append({
                "month_end": month_end.date(),
                series_id: float(row["value"]),
                f"{series_id}__vintage_id": str(row["vintage_id"]),
            })
        if not picks:
            continue
        wide = pd.DataFrame(picks).groupby("month_end", as_index=False).first()
        validate_frame(wide, schema_path)
        part_key = f"month={month}"
        dest, fp, rows_count = replace_partition(
            root=published_root,
            dataset="macro_asof_monthly",
            partition_key=part_key,
            df=wide,
            sort_keys=["month_end"],
        )
        metadata.update_partition("gold.macro_asof_monthly", part_key, fp, rows_count, str(dest))
        outputs[part_key] = fp
    return outputs
