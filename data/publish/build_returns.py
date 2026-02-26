from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.normalize.validation import validate_frame
from data.store.metadata import MetadataStore
from data.store.upsert import replace_partition


def _load_prices_for_dates(canonical_root: Path, dates: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dt in dates:
        path = canonical_root / "market_prices" / f"dt={dt}" / "part-00000.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["dt", "instrument", "px_last"])
    return pd.concat(frames, ignore_index=True)


def build_returns(
    datastore_root: Path,
    dates_to_rebuild: list[str],
    schema_path: Path,
) -> dict[str, str]:
    canonical_root = Path(datastore_root) / "silver"
    published_root = Path(datastore_root) / "gold"
    metadata = MetadataStore(Path(datastore_root) / "meta")

    outputs: dict[str, str] = {}
    if not dates_to_rebuild:
        return outputs

    all_needed = sorted(set(dates_to_rebuild))
    min_date = pd.to_datetime(min(all_needed))
    prior = (min_date - pd.Timedelta(days=1)).date().isoformat()
    load_dates = sorted(set(all_needed + [prior]))

    prices = _load_prices_for_dates(canonical_root, load_dates)
    if prices.empty:
        return outputs

    prices["dt"] = pd.to_datetime(prices["dt"])
    prices = prices.sort_values(["instrument", "dt"])
    prices["px_t_1"] = prices.groupby("instrument")["px_last"].shift(1)
    prices["return"] = prices["px_last"] / prices["px_t_1"] - 1.0

    for dt in all_needed:
        target = prices[prices["dt"] == pd.Timestamp(dt)][["dt", "instrument", "return", "px_last", "px_t_1"]].copy()
        target = target.rename(columns={"px_last": "px_t"})
        target["dt"] = pd.to_datetime(target["dt"]).dt.date
        target = target.dropna(subset=["return", "px_t", "px_t_1"])
        validate_frame(target, schema_path)
        part_key = f"dt={dt}"
        dest, fp, rows = replace_partition(
            root=published_root,
            dataset="returns_daily",
            partition_key=part_key,
            df=target,
            sort_keys=["dt", "instrument"],
        )
        metadata.update_partition(
            dataset_name="gold.returns_daily",
            partition_key=part_key,
            fingerprint=fp,
            row_count=rows,
            path=str(dest),
        )
        outputs[part_key] = fp

    return outputs
