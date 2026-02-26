from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.normalize.validation import validate_frame
from data.store.metadata import MetadataStore
from data.store.upsert import replace_partition


def build_cot_zscores(
    datastore_root: Path,
    weeks_to_rebuild: list[str],
    lookback_weeks: int,
    schema_path: Path,
) -> dict[str, str]:
    canonical_root = Path(datastore_root) / "silver" / "cot_weekly"
    published_root = Path(datastore_root) / "gold"
    metadata = MetadataStore(Path(datastore_root) / "meta")

    rows = []
    for path in canonical_root.glob("week=*/part-00000.parquet"):
        rows.append(pd.read_parquet(path))
    if not rows:
        return {}

    cot = pd.concat(rows, ignore_index=True)
    cot["report_date"] = pd.to_datetime(cot["report_date"])
    cot["week"] = cot["report_date"].dt.strftime("%Y-%W")
    cot = cot.sort_values(["contract", "group", "report_date"])

    cot["zscore"] = cot.groupby(["contract", "group"])["net_pos"].transform(
        lambda s: (s - s.rolling(lookback_weeks, min_periods=4).mean()) / s.rolling(lookback_weeks, min_periods=4).std()
    )

    outputs: dict[str, str] = {}
    for week in sorted(set(weeks_to_rebuild)):
        data = cot[cot["week"] == week][["week", "contract", "group", "net_pos", "zscore"]].copy()
        if data.empty:
            continue
        validate_frame(data, schema_path)
        part_key = f"week={week}"
        dest, fp, rows_count = replace_partition(
            root=published_root,
            dataset="cot_z_weekly",
            partition_key=part_key,
            df=data,
            sort_keys=["week", "contract", "group"],
        )
        metadata.update_partition("gold.cot_z_weekly", part_key, fp, rows_count, str(dest))
        outputs[part_key] = fp
    return outputs
