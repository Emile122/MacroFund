from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.normalize.validation import validate_frame
from data.store.metadata import MetadataStore
from data.store.upsert import replace_partition


def build_policy_features(
    datastore_root: Path,
    dates_to_rebuild: list[str],
    schema_path: Path,
) -> dict[str, str]:
    canonical_root = Path(datastore_root) / "silver" / "policy_curve"
    published_root = Path(datastore_root) / "gold"
    metadata = MetadataStore(Path(datastore_root) / "meta")
    outputs: dict[str, str] = {}

    for dt in sorted(set(dates_to_rebuild)):
        path = canonical_root / f"dt={dt}" / "part-00000.parquet"
        if not path.exists():
            continue
        curve = pd.read_parquet(path)
        curve = curve.sort_values("tenor_days")

        r30 = _nearest_rate(curve, 30)
        r90 = _nearest_rate(curve, 90)
        r180 = _nearest_rate(curve, 180)
        rt = _nearest_rate(curve, 365)

        row = pd.DataFrame(
            [{
                "dt": pd.Timestamp(dt).date(),
                "implied_cut_prob_3m": float(1 / (1 + pow(2.71828, (r90 - r30) * 100))),
                "implied_cut_prob_6m": float(1 / (1 + pow(2.71828, (r180 - r30) * 100))),
                "term_front_slope": float(r180 - r30),
                "terminal_shift_30d": float(rt - r30),
                "source": str(curve["source"].iloc[0]),
            }]
        )

        validate_frame(row, schema_path)
        part_key = f"dt={dt}"
        dest, fp, rows_count = replace_partition(
            root=published_root,
            dataset="policy_features",
            partition_key=part_key,
            df=row,
            sort_keys=["dt"],
        )
        metadata.update_partition("gold.policy_features", part_key, fp, rows_count, str(dest))
        outputs[part_key] = fp
    return outputs


def _nearest_rate(curve: pd.DataFrame, tenor: int) -> float:
    idx = (curve["tenor_days"] - tenor).abs().idxmin()
    return float(curve.loc[idx, "implied_rate"])
