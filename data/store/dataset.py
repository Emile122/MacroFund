from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


def _maybe_duckdb_query(path_glob: str) -> pd.DataFrame:
    try:
        import duckdb  # type: ignore
    except Exception:
        return pd.DataFrame()
    query = f"SELECT * FROM parquet_scan('{path_glob}')"
    return duckdb.sql(query).df()


def scan_partitioned_dataset(dataset_root: Path) -> pd.DataFrame:
    path = Path(dataset_root)
    pattern = str(path / "*" / "*.parquet")
    duck = _maybe_duckdb_query(pattern)
    if not duck.empty:
        return duck
    if not path.exists():
        return pd.DataFrame()
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()


def _resolve_layer(layer: str) -> str:
    if layer == "published":
        return "gold"
    if layer == "canonical":
        return "silver"
    if layer == "raw":
        return "bronze"
    return layer


def read_dataset_table(base_path: Path, dataset: str, layer: str = "gold") -> pd.DataFrame:
    root = Path(base_path) / _resolve_layer(layer) / dataset
    return scan_partitioned_dataset(root)


def legacy_single_table_view(base_path: Path, dataset: str, layer: str = "gold") -> pd.DataFrame:
    """Backwards-compatibility shim for code expecting one table per dataset."""
    return read_dataset_table(base_path=base_path, dataset=dataset, layer=layer)
