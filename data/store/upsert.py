from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Sequence

import pandas as pd

from data.store.locks import FileLock


def deterministic_fingerprint(df: pd.DataFrame, sort_keys: Sequence[str] | None = None) -> str:
    frame = df.copy()
    if sort_keys:
        keys = [k for k in sort_keys if k in frame.columns]
        if keys:
            frame = frame.sort_values(keys, kind="mergesort")
    csv = frame.to_csv(index=False, na_rep="__NULL__", float_format="%.12g", date_format="%Y-%m-%dT%H:%M:%S")
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def partition_path(root: Path, dataset: str, partition_key: str) -> Path:
    return Path(root) / dataset / partition_key


def replace_partition(
    root: Path,
    dataset: str,
    partition_key: str,
    df: pd.DataFrame,
    sort_keys: Sequence[str] | None = None,
    filename: str = "part-00000.parquet",
) -> tuple[Path, str, int]:
    dest_dir = partition_path(root, dataset, partition_key)
    dest_file = dest_dir / filename
    tmp_dir = dest_dir.parent / f"{dest_dir.name}.tmp"
    lock_path = Path(root) / ".locks" / f"{dataset}_{partition_key.replace('=', '_')}.lock"

    with FileLock(lock_path):
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / filename
        df.to_parquet(tmp_file, index=False)
        fp = deterministic_fingerprint(df, sort_keys=sort_keys)

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        tmp_dir.rename(dest_dir)

    return dest_file, fp, int(len(df))
