from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.athena_regime.data.lake import DataLake


def _lake(tmp_path: Path) -> DataLake:
    return DataLake(tmp_path / "datastore")


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    frame = pd.DataFrame(
        [
            {"dt": "2026-01-02", "instrument": "SP500", "return": 0.01, "px_t": 101.0, "px_t_1": 100.0},
            {"dt": "2026-01-02", "instrument": "US10Y", "return": -0.01, "px_t": 99.0, "px_t_1": 100.0},
        ]
    )

    lake.write_partition("returns_daily", "dt=2026-01-02", frame, level="gold", validate_schema=True)
    out = lake.load("returns_daily", start="2026-01-01", end="2026-01-10")

    assert len(out) == 2
    assert set(out["instrument"]) == {"SP500", "US10Y"}


def test_partition_detection_and_latest(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    for dt in ["2026-01-02", "2026-01-03"]:
        frame = pd.DataFrame(
            [{"dt": dt, "instrument": "SP500", "return": 0.01, "px_t": 101.0, "px_t_1": 100.0}]
        )
        lake.write_partition("returns_daily", f"dt={dt}", frame, level="gold", validate_schema=True)

    parts = lake.list_partitions("returns_daily", level="gold")
    latest = lake.latest_partition("returns_daily", level="gold")

    assert parts == ["dt=2026-01-02", "dt=2026-01-03"]
    assert latest == "dt=2026-01-03"


def test_schema_validation_rejects_bad_dtypes(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    bad = pd.DataFrame(
        [{"dt": "2026-01-02", "instrument": "SP500", "return": "bad", "px_t": 101.0, "px_t_1": 100.0}]
    )
    with pytest.raises(Exception):
        lake.write_partition("returns_daily", "dt=2026-01-02", bad, level="gold", validate_schema=True)


def test_write_is_idempotent_by_primary_key(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    frame = pd.DataFrame(
        [{"dt": "2026-01-02", "instrument": "SP500", "return": 0.01, "px_t": 101.0, "px_t_1": 100.0}]
    )

    lake.write_partition("returns_daily", "dt=2026-01-02", frame, level="gold", validate_schema=True)
    lake.write_partition("returns_daily", "dt=2026-01-02", frame, level="gold", validate_schema=True)
    out = lake.load("returns_daily")

    assert len(out) == 1


def test_prune_older_than_keeps_recent(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    for dt in ["2025-01-01", "2026-01-01", "2026-01-02"]:
        frame = pd.DataFrame(
            [{"dt": dt, "instrument": "SP500", "return": 0.01, "px_t": 101.0, "px_t_1": 100.0}]
        )
        lake.write_partition("returns_daily", f"dt={dt}", frame, level="gold", validate_schema=True)

    deleted = lake.prune("returns_daily", level="gold", older_than="2026-01-01", keep_last_n=1)
    assert "dt=2025-01-01" in deleted
