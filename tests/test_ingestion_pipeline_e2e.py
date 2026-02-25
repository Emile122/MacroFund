from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.athena_regime.data.ingestion import DEFAULT_UPDATE_DATASETS, update_dataset
from src.athena_regime.data.lake import DataLake
from src.athena_regime.data.pipeline import build_feature_matrix
from src.athena_regime.data.providers import MockProvider


def test_update_is_idempotent(tmp_path: Path) -> None:
    lake = DataLake(tmp_path / "datastore")
    provider = MockProvider()

    first = update_dataset(
        lake=lake,
        dataset="returns_daily",
        provider=provider,
        start="2026-01-02",
        end="2026-01-15",
    )
    second = update_dataset(
        lake=lake,
        dataset="returns_daily",
        provider=provider,
        start="2026-01-02",
        end="2026-01-15",
    )

    out = lake.load("returns_daily", start="2026-01-02", end="2026-01-15")
    deduped = out.drop_duplicates(["dt", "instrument"])

    assert first["status"] == "updated"
    assert second["status"] == "updated"
    assert len(out) == len(deduped)


def test_end_to_end_feature_pipeline_deterministic(tmp_path: Path) -> None:
    lake = DataLake(tmp_path / "datastore")
    provider = MockProvider()

    for dataset in DEFAULT_UPDATE_DATASETS:
        update_dataset(
            lake=lake,
            dataset=dataset,
            provider=provider,
            start="2025-01-01",
            end="2026-01-31",
        )

    x1 = build_feature_matrix(lake=lake, start_date="2025-01-01", end_date="2026-01-31")
    x2 = build_feature_matrix(lake=lake, start_date="2025-01-01", end_date="2026-01-31")

    assert not x1.empty
    pd.testing.assert_frame_equal(x1, x2)
