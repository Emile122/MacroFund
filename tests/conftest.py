# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.athena_regime.config.schema import AppConfig, DataConfig, RegimeConfig, BacktestConfig, RunConfig
from src.athena_regime.data.models import FeatureMatrix


@pytest.fixture()
def tmp_data_dirs(tmp_path):
    (tmp_path / "datastore").mkdir()
    (tmp_path / "canonical").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw_archive").mkdir()
    (tmp_path / "runs").mkdir()
    return tmp_path


@pytest.fixture()
def app_config(tmp_data_dirs):
    mapping = Path(__file__).parent.parent / "src/athena_regime/data/sources/mapping.yaml"
    return AppConfig(
        data=DataConfig(
            datastore_root=tmp_data_dirs / "datastore",
            canonical_dir=tmp_data_dirs / "canonical",
            raw_dir=tmp_data_dirs / "raw",
            raw_archive_dir=tmp_data_dirs / "raw_archive",
            mapping_yaml=mapping,
        ),
        regime=RegimeConfig(),
        backtest=BacktestConfig(),
        run=RunConfig(runs_dir=tmp_data_dirs / "runs"),
    )


@pytest.fixture()
def sample_feature_matrix():
    dates = pd.date_range("2020-01-02", periods=300, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((300, 8)), index=dates,
                     columns=["f1","f2","f3","f4","f5","f6","f7","f8"])
    return FeatureMatrix(X=X, feature_names=list(X.columns), metadata={})
