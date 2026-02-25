# src/athena_regime/config/loader.py
from __future__ import annotations
import os
from pathlib import Path
import yaml
from src.athena_regime.config.schema import AppConfig, DataConfig, RegimeConfig, BacktestConfig, RunConfig
from src.athena_regime.config.utils import expand_env, deep_merge

def load_config(path: str | Path, overrides: dict | None = None) -> AppConfig:
    """Load config from YAML, apply env-var substitutions and dict overrides."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Environment variable substitution: ${VAR_NAME} in values
    raw = expand_env(raw)


    if overrides:
        deep_merge(raw, overrides)

    root = Path(path).parent.parent  # project root relative to configs/
    return _build(raw, root)


def _build(raw: dict, root: Path) -> AppConfig:
    def p(key: str, default: str) -> Path:
        val = raw.get("paths", {}).get(key, default)
        pth = Path(val)
        return pth if pth.is_absolute() else (root / pth).resolve()

    datastore_root = p("datastore_root", "datastore")

    data = DataConfig(
        datastore_root=datastore_root,
        canonical_dir=p("canonical_dir", str(datastore_root / "silver")),
        raw_dir=p("raw_dir", str(datastore_root / "bronze")),
        raw_archive_dir=p("raw_archive_dir", str(datastore_root / "bronze_archive")),
        mapping_yaml=p("mapping_yaml", "src/athena_regime/data/sources/mapping.yaml"),
        ffill_limits=raw.get("ffill_limits", {}),
    )
    regime = RegimeConfig(**raw.get("regime", {}))
    backtest = BacktestConfig(**raw.get("backtest", {}))
    run = RunConfig(
        runs_dir=p("runs_dir", "runs"),
        log_level=raw.get("log_level", "INFO"),
    )
    return AppConfig(data=data, regime=regime, backtest=backtest, run=run)
