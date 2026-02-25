# src/athena_regime/config/schema.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataConfig:
    datastore_root: Path = Path("datastore")
    canonical_dir: Path = Path("datastore/silver")
    raw_dir: Path = Path("datastore/bronze")
    raw_archive_dir: Path = Path("datastore/bronze_archive")
    mapping_yaml: Path = Path("src/athena_regime/data/sources/mapping.yaml")
    ffill_limits: dict[str, int] = field(default_factory=lambda: {
        "prices": 5, "cot": 7, "policy": 5, "macro": 5
    })


@dataclass(frozen=True)
class RegimeConfig:
    n_states: int = 4
    n_hmm_iter: int = 200
    n_restarts: int = 5
    min_train_days: int = 252
    prob_temperature: float = 1.0
    min_state_occupancy: float = 0.02


@dataclass(frozen=True)
class BacktestConfig:
    train_window: int = 756      # 3Y in trading days
    step_size: int = 63          # quarterly refit
    refit_every: int = 63
    expanding: bool = True
    transaction_cost_bps: float = 5.0
    impact_bps: float = 2.0
    bid_ask_bps: float = 1.0
    max_leverage: float = 1.0
    max_weight: float = 0.40
    long_only: bool = False
    target_net_exposure: Optional[float] = None
    turnover_cap: float = 0.20
    vol_target: Optional[float] = None
    rebalance_threshold: float = 0.10
    borrow_spread_bps: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    runs_dir: Path
    log_level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    regime: RegimeConfig
    backtest: BacktestConfig
    run: RunConfig
