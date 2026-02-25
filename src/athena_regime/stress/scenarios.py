# src/athena_regime/stress/scenarios.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass(frozen=True)
class ScenarioDefinition:
    """
    Defines a stress test by specifying additive z-score shocks
    to named features. The runner injects these into a FeatureMatrix
    before passing it through the regime+allocation pipeline.
    """
    name: str
    description: str
    # feature_name -> additive shock in z-score units
    shocks: dict[str, float]
    # If set, shock is applied only within this date window
    shock_start: Optional[pd.Timestamp] = None
    shock_end: Optional[pd.Timestamp] = None
    seed: Optional[int] = None


class ScenarioRegistry:
    """Simple name->scenario lookup. Load from YAML or register inline."""

    _registry: dict[str, ScenarioDefinition] = {}

    @classmethod
    def register(cls, scenario: ScenarioDefinition) -> None:
        cls._registry[scenario.name] = scenario

    @classmethod
    def get(cls, name: str) -> ScenarioDefinition:
        if name not in cls._registry:
            raise KeyError(f"Scenario '{name}' not registered. "
                           f"Available: {list(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def from_yaml(cls, path) -> None:
        import yaml
        from pathlib import Path
        with open(path) as f:
            raw = yaml.safe_load(f)
        for entry in raw.get("scenarios", []):
            cls.register(ScenarioDefinition(**entry))


# Built-in scenarios
# Feature names match pipeline.py output: ret_{tag} where tag = col.replace("_price","")
# e.g. SP500_price -> ret_SP500, US10Y_price -> ret_US10Y, GOLD_price -> ret_GOLD
ScenarioRegistry.register(ScenarioDefinition(
    name="rates_shock_+200bps",
    description="Parallel shift up 200bps in all rate features",
    shocks={"ret_US10Y": 2.0, "ret_US2Y": 2.0},
))
ScenarioRegistry.register(ScenarioDefinition(
    name="risk_off",
    description="Equity crash + flight-to-quality",
    shocks={"ret_SP500": -3.0, "ret_GOLD": 2.0, "ret_US10Y": -1.5},
))
ScenarioRegistry.register(ScenarioDefinition(
    name="hawkish-surprise",
    description="Sudden hawkish pivot: rates spike, equities sell off, gold weakens",
    shocks={"ret_US10Y": 3.0, "ret_US2Y": 4.0, "ret_SP500": -2.0,
            "ret_GOLD": -1.0, "fw_net_cut_bias": -0.5},
))
