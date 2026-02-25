# src/athena_regime/stress/runner.py
from __future__ import annotations
import copy
import pandas as pd
from src.athena_regime.data.models import FeatureMatrix
from src.athena_regime.backtest.engine import BacktestResult
from src.athena_regime.stress.scenarios import ScenarioDefinition
from src.athena_regime.run_context.context import RunContext


class StressTestRunner:
    """
    Applies scenario shocks to a FeatureMatrix, then delegates to
    the standard backtest engine. Returns a BacktestResult tagged
    with the scenario name.
    """

    def __init__(self, backtest_factory, ctx: RunContext) -> None:
        # backtest_factory: callable(fm: FeatureMatrix) -> BacktestResult
        self._factory = backtest_factory
        self._ctx = ctx

    def run(
        self,
        base_fm: FeatureMatrix,
        scenario: ScenarioDefinition,
    ) -> BacktestResult:
        shocked_fm = self._apply_shocks(base_fm, scenario)
        self._ctx.logger.info(
            "stress_test scenario=%s shocks=%s", scenario.name, scenario.shocks
        )
        result = self._factory(shocked_fm)
        # Tag the result — BacktestResult.performance is the metrics dict
        result.performance["scenario"] = scenario.name
        result.performance["scenario_description"] = scenario.description
        return result

    def _apply_shocks(
        self, fm: FeatureMatrix, scenario: ScenarioDefinition
    ) -> FeatureMatrix:
        X_shocked = fm.X.copy()
        for col, shock in scenario.shocks.items():
            if col not in X_shocked.columns:
                self._ctx.logger.warning(
                    "stress: feature '%s' not in matrix — skipping", col
                )
                continue
            if scenario.shock_start or scenario.shock_end:
                mask = pd.Series(True, index=X_shocked.index)
                if scenario.shock_start:
                    mask &= X_shocked.index >= scenario.shock_start
                if scenario.shock_end:
                    mask &= X_shocked.index <= scenario.shock_end
                X_shocked.loc[mask, col] += shock
            else:
                X_shocked[col] += shock
        return FeatureMatrix(
            X=X_shocked,
            feature_names=fm.feature_names,
            metadata={**fm.metadata, "scenario": scenario.name},
        )
