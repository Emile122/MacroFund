from __future__ import annotations

import numpy as np
import pandas as pd

from src.athena_regime.regimes.inference import RegimeInferenceEngine


class _FakeResult:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


class _FakeHMM:
    _fitted = True

    def infer(self, X: pd.DataFrame) -> _FakeResult:
        n = len(X)
        df = pd.DataFrame(
            {
                "state": [0] * n,
                "p_0": [0.8] * n,
                "p_1": [0.2] * n,
            },
            index=X.index,
        )
        return _FakeResult(df)


def test_regime_keys_by_state_id_requires_fit() -> None:
    eng = RegimeInferenceEngine(n_states=2, n_hmm_iter=10, n_restarts=1)
    try:
        eng.regime_keys_by_state_id()
    except RuntimeError as exc:
        assert "fit() must be called" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError before fit")


def test_infer_adds_collapse_diagnostics_columns() -> None:
    eng = RegimeInferenceEngine(n_states=2, n_hmm_iter=10, n_restarts=1)
    eng.hmm = _FakeHMM()
    eng.labels = [{"state_id": 0, "label": "Easing"}, {"state_id": 1, "label": "Risk-Off"}]
    eng.state_id_to_key = ["Easing", "Risk-Off"]

    X = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
    out = eng.infer(X)
    assert "label" in out.columns
    assert "max_prob" in out.columns
    assert "entropy" in out.columns
    assert np.isclose(float(out["max_prob"].iloc[0]), 0.8)
