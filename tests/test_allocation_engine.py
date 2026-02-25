from __future__ import annotations

import numpy as np
import pandas as pd

from src.athena_regime.allocation.engine import AllocationEngine, enforce_constraints


def test_turnover_cap_uses_one_way_definition() -> None:
    prev = {"A": 0.5, "B": 0.3, "C": 0.2}
    new = {"A": 0.2, "B": 0.0, "C": 0.8}
    capped = enforce_constraints(new, max_leverage=2.0, max_weight=1.0, turnover_cap=0.30, prev_weights=prev)
    one_way = 0.5 * sum(abs(capped.get(a, 0.0) - prev.get(a, 0.0)) for a in set(capped) | set(prev))
    assert one_way <= 0.30 + 1e-9


def test_allocation_step_requires_state_aligned_keys() -> None:
    eng = AllocationEngine(mode="soft")
    probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    try:
        eng.step(
            date=pd.Timestamp("2024-01-01"),
            probs=probs,
            regime_keys=["Easing", "Reacceleration"],  # wrong length
            force_rebalance=True,
        )
    except ValueError as exc:
        assert "regime_keys length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched regime_keys length")
