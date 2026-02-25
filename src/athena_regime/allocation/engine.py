"""
allocation/engine.py
====================
Portfolio allocation engine implementing Wt = Ï†(Et).

Two allocation modes:
  * Hard mapping  : argmax regime if max(p_t) >= confidence_threshold;
                    else fall back to prior or baseline weights.
  * Soft mapping  : probability-weighted blend of per-regime target weights.

Constraints:
  * Leverage limit:     Î£|w_i| <= L
  * Max weight per asset: |w_i| <= w_max
  * Turnover cap:        Î£|Î”w_i| <= cap (per rebalance)
  * Vol targeting hook:  placeholder interface (plugs into RiskGovernor in prod)

Rebalancing trigger:
  * L1 distance between regime probability vectors: ||p_t - p_{t-1}||_1
  * Jensen-Shannon divergence (JSD): more principled information-theoretic measure
  * Threshold Î¸ (configurable): only rebalance when distance > Î¸
  * Macro surprise override: allow rebalance even if distance < Î¸
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# â”€â”€ Per-regime target weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Default archetypal weights per regime state label.
# Units: notional weight (positive = long, negative = short)
# Sum of absolute weights â‰¤ 2.0 (moderate leverage)
DEFAULT_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "Easing": {
        "US10Y"   : +0.30,
        "US2Y"    : +0.10,
        "GOLD"    : +0.20,
        "USD_index": -0.15,
        "SP500"   : -0.05,
        "OIL"     : -0.10,
        "CARRY_ETF": -0.10,
        "TIPS"    : +0.25,
    },
    "Reacceleration": {
        "US10Y"   : -0.20,
        "US2Y"    : -0.15,
        "GOLD"    : -0.05,
        "USD_index": +0.10,
        "SP500"   : +0.30,
        "OIL"     : +0.20,
        "CARRY_ETF": +0.20,
        "TIPS"    : -0.05,
    },
    "Stagflation": {
        "US10Y"   : -0.20,
        "US2Y"    : -0.10,
        "GOLD"    : +0.25,
        "USD_index": +0.10,
        "SP500"   : -0.15,
        "OIL"     : +0.30,
        "CARRY_ETF": -0.10,
        "TIPS"    : +0.10,
    },
    "Risk-Off": {
        "US10Y"   : +0.30,
        "US2Y"    : +0.15,
        "GOLD"    : +0.20,
        "USD_index": +0.10,
        "SP500"   : -0.25,
        "OIL"     : -0.15,
        "CARRY_ETF": -0.20,
        "TIPS"    : +0.15,
    },
}

# Baseline (neutral) weights â€” used as fallback if regime confidence is low
BASELINE_WEIGHTS: dict[str, float] = {
    "US10Y"   : +0.10,
    "US2Y"    : +0.05,
    "GOLD"    : +0.05,
    "USD_index": 0.00,
    "SP500"   : +0.10,
    "OIL"     : 0.00,
    "CARRY_ETF": +0.05,
    "TIPS"    : +0.05,
}


# â”€â”€ Probability distance metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    L1 (total variation) distance between two probability vectors.
    Range [0, 2]. A shift of 1 from one regime to another gives L1 = 2.
    """
    return float(np.abs(p - q).sum())


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence between two probability vectors.
    Range [0, log(2)] â‰ˆ [0, 0.693].
    Symmetric and bounded â€” preferred for rebalancing triggers.

    JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)  where M = (P+Q)/2
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    eps = 1e-12

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log(a[mask] / (b[mask] + eps))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def regime_distance(
    p        : np.ndarray,
    q        : np.ndarray,
    metric   : str = "jsd",
) -> float:
    """
    Compute distance between two regime probability vectors.

    Parameters
    ----------
    p, q   : Probability vectors (must sum to 1, same length).
    metric : 'l1' or 'jsd'.
    """
    if metric == "l1":
        return l1_distance(p, q)
    elif metric == "jsd":
        return jsd(p, q)
    else:
        raise ValueError(f"Unknown distance metric: {metric}. Choose 'l1' or 'jsd'.")


# â”€â”€ Rebalance decision â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class RebalanceDecision:
    """Output of the rebalancing logic for a single date."""
    date               : pd.Timestamp
    do_rebalance       : bool
    reason             : str
    distance           : float
    metric             : str
    threshold          : float
    prev_regime_probs  : np.ndarray
    new_regime_probs   : np.ndarray
    surprise_triggered : bool = False


def should_rebalance(
    prev_probs         : np.ndarray,
    new_probs          : np.ndarray,
    date               : pd.Timestamp,
    threshold          : float = 0.10,
    metric             : str   = "jsd",
    surprise_score     : float = 0.0,
    surprise_threshold : float = 1.5,
) -> RebalanceDecision:
    """
    Decide whether to rebalance based on regime distribution shift.

    Primary trigger: ||p_t - p_{t-1}|| > Î¸ (in probability space).
    Optional trigger: |surprise_score| >= surprise_threshold (macro override).

    Parameters
    ----------
    prev_probs         : Previous regime probability vector.
    new_probs          : Current regime probability vector.
    date               : Current date (for logging).
    threshold          : Distance threshold Î¸.
    metric             : Distance metric ('l1' or 'jsd').
    surprise_score     : Macro surprise score for this date.
    surprise_threshold : Minimum |surprise| to force a rebalance.

    Returns
    -------
    RebalanceDecision
    """
    dist = regime_distance(prev_probs, new_probs, metric)
    surprise_triggered = abs(surprise_score) >= surprise_threshold

    do_rebalance = dist > threshold or surprise_triggered
    if surprise_triggered and dist <= threshold:
        reason = f"Surprise-triggered (score={surprise_score:.2f}, dist={dist:.4f}<Î¸)"
    elif do_rebalance:
        reason = f"Distribution shift {metric}={dist:.4f} > Î¸={threshold:.4f}"
    else:
        reason = f"No rebalance ({metric}={dist:.4f} â‰¤ Î¸={threshold:.4f})"

    return RebalanceDecision(
        date               = date,
        do_rebalance       = do_rebalance,
        reason             = reason,
        distance           = dist,
        metric             = metric,
        threshold          = threshold,
        prev_regime_probs  = prev_probs.copy(),
        new_regime_probs   = new_probs.copy(),
        surprise_triggered = surprise_triggered,
    )


# â”€â”€ Portfolio weight constraints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def enforce_constraints(
    weights     : dict[str, float],
    max_leverage: float = 2.0,
    max_weight  : float = 0.40,
    turnover_cap: float | None = None,
    prev_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Enforce portfolio constraints:
      1. Per-asset weight cap: |w_i| <= max_weight.
      2. Gross leverage: Î£|w_i| <= max_leverage (rescale if exceeded).
      3. Turnover cap: |Î”w|_1 <= turnover_cap (linear interpolation toward prev).

    Parameters
    ----------
    weights      : Target weight dict {asset â†’ weight}.
    max_leverage : Maximum gross exposure (Î£|w_i|).
    max_weight   : Maximum single-asset exposure.
    turnover_cap : Maximum one-way turnover per rebalance (None = uncapped).
    prev_weights : Previous weights (required if turnover_cap is set).

    Returns
    -------
    dict[str, float] : Constrained weights.
    """
    w = weights.copy()

    # 1. Per-asset cap
    for asset in w:
        w[asset] = float(np.clip(w[asset], -max_weight, max_weight))

    # 2. Leverage cap
    gross = sum(abs(v) for v in w.values())
    if gross > max_leverage:
        sf = max_leverage / gross
        w = {k: v * sf for k, v in w.items()}

    # 3. Turnover cap
    if turnover_cap is not None and prev_weights is not None:
        all_assets = set(w.keys()) | set(prev_weights.keys())
        turnover = sum(
            abs(w.get(a, 0.0) - prev_weights.get(a, 0.0))
            for a in all_assets
        )
        if turnover > turnover_cap:
            # Linear interpolation toward prev to exactly hit the cap
            alpha = turnover_cap / turnover
            w = {
                a: prev_weights.get(a, 0.0) + alpha * (w.get(a, 0.0) - prev_weights.get(a, 0.0))
                for a in all_assets
            }
            logger.debug("Turnover capped: raw=%.4f cap=%.4f Î±=%.4f", turnover, turnover_cap, alpha)

    return w


def vol_targeting_hook(
    weights          : dict[str, float],
    target_vol       : float = 0.08,
    estimated_port_vol: float | None = None,
) -> dict[str, float]:
    """
    Placeholder vol-targeting interface.

    In production, plug in the ATHENA RiskGovernor.apply_vol_target here.
    Here we apply a simple scaling if estimated_port_vol is provided.

    Parameters
    ----------
    weights            : Current portfolio weights.
    target_vol         : Annualised vol target (default 8%).
    estimated_port_vol : Current estimated portfolio volatility.
                         If None, no scaling is applied.
    """
    if estimated_port_vol is None or estimated_port_vol <= 0:
        return weights
    sf = min(target_vol / estimated_port_vol, 3.0)   # cap at 3Ã— leverage
    return {k: v * sf for k, v in weights.items()}


# â”€â”€ Allocation engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AllocationEngine:
    """Regime-to-weights translator with constraints and rebalance trigger."""

    def __init__(
        self,
        regime_weights      : dict[str, dict[str, float]] | None = None,
        baseline_weights    : dict[str, float] | None = None,
        mode                : str   = "soft",
        confidence_threshold: float = 0.50,
        max_leverage        : float = 2.0,
        max_weight          : float = 0.40,
        turnover_cap        : float | None = 0.30,
        rebal_threshold     : float = 0.10,
        rebal_metric        : str   = "jsd",
        surprise_threshold  : float = 1.5,
    ) -> None:
        self.regime_weights       = regime_weights or DEFAULT_REGIME_WEIGHTS
        self.baseline_weights     = dict(baseline_weights or BASELINE_WEIGHTS)
        self.mode                 = mode
        self.confidence_threshold = confidence_threshold
        self.max_leverage         = max_leverage
        self.max_weight           = max_weight
        self.turnover_cap         = turnover_cap
        self.rebal_threshold      = rebal_threshold
        self.rebal_metric         = rebal_metric
        self.surprise_threshold   = surprise_threshold

        self._prev_probs  : np.ndarray | None = None
        self._current_weights: dict[str, float] = dict(self.baseline_weights)
        self._rebal_log   : list[RebalanceDecision] = []

    def current_weights(self) -> dict[str, float]:
        return dict(self._current_weights)

    def _weights_for_state_key(self, regime_key: str) -> dict[str, float]:
        if regime_key not in self.regime_weights:
            raise KeyError(f"Unknown regime_key '{regime_key}'. Available: {list(self.regime_weights.keys())}")
        return dict(self.regime_weights[regime_key])

    def _soft_weights(
        self,
        probs : np.ndarray,
        regime_keys: list[str],
    ) -> dict[str, float]:
        K = len(probs)
        if len(regime_keys) != K:
            raise ValueError(f"regime_keys length {len(regime_keys)} != K {K}")

        all_assets = set()
        for key in regime_keys:
            all_assets |= set(self._weights_for_state_key(key).keys())

        blended: dict[str, float] = {a: 0.0 for a in all_assets}
        for k in range(K):
            p_k = float(probs[k])
            regime_w = self._weights_for_state_key(regime_keys[k])
            for asset in all_assets:
                blended[asset] += p_k * regime_w.get(asset, 0.0)
        return blended

    def _hard_weights(
        self,
        probs : np.ndarray,
        regime_keys: list[str],
    ) -> dict[str, float]:
        K = len(probs)
        if len(regime_keys) != K:
            raise ValueError(f"regime_keys length {len(regime_keys)} != K {K}")

        k_star = int(np.argmax(probs))
        max_p  = float(probs[k_star])
        if max_p >= self.confidence_threshold:
            return self._weights_for_state_key(regime_keys[k_star])
        return dict(self.baseline_weights)

    def step(
        self,
        date           : pd.Timestamp,
        probs          : np.ndarray,
        regime_keys    : list[str],
        surprise_score : float = 0.0,
        force_rebalance: bool  = False,
    ) -> dict[str, float]:
        if self._prev_probs is None:
            self._prev_probs = probs.copy()

        if force_rebalance:
            decision = RebalanceDecision(
                date=date, do_rebalance=True, reason="Forced",
                distance=0.0, metric=self.rebal_metric,
                threshold=self.rebal_threshold,
                prev_regime_probs=self._prev_probs,
                new_regime_probs=probs,
            )
        else:
            decision = should_rebalance(
                prev_probs        = self._prev_probs,
                new_probs         = probs,
                date              = date,
                threshold         = self.rebal_threshold,
                metric            = self.rebal_metric,
                surprise_score    = surprise_score,
                surprise_threshold= self.surprise_threshold,
            )

        self._rebal_log.append(decision)

        if decision.do_rebalance:
            if self.mode == "soft":
                raw_w = self._soft_weights(probs, regime_keys)
            else:
                raw_w = self._hard_weights(probs, regime_keys)

            prev_w = self._current_weights
            final_w = enforce_constraints(
                weights      = raw_w,
                max_leverage = self.max_leverage,
                max_weight   = self.max_weight,
                turnover_cap = self.turnover_cap,
                prev_weights = prev_w if self.turnover_cap is not None else None,
            )

            self._current_weights = final_w
            self._prev_probs      = probs.copy()

            logger.debug(
                "REBALANCE | %s | reason=%s | gross=%.3f",
                date.date(), decision.reason[:60],
                sum(abs(v) for v in final_w.values()),
            )

        return dict(self._current_weights)

    def rebal_log_df(self) -> pd.DataFrame:
        """Return rebalancing log as a DataFrame."""
        rows = []
        for d in self._rebal_log:
            rows.append({
                "date"              : d.date,
                "do_rebalance"      : d.do_rebalance,
                "reason"            : d.reason,
                "distance"          : round(d.distance, 6),
                "surprise_triggered": d.surprise_triggered,
                "max_prob"          : round(float(d.new_regime_probs.max()), 4),
                "dominant_state"    : int(d.new_regime_probs.argmax()),
            })
        return pd.DataFrame(rows).set_index("date")

    def reset(self) -> None:
        """Reset engine state (for walk-forward resets between windows)."""
        self._prev_probs   = None
        self._current_weights = dict(self.baseline_weights)
        self._rebal_log    = []

def run_allocation_unit_tests() -> None:
    """Verify probability distance metrics and turnover cap enforcement."""

    # Test 1: L1 distance correctness
    p = np.array([0.7, 0.1, 0.1, 0.1])
    q = np.array([0.1, 0.7, 0.1, 0.1])
    d = l1_distance(p, q)
    assert abs(d - 1.2) < 1e-9, f"L1 wrong: {d}"
    print("  âœ“ L1 distance: correct (0.6+0.6=1.2)")

    # Test 2: JSD symmetry
    j1 = jsd(p, q)
    j2 = jsd(q, p)
    assert abs(j1 - j2) < 1e-9, "JSD not symmetric"
    assert 0 <= j1 <= np.log(2) + 1e-9, f"JSD out of [0, log2]: {j1}"
    print(f"  âœ“ JSD: symmetric, in [0, log2] (value={j1:.4f})")

    # Test 3: JSD = 0 for identical distributions
    j_same = jsd(p, p)
    assert j_same < 1e-9, f"JSD(p,p) should be 0, got {j_same}"
    print("  âœ“ JSD(p, p) = 0")

    # Test 4: Turnover cap enforcement
    prev = {"A": 0.5, "B": 0.3, "C": 0.2}
    new  = {"A": 0.2, "B": 0.0, "C": 0.8}  # sum abs deltas = 1.2 => one-way = 0.6
    capped = enforce_constraints(
        new, max_leverage=2.0, max_weight=1.0,
        turnover_cap=0.30, prev_weights=prev,
    )
    one_way_turnover = 0.5 * sum(abs(capped.get(a, 0.0) - prev.get(a, 0.0)) for a in set(capped) | set(prev))
    assert one_way_turnover <= 0.30 + 1e-9, (
        f"Turnover cap violated: one_way={one_way_turnover:.4f} > cap=0.30"
    )
    print(f"  OK Turnover cap: enforced (one_way={one_way_turnover:.4f} <= 0.30)")

    # Test 5: Leverage cap
    big = {"A": 0.8, "B": -0.8, "C": 0.8}
    constrained = enforce_constraints(big, max_leverage=1.5, max_weight=0.5)
    gross = sum(abs(v) for v in constrained.values())
    assert gross <= 1.5 + 1e-9, f"Leverage cap violated: gross={gross:.4f}"
    print(f"  âœ“ Leverage cap: enforced (gross={gross:.4f} â‰¤ 1.5)")

    # Test 6: Rebalance triggers
    p_stable = np.array([0.7, 0.1, 0.1, 0.1])
    p_shift  = np.array([0.1, 0.7, 0.1, 0.1])
    dec1 = should_rebalance(p_stable, p_stable, pd.Timestamp("2024-01-01"), threshold=0.10)
    dec2 = should_rebalance(p_stable, p_shift,  pd.Timestamp("2024-01-01"), threshold=0.10)
    assert not dec1.do_rebalance, "Should NOT rebalance on identical probs"
    assert dec2.do_rebalance, "Should rebalance on large distribution shift"
    print("  âœ“ Rebalance trigger: correctly fires on distribution shift")

    # Test 7: Surprise override
    dec3 = should_rebalance(
        p_stable, p_stable, pd.Timestamp("2024-01-01"),
        threshold=0.10, surprise_score=2.0, surprise_threshold=1.5
    )
    assert dec3.do_rebalance, "Should rebalance due to surprise override"
    assert dec3.surprise_triggered, "Should flag as surprise-triggered"
    print("  âœ“ Surprise override: correctly triggers rebalance")

    print("All allocation unit tests PASSED.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running allocation unit tests...")
    run_allocation_unit_tests()

