"""
regimes/inference.py
====================
Regime inference module combining:
  1. GaussianHMM (primary — temporal structure via Markov transitions)
  2. Gaussian Mixture Model baseline (no temporal structure)
  3. Expectation-regime labeling (rule-based, forward-looking framing)
  4. Per-state diagnostics (feature means, discriminating features)

Interpretation constraint (from spec):
  All labels are prefixed "Market pricing:" or "Positioning implies:"
  Labels NEVER say "the economy is in regime X."
  They say "institutions are positioned AS IF X is imminent."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.athena_regime.regimes.hmm import GaussianHMM, HMMResult

logger = logging.getLogger(__name__)


# ── Gaussian Mixture Model baseline ──────────────────────────────────────────

class GMMRegimeModel:
    """
    Gaussian Mixture Model baseline for regime detection.

    Unlike HMM, GMM has no temporal structure — each observation is
    classified independently. Useful as a diagnostic cross-check.

    Wraps sklearn.GaussianMixture with a compatible interface.
    """

    def __init__(
        self,
        n_states    : int   = 4,
        n_init      : int   = 10,
        random_state: int   = 42,
        covariance  : str   = "diag",
    ) -> None:
        self.n_states = n_states
        self._gmm = GaussianMixture(
            n_components     = n_states,
            covariance_type  = covariance,
            n_init           = n_init,
            random_state     = random_state,
            max_iter         = 500,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "GMMRegimeModel":
        X_ = X.values if isinstance(X, pd.DataFrame) else X
        self._gmm.fit(X_)
        self._fitted = True
        logger.info("GMM fitted | BIC=%.2f AIC=%.2f", self._gmm.bic(X_), self._gmm.aic(X_))
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_ = X.values if isinstance(X, pd.DataFrame) else X
        return self._gmm.predict_proba(X_)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_ = X.values if isinstance(X, pd.DataFrame) else X
        return self._gmm.predict(X_)

    def to_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(X)
        state = self.predict(X)
        K     = self.n_states
        cols  = {f"p_{k}": proba[:, k] for k in range(K)}
        cols["state"] = state
        return pd.DataFrame(cols, index=X.index)

    def save(self, path: str | Path) -> None:
        """Persist GMM parameters (means only; full persistence via pickle if needed)."""
        d = {
            "n_states": self.n_states,
            "means"   : self._gmm.means_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


# ── Expectation-regime labeler ────────────────────────────────────────────────

# Feature groups used for labeling
FEATURE_GROUPS = {
    "rates_direction": ["ret_US10Y", "ret_GER10Y", "ret_UK10Y", "ma5_US10Y", "ma20_US10Y"],
    "growth_assets"  : ["ret_SP500", "ret_OIL", "ret_CARRY_ETF", "ma5_SP500"],
    "safe_havens"    : ["ret_GOLD", "ret_US2Y", "ret_TIPS"],
    "fx_usd"         : ["ret_USD_index", "gold_vs_usd"],
    "cot_bonds"      : ["cot_z_bonds"],
    "cot_equities"   : ["cot_z_sp500"],
    "cot_commodities": ["cot_z_oil", "cot_z_gold"],
    "cut_probability": ["fw_prob_cut_1m", "fw_prob_cut_3m", "fw_net_cut_bias"],
    "hike_risk"      : ["fw_prob_hike_1m"],
    "policy_uncert"  : ["fw_policy_uncertainty"],
}


def _group_score(
    state_means: pd.Series,
    feature_names: list[str],
    direction: float = +1.0,
) -> float:
    """Average z-score of available features in direction."""
    vals = [state_means.get(f, np.nan) for f in feature_names]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return 0.0
    return direction * float(np.mean(vals))


def label_regime(
    state_id    : int,
    state_means : pd.Series,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Assign a human-readable label to a regime state based on its feature
    signature.

    Uses the INTERPRETATION CONSTRAINT: labels frame what institutions
    are POSITIONED FOR (not what the economy currently is).

    Returns
    -------
    dict with keys:
      label            : short label
      description      : one-line description (prefixed "Market pricing:")
      supporting_features: list of (feature, z-score) for top 5 features
      top_feature_scores: dict of group scores
    """
    # Compute group scores
    g = {}
    g["rates_up"]   = _group_score(state_means, FEATURE_GROUPS["rates_direction"], +1.0)
    g["growth_up"]  = _group_score(state_means, FEATURE_GROUPS["growth_assets"],   +1.0)
    g["havens_up"]  = _group_score(state_means, FEATURE_GROUPS["safe_havens"],      +1.0)
    g["usd_up"]     = _group_score(state_means, FEATURE_GROUPS["fx_usd"],           +1.0)
    g["cot_bonds"]  = _group_score(state_means, FEATURE_GROUPS["cot_bonds"],        +1.0)
    g["cot_eq"]     = _group_score(state_means, FEATURE_GROUPS["cot_equities"],     +1.0)
    g["cot_comm"]   = _group_score(state_means, FEATURE_GROUPS["cot_commodities"],  +1.0)
    g["cut_prob"]   = _group_score(state_means, FEATURE_GROUPS["cut_probability"],  +1.0)
    g["hike_risk"]  = _group_score(state_means, FEATURE_GROUPS["hike_risk"],        +1.0)

    # Rule-based classification
    # Aggressive easing: bonds up, gold up, equities weak, high cut probs
    easing_score = (
        g["rates_up"]    * 1.5
        + g["havens_up"] * 1.0
        + g["cut_prob"]  * 2.0
        - g["growth_up"] * 0.5
        - g["hike_risk"] * 1.0
        + g["cot_bonds"] * 1.0
    )
    # Reacceleration: equities up, commodities up, bonds weak, cuts priced out
    reaccel_score = (
        g["growth_up"]   * 2.0
        + g["cot_eq"]    * 1.5
        + g["cot_comm"]  * 0.8
        - g["cut_prob"]  * 1.0
        + g["hike_risk"] * 0.5
        - g["rates_up"]  * 1.0
    )
    # Stagflation: commodities up, bonds weak, equities weak, hike risk
    stagflation_score = (
        g["cot_comm"]    * 2.0
        - g["rates_up"]  * 1.5
        - g["growth_up"] * 1.0
        + g["hike_risk"] * 2.0
        + g["usd_up"]    * 0.5
    )
    # Risk-off: safe havens up, equities crushed, USD up, cut probs moderate
    riskoff_score = (
        g["havens_up"]   * 2.0
        + g["cot_bonds"] * 1.5
        - g["cot_eq"]    * 2.0
        - g["growth_up"] * 1.5
        + g["usd_up"]    * 0.8
    )

    scores = {
        "aggressive_easing" : easing_score,
        "reacceleration"    : reaccel_score,
        "stagflation"       : stagflation_score,
        "risk_off_deflation": riskoff_score,
    }
    best_label = max(scores, key=scores.get)

    label_map = {
        "aggressive_easing" : "Market pricing: aggressive rate cuts imminent",
        "reacceleration"    : "Positioning implies: growth reacceleration, cuts repriced out",
        "stagflation"       : "Market pricing: stagflationary pressure — commodities strong, rates under pressure",
        "risk_off_deflation": "Positioning implies: risk-off / deflation scare — flight to safe havens",
    }
    short_map = {
        "aggressive_easing" : "Easing",
        "reacceleration"    : "Reacceleration",
        "stagflation"       : "Stagflation",
        "risk_off_deflation": "Risk-Off",
    }

    # Top 5 supporting features (by absolute z-score)
    top_feats = (
        state_means.dropna()
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )
    supporting = [(feat, float(state_means[feat])) for feat in top_feats.index]

    return {
        "state_id"           : state_id,
        "label"              : short_map[best_label],
        "description"        : label_map[best_label],
        "group_scores"       : {k: round(float(v), 3) for k, v in scores.items()},
        "supporting_features": supporting,
        "top_feature_z"      : {feat: round(float(state_means.get(feat, np.nan)), 3)
                                 for feat in FEATURE_GROUPS["rates_direction"][:3]},
    }


def label_all_regimes(
    state_means  : np.ndarray,
    feature_names: list[str],
) -> list[dict]:
    """
    Label all K states.

    Parameters
    ----------
    state_means   : (K, D) array of per-state feature means.
    feature_names : D feature names.

    Returns
    -------
    list of K label dicts.
    """
    K = state_means.shape[0]
    labels = []
    for k in range(K):
        ms = pd.Series(state_means[k], index=feature_names)
        labels.append(label_regime(k, ms, feature_names))
    return labels


def print_regime_diagnostics(
    result        : HMMResult,
    feature_names : list[str],
    labels        : list[dict],
) -> None:
    """Print a formatted diagnostic table for all regime states."""
    K = len(labels)
    print("\n" + "═" * 70)
    print("  REGIME DIAGNOSTICS")
    print("═" * 70)

    for k, lab in enumerate(labels):
        print(f"\n  State {k}: {lab['label']}")
        print(f"  {lab['description']}")
        freq = (result.state == k).mean()
        print(f"  Frequency: {freq:.1%}")
        print(f"  Top supporting features:")
        for feat, zval in lab["supporting_features"]:
            bar = "▲" * int(abs(zval) + 0.5) if zval > 0 else "▼" * int(abs(zval) + 0.5)
            print(f"    {feat:<35}  z={zval:+.3f}  {bar}")

    print("\n  Transition Matrix (row=from, col=to):")
    header = "       " + "".join(f"S{k:<6}" for k in range(K))
    print(f"  {header}")
    for i in range(K):
        row = "".join(f"{result.transition_matrix[i,j]:.4f} " for j in range(K))
        print(f"  S{i}  →  {row}")
    print("═" * 70 + "\n")


def discriminating_features_table(
    result        : HMMResult,
    feature_names : list[str],
    top_n         : int = 8,
) -> pd.DataFrame:
    """
    Table of top discriminating features per state.

    Computes |mean_k - global_mean| / global_std for each feature × state.
    Returns a (top_n features) × K state DataFrame.
    """
    K, D = result.state_means.shape
    global_mean = result.state_means.mean(axis=0)
    global_std  = result.state_means.std(axis=0).clip(1e-9)

    discrimination = np.abs((result.state_means - global_mean) / global_std)

    # Score = mean discrimination across states
    feature_scores = discrimination.mean(axis=0)
    top_idx = np.argsort(feature_scores)[::-1][:top_n]

    rows = []
    for fi in top_idx:
        row = {"feature": feature_names[fi]}
        for k in range(K):
            row[f"S{k}_z"] = round(float(result.state_means[k, fi]), 3)
        row["discrimination"] = round(float(feature_scores[fi]), 3)
        rows.append(row)

    return pd.DataFrame(rows).set_index("feature")


# ── Cost model ────────────────────────────────────────────────────────────────

class CostModel:
    """
    Transaction cost model for regime-driven portfolio rebalancing.

    Implements two cost components:
      1. Proportional (bps per unit turnover): linear in |Δw|
      2. Market impact (bps × sqrt(turnover)): price impact for larger trades
      3. Bid-ask spread cost: fixed per transaction

    Parameters
    ----------
    prop_cost_bps   : Proportional cost in basis points per unit weight turnover.
    impact_bps      : Market impact coefficient in bps.
    bid_ask_bps     : Bid-ask spread cost per instrument per rebalance.
    min_trade_size  : Below this weight change, no cost is incurred (slippage dead zone).
    """

    def __init__(
        self,
        prop_cost_bps : float = 5.0,
        impact_bps    : float = 2.0,
        bid_ask_bps   : float = 1.0,
        min_trade_size: float = 0.001,
    ) -> None:
        self.prop_cost_bps  = prop_cost_bps
        self.impact_bps     = impact_bps
        self.bid_ask_bps    = bid_ask_bps
        self.min_trade_size = min_trade_size

    def compute_cost(
        self,
        w_prev    : dict[str, float],
        w_new     : dict[str, float],
        nav       : float = 1.0,
        verbose   : bool  = False,
    ) -> float:
        """
        Compute total transaction cost for a rebalance from w_prev to w_new.

        Parameters
        ----------
        w_prev : Previous portfolio weights {asset → weight}.
        w_new  : New target weights {asset → weight}.
        nav    : Portfolio NAV (used to scale impact cost; default=1 for pct returns).

        Returns
        -------
        float : Total cost as a fraction of NAV (e.g., 0.0003 = 3bps).
        """
        all_assets = set(w_prev.keys()) | set(w_new.keys())
        total_cost = 0.0
        total_turnover = 0.0
        n_trades = 0

        for asset in all_assets:
            dw = abs(w_new.get(asset, 0.0) - w_prev.get(asset, 0.0))
            if dw < self.min_trade_size:
                continue

            # Proportional cost
            prop_cost = dw * self.prop_cost_bps / 10_000
            # Market impact (Almgren-Chriss square-root law)
            impact_cost = (dw ** 0.5) * self.impact_bps / 10_000
            # Bid-ask
            ba_cost = self.bid_ask_bps / 10_000

            asset_cost = prop_cost + impact_cost + ba_cost
            total_cost += asset_cost
            total_turnover += dw
            n_trades += 1

        if verbose and n_trades > 0:
            logger.debug(
                "CostModel | trades=%d turnover=%.4f total_cost_bps=%.2f",
                n_trades, total_turnover, total_cost * 10_000,
            )

        return float(total_cost)

    def estimate_annual_drag(
        self,
        avg_turnover_per_rebal: float,
        rebal_freq_per_year   : float,
    ) -> float:
        """
        Estimate annualised cost drag.

        Parameters
        ----------
        avg_turnover_per_rebal : Average one-way turnover per rebalance.
        rebal_freq_per_year    : Number of rebalances per year.

        Returns
        -------
        float : Annualised cost as fraction of NAV.
        """
        per_rebal = (
            avg_turnover_per_rebal * self.prop_cost_bps / 10_000
            + (avg_turnover_per_rebal ** 0.5) * self.impact_bps / 10_000
            + self.bid_ask_bps / 10_000
        )
        return per_rebal * rebal_freq_per_year


# ── Regime inference orchestrator ────────────────────────────────────────────

class RegimeInferenceEngine:
    """
    Orchestrates HMM + GMM regime detection, labeling, and diagnostics.

    Parameters
    ----------
    n_states    : Number of regime states.
    n_hmm_iter  : Max Baum-Welch EM iterations.
    n_restarts  : Multi-restart count for HMM (avoids local optima).
    save_dir    : Directory to persist model artifacts.
    """

    def __init__(
        self,
        n_states   : int = 4,
        n_hmm_iter : int = 200,
        n_restarts : int = 5,
        save_dir   : str = "outputs/models",
        prob_temperature: float = 1.0,
        min_state_occupancy: float = 0.02,
    ) -> None:
        self.n_states   = n_states
        self.n_hmm_iter = n_hmm_iter
        self.n_restarts = n_restarts
        self.prob_temperature = prob_temperature
        self.min_state_occupancy = min_state_occupancy
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.hmm     : GaussianHMM | None = None
        self.gmm     : GMMRegimeModel | None = None
        self.labels  : list[dict] = []
        self.state_id_to_key: list[str] = []
        self._feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame) -> "RegimeInferenceEngine":
        """
        Fit both HMM and GMM on training data.

        Parameters
        ----------
        X_train : Scaled feature matrix (output of RollingZScaler.fit_transform).
        """
        self._feature_names = list(X_train.columns)
        Xarr = X_train.values.astype(float)

        logger.info("Fitting HMM | n_states=%d n_restarts=%d", self.n_states, self.n_restarts)
        self.hmm = GaussianHMM(
            n_states=self.n_states,
            n_iter=self.n_hmm_iter,
            random_state=42,
        )
        self.hmm.fit_best_of(Xarr, n_restarts=self.n_restarts)

        logger.info("Fitting GMM baseline | n_states=%d", self.n_states)
        self.gmm = GMMRegimeModel(n_states=self.n_states)
        self.gmm.fit(X_train)

        # Generate labels
        self.labels = label_all_regimes(self.hmm._mu, self._feature_names)
        # IMPORTANT: aligned by state_id (k=0..K-1)
        self.state_id_to_key = [lab["label"] for lab in self.labels]

        # Collapse diagnostics on training sample
        train_result = self.hmm.infer(X_train)
        state_counts = pd.Series(train_result.state).value_counts(normalize=True).sort_index()
        occupancy = {int(k): float(v) for k, v in state_counts.to_dict().items()}
        low_occ = [k for k, v in occupancy.items() if v < self.min_state_occupancy]
        if low_occ:
            logger.warning(
                "Low state occupancy detected: %s (threshold=%.3f)",
                {k: round(occupancy[k], 4) for k in low_occ},
                self.min_state_occupancy,
            )

        # Persist models
        self.hmm.save(self.save_dir / "hmm.json")
        self.gmm.save(self.save_dir / "gmm_summary.json")

        # Save labels
        with open(self.save_dir / "regime_labels.json", "w") as f:
            json.dump(self.labels, f, indent=2)

        logger.info("Models saved to %s", self.save_dir)
        return self

    def infer(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Infer regime probability vectors for each date in X.

        Returns
        -------
        pd.DataFrame with columns: state, p_0..p_{K-1}, label
        """
        assert self.hmm is not None and self.hmm._fitted, "Must call fit() first."
        result = self.hmm.infer(X)
        df     = result.to_dataframe()

        # Optional posterior smoothing to avoid regime-probability collapse.
        if self.prob_temperature > 1.0:
            pcols = [c for c in df.columns if c.startswith("p_")]
            if pcols:
                p = df[pcols].to_numpy(dtype=float)
                eps = 1e-12
                logp = np.log(np.clip(p, eps, 1.0))
                tempered = np.exp(logp / self.prob_temperature)
                tempered = tempered / tempered.sum(axis=1, keepdims=True)
                df[pcols] = tempered
                df["state"] = np.argmax(tempered, axis=1).astype(int)

        # Attach human-readable label
        label_map = {lab["state_id"]: lab["label"] for lab in self.labels}
        df["label"] = df["state"].map(label_map)
        # Collapse diagnostics (for plotting/monitoring)
        pcols = [c for c in df.columns if c.startswith("p_")]
        if pcols:
            df["max_prob"] = df[pcols].max(axis=1)
            p = df[pcols].to_numpy(dtype=float)
            eps = 1e-12
            p_clip = np.clip(p, eps, 1.0)
            df["entropy"] = -np.sum(p_clip * np.log(p_clip), axis=1)

        return df

    def regime_keys_by_state_id(self) -> list[str]:
        if not self.state_id_to_key:
            raise RuntimeError("fit() must be called before requesting regime keys")
        return list(self.state_id_to_key)

    def diagnostics(self, X: pd.DataFrame) -> dict:
        """Run and return full diagnostics."""
        assert self.hmm and self.hmm._fitted
        result = self.hmm.infer(X)

        disc_table = discriminating_features_table(result, self._feature_names)
        print_regime_diagnostics(result, self._feature_names, self.labels)

        return {
            "hmm_result"        : result,
            "discriminating"    : disc_table,
            "labels"            : self.labels,
            "transition_matrix" : pd.DataFrame(
                result.transition_matrix,
                index=[f"From_S{k}" for k in range(self.n_states)],
                columns=[f"To_S{k}" for k in range(self.n_states)],
            ),
        }
