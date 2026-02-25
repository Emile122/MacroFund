"""
backtest/engine.py
==================
Walk-forward backtest framework for the ATHENA expectation-regime strategy.

Walk-forward procedure:
  For each step:
    1. Fit RollingZScaler on train window.
    2. Fit RegimeInferenceEngine on scaled train data.
    3. Infer regime probabilities on test (out-of-sample) window.
    4. Run AllocationEngine → weights per date.
    5. Compute portfolio returns with transaction costs.
    6. Advance train window (expanding or rolling).

Reports:
  * CAGR, annualised vol, Sharpe, max drawdown
  * Turnover, average rebalance frequency
  * Regime durations, transition stats
  * Probability entropy, average max probability (regime stability)
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.athena_regime.features.engineering import RollingZScaler, run_qa
from src.athena_regime.regimes.inference import RegimeInferenceEngine, CostModel
from src.athena_regime.allocation.engine import AllocationEngine

logger = logging.getLogger(__name__)


def _evaluate_acceptance(perf: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "max_drawdown_ok": perf.get("max_drawdown", 0.0) > -0.20,
        "state_confidence_not_collapsed": perf.get("avg_max_regime_prob", 1.0) < 0.995,
        "enough_rebalances": perf.get("n_rebalances", 0) >= 5,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _apply_exposure_policy(
    weights: dict[str, float],
    *,
    long_only: bool,
    target_net_exposure: float | None,
    max_weight: float,
    max_leverage: float,
) -> dict[str, float]:
    """Project weights to exposure policy without changing asset universe."""
    w = {k: float(v) for k, v in weights.items()}
    if long_only:
        w = {k: max(v, 0.0) for k, v in w.items()}

    if target_net_exposure is not None:
        net = float(sum(w.values()))
        if abs(net) > 1e-9:
            sf = target_net_exposure / net
            w = {k: v * sf for k, v in w.items()}

    lower = 0.0 if long_only else -max_weight
    upper = max_weight
    w = {k: float(np.clip(v, lower, upper)) for k, v in w.items()}

    gross = float(sum(abs(v) for v in w.values()))
    if gross > max_leverage and gross > 0:
        sf = max_leverage / gross
        w = {k: v * sf for k, v in w.items()}

    return w


# ── Walk-forward configuration ────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest."""
    mode              : str   = "expanding"    # 'expanding' or 'rolling'
    train_periods     : int   = 252            # Initial training period (days)
    step_size         : int   = 63             # Roll step (quarter)
    refit_every       : int   = 63             # Refit model every N steps
    n_states          : int   = 4
    n_hmm_iter        : int   = 200
    n_hmm_restarts    : int   = 3
    prob_temperature  : float = 1.0
    min_state_occupancy: float = 0.02
    allocation_mode   : str   = "soft"
    confidence_threshold: float = 0.50
    rebal_threshold   : float = 0.08
    rebal_metric      : str   = "jsd"
    surprise_threshold: float = 1.5
    max_leverage      : float = 2.0
    max_weight        : float = 0.40
    long_only         : bool = False
    target_net_exposure: float | None = None
    turnover_cap      : float = 0.30
    prop_cost_bps     : float = 5.0
    impact_bps        : float = 2.0
    bid_ask_bps       : float = 1.0
    vol_target        : float | None = None    # None = no vol targeting
    risk_free_rate    : float = 0.04           # Annual
    borrow_spread_bps : float = 0.0


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_performance(
    returns         : pd.Series,
    rebal_log       : pd.DataFrame,
    regime_labels   : pd.Series,
    risk_free_rate  : float = 0.0,
    freq            : int   = 252,
) -> dict[str, Any]:
    ret = returns.dropna()
    n = len(ret)

    total_ret = float((1 + ret).prod() - 1)
    cagr = float((1 + total_ret) ** (freq / n) - 1) if n > 0 else 0.0
    ann_vol = float(ret.std() * np.sqrt(freq))
    daily_rf = risk_free_rate / freq
    sharpe = float((ret.mean() - daily_rf) / ret.std() * np.sqrt(freq)) if ret.std() > 0 else 0.0

    downside = ret[ret < daily_rf]
    downside_std = downside.std()
    sortino = float((ret.mean() - daily_rf) / downside_std * np.sqrt(freq)) if downside_std and downside_std > 0 else 0.0

    cumret = (1 + ret).cumprod()
    rolling_max = cumret.cummax()
    dd = (cumret - rolling_max) / rolling_max
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    rebal_dates = rebal_log[rebal_log["do_rebalance"]].index if "do_rebalance" in rebal_log else pd.DatetimeIndex([])
    n_rebalances = len(rebal_dates)
    avg_rebal_freq = n / n_rebalances if n_rebalances > 0 else np.inf
    avg_turnover = float(rebal_log["distance"].mean()) if "distance" in rebal_log else 0.0
    surprise_count = int(rebal_log["surprise_triggered"].sum()) if "surprise_triggered" in rebal_log else 0
    avg_max_prob = float(rebal_log["max_prob"].mean()) if "max_prob" in rebal_log else 0.0

    # Tail risk diagnostics
    var_95 = float(ret.quantile(0.05)) if n > 0 else 0.0
    cvar_95 = float(ret[ret <= var_95].mean()) if n > 0 and (ret <= var_95).any() else 0.0
    hit_rate = float((ret > 0).mean()) if n > 0 else 0.0
    tail_ratio = float(ret.quantile(0.95) / abs(var_95)) if var_95 < 0 else 0.0

    cond_ret: dict[str, dict] = {}
    if regime_labels is not None and len(regime_labels) > 0:
        labels_aligned = regime_labels.reindex(ret.index).dropna()
        for label in labels_aligned.unique():
            idx = labels_aligned[labels_aligned == label].index
            r_label = ret.reindex(idx).dropna()
            if len(r_label) > 0:
                cond_ret[str(label)] = {
                    "mean_daily_ret": round(float(r_label.mean()), 6),
                    "ann_ret": round(float(r_label.mean() * freq), 4),
                    "vol": round(float(r_label.std() * np.sqrt(freq)), 4),
                    "count": int(len(r_label)),
                }

    regime_durations: dict[str, float] = {}
    if regime_labels is not None and len(regime_labels) > 0:
        rl = regime_labels.dropna()
        changes = (rl != rl.shift()).fillna(True)
        regime_groups = rl.groupby(changes.cumsum())
        durations_list: dict[str, list[int]] = {}
        for _, grp in regime_groups:
            label = str(grp.iloc[0])
            durations_list.setdefault(label, []).append(len(grp))
        regime_durations = {
            k: round(float(np.mean(v)), 1)
            for k, v in durations_list.items()
        }

    return {
        "total_return"   : round(total_ret, 4),
        "cagr"           : round(cagr, 4),
        "ann_vol"        : round(ann_vol, 4),
        "sharpe"         : round(sharpe, 4),
        "sortino"        : round(sortino, 4),
        "max_drawdown"   : round(max_dd, 4),
        "calmar"         : round(calmar, 4),
        "n_rebalances"   : n_rebalances,
        "avg_rebal_freq_days" : round(avg_rebal_freq, 1),
        "avg_jsd_distance"    : round(avg_turnover, 5),
        "surprise_rebalances" : surprise_count,
        "avg_max_regime_prob" : round(avg_max_prob, 4),
        "hit_rate"            : round(hit_rate, 4),
        "var_95"              : round(var_95, 5),
        "cvar_95"             : round(cvar_95, 5),
        "tail_ratio"          : round(tail_ratio, 4),
        "n_trading_days" : n,
        "conditional_returns" : cond_ret,
        "regime_durations_days": regime_durations,
    }


# ── Walk-forward backtest ─────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Full backtest output."""
    daily_returns  : pd.Series
    nav            : pd.Series
    weights_history: pd.DataFrame
    regime_history : pd.DataFrame
    rebal_log      : pd.DataFrame
    performance    : dict[str, Any]
    step_logs      : list[dict] = field(default_factory=list)


class WalkForwardBacktest:
    """
    Walk-forward backtest engine.

    Fits the regime model and allocation engine on a rolling/expanding
    training window, then generates out-of-sample portfolio returns.

    Parameters
    ----------
    X         : Full feature matrix (all dates).
    prices    : Daily price levels (for return computation).
    config    : WalkForwardConfig.
    save_dir  : Directory to save results.
    surprise  : Optional macro surprise series (date-indexed).
    """

    def __init__(
        self,
        X         : pd.DataFrame,
        prices    : pd.DataFrame,
        config    : WalkForwardConfig,
        save_dir  : str = "outputs",
        surprise  : pd.Series | None = None,
    ) -> None:
        self.X       = X
        self.prices  = prices
        self.config  = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.surprise = surprise

        self.cost_model = CostModel(
            prop_cost_bps  = config.prop_cost_bps,
            impact_bps     = config.impact_bps,
            bid_ask_bps    = config.bid_ask_bps,
        )

    def run(self) -> BacktestResult:
        """Execute the full walk-forward backtest."""
        cfg    = self.config
        X      = self.X
        prices = self.prices.reindex(X.index)

        # Align prices to X index
        price_cols = [c.replace("_price", "") for c in prices.columns]
        prices.columns = price_cols

        n_total = len(X)
        if n_total < cfg.train_periods + cfg.step_size:
            raise ValueError(
                f"Not enough data: need {cfg.train_periods + cfg.step_size} "
                f"rows, have {n_total}"
            )

        # Storage
        all_returns  : list[tuple[pd.Timestamp, float]] = []
        all_weights  : list[dict] = []
        all_regimes  : list[dict] = []
        step_logs    : list[dict] = []

        # Allocation engine (persistent across steps — accumulates state)
        alloc = AllocationEngine(
            mode                 = cfg.allocation_mode,
            confidence_threshold = cfg.confidence_threshold,
            max_leverage         = cfg.max_leverage,
            max_weight           = cfg.max_weight,
            turnover_cap         = cfg.turnover_cap,
            rebal_threshold      = cfg.rebal_threshold,
            rebal_metric         = cfg.rebal_metric,
            surprise_threshold   = cfg.surprise_threshold,
        )

        # Walk-forward loop
        train_start = 0
        test_start  = cfg.train_periods

        step = 0
        scaler: RollingZScaler | None = None
        engine: RegimeInferenceEngine | None = None
        regime_keys: list[str] = []
        held_weights: dict[str, float] = alloc.current_weights()
        last_probs: np.ndarray | None = None
        refit_steps = max(1, cfg.refit_every // cfg.step_size)

        while test_start < n_total:
            test_end = min(test_start + cfg.step_size, n_total)

            # ── Refit on train window ─────────────────────────────────────
            if step == 0 or (step % refit_steps == 0):
                logger.info(
                    "Step %d | train=[%d:%d] test=[%d:%d]",
                    step, train_start, test_start, test_start, test_end,
                )

                X_train = X.iloc[train_start:test_start]
                X_train_c, _ = run_qa(X_train, logger=logger)
                X_train_c = X_train_c.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                scaler = RollingZScaler()
                X_train_z = scaler.fit_transform(X_train_c)

                engine = RegimeInferenceEngine(
                    n_states   = cfg.n_states,
                    n_hmm_iter = cfg.n_hmm_iter,
                    n_restarts = cfg.n_hmm_restarts,
                    prob_temperature=cfg.prob_temperature,
                    min_state_occupancy=cfg.min_state_occupancy,
                    save_dir   = str(self.save_dir / f"models/step_{step:03d}"),
                )
                engine.fit(X_train_z)
                regime_keys = (
                    engine.regime_keys_by_state_id()
                    if hasattr(engine, "regime_keys_by_state_id")
                    else [lab["label"] for lab in engine.labels]
                )
                last_train_day = X.index[test_start - 1]
                X_last = X.loc[[last_train_day]]
                X_last_c = X_last[[c for c in X_last.columns if c in scaler._columns]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                X_last_z = scaler.transform(X_last_c)
                last_row = engine.infer(X_last_z).iloc[0]
                last_probs = np.array([last_row[f"p_{k}"] for k in range(cfg.n_states)], dtype=float)

                step_logs.append({
                    "step"       : step,
                    "train_start": str(X.index[train_start].date()),
                    "train_end"  : str(X.index[test_start - 1].date()),
                    "test_start" : str(X.index[test_start].date()),
                    "test_end"   : str(X.index[test_end - 1].date()),
                })

            # ── Infer regimes on test window ──────────────────────────────
            assert scaler is not None and engine is not None and last_probs is not None
            X_test     = X.iloc[test_start:test_end]
            X_test_c   = X_test[[c for c in X_test.columns if c in scaler._columns]]
            X_test_c   = X_test_c.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X_test_z   = scaler.transform(X_test_c)
            regime_df  = engine.infer(X_test_z)
            daily_rf = cfg.risk_free_rate / 252.0
            borrow_spread = cfg.borrow_spread_bps / 10_000 / 252.0

            # ── Generate weights and returns ──────────────────────────────
            for i, date in enumerate(X_test.index):
                row_probs = regime_df.loc[date, [f"p_{k}" for k in range(cfg.n_states)]].values
                probs_for_trade = last_probs

                # Macro surprise for this date
                surprise_score = 0.0
                if self.surprise is not None and date in self.surprise.index:
                    surprise_score = float(self.surprise.loc[date])

                prev_weights_before_step = alloc.current_weights()

                # Get portfolio weights
                proposed_weights = alloc.step(
                    date           = date,
                    probs          = probs_for_trade,
                    regime_keys    = regime_keys,
                    surprise_score = surprise_score,
                    force_rebalance= (i == 0),   # always rebalance at start of test window
                )
                weights = _apply_exposure_policy(
                    proposed_weights,
                    long_only=cfg.long_only,
                    target_net_exposure=cfg.target_net_exposure,
                    max_weight=cfg.max_weight,
                    max_leverage=cfg.max_leverage,
                )

                # Compute portfolio return for this date
                curr_prices = prices.loc[date]
                prev_row    = prices.iloc[max(test_start + i - 1, 0)]
                port_return = 0.0
                tc_cost = 0.0

                rebal_decisions = alloc._rebal_log
                if rebal_decisions and rebal_decisions[-1].date == date and rebal_decisions[-1].do_rebalance:
                    held_weights = dict(weights)

                for asset, w in held_weights.items():
                    if asset in curr_prices.index and asset in prev_row.index:
                        if prev_row[asset] > 0:
                            asset_ret = float((curr_prices[asset] - prev_row[asset]) / prev_row[asset])
                            port_return += w * asset_ret

                # Financing: no positive carry on underinvestment.
                # If gross > 1, apply borrowing cost on financed notional.
                gross_exposure = float(sum(abs(v) for v in held_weights.values()))
                borrow_notional = max(gross_exposure - 1.0, 0.0)
                port_return -= borrow_notional * (daily_rf + borrow_spread)

                # Compute TC on rebalance days
                if rebal_decisions and rebal_decisions[-1].date == date and rebal_decisions[-1].do_rebalance:
                    tc_cost = self.cost_model.compute_cost(
                        w_prev = prev_weights_before_step,
                        w_new  = held_weights,
                    )

                net_return = port_return - tc_cost

                all_returns.append((date, net_return))
                all_weights.append({"date": date, **held_weights})
                all_regimes.append({
                    "date"  : date,
                    "state" : int(regime_df.loc[date, "state"]),
                    "label" : regime_df.loc[date, "label"],
                    **{f"p_{k}": float(row_probs[k]) for k in range(cfg.n_states)},
                    "max_prob": float(np.max(row_probs)),
                    "entropy": float(regime_df.loc[date, "entropy"]) if "entropy" in regime_df.columns else float("nan"),
                })
                last_probs = np.array(row_probs, dtype=float)

            # Advance window
            if cfg.mode == "rolling":
                train_start += cfg.step_size
            test_start = test_end
            step += 1

        # Collect results
        ret_df = pd.DataFrame(all_returns, columns=["date", "return"]).set_index("date")
        ret_df.index = pd.DatetimeIndex(ret_df.index)
        ret_series = ret_df.groupby(level=0)["return"].sum()
        nav = (1 + ret_series).cumprod()
        bench_ret = prices.pct_change().reindex(ret_series.index).mean(axis=1).fillna(0.0)
        bench_nav = (1 + bench_ret).cumprod()

        weights_df = pd.DataFrame(all_weights).set_index("date")
        regime_df  = pd.DataFrame(all_regimes).set_index("date")
        rebal_df   = alloc.rebal_log_df()

        # Performance attribution
        perf = compute_performance(
            returns        = ret_series,
            rebal_log      = rebal_df,
            regime_labels  = regime_df.get("label"),
            risk_free_rate = 0.0,
        )
        n = len(ret_series.dropna())
        bench_total = float(bench_nav.iloc[-1] - 1.0) if len(bench_nav) else 0.0
        bench_cagr = float((1 + bench_total) ** (252 / n) - 1) if n > 0 else 0.0
        active = (ret_series - bench_ret).dropna()
        ir = float(active.mean() / active.std() * np.sqrt(252)) if len(active) and active.std() > 0 else 0.0
        perf["benchmark_total_return"] = round(bench_total, 4)
        perf["benchmark_cagr"] = round(bench_cagr, 4)
        perf["alpha_cagr"] = round(perf["cagr"] - bench_cagr, 4)
        perf["information_ratio"] = round(ir, 4)
        perf["acceptance_gates"] = _evaluate_acceptance(perf)

        result = BacktestResult(
            daily_returns  = ret_series,
            nav            = nav,
            weights_history= weights_df,
            regime_history = regime_df,
            rebal_log      = rebal_df,
            performance    = perf,
            step_logs      = step_logs,
        )

        # Save outputs
        self._save_results(result)
        return result

    def _save_results(self, result: BacktestResult) -> None:
        result.daily_returns.to_csv(self.save_dir / "daily_returns.csv")
        result.nav.to_csv(self.save_dir / "nav.csv")
        result.regime_history.to_csv(self.save_dir / "regime_history.csv")
        result.rebal_log.to_csv(self.save_dir / "rebalance_log.csv")
        result.weights_history.to_csv(self.save_dir / "weights_history.csv")

        # Performance summary
        perf = result.performance.copy()
        cond = perf.pop("conditional_returns", {})
        dur  = perf.pop("regime_durations_days", {})
        gates = perf.pop("acceptance_gates", {})
        perf_df = pd.Series(perf).to_frame("value")
        perf_df.to_csv(self.save_dir / "performance_summary.csv")

        if cond:
            pd.DataFrame(cond).T.to_csv(self.save_dir / "conditional_returns.csv")
        if gates:
            with open(self.save_dir / "acceptance_gates.json", "w", encoding="utf-8") as f:
                json.dump(gates, f, indent=2)

        logger.info("Results saved to %s", self.save_dir)


def print_performance_report(perf: dict, result: BacktestResult) -> None:
    """Print formatted performance report."""
    print("\n" + "═" * 60)
    print("  BACKTEST PERFORMANCE REPORT")
    print("═" * 60)

    metrics = [
        ("Total Return",     f"{perf['total_return']*100:.2f}%"),
        ("CAGR",             f"{perf['cagr']*100:.2f}%"),
        ("Annualised Vol",   f"{perf['ann_vol']*100:.2f}%"),
        ("Sharpe Ratio",     f"{perf['sharpe']:.3f}"),
        ("Sortino Ratio",    f"{perf['sortino']:.3f}"),
        ("Max Drawdown",     f"{perf['max_drawdown']*100:.2f}%"),
        ("Calmar Ratio",     f"{perf['calmar']:.3f}"),
        ("",                 ""),
        ("Rebalances",       f"{perf['n_rebalances']}"),
        ("Avg Rebal Freq",   f"{perf['avg_rebal_freq_days']:.1f} days"),
        ("Surprise Rebal",   f"{perf['surprise_rebalances']}"),
        ("Avg JSD Distance", f"{perf['avg_jsd_distance']:.5f}"),
        ("Avg Max Prob",     f"{perf['avg_max_regime_prob']:.4f}"),
        ("",                 ""),
        ("Trading Days",     f"{perf['n_trading_days']}"),
    ]
    for label, val in metrics:
        if label:
            print(f"  {label:<30} {val:>12}")

    if perf.get("conditional_returns"):
        print("\n  Conditional Returns by Regime:")
        print(f"  {'Regime':<25} {'Ann Ret':>10} {'Vol':>8} {'Days':>6}")
        print("  " + "-" * 55)
        for label, stats in perf["conditional_returns"].items():
            print(f"  {label:<25} {stats['ann_ret']*100:>9.2f}% {stats['vol']*100:>7.2f}% {stats['count']:>6}")

    if perf.get("regime_durations_days"):
        print("\n  Average Regime Duration (days):")
        for label, dur in perf["regime_durations_days"].items():
            print(f"    {label:<25} {dur:.1f} days")

    print("═" * 60 + "\n")
