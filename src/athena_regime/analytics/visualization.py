from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _try_import_matplotlib():
    try:
        import matplotlib.dates as mdates
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        return plt, mdates, mcolors
    except Exception:
        return None, None, None


def _coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df[~df.index.isna()]


def _first_numeric_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().any():
            df[col] = coerced
            return col
    return None


def _load_table(artifacts_dir: Path, stem: str, index_col: int = 0) -> pd.DataFrame:
    csv_path = artifacts_dir / f"{stem}.csv"
    xlsx_path = artifacts_dir / f"{stem}.xlsx"
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=index_col)
    if xlsx_path.exists():
        try:
            return pd.read_excel(xlsx_path, index_col=index_col)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _run_ts_from_id(run_id: str) -> datetime | None:
    prefix = run_id.split("_", 1)[0]
    try:
        return datetime.strptime(prefix, "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def _latest_run_dir(runs_dir: Path) -> Path | None:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None

    def _key(p: Path):
        parsed = _run_ts_from_id(p.name)
        return parsed or datetime.min

    return sorted(candidates, key=_key)[-1]


def _prep_run_data(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    art = run_path / "artifacts"
    payload: dict[str, Any] = {"run_path": run_path, "artifacts_dir": art}
    if not art.exists():
        return payload

    returns_df = _load_table(art, "daily_returns")
    if not returns_df.empty:
        returns_df = _coerce_datetime_index(returns_df)
        col = _first_numeric_column(returns_df)
        if col:
            payload["returns"] = returns_df[col].astype(float).sort_index()

    nav_df = _load_table(art, "nav")
    if not nav_df.empty:
        nav_df = _coerce_datetime_index(nav_df)
        col = _first_numeric_column(nav_df)
        if col:
            payload["nav"] = nav_df[col].astype(float).sort_index()

    regime_df = _load_table(art, "regime_history")
    if not regime_df.empty:
        payload["regime"] = _coerce_datetime_index(regime_df).sort_index()

    weights_df = _load_table(art, "weights_history")
    if not weights_df.empty:
        payload["weights"] = _coerce_datetime_index(weights_df).sort_index()

    rebal_df = _load_table(art, "rebalance_log")
    if not rebal_df.empty:
        payload["rebal"] = _coerce_datetime_index(rebal_df).sort_index()

    return payload


def _save_fig(fig, out_path: Path, dpi: int = 150) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    return out_path


def _transform_signal(returns: pd.Series, mode: str) -> tuple[pd.Series, str]:
    r = returns.astype(float).copy()
    if mode == "log":
        # log(1+r) keeps compounding structure and magnifies subtle variation.
        out = np.log1p(r.clip(lower=-0.999999))
        return out, "Log Return"
    if mode == "zscore":
        mu = float(r.mean())
        sd = float(r.std())
        out = (r - mu) / sd if sd > 0 else r * 0.0
        return out, "Return Z-Score"
    return r, "Daily Return"


def _plot_returns_nav_events(data: dict[str, Any], signal_mode: str = "log") -> Path | None:
    plt, mdates, _ = _try_import_matplotlib()
    if plt is None:
        return None

    run_path: Path = data["run_path"]
    art: Path = data["artifacts_dir"]
    returns: pd.Series | None = data.get("returns")
    nav: pd.Series | None = data.get("nav")
    regime_df: pd.DataFrame = data.get("regime", pd.DataFrame())
    rebal_df: pd.DataFrame = data.get("rebal", pd.DataFrame())
    if returns is None or returns.empty:
        return None

    cumulative_returns = (1.0 + returns).cumprod() - 1.0
    signal, signal_label = _transform_signal(returns, signal_mode)
    if nav is None or nav.empty:
        nav = cumulative_returns + 1.0
    else:
        nav = nav.reindex(cumulative_returns.index).ffill()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    ax_signal, ax_nav = axes
    ax_nav2 = ax_nav.twinx()

    signal = signal.sort_index()
    roll_21 = signal.rolling(21, min_periods=5).mean()
    roll_63 = signal.rolling(63, min_periods=10).mean()

    # Use bars for day-to-day moves and smoothed lines for the actual "signal" shape.
    ax_signal.bar(signal.index, signal.values, width=1.0, color="#9aa0aa", alpha=0.25, label=f"{signal_label} (Daily)")
    ax_signal.plot(roll_21.index, roll_21.values, color="#6f42c1", lw=1.8, label=f"{signal_label} 21D Mean")
    ax_signal.plot(roll_63.index, roll_63.values, color="#1f77b4", lw=1.5, alpha=0.9, label=f"{signal_label} 63D Mean")

    ax_signal.axhline(0.0, color="#333333", lw=0.8, alpha=0.5)
    ql, qh = signal.quantile(0.01), signal.quantile(0.99)
    if np.isfinite(ql) and np.isfinite(qh) and qh > ql:
        pad = (qh - ql) * 0.20
        ax_signal.set_ylim(ql - pad, qh + pad)
    ax_signal.set_ylabel(signal_label)
    ax_signal.set_title("Return Signal (Daily + Rolling Means)")
    ax_signal.grid(alpha=0.2)
    ax_signal.legend(loc="upper left")

    ax_nav.plot(cumulative_returns.index, cumulative_returns.values, color="#1f77b4", lw=1.8, label="Cumulative Return")
    ax_nav2.plot(nav.index, nav.values, color="#ff7f0e", lw=1.4, alpha=0.95, label="NAV")
    ax_nav.axhline(0.0, color="#333333", lw=0.8, alpha=0.5)

    if not rebal_df.empty and "do_rebalance" in rebal_df.columns:
        trade_dates = pd.DatetimeIndex(rebal_df[rebal_df["do_rebalance"] == True].index.unique())
        signal_y = signal.reindex(trade_dates).dropna()
        nav_y = cumulative_returns.reindex(trade_dates).dropna()
        if len(signal_y) > 0:
            ax_signal.scatter(signal_y.index, signal_y.values, s=18, marker="o", color="#d62728", label="Trade/Rebalance")
        if len(nav_y) > 0:
            ax_nav.scatter(nav_y.index, nav_y.values, s=18, marker="o", color="#d62728", label="Trade/Rebalance")

    if not regime_df.empty:
        state_col = "state" if "state" in regime_df.columns else ("label" if "label" in regime_df.columns else None)
        if state_col is not None:
            regime_series = regime_df[state_col]
            changes = regime_series.ne(regime_series.shift(1))
            change_dates = pd.DatetimeIndex(regime_series.index[changes.fillna(False)])
            if len(change_dates) > 0:
                for dt in change_dates:
                    ax_signal.axvline(dt, color="#2ca02c", lw=0.7, alpha=0.18)
                    ax_nav.axvline(dt, color="#2ca02c", lw=0.7, alpha=0.18)
                signal_rc = signal.reindex(change_dates).dropna()
                nav_rc = cumulative_returns.reindex(change_dates).dropna()
                if len(signal_rc) > 0:
                    ax_signal.scatter(signal_rc.index, signal_rc.values, s=18, marker="x", color="#2ca02c", label="Regime Change")
                if len(nav_rc) > 0:
                    ax_nav.scatter(nav_rc.index, nav_rc.values, s=18, marker="x", color="#2ca02c", label="Regime Change")

    ax_nav.set_title(f"Return Evolution + NAV + Events | {run_path.name}")
    ax_nav.set_ylabel("Cumulative Return")
    ax_nav2.set_ylabel("NAV")
    ax_nav.grid(alpha=0.2)
    h1, l1 = ax_nav.get_legend_handles_labels()
    h2, l2 = ax_nav2.get_legend_handles_labels()
    ax_nav.legend(h1 + h2, l1 + l2, loc="upper left")
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax_nav.xaxis.set_major_locator(locator)
    ax_nav.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    out_path = _save_fig(fig, art / "returns_nav_events.png", dpi=165)
    plt.close(fig)
    return out_path


def _plot_nav_only(data: dict[str, Any]) -> Path | None:
    plt, mdates, _ = _try_import_matplotlib()
    if plt is None:
        return None
    art: Path = data["artifacts_dir"]
    returns: pd.Series | None = data.get("returns")
    nav: pd.Series | None = data.get("nav")
    if returns is None or returns.empty:
        return None
    if nav is None or nav.empty:
        nav = (1.0 + returns).cumprod()

    fig, ax = plt.subplots(1, 1, figsize=(14, 4.8), constrained_layout=True)
    ax.plot(nav.index, nav.values, color="#1f77b4", lw=1.8, label="NAV")
    ax.set_title("NAV Over Time")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    out_path = _save_fig(fig, art / "nav_timeseries.png", dpi=165)
    plt.close(fig)
    return out_path


def _plot_log_returns_only(data: dict[str, Any]) -> Path | None:
    plt, mdates, _ = _try_import_matplotlib()
    if plt is None:
        return None
    art: Path = data["artifacts_dir"]
    returns: pd.Series | None = data.get("returns")
    if returns is None or returns.empty:
        return None
    log_ret = np.log1p(returns.clip(lower=-0.999999)).sort_index()
    roll_21 = log_ret.rolling(21, min_periods=5).mean()
    roll_63 = log_ret.rolling(63, min_periods=10).mean()

    fig, ax = plt.subplots(1, 1, figsize=(14, 4.8), constrained_layout=True)
    ax.bar(log_ret.index, log_ret.values, width=1.0, color="#9aa0aa", alpha=0.22, label="Log Return (Daily)")
    ax.plot(roll_21.index, roll_21.values, color="#6f42c1", lw=1.8, label="Log Return 21D Mean")
    ax.plot(roll_63.index, roll_63.values, color="#1f77b4", lw=1.5, label="Log Return 63D Mean")
    ax.axhline(0.0, color="#333333", lw=0.8, alpha=0.6)
    ax.set_title("Log Returns Signal")
    ax.set_ylabel("log(1+r)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    out_path = _save_fig(fig, art / "log_returns_signal.png", dpi=165)
    plt.close(fig)
    return out_path


def _plot_underwater_and_rolling(data: dict[str, Any]) -> Path | None:
    plt, mdates, _ = _try_import_matplotlib()
    if plt is None:
        return None

    art: Path = data["artifacts_dir"]
    returns: pd.Series | None = data.get("returns")
    if returns is None or returns.empty:
        return None

    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    rolling_63_vol = returns.rolling(63, min_periods=20).std() * np.sqrt(252)
    rolling_63_sharpe = (
        returns.rolling(63, min_periods=20).mean()
        / returns.rolling(63, min_periods=20).std().replace(0, np.nan)
    ) * np.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    ax_top, ax_bottom = axes

    ax_top.fill_between(drawdown.index, drawdown.values, 0.0, color="#d62728", alpha=0.32)
    ax_top.plot(drawdown.index, drawdown.values, color="#b71c1c", lw=1.0)
    ax_top.set_ylabel("Drawdown")
    ax_top.set_title("Underwater Curve")
    ax_top.grid(alpha=0.2)

    ax_bottom.plot(rolling_63_vol.index, rolling_63_vol.values, color="#1f77b4", lw=1.5, label="63D Ann. Vol")
    ax_bottom.plot(rolling_63_sharpe.index, rolling_63_sharpe.values, color="#2ca02c", lw=1.2, label="63D Ann. Sharpe")
    ax_bottom.set_ylabel("Rolling Metrics")
    ax_bottom.set_title("Rolling Risk and Risk-Adjusted Return")
    ax_bottom.grid(alpha=0.2)
    ax_bottom.legend(loc="upper left")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax_bottom.xaxis.set_major_locator(locator)
    ax_bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    out_path = _save_fig(fig, art / "risk_diagnostics.png", dpi=155)
    plt.close(fig)
    return out_path


def _plot_regime_diagnostics(data: dict[str, Any]) -> Path | None:
    plt, mdates, _ = _try_import_matplotlib()
    if plt is None:
        return None
    art: Path = data["artifacts_dir"]
    regime_df: pd.DataFrame = data.get("regime", pd.DataFrame())
    if regime_df.empty:
        return None

    prob_cols = [c for c in regime_df.columns if str(c).startswith("p_")]
    if not prob_cols and "state" not in regime_df.columns:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    ax_prob, ax_state = axes

    if prob_cols:
        prob_df = regime_df[prob_cols].fillna(0.0).clip(0.0, 1.0)
        ax_prob.stackplot(prob_df.index, [prob_df[c] for c in prob_cols], labels=prob_cols, alpha=0.85)
        ax_prob.set_ylim(0, 1.0)
        ax_prob.legend(loc="upper left", ncol=min(4, len(prob_cols)))
    ax_prob.set_ylabel("Probability")
    ax_prob.set_title("Regime Probabilities")
    ax_prob.grid(alpha=0.2)

    if "state" in regime_df.columns:
        states = regime_df["state"].astype(int)
        ax_state.step(states.index, states.values, where="post", color="#2ca02c", lw=1.3)
        ax_state.set_yticks(sorted(states.unique()))
    ax_state.set_ylabel("State")
    ax_state.set_title("Inferred State Path")
    ax_state.grid(alpha=0.2)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax_state.xaxis.set_major_locator(locator)
    ax_state.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    out_path = _save_fig(fig, art / "regime_diagnostics.png", dpi=155)
    plt.close(fig)
    return out_path


def _plot_regime_transition_matrix(data: dict[str, Any]) -> Path | None:
    plt, _, mcolors = _try_import_matplotlib()
    if plt is None:
        return None
    art: Path = data["artifacts_dir"]
    regime_df: pd.DataFrame = data.get("regime", pd.DataFrame())
    if regime_df.empty or "state" not in regime_df.columns:
        return None

    states = regime_df["state"].astype(int)
    unique_states = sorted(states.unique())
    if len(unique_states) <= 1:
        return None
    idx_map = {s: i for i, s in enumerate(unique_states)}
    mat = np.zeros((len(unique_states), len(unique_states)), dtype=float)
    prev = states.shift(1)
    for p, c in zip(prev.iloc[1:], states.iloc[1:]):
        if pd.isna(p) or pd.isna(c):
            continue
        mat[idx_map[int(p)], idx_map[int(c)]] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(mat, cmap="Blues", norm=mcolors.Normalize(vmin=0, vmax=max(0.01, float(np.nanmax(mat)))))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="#111111", fontsize=9)
    ax.set_xticks(range(len(unique_states)))
    ax.set_yticks(range(len(unique_states)))
    ax.set_xticklabels([str(s) for s in unique_states])
    ax.set_yticklabels([str(s) for s in unique_states])
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")
    ax.set_title("Regime Transition Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = _save_fig(fig, art / "regime_transition_matrix.png", dpi=165)
    plt.close(fig)
    return out_path


def _plot_weight_diagnostics(data: dict[str, Any]) -> Path | None:
    plt, mdates, mcolors = _try_import_matplotlib()
    if plt is None:
        return None
    art: Path = data["artifacts_dir"]
    weights_df: pd.DataFrame = data.get("weights", pd.DataFrame())
    if weights_df.empty:
        return None

    weight_cols = [c for c in weights_df.columns if c != "date"]
    if not weight_cols:
        return None
    w = weights_df[weight_cols].astype(float).sort_index()
    gross = w.abs().sum(axis=1)
    net = w.sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    top_assets = list(w.abs().mean().sort_values(ascending=False).head(min(10, len(weight_cols))).index)
    w_top = w[top_assets]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    ax_top, ax_mid, ax_bottom = axes

    ax_top.plot(gross.index, gross.values, color="#1f77b4", lw=1.5, label="Gross Exposure")
    ax_top.plot(net.index, net.values, color="#ff7f0e", lw=1.2, label="Net Exposure")
    ax_top.set_title("Exposure Evolution")
    ax_top.set_ylabel("Exposure")
    ax_top.grid(alpha=0.2)
    ax_top.legend(loc="upper left")

    vmin = float(w_top.min().min())
    vmax = float(w_top.max().max())
    if vmin < 0.0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)
    im = ax_mid.imshow(
        w_top.T.values,
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
    )
    ax_mid.set_yticks(range(len(top_assets)))
    ax_mid.set_yticklabels(top_assets)
    ax_mid.set_title("Top-Asset Weight Heatmap")
    fig.colorbar(im, ax=ax_mid, fraction=0.025, pad=0.02)

    ax_bottom.plot(turnover.index, turnover.values, color="#9467bd", lw=1.2)
    ax_bottom.set_title("Portfolio Turnover")
    ax_bottom.set_ylabel("L1 Turnover")
    ax_bottom.grid(alpha=0.2)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax_bottom.xaxis.set_major_locator(locator)
    ax_bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    out_path = _save_fig(fig, art / "weights_diagnostics.png", dpi=155)
    plt.close(fig)
    return out_path


def _plot_rebalance_diagnostics(data: dict[str, Any]) -> Path | None:
    plt, mdates, _ = _try_import_matplotlib()
    if plt is None:
        return None
    art: Path = data["artifacts_dir"]
    rebal_df: pd.DataFrame = data.get("rebal", pd.DataFrame())
    if rebal_df.empty:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    ax_top, ax_bottom = axes

    if "distance" in rebal_df.columns:
        dist = pd.to_numeric(rebal_df["distance"], errors="coerce").fillna(0.0)
        ax_top.plot(dist.index, dist.values, color="#9467bd", lw=1.3, label="Rebalance Distance")
    if "do_rebalance" in rebal_df.columns:
        flags = rebal_df["do_rebalance"].astype(bool)
        marker_dates = rebal_df.index[flags]
        marker_y = (
            pd.to_numeric(rebal_df["distance"], errors="coerce").reindex(marker_dates).fillna(0.0)
            if "distance" in rebal_df.columns else pd.Series(0.0, index=marker_dates)
        )
        if len(marker_dates) > 0:
            ax_top.scatter(marker_dates, marker_y.values, color="#d62728", s=18, label="Rebalance")
    ax_top.set_title("Rebalance Distance and Trigger Points")
    ax_top.set_ylabel("Distance")
    ax_top.grid(alpha=0.2)
    ax_top.legend(loc="upper left")

    if "max_prob" in rebal_df.columns:
        mp = pd.to_numeric(rebal_df["max_prob"], errors="coerce")
        ax_bottom.plot(mp.index, mp.values, color="#2ca02c", lw=1.2, label="Max Regime Probability")
    if "surprise_triggered" in rebal_df.columns:
        sp = rebal_df["surprise_triggered"].astype(bool)
        if sp.any():
            sp_dates = rebal_df.index[sp]
            ax_bottom.scatter(sp_dates, [1.0] * len(sp_dates), marker="x", s=30, color="#ff7f0e", label="Surprise Triggered")
    ax_bottom.set_title("Decision Confidence and Surprise Triggers")
    ax_bottom.set_ylabel("Confidence")
    ax_bottom.grid(alpha=0.2)
    ax_bottom.legend(loc="upper left")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax_bottom.xaxis.set_major_locator(locator)
    ax_bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    out_path = _save_fig(fig, art / "rebalance_diagnostics.png", dpi=155)
    plt.close(fig)
    return out_path


def _plot_return_distribution_by_regime(data: dict[str, Any]) -> Path | None:
    plt, _, _ = _try_import_matplotlib()
    if plt is None:
        return None
    art: Path = data["artifacts_dir"]
    returns: pd.Series | None = data.get("returns")
    regime_df: pd.DataFrame = data.get("regime", pd.DataFrame())
    if returns is None or returns.empty or regime_df.empty:
        return None
    label_col = "label" if "label" in regime_df.columns else ("state" if "state" in regime_df.columns else None)
    if label_col is None:
        return None

    aligned = pd.DataFrame({"return": returns}).join(regime_df[[label_col]], how="left").dropna()
    if aligned.empty:
        return None
    groups = [(str(k), v["return"].values) for k, v in aligned.groupby(label_col)]
    if len(groups) == 0:
        return None

    labels = [g[0] for g in groups]
    values = [g[1] for g in groups]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax_box, ax_hist = axes

    ax_box.boxplot(values, tick_labels=labels, showfliers=False)
    ax_box.axhline(0.0, color="#333333", lw=0.8, alpha=0.5)
    ax_box.set_title("Return Distribution by Regime")
    ax_box.set_ylabel("Daily Return")
    ax_box.tick_params(axis="x", rotation=20)
    ax_box.grid(alpha=0.2, axis="y")

    ax_hist.hist(returns.values, bins=60, color="#1f77b4", alpha=0.8)
    ax_hist.axvline(float(returns.mean()), color="#d62728", lw=1.3, label="Mean")
    ax_hist.axvline(float(returns.quantile(0.05)), color="#ff7f0e", lw=1.0, linestyle="--", label="5%")
    ax_hist.axvline(float(returns.quantile(0.95)), color="#2ca02c", lw=1.0, linestyle="--", label="95%")
    ax_hist.set_title("Overall Daily Return Distribution")
    ax_hist.set_xlabel("Daily Return")
    ax_hist.grid(alpha=0.2)
    ax_hist.legend(loc="upper left")

    out_path = _save_fig(fig, art / "return_distribution.png", dpi=165)
    plt.close(fig)
    return out_path


def generate_visual_report(
    runs_dir: str | Path,
    run_id: str | None = None,
    signal_mode: str = "log",
) -> dict[str, str]:
    runs_root = Path(runs_dir)
    run_dir = (runs_root / run_id) if run_id else _latest_run_dir(runs_root)
    outputs: dict[str, str] = {}

    plt, _, _ = _try_import_matplotlib()
    if plt is None:
        outputs["plotting_backend"] = "missing: install matplotlib to generate PNG charts"
        return outputs
    if run_dir is None or not run_dir.exists():
        outputs["run_id"] = f"not found: {run_id}"
        return outputs

    allowed_modes = {"raw", "log", "zscore"}
    if signal_mode not in allowed_modes:
        outputs["signal_mode"] = f"invalid: {signal_mode} (allowed: {sorted(allowed_modes)})"
        return outputs

    data = _prep_run_data(run_dir)
    plotters: list[tuple[str, Any]] = [
        ("nav_timeseries", _plot_nav_only),
        ("log_returns_signal", _plot_log_returns_only),
        ("returns_nav_events", lambda d: _plot_returns_nav_events(d, signal_mode=signal_mode)),
        ("risk_diagnostics", _plot_underwater_and_rolling),
        ("regime_diagnostics", _plot_regime_diagnostics),
        ("regime_transition_matrix", _plot_regime_transition_matrix),
        ("weights_diagnostics", _plot_weight_diagnostics),
        ("rebalance_diagnostics", _plot_rebalance_diagnostics),
        ("return_distribution", _plot_return_distribution_by_regime),
    ]
    for name, fn in plotters:
        out = fn(data)
        if out:
            outputs[name] = str(out)

    return outputs
