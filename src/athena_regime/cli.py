from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from typing import Any

import pandas as pd

from src.athena_regime.config.loader import load_config
from src.athena_regime.data.ingestion import DEFAULT_UPDATE_DATASETS, prune_dataset, update_dataset
from src.athena_regime.data.lake import DATASET_SPECS, DataLake
from src.athena_regime.data.pipeline import build_feature_matrix, build_prices_matrix
from src.athena_regime.data.providers import MockProvider, provider_from_env
from src.athena_regime.run_context.context import RunContext


def _console():
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None


def _print_banner(console, mode: str, run_id: str) -> None:
    try:
        from rich.panel import Panel
        from rich.text import Text

        t = Text()
        t.append("ATHENA", style="bold cyan")
        t.append(f"  mode={mode}  run_id={run_id}", style="dim")
        console.print(Panel(t, border_style="cyan", padding=(0, 2)))
    except ImportError:
        print(f"\n=== ATHENA | mode={mode} | run_id={run_id} ===\n")


def _print_metrics_table(metrics: dict[str, Any], title: str = "Results") -> None:
    console = _console()
    if console is None:
        print(f"\n--- {title} ---")
        for k, v in metrics.items():
            if not isinstance(v, dict):
                print(f"  {k:<30} {v}")
        return
    try:
        from rich import box
        from rich.table import Table

        table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")
        for k, v in metrics.items():
            if isinstance(v, dict):
                continue
            table.add_row(str(k), f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)
    except ImportError:
        print(f"\n--- {title} ---")
        for k, v in metrics.items():
            if not isinstance(v, dict):
                print(f"  {k:<30} {v}")


def _lake_from_cfg(cfg) -> DataLake:
    return DataLake(cfg.data.datastore_root)


def _load_feature_matrix(cfg, ctx):
    lake = _lake_from_cfg(cfg)
    X = build_feature_matrix(lake=lake)
    prices = build_prices_matrix(lake=lake, target_idx=X.index)
    if prices is not None:
        prices = prices.reindex(X.index).ffill(limit=3)
    ctx.logger.info("Loaded feature matrix from gold datasets: shape=%s", X.shape)
    return X, prices


def _cmd_generate(cfg, ctx):
    lake = _lake_from_cfg(cfg)
    provider = MockProvider()
    end = pd.Timestamp.now(tz="UTC").date().isoformat()
    start = (pd.Timestamp(end) - pd.Timedelta(days=1500)).date().isoformat()
    results = {}
    for dataset in DEFAULT_UPDATE_DATASETS:
        results[dataset] = update_dataset(lake=lake, dataset=dataset, provider=provider, start=start, end=end)
    ctx.log_metrics({k: v["status"] for k, v in results.items()})


def _cmd_infer(cfg, ctx):
    from src.athena_regime.features.engineering import RollingZScaler, run_qa
    from src.athena_regime.regimes.inference import RegimeInferenceEngine

    X, _ = _load_feature_matrix(cfg, ctx)
    X, _ = run_qa(X, logger=ctx.logger)
    X = X.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)

    scaler = RollingZScaler()
    X_z = scaler.fit_transform(X)

    engine = RegimeInferenceEngine(
        n_states=cfg.regime.n_states,
        n_hmm_iter=cfg.regime.n_hmm_iter,
        n_restarts=cfg.regime.n_restarts,
        prob_temperature=cfg.regime.prob_temperature,
        min_state_occupancy=cfg.regime.min_state_occupancy,
    )
    engine.fit(X_z)
    result_df = engine.infer(X_z)

    out = ctx.artifact("regime_history.parquet")
    result_df.to_parquet(out)
    ctx.logger.info("regime_history written -> %s", out)
    _print_metrics_table({"rows": len(result_df), "states": result_df["state"].nunique()}, title="Infer Complete")


def _cmd_backtest(cfg, ctx):
    from src.athena_regime.backtest.engine import WalkForwardBacktest, WalkForwardConfig
    from src.athena_regime.analytics.visualization import generate_visual_report

    X, prices = _load_feature_matrix(cfg, ctx)
    if prices is None:
        raise RuntimeError("prices not available in gold returns_daily dataset")

    wf_cfg = WalkForwardConfig(
        mode="expanding" if cfg.backtest.expanding else "rolling",
        train_periods=cfg.backtest.train_window,
        step_size=cfg.backtest.step_size,
        refit_every=cfg.backtest.refit_every,
        n_hmm_iter=cfg.regime.n_hmm_iter,
        n_hmm_restarts=cfg.regime.n_restarts,
        prob_temperature=cfg.regime.prob_temperature,
        min_state_occupancy=cfg.regime.min_state_occupancy,
        n_states=cfg.regime.n_states,
        prop_cost_bps=cfg.backtest.transaction_cost_bps,
        impact_bps=cfg.backtest.impact_bps,
        bid_ask_bps=cfg.backtest.bid_ask_bps,
        max_leverage=cfg.backtest.max_leverage,
        max_weight=cfg.backtest.max_weight,
        long_only=cfg.backtest.long_only,
        target_net_exposure=cfg.backtest.target_net_exposure,
        turnover_cap=cfg.backtest.turnover_cap,
        rebal_threshold=cfg.backtest.rebalance_threshold,
        vol_target=cfg.backtest.vol_target,
        borrow_spread_bps=cfg.backtest.borrow_spread_bps,
    )
    print("backtesting")
    bt = WalkForwardBacktest(X, prices, wf_cfg, save_dir=str(ctx.run_dir / "artifacts"))
    result = bt.run()

    metrics = result.performance
    ctx.log_metrics(metrics)
    _print_metrics_table(metrics, title="Backtest Results")

    returns_df = result.daily_returns.to_frame("returns")
    parquet_path = ctx.artifact("backtest_returns.parquet")
    returns_df.to_parquet(parquet_path)
    ctx.logger.info("backtest complete sharpe=%.2f", metrics.get("sharpe", float("nan")))
    viz_outputs = generate_visual_report(cfg.run.runs_dir, run_id=ctx.run_id)
    if viz_outputs:
        ctx.logger.info("visual artifacts: %s", viz_outputs)


def _cmd_stress(cfg, ctx, scenario_name: str | None):
    if not scenario_name:
        raise ValueError("--scenario is required for stress mode")

    from src.athena_regime.backtest.engine import WalkForwardBacktest, WalkForwardConfig
    from src.athena_regime.data.models import FeatureMatrix
    from src.athena_regime.stress.runner import StressTestRunner
    from src.athena_regime.stress.scenarios import ScenarioRegistry

    X, prices = _load_feature_matrix(cfg, ctx)
    if prices is None:
        raise RuntimeError("prices not available in gold returns_daily dataset")

    wf_cfg = WalkForwardConfig(
        mode="expanding" if cfg.backtest.expanding else "rolling",
        train_periods=cfg.backtest.train_window,
        step_size=cfg.backtest.step_size,
        refit_every=cfg.backtest.refit_every,
        n_hmm_iter=cfg.regime.n_hmm_iter,
        n_hmm_restarts=cfg.regime.n_restarts,
        prob_temperature=cfg.regime.prob_temperature,
        min_state_occupancy=cfg.regime.min_state_occupancy,
        n_states=cfg.regime.n_states,
        prop_cost_bps=cfg.backtest.transaction_cost_bps,
        impact_bps=cfg.backtest.impact_bps,
        bid_ask_bps=cfg.backtest.bid_ask_bps,
        max_leverage=cfg.backtest.max_leverage,
        max_weight=cfg.backtest.max_weight,
        long_only=cfg.backtest.long_only,
        target_net_exposure=cfg.backtest.target_net_exposure,
        turnover_cap=cfg.backtest.turnover_cap,
        rebal_threshold=cfg.backtest.rebalance_threshold,
        vol_target=cfg.backtest.vol_target,
        borrow_spread_bps=cfg.backtest.borrow_spread_bps,
    )
    fm = FeatureMatrix(X=X, feature_names=list(X.columns), metadata={})

    def backtest_factory(shocked_fm):
        return WalkForwardBacktest(shocked_fm.X, prices, wf_cfg, save_dir=str(ctx.run_dir / "artifacts")).run()

    scenario = ScenarioRegistry.get(scenario_name)
    runner = StressTestRunner(backtest_factory, ctx)
    result = runner.run(fm, scenario)

    ctx.log_metrics(result.performance)
    _print_metrics_table(result.performance, title=f"Stress: {scenario_name}")
    from src.athena_regime.analytics.visualization import generate_visual_report

    viz_outputs = generate_visual_report(cfg.run.runs_dir, run_id=ctx.run_id)
    if viz_outputs:
        ctx.logger.info("visual artifacts: %s", viz_outputs)


def _cmd_visualize(cfg, ctx, run_id: str | None, signal_mode: str):
    from src.athena_regime.analytics.visualization import generate_visual_report

    if not run_id:
        raise ValueError("--target-run-id is required for visualize")
    outputs = generate_visual_report(cfg.run.runs_dir, run_id=run_id, signal_mode=signal_mode)
    if not outputs:
        ctx.logger.warning("No visualization outputs produced. Missing run artifacts or plotting dependency.")
        return
    for name, path in outputs.items():
        ctx.logger.info("%s -> %s", name, path)
    _print_metrics_table({"artifacts": len(outputs), **{k: str(v) for k, v in outputs.items()}}, title="Visualize")


def _cmd_sweep(cfg, ctx, max_runs: int):
    from src.athena_regime.backtest.engine import WalkForwardBacktest, WalkForwardConfig

    X, prices = _load_feature_matrix(cfg, ctx)
    if prices is None:
        raise RuntimeError("prices not available in gold returns_daily dataset")

    grid = {
        "rebal_threshold": [0.05, 0.10, 0.15],
        "turnover_cap": [0.10, 0.20, 0.30],
        "prob_temperature": [1.0, 1.25],
    }
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    combos = combos[:max_runs]

    rows: list[dict[str, Any]] = []
    for i, params in enumerate(combos, start=1):
        ctx.logger.info("sweep run %d/%d params=%s", i, len(combos), params)
        wf_cfg = WalkForwardConfig(
            mode="expanding" if cfg.backtest.expanding else "rolling",
            train_periods=cfg.backtest.train_window,
            step_size=cfg.backtest.step_size,
            refit_every=cfg.backtest.refit_every,
            n_hmm_iter=cfg.regime.n_hmm_iter,
            n_hmm_restarts=cfg.regime.n_restarts,
            prob_temperature=params["prob_temperature"],
            min_state_occupancy=cfg.regime.min_state_occupancy,
            n_states=cfg.regime.n_states,
            prop_cost_bps=cfg.backtest.transaction_cost_bps,
            impact_bps=cfg.backtest.impact_bps,
            bid_ask_bps=cfg.backtest.bid_ask_bps,
            max_leverage=cfg.backtest.max_leverage,
            max_weight=cfg.backtest.max_weight,
            long_only=cfg.backtest.long_only,
            target_net_exposure=cfg.backtest.target_net_exposure,
            turnover_cap=params["turnover_cap"],
            rebal_threshold=params["rebal_threshold"],
            vol_target=cfg.backtest.vol_target,
            borrow_spread_bps=cfg.backtest.borrow_spread_bps,
        )
        run_art_dir = ctx.run_dir / "artifacts" / f"sweep_{i:02d}"
        bt = WalkForwardBacktest(X, prices, wf_cfg, save_dir=str(run_art_dir))
        result = bt.run()
        row = {
            **params,
            "sharpe": result.performance.get("sharpe"),
            "cagr": result.performance.get("cagr"),
            "max_drawdown": result.performance.get("max_drawdown"),
            "avg_max_regime_prob": result.performance.get("avg_max_regime_prob"),
            "n_rebalances": result.performance.get("n_rebalances"),
            "artifact_dir": str(run_art_dir),
        }
        rows.append(row)

    out = ctx.artifact("sweep_summary.csv")
    df = pd.DataFrame(rows).sort_values(["sharpe", "cagr"], ascending=False)
    df.to_csv(out, index=False)
    ctx.logger.info("sweep summary written -> %s", out)
    _print_metrics_table({"trials": len(rows), "summary": str(out)}, title="Sweep Complete")


def _run_pipeline_command(args) -> int:
    cfg = load_config(args.config)
    ctx = RunContext(cfg, run_id=args.run_id)
    ctx.log_config(dataclasses.asdict(cfg))
    ctx.logger.info("command=%s run_id=%s", args.command, ctx.run_id)

    console = _console()
    if console:
        _print_banner(console, args.command, ctx.run_id)

    try:
        if args.command == "generate":
            _cmd_generate(cfg, ctx)
        elif args.command == "infer":
            _cmd_infer(cfg, ctx)
        elif args.command == "backtest":
            _cmd_backtest(cfg, ctx)
        elif args.command == "stress":
            _cmd_stress(cfg, ctx, args.scenario)
        elif args.command == "visualize":
            _cmd_visualize(cfg, ctx, args.target_run_id, args.signal_mode)
        elif args.command == "sweep":
            _cmd_sweep(cfg, ctx, args.max_runs)
        else:
            raise ValueError(f"Unknown run command: {args.command}")
    except Exception:
        ctx.logger.exception("Fatal error in command=%s", args.command)
        return 1

    ctx.logger.info("run complete run_id=%s", ctx.run_id)
    return 0


def _run_data_command(args) -> int:
    cfg = load_config(args.config)
    lake = DataLake(cfg.data.datastore_root)

    if args.data_command == "update":
        provider = provider_from_env(args.dataset)
        result = update_dataset(
            lake=lake,
            dataset=args.dataset,
            provider=provider,
            start=args.start,
            end=args.end,
        )
    elif args.data_command == "update-all":
        result = {}
        for dataset in DEFAULT_UPDATE_DATASETS:
            provider = provider_from_env(dataset)
            result[dataset] = update_dataset(
                lake=lake,
                dataset=dataset,
                provider=provider,
                start=args.start,
                end=args.end,
            )
    elif args.data_command == "prune":
        if args.level == "gold" and not args.force_gold:
            raise ValueError("Refusing to prune gold without --force-gold.")
        result = prune_dataset(
            lake=lake,
            dataset=args.dataset,
            level=args.level,
            older_than=args.older_than,
            keep_last_n=args.keep_last_n,
        )
    else:
        raise ValueError(f"Unknown data command: {args.data_command}")

    print(json.dumps(result, indent=2, default=str))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="athena", description="ATHENA CLI")
    sub = parser.add_subparsers(dest="group", required=True)

    p_run = sub.add_parser("run", help="Run modeling workflows")
    p_run.add_argument("--config", required=True, help="Path to YAML config file")
    p_run.add_argument("--run-id", default=None, help="Override auto-generated run ID")
    run_sub = p_run.add_subparsers(dest="command", required=True)
    run_sub.add_parser("generate", help="Generate synthetic data via MockProvider")
    run_sub.add_parser("infer", help="Run regime inference")
    run_sub.add_parser("backtest", help="Run backtest")
    p_stress = run_sub.add_parser("stress", help="Run stress scenario")
    p_stress.add_argument("--scenario", required=True, help="Stress scenario name")
    p_visualize = run_sub.add_parser("visualize", help="Generate per-run analysis chart suite")
    p_visualize.add_argument(
        "--target-run-id",
        required=True,
        help="Run ID to visualize; charts are written to runs/<run_id>/artifacts.",
    )
    p_visualize.add_argument(
        "--signal-mode",
        default="log",
        choices=["raw", "log", "zscore"],
        help="How to plot return signal on the top panel.",
    )
    p_sweep = run_sub.add_parser("sweep", help="Run a small parameter sweep and rank configurations")
    p_sweep.add_argument("--max-runs", type=int, default=18, help="Maximum sweep trials to execute")

    p_data = sub.add_parser("data", help="Data lake commands")
    p_data.add_argument("--config", default="configs/base.yaml", help="Path to YAML config file")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    p_update = data_sub.add_parser("update", help="Incremental update for one dataset")
    p_update.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS.keys()))
    p_update.add_argument("--start", default=None)
    p_update.add_argument("--end", default=None)

    p_update_all = data_sub.add_parser("update-all", help="Incremental update for default datasets")
    p_update_all.add_argument("--start", default=None)
    p_update_all.add_argument("--end", default=None)

    p_prune = data_sub.add_parser("prune", help="Prune old partitions")
    p_prune.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS.keys()))
    p_prune.add_argument("--level", required=True, choices=["bronze", "silver", "gold"])
    p_prune.add_argument("--older-than", default=None)
    p_prune.add_argument("--keep-last-n", type=int, default=None)
    p_prune.add_argument("--force-gold", action="store_true", help="Explicitly allow gold pruning")

    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.group == "run":
        return _run_pipeline_command(args)
    if args.group == "data":
        return _run_data_command(args)
    raise ValueError(f"Unknown command group: {args.group}")


if __name__ == "__main__":
    raise SystemExit(main())
