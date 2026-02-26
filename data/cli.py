from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from data.ingest.jobs.backfill import run_backfill
from data.ingest.jobs.update_cot import run_update_cot
from data.ingest.jobs.update_macro import run_update_macro
from data.ingest.jobs.update_policy import run_update_policy
from data.ingest.jobs.update_prices import run_update_prices
from data.publish.build_cot_zscores import build_cot_zscores
from data.publish.build_macro_asof import build_macro_asof
from data.publish.build_policy_features import build_policy_features
from data.publish.build_returns import build_returns
from data.publish.dependency_tracker import compute_minimal_recompute
from data.store.metadata import MetadataStore


# ---------------------------------------------------------------------------
# Rich helpers (graceful fallback if rich not installed yet)
# ---------------------------------------------------------------------------

def _stderr_console():
    try:
        from rich.console import Console
        return Console(stderr=True)
    except ImportError:
        return None


def _print_job_result(console, command: str, result: dict, elapsed: float) -> None:
    """Print a formatted job summary to stderr; JSON result goes to stdout."""
    if console is None:
        print(f"[{command}] done in {elapsed:.1f}s", file=sys.stderr)
        return
    try:
        from rich.table import Table
        from rich import box
        table = Table(
            title=f"[bold cyan]{command}[/bold cyan]  ({elapsed:.1f}s)",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value")
        for k, v in result.items():
            if isinstance(v, list):
                table.add_row(str(k), f"{len(v)} partition(s)")
            elif isinstance(v, dict):
                table.add_row(str(k), str(v))
            else:
                table.add_row(str(k), str(v))
        console.print(table)
    except ImportError:
        print(f"[{command}] done in {elapsed:.1f}s", file=sys.stderr)


def _load_storage_config() -> dict:
    return yaml.safe_load(Path("data/configs/storage.yaml").read_text(encoding="utf-8"))


def _snapshot_path(base_path: Path) -> Path:
    return Path(base_path) / "meta" / "last_build_snapshot.json"


def _load_snapshot(base_path: Path) -> dict:
    path = _snapshot_path(base_path)
    if not path.exists():
        return {"datasets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_snapshot(base_path: Path, payload: dict) -> None:
    path = _snapshot_path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_build_all(base_path: Path, smart: bool = False) -> dict:
    storage = _load_storage_config()
    lookback = int(storage.get("cot_z_lookback_weeks", 260))
    metadata = MetadataStore(Path(base_path) / "meta")
    current = metadata.load_versions()

    if smart:
        previous = _load_snapshot(base_path)
        changed = {
            "silver.market_prices": metadata.partitions_with_changed_fingerprints("silver.market_prices", previous, current),
            "silver.macro_releases": metadata.partitions_with_changed_fingerprints("silver.macro_releases", previous, current),
            "silver.cot_weekly": metadata.partitions_with_changed_fingerprints("silver.cot_weekly", previous, current),
            "silver.policy_curve": metadata.partitions_with_changed_fingerprints("silver.policy_curve", previous, current),
        }
        impacted = compute_minimal_recompute(changed, cot_lookback_weeks=lookback)
    else:
        impacted = {
            "gold.returns_daily": sorted(current.get("datasets", {}).get("silver.market_prices", {}).get("partitions", {}).keys()),
            "gold.macro_asof_monthly": sorted(current.get("datasets", {}).get("silver.macro_releases", {}).get("partitions", {}).keys()),
            "gold.cot_z_weekly": sorted(current.get("datasets", {}).get("silver.cot_weekly", {}).get("partitions", {}).keys()),
            "gold.policy_features": sorted(current.get("datasets", {}).get("silver.policy_curve", {}).get("partitions", {}).keys()),
        }

    returns_dates = [p.split("=", 1)[1] for p in impacted.get("gold.returns_daily", []) if p.startswith("dt=")]
    returns = build_returns(Path(base_path), returns_dates, Path("data/schemas/returns.schema.json")) if returns_dates else {}

    macro_months = [p.split("=", 1)[1] for p in impacted.get("gold.macro_asof_monthly", []) if p.startswith("month=")]
    macro = build_macro_asof(
        datastore_root=Path(base_path),
        months_to_rebuild=macro_months,
        lags_path=Path("data/configs/lags.yaml"),
        schema_path=Path("data/schemas/macro_asof.schema.json"),
    ) if macro_months else {}

    cot_weeks = [p.split("=", 1)[1] for p in impacted.get("gold.cot_z_weekly", []) if p.startswith("week=")]
    cot = build_cot_zscores(
        datastore_root=Path(base_path),
        weeks_to_rebuild=cot_weeks,
        lookback_weeks=lookback,
        schema_path=Path("data/schemas/cot_z.schema.json"),
    ) if cot_weeks else {}

    policy_dates = [p.split("=", 1)[1] for p in impacted.get("gold.policy_features", []) if p.startswith("dt=")]
    policy = build_policy_features(
        datastore_root=Path(base_path),
        dates_to_rebuild=policy_dates,
        schema_path=Path("data/schemas/policy_features.schema.json"),
    ) if policy_dates else {}

    _save_snapshot(base_path, metadata.load_versions())
    return {
        "returns": sorted(returns.keys()),
        "macro_asof": sorted(macro.keys()),
        "cot_z": sorted(cot.keys()),
        "policy_features": sorted(policy.keys()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="data", description="ATHENA DataPack incremental data lake CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prices = sub.add_parser("update_prices")
    p_prices.add_argument("--base-path", default="datastore")
    p_prices.add_argument("--vendor", default="fred")
    p_prices.add_argument("--last-n-days", type=int, default=7)
    p_prices.add_argument("--input-file", default=None)
    p_prices.add_argument("--person", default="unknown")

    p_macro = sub.add_parser("update_macro")
    p_macro.add_argument("--base-path", default="datastore")
    p_macro.add_argument("--vendor", default="fred")
    p_macro.add_argument("--input-file", default=None)
    p_macro.add_argument("--start", default=None)
    p_macro.add_argument("--end", default=None)
    p_macro.add_argument("--person", default="unknown")

    p_cot = sub.add_parser("update_cot")
    p_cot.add_argument("--base-path", default="datastore")
    p_cot.add_argument("--source-url", default="https://www.cftc.gov/dea/newcot/FinFutWk.txt")
    p_cot.add_argument("--input-file", default=None)
    p_cot.add_argument("--lookback-weeks", type=int, default=260)
    p_cot.add_argument("--person", default="unknown")

    p_policy = sub.add_parser("update_policy")
    p_policy.add_argument("--base-path", default="datastore")
    p_policy.add_argument("--input-file", default="data/raw/fedwatch.csv")
    p_policy.add_argument("--dt", default=None)
    p_policy.add_argument("--person", default="unknown")

    p_build = sub.add_parser("build_all")
    p_build.add_argument("--base-path", default="datastore")
    p_build.add_argument("--smart", action="store_true")

    p_backfill = sub.add_parser("backfill")
    p_backfill.add_argument("--base-path", default="datastore")
    p_backfill.add_argument("--start", required=True)
    p_backfill.add_argument("--end", required=True)
    p_backfill.add_argument("--person", default="unknown")

    args = parser.parse_args()

    console = _stderr_console()
    t0 = time.monotonic()

    if args.command == "update_prices":
        result = run_update_prices(
            base_path=Path(args.base_path),
            vendor=args.vendor,
            last_n_days=args.last_n_days,
            input_file=Path(args.input_file) if args.input_file else None,
            person=args.person,
        )
    elif args.command == "update_macro":
        result = run_update_macro(
            base_path=Path(args.base_path),
            vendor=args.vendor,
            input_file=Path(args.input_file) if args.input_file else None,
            start=args.start,
            end=args.end,
            person=args.person,
        )
    elif args.command == "update_cot":
        result = run_update_cot(
            base_path=Path(args.base_path),
            source_url=args.source_url,
            input_file=Path(args.input_file) if args.input_file else None,
            lookback_weeks=args.lookback_weeks,
            person=args.person,
        )
    elif args.command == "update_policy":
        result = run_update_policy(
            base_path=Path(args.base_path),
            input_file=Path(args.input_file),
            dt=args.dt,
            person=args.person,
        )
    elif args.command == "build_all":
        result = run_build_all(Path(args.base_path), smart=args.smart)
    elif args.command == "backfill":
        result = run_backfill(args.start, args.end, Path(args.base_path), args.person)
    else:
        raise ValueError("Unknown command")

    elapsed = time.monotonic() - t0
    _print_job_result(console, args.command, result, elapsed)
    # JSON result on stdout (machine-readable; status summary already on stderr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
