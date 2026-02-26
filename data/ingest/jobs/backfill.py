from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from data.ingest.jobs.update_cot import run_update_cot
from data.ingest.jobs.update_macro import run_update_macro
from data.ingest.jobs.update_policy import run_update_policy
from data.ingest.jobs.update_prices import run_update_prices


def run_backfill(start: str, end: str, base_path: Path = Path("datastore"), person: str = "unknown") -> dict:
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    if start_dt > end_dt:
        raise ValueError("start must be <= end")

    # Backfill orchestrates job invocations; providers may still fetch range internally.
    prices = run_update_prices(base_path=base_path, last_n_days=(end_dt - start_dt).days + 1, person=person)
    macro = run_update_macro(base_path=base_path, start=start, end=end, person=person)
    cot = run_update_cot(base_path=base_path, person=person)
    policy = run_update_policy(base_path=base_path, dt=end, person=person)
    return {
        "prices": prices,
        "macro": macro,
        "cot": cot,
        "policy": policy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill ATHENA data module")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--base-path", default="datastore")
    parser.add_argument("--person", default="unknown")
    args = parser.parse_args()

    result = run_backfill(args.start, args.end, base_path=Path(args.base_path), person=args.person)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
