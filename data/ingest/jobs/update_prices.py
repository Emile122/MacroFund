from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from data.ingest.providers import fred, polygon
from data.normalize.canonicalize_prices import canonicalize_prices
from data.normalize.validation import validate_frame
from data.publish.build_returns import build_returns
from data.store.manifest import build_run_manifest, make_datapack_id, sha256_file, utc_now_iso, write_datapack_manifest, write_run_manifest
from data.store.metadata import MetadataStore
from data.store.upsert import replace_partition


def run_update_prices(
    base_path: Path = Path("datastore"),
    vendor: str = "fred",
    last_n_days: int = 7,
    input_file: Path | None = None,
    person: str = "unknown",
) -> dict:
    start_ts = utc_now_iso()
    run_date = datetime.now(timezone.utc).date().isoformat()
    run_id = f"update_prices_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    cfg_storage = yaml.safe_load(Path("data/configs/storage.yaml").read_text(encoding="utf-8"))
    universe_path = Path("data/configs/universe.yaml")
    metadata = MetadataStore(Path(base_path) / "meta")

    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=last_n_days)

    if input_file:
        raw_df = pd.read_csv(input_file)
    elif vendor == "polygon":
        universe = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
        tickers = [v.get("vendor", {}).get("polygon") for v in universe.get("instruments", {}).values() if v.get("vendor", {}).get("polygon")]
        raw_df = polygon.fetch_grouped_daily_prices(tickers, start_date.isoformat(), today.isoformat())
    else:
        universe = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
        series = [v.get("vendor", {}).get("fred") for v in universe.get("instruments", {}).values() if v.get("vendor", {}).get("fred")]
        raw_df = fred.fetch_series(series, start_date.isoformat(), today.isoformat())

    fetch_ts = utc_now_iso()
    canonical = canonicalize_prices(raw_df, vendor=vendor, universe_path=universe_path, fetch_ts=fetch_ts)
    schema_path = Path("data/schemas/market_prices.schema.json")
    validate_frame(canonical.sort_values(["dt", "instrument"]), schema_path)

    raw_dir = Path(base_path) / "bronze" / "prices" / f"ingest_date={run_date}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "part-00000.parquet"
    raw_df.to_parquet(raw_file, index=False)

    datapack = {
        "datapack_id": make_datapack_id(run_date, 1),
        "extraction_date": run_date,
        "person": person,
        "vendor_functions_used": [f"{vendor}.fetch"],
        "tickers_and_fields": {"fields": ["PX_LAST"]},
        "frequencies": {"prices": "daily"},
        "currency_assumptions": {"default": "USD"},
        "lag_assumptions": {},
        "notes": "",
        "raw_file_hashes": {str(raw_file): sha256_file(raw_file)},
    }
    write_datapack_manifest(raw_dir / "manifest.json", datapack)

    changed_dates: list[str] = []
    part_fps: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    outputs: list[str] = []
    for dt, part in canonical.groupby("dt"):
        dt_str = pd.Timestamp(dt).date().isoformat()
        part_key = f"dt={dt_str}"
        dest, fp, rows = replace_partition(
            root=Path(base_path) / "silver",
            dataset="market_prices",
            partition_key=part_key,
            df=part.sort_values(["instrument", "vendor"]),
            sort_keys=["dt", "instrument", "vendor"],
            filename=cfg_storage.get("partition_file_name", "part-00000.parquet"),
        )
        result = metadata.update_partition("silver.market_prices", part_key, fp, rows, str(dest))
        if result["changed"]:
            changed_dates.append(dt_str)
        part_fps[f"silver.market_prices/{part_key}"] = fp
        row_counts[f"silver.market_prices/{part_key}"] = rows
        outputs.append(str(dest))

    rebuild_dates = sorted(set(changed_dates + [(pd.Timestamp(d) + pd.Timedelta(days=1)).date().isoformat() for d in changed_dates]))
    returns_fps = build_returns(
        datastore_root=Path(base_path),
        dates_to_rebuild=rebuild_dates,
        schema_path=Path("data/schemas/returns.schema.json"),
    )
    for k, v in returns_fps.items():
        part_fps[f"gold.returns_daily/{k}"] = v

    end_ts = utc_now_iso()
    run_manifest = build_run_manifest(
        job="update_prices",
        run_id=run_id,
        start_ts=start_ts,
        end_ts=end_ts,
        inputs=[str(raw_file)],
        outputs=outputs,
        partition_fingerprints=part_fps,
        row_counts=row_counts,
    )
    run_manifest_path = write_run_manifest(Path(base_path) / "meta", run_date, run_manifest)
    return {"run_manifest": str(run_manifest_path), "changed_dates": changed_dates, "returns_rebuilt": sorted(returns_fps)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental price ingestion job")
    parser.add_argument("--base-path", default="datastore")
    parser.add_argument("--vendor", default="fred")
    parser.add_argument("--last-n-days", type=int, default=7)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--person", default="unknown")
    args = parser.parse_args()

    result = run_update_prices(
        base_path=Path(args.base_path),
        vendor=args.vendor,
        last_n_days=args.last_n_days,
        input_file=Path(args.input_file) if args.input_file else None,
        person=args.person,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
