from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data.ingest.providers.ois_curve import load_curve
from data.normalize.canonicalize_policy import canonicalize_policy
from data.publish.build_policy_features import build_policy_features
from data.store.manifest import build_run_manifest, make_datapack_id, sha256_file, utc_now_iso, write_datapack_manifest, write_run_manifest
from data.store.metadata import MetadataStore
from data.store.upsert import replace_partition


def run_update_policy(
    base_path: Path = Path("datastore"),
    input_file: Path = Path("data/raw/fedwatch.csv"),
    dt: str | None = None,
    person: str = "unknown",
) -> dict:
    start_ts = utc_now_iso()
    run_date = dt or datetime.now(timezone.utc).date().isoformat()
    run_id = f"update_policy_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    raw = load_curve(input_file, dt=run_date)
    source = str(raw["source"].iloc[0]) if "source" in raw.columns and len(raw) else "cme"
    canonical = canonicalize_policy(raw=raw, dt=run_date, source=source)

    raw_dir = Path(base_path) / "bronze" / "policy" / f"ingest_date={run_date}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "part-00000.parquet"
    raw.to_parquet(raw_file, index=False)

    part_key = f"dt={run_date}"
    dest, fp, rows = replace_partition(
        root=Path(base_path) / "silver",
        dataset="policy_curve",
        partition_key=part_key,
        df=canonical,
        sort_keys=["dt", "tenor_days"],
    )

    metadata = MetadataStore(Path(base_path) / "meta")
    metadata.update_partition("silver.policy_curve", part_key, fp, rows, str(dest))
    feature_fps = build_policy_features(
        datastore_root=Path(base_path),
        dates_to_rebuild=[run_date],
        schema_path=Path("data/schemas/policy_features.schema.json"),
    )

    datapack = {
        "datapack_id": make_datapack_id(run_date, 1),
        "extraction_date": run_date,
        "person": person,
        "vendor_functions_used": ["ois_curve.load_curve"],
        "tickers_and_fields": {"policy_curve": ["tenor_days", "implied_rate"]},
        "frequencies": {"policy": "daily"},
        "currency_assumptions": {},
        "lag_assumptions": {},
        "notes": "",
        "raw_file_hashes": {str(raw_file): sha256_file(raw_file)},
    }
    write_datapack_manifest(raw_dir / "manifest.json", datapack)

    run_manifest = build_run_manifest(
        job="update_policy",
        run_id=run_id,
        start_ts=start_ts,
        end_ts=utc_now_iso(),
        inputs=[str(raw_file)],
        outputs=[str(dest)],
        partition_fingerprints={"silver.policy_curve/" + part_key: fp, **{"gold.policy_features/" + k: v for k, v in feature_fps.items()}},
        row_counts={"silver.policy_curve/" + part_key: rows},
    )
    run_manifest_path = write_run_manifest(Path(base_path) / "meta", run_date, run_manifest)
    return {"run_manifest": str(run_manifest_path), "features_rebuilt": sorted(feature_fps.keys())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental policy ingestion job")
    parser.add_argument("--base-path", default="datastore")
    parser.add_argument("--input-file", default="data/raw/fedwatch.csv")
    parser.add_argument("--dt", default=None)
    parser.add_argument("--person", default="unknown")
    args = parser.parse_args()

    result = run_update_policy(
        base_path=Path(args.base_path),
        input_file=Path(args.input_file),
        dt=args.dt,
        person=args.person,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
