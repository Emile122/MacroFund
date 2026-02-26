from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from data.ingest.providers import cftc_cot
from data.normalize.canonicalize_cot import canonicalize_cot
from data.normalize.validation import validate_frame
from data.publish.build_cot_zscores import build_cot_zscores
from data.store.manifest import build_run_manifest, make_datapack_id, sha256_file, utc_now_iso, write_datapack_manifest, write_run_manifest
from data.store.metadata import MetadataStore
from data.store.upsert import replace_partition


def _apply_contract_mapping(canonical: pd.DataFrame, universe_path: Path) -> pd.DataFrame:
    universe = yaml.safe_load(Path(universe_path).read_text(encoding="utf-8")) or {}
    contracts_cfg = universe.get("cot_contracts", {}) or {}
    if not contracts_cfg:
        return canonical

    frame = canonical.copy()
    raw = frame["contract"].astype(str)
    mapped = pd.Series(index=frame.index, dtype="object")

    for contract_name, cfg in contracts_cfg.items():
        internal_id = str((cfg or {}).get("internal_id", contract_name))
        pattern = str((cfg or {}).get("cftc_name_pattern", contract_name)).upper()
        mask = raw.str.upper().str.contains(pattern, regex=False, na=False)
        mapped.loc[mask] = internal_id

    out = frame[mapped.notna()].copy()
    out["contract"] = mapped.loc[out.index].astype(str)
    return out


def run_update_cot(
    base_path: Path = Path("datastore"),
    source_url: str = "https://www.cftc.gov/dea/newcot/FinFutWk.txt",
    input_file: Path | None = None,
    lookback_weeks: int = 260,
    person: str = "unknown",
) -> dict:
    start_ts = utc_now_iso()
    run_date = datetime.now(timezone.utc).date().isoformat()
    run_id = f"update_cot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    if input_file:
        raw = pd.read_csv(input_file)
    else:
        raw = cftc_cot.fetch_weekly_csv(source_url)

    raw_dir = Path(base_path) / "bronze" / "cot" / f"ingest_date={run_date}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "part-00000.parquet"
    raw.to_parquet(raw_file, index=False)

    canonical = canonicalize_cot(raw, dt_ingest=run_date)
    canonical = _apply_contract_mapping(canonical, Path("data/configs/universe.yaml"))
    if canonical.empty:
        raise ValueError("COT contract mapping filtered out all rows; update data/configs/universe.yaml cot_contracts patterns.")
    validate_frame(canonical, Path("data/schemas/cot.schema.json"))

    metadata = MetadataStore(Path(base_path) / "meta")
    outputs: list[str] = []
    part_fps: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    changed_weeks: list[str] = []

    canonical["week"] = pd.to_datetime(canonical["report_date"]).dt.strftime("%Y-%W")
    for week, part in canonical.groupby("week"):
        part_key = f"week={week}"
        dest, fp, rows = replace_partition(
            root=Path(base_path) / "silver",
            dataset="cot_weekly",
            partition_key=part_key,
            df=part.drop(columns=["week"]),
            sort_keys=["report_date", "contract", "group"],
        )
        result = metadata.update_partition("silver.cot_weekly", part_key, fp, rows, str(dest))
        if result["changed"]:
            changed_weeks.append(week)
        outputs.append(str(dest))
        part_fps[f"silver.cot_weekly/{part_key}"] = fp
        row_counts[f"silver.cot_weekly/{part_key}"] = rows

    if changed_weeks:
        latest = pd.Timestamp(datetime.strptime(max(changed_weeks) + "-1", "%Y-%W-%w"))
        rebuild = [(latest - pd.Timedelta(weeks=i)).strftime("%Y-%W") for i in range(lookback_weeks)]
    else:
        rebuild = []

    z_fps = build_cot_zscores(
        datastore_root=Path(base_path),
        weeks_to_rebuild=rebuild,
        lookback_weeks=lookback_weeks,
        schema_path=Path("data/schemas/cot_z.schema.json"),
    )
    for k, v in z_fps.items():
        part_fps[f"gold.cot_z_weekly/{k}"] = v

    datapack = {
        "datapack_id": make_datapack_id(run_date, 1),
        "extraction_date": run_date,
        "person": person,
        "vendor_functions_used": ["cftc_cot.fetch_weekly_csv"],
        "tickers_and_fields": {"cot": ["long", "short", "open_interest"]},
        "frequencies": {"cot": "weekly"},
        "currency_assumptions": {},
        "lag_assumptions": {},
        "notes": "",
        "raw_file_hashes": {str(raw_file): sha256_file(raw_file)},
    }
    write_datapack_manifest(raw_dir / "manifest.json", datapack)

    run_manifest = build_run_manifest(
        job="update_cot",
        run_id=run_id,
        start_ts=start_ts,
        end_ts=utc_now_iso(),
        inputs=[str(raw_file)],
        outputs=outputs,
        partition_fingerprints=part_fps,
        row_counts=row_counts,
    )
    run_manifest_path = write_run_manifest(Path(base_path) / "meta", run_date, run_manifest)
    return {"run_manifest": str(run_manifest_path), "weeks_rebuilt": sorted(z_fps.keys())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental COT ingestion job")
    parser.add_argument("--base-path", default="datastore")
    parser.add_argument("--source-url", default="https://www.cftc.gov/dea/newcot/FinFutWk.txt")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--lookback-weeks", type=int, default=260)
    parser.add_argument("--person", default="unknown")
    args = parser.parse_args()

    result = run_update_cot(
        base_path=Path(args.base_path),
        source_url=args.source_url,
        input_file=Path(args.input_file) if args.input_file else None,
        lookback_weeks=args.lookback_weeks,
        person=args.person,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
