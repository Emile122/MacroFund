from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from data.ingest.providers import bloomberg_excel, ecb_sdmx, fred
from data.normalize.canonicalize_macro import canonicalize_macro
from data.normalize.validation import validate_frame
from data.publish.build_macro_asof import build_macro_asof
from data.store.manifest import build_run_manifest, make_datapack_id, sha256_file, utc_now_iso, write_datapack_manifest, write_run_manifest
from data.store.metadata import MetadataStore
from data.store.upsert import replace_partition


def _incremental_start_date(metadata: MetadataStore, explicit_start: str | None) -> str:
    """
    Derive start date for FRED fetch.
    - If explicit_start is given, use it.
    - Otherwise find the latest already-ingested partition and start from
      one month before (overlap to catch any revisions to recent releases).
    - Fall back to "2010-01-01" if no prior data exists.
    """
    if explicit_start:
        return explicit_start
    versions = metadata.load_versions()
    partitions = (
        versions.get("datasets", {})
        .get("silver.macro_releases", {})
        .get("partitions", {})
    )
    if not partitions:
        return "2010-01-01"
    dates = []
    for key in partitions:
        date_part = key.replace("dt=", "")
        try:
            dates.append(pd.Timestamp(date_part))
        except Exception:
            pass
    if not dates:
        return "2010-01-01"
    latest = max(dates)
    # Start one month back to catch revisions to the most recent release
    start_ts = (latest - pd.DateOffset(months=1)).normalize()
    return start_ts.date().isoformat()


def _ensure_macro_shape(raw: pd.DataFrame) -> pd.DataFrame:
    expected = {"period", "series_id", "value"}
    if expected.issubset(raw.columns):
        return raw
    fred_like = {"dt", "ticker", "px_last"}
    if fred_like.issubset(raw.columns):
        return raw.rename(columns={"dt": "period", "ticker": "series_id", "px_last": "value"})[
            ["period", "series_id", "value"]
        ]
    return raw


def run_update_macro(
    base_path: Path = Path("datastore"),
    vendor: str = "fred",
    input_file: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    person: str = "unknown",
) -> dict:
    start_ts = utc_now_iso()
    run_date = datetime.now(timezone.utc).date().isoformat()
    run_id = f"update_macro_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    metadata = MetadataStore(Path(base_path) / "meta")

    universe = yaml.safe_load(Path("data/configs/universe.yaml").read_text(encoding="utf-8"))
    series_ids = list(universe.get("macro_series", {}).keys())

    if vendor == "ecb" and input_file is None:
        raise ValueError(
            "ECB vendor is disabled by default. Provide --input-file with explicitly mapped macro rows to opt in."
        )

    # Determine incremental start date (avoids full re-fetch from 2010 every run)
    effective_start = _incremental_start_date(metadata, start)

    if input_file and input_file.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        raw = bloomberg_excel.load_export(input_file)
    elif input_file:
        raw = pd.read_csv(input_file)
    elif vendor == "ecb":
        raw = ecb_sdmx.fetch_series("EXR", effective_start, end or run_date, series_key="M.USD.EUR.SP00.A")
    else:
        raw = fred.fetch_series(series_ids, effective_start, end or run_date)
        raw = _ensure_macro_shape(raw)

    raw_dir = Path(base_path) / "bronze" / "macro" / f"ingest_date={run_date}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "part-00000.parquet"
    raw.to_parquet(raw_file, index=False)

    vintage_id = f"{run_date.replace('-', '')}_{vendor}"
    canonical = canonicalize_macro(raw=raw, vendor=vendor, universe_path=Path("data/configs/universe.yaml"), dt_ingest=run_date, vintage_id=vintage_id)
    # Read missing_critical_threshold from storage config (governs how many
    # missing critical series are tolerated before raising a ValidationError).
    storage_cfg = yaml.safe_load(Path("data/configs/storage.yaml").read_text(encoding="utf-8"))
    base_threshold = storage_cfg.get("missing_critical_threshold", 0)
    # For FRED without an input file allow at most 1 missing series so that a
    # single unavailable ticker (e.g. recently retired) doesn't block the run.
    if vendor == "fred" and input_file is None:
        missing_threshold = max(base_threshold, 1)
    else:
        missing_threshold = base_threshold
    validate_frame(
        canonical.sort_values(["dt_ingest", "series_id", "period", "vintage_id"]),
        schema_path=Path("data/schemas/macro_releases.schema.json"),
        critical_series=[v["internal_id"] for v in universe.get("macro_series", {}).values()],
        missing_threshold=missing_threshold,
    )

    part_key = f"dt={run_date}"
    dest, fp, rows = replace_partition(
        root=Path(base_path) / "silver",
        dataset="macro_releases",
        partition_key=part_key,
        df=canonical,
        sort_keys=["dt_ingest", "series_id", "period", "vintage_id"],
    )
    metadata.update_partition("silver.macro_releases", part_key, fp, rows, str(dest))

    changed_periods = sorted(pd.to_datetime(canonical["period"]).dt.strftime("%Y-%m").unique().tolist())
    if changed_periods:
        min_month = pd.Timestamp(changed_periods[0] + "-01")
        max_month = pd.Timestamp(run_date).to_period("M").to_timestamp()
        months = pd.period_range(min_month, max_month, freq="M").astype(str).tolist()
    else:
        months = []

    macro_fps = build_macro_asof(
        datastore_root=Path(base_path),
        months_to_rebuild=months,
        lags_path=Path("data/configs/lags.yaml"),
        schema_path=Path("data/schemas/macro_asof.schema.json"),
    )

    datapack = {
        "datapack_id": make_datapack_id(run_date, 1),
        "extraction_date": run_date,
        "person": person,
        "vendor_functions_used": [f"{vendor}.fetch"],
        "tickers_and_fields": {"series": series_ids},
        "frequencies": {"macro": "monthly"},
        "currency_assumptions": {},
        "lag_assumptions": yaml.safe_load(Path("data/configs/lags.yaml").read_text(encoding="utf-8")).get("series_lags", {}),
        "notes": "",
        "raw_file_hashes": {str(raw_file): sha256_file(raw_file)},
    }
    write_datapack_manifest(raw_dir / "manifest.json", datapack)

    end_ts = utc_now_iso()
    run_manifest = build_run_manifest(
        job="update_macro",
        run_id=run_id,
        start_ts=start_ts,
        end_ts=end_ts,
        inputs=[str(raw_file)],
        outputs=[str(dest)],
        partition_fingerprints={"silver.macro_releases/" + part_key: fp, **{"gold.macro_asof_monthly/" + k: v for k, v in macro_fps.items()}},
        row_counts={"silver.macro_releases/" + part_key: rows},
    )
    run_manifest_path = write_run_manifest(Path(base_path) / "meta", run_date, run_manifest)
    return {"run_manifest": str(run_manifest_path), "months_rebuilt": sorted(macro_fps.keys())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental macro ingestion job")
    parser.add_argument("--base-path", default="datastore")
    parser.add_argument("--vendor", default="fred")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--person", default="unknown")
    args = parser.parse_args()

    result = run_update_macro(
        base_path=Path(args.base_path),
        vendor=args.vendor,
        input_file=Path(args.input_file) if args.input_file else None,
        start=args.start,
        end=args.end,
        person=args.person,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
