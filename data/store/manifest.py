from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_datapack_id(run_date: str, version: int = 1) -> str:
    return f"{run_date.replace('-', '')}_v{version}"


def write_datapack_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_run_manifest(
    job: str,
    run_id: str,
    start_ts: str,
    end_ts: str,
    inputs: list[str],
    outputs: list[str],
    partition_fingerprints: dict[str, str],
    row_counts: dict[str, int],
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "job": job,
        "run_id": run_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "inputs": inputs,
        "outputs": outputs,
        "partition_fingerprints": partition_fingerprints,
        "row_counts": row_counts,
        "errors": errors or [],
        "warnings": warnings or [],
    }


def write_run_manifest(meta_root: Path, run_date: str, manifest: dict[str, Any]) -> Path:
    out = Path(meta_root) / "runs" / f"dt={run_date}" / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
