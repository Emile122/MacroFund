from __future__ import annotations

from pathlib import Path

import yaml


def load_universe(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def vendor_to_internal(
    vendor: str,
    ticker: str,
    universe_path: Path,
    strict: bool = False,
) -> str:
    """
    Map a vendor ticker to its canonical internal_id.

    Parameters
    ----------
    vendor        : Vendor name (e.g. 'fred', 'polygon').
    ticker        : Vendor-specific ticker or series key.
    universe_path : Path to universe.yaml.
    strict        : If True, raise KeyError for unknown tickers instead of
                    silently passing them through. Use strict=True in
                    production ingestion to catch mapping gaps early.
    """
    data = load_universe(universe_path)
    for _, entry in data.get("instruments", {}).items():
        if entry.get("vendor", {}).get(vendor) == ticker:
            return entry["internal_id"]
    for key, entry in data.get("macro_series", {}).items():
        if key == ticker:
            return entry["internal_id"]
    if strict:
        raise KeyError(
            f"Unknown ticker '{ticker}' for vendor '{vendor}'. "
            "Add it to data/configs/universe.yaml or pass strict=False to allow passthrough."
        )
    return ticker  # passthrough (legacy behaviour when strict=False)
