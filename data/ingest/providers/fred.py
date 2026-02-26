from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def fetch_series(series_ids: list[str], start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch FRED series and return a wide Date-indexed frame."""
    key = api_key or os.getenv("FRED_API_KEY", "")
    rows = []
    for series_id in series_ids:
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
        url = "https://api.stlouisfed.org/fred/series/observations"
        try:
            resp = _request_with_backoff(url, params=params)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (400, 404):
                logger.warning("Skipping unavailable FRED series_id=%s status=%s", series_id, status)
                continue
            raise
        data = resp.json().get("observations", [])
        for item in data:
            rows.append({"dt": item.get("date"), "ticker": series_id, "px_last": item.get("value")})
    return pd.DataFrame(rows, columns=["dt", "ticker", "px_last"])


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _request_with_backoff(url: str, params: dict, max_retries: int = 3) -> requests.Response:
    delay = 1.0
    for i in range(max_retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code < 400:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504) and i < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
    raise RuntimeError("unreachable")
