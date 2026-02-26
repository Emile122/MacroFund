from __future__ import annotations

import logging
import os
import time

import pandas as pd
import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)


def fetch_grouped_daily_prices(tickers: list[str], start: str, end: str, api_key: str | None = None) -> pd.DataFrame:
    key = api_key or os.getenv("MASSIVE_API_KEY", "")
    rows = []
    for ticker in tickers:
        safe_ticker = quote(ticker, safe="")  # encode ':' '.' etc safely
        url = f"https://api.massive.com/v2/aggs/ticker/{safe_ticker}/range/1/day/{start}/{end}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "apiKey": key,
        }
        try:
            resp = _request_with_backoff(url, params=params)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            logger.warning("Skipping unavailable Massive ticker=%s status=%s", ticker, status)
            continue
        payload = resp.json()
        for entry in payload.get("results", []):
            dt = pd.to_datetime(entry["t"], unit="ms", utc=True).date().isoformat()
            rows.append({"dt": dt, "ticker": ticker, "px_last": entry.get("c")})
    return pd.DataFrame(rows, columns=["dt", "ticker", "px_last"])


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
