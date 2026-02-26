from __future__ import annotations

import time

import pandas as pd
import requests


from urllib.parse import urlencode

def fetch_series(
    dataflow: str,
    start: str,
    end: str,
    series_key: str | None = None,
    *,
    fmt: str = "csvdata",
    params: dict | None = None,
) -> pd.DataFrame:
    if series_key is None:
        if "/" in dataflow:
            path = dataflow
        else:
            raise ValueError(
                "ECB SDMX requires a series key. "
                "Use fetch_series('EXR', start, end, series_key='M.USD.EUR.SP00.A') "
                "or pass dataflow='EXR/M.USD.EUR.SP00.A'."
            )
    else:
        path = f"{dataflow}/{series_key}"

    q = {"startPeriod": start, "endPeriod": end, "format": fmt}
    if params:
        q.update(params)

    url = f"https://data-api.ecb.europa.eu/service/data/{path}?{urlencode(q)}"
    resp = _request_with_backoff(url)
    return pd.read_csv(pd.io.common.StringIO(resp.text))

def _request_with_backoff(url: str, max_retries: int = 3) -> requests.Response:
    delay = 1.0
    for i in range(max_retries):
        resp = requests.get(url, timeout=30)
        if resp.status_code < 400:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504) and i < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
    raise RuntimeError("unreachable")
