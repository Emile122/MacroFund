from __future__ import annotations

import importlib
import os
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Provider(Protocol):
    def fetch(self, dataset: str, start: str, end: str) -> pd.DataFrame:
        ...


class MockProvider:
    """
    Deterministic fixture provider for local tests.
    """

    def fetch(self, dataset: str, start: str, end: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts > end_ts:
            return pd.DataFrame()

        if dataset == "returns_daily":
            return self._returns(start_ts, end_ts)
        if dataset == "cot_z_weekly":
            return self._cot(start_ts, end_ts)
        if dataset == "policy_features":
            return self._policy(start_ts, end_ts)
        if dataset == "macro_asof_monthly":
            return self._macro(start_ts, end_ts)
        raise ValueError(f"MockProvider does not support dataset '{dataset}'")

    @staticmethod
    def _returns(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        dates = pd.bdate_range(start, end)
        instruments = ["SP500", "US10Y", "US2Y", "OIL", "GOLD", "USD_index"]
        rows: list[dict] = []
        for i, inst in enumerate(instruments):
            px = 100.0 + 5.0 * i + np.arange(len(dates)) * (0.05 + i * 0.01)
            prev = np.roll(px, 1)
            prev[0] = np.nan
            ret = px / prev - 1.0
            for dt, px_t, px_t_1, r in zip(dates, px, prev, ret):
                if np.isnan(r):
                    continue
                rows.append(
                    {
                        "dt": dt.date().isoformat(),
                        "instrument": inst,
                        "return": float(r),
                        "px_t": float(px_t),
                        "px_t_1": float(px_t_1),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _cot(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        weeks = pd.date_range(start, end, freq="W-FRI")
        contracts = ["BONDS", "EQUITY", "USD"]
        rows: list[dict] = []
        for w_idx, week_end in enumerate(weeks):
            week = f"{week_end.isocalendar().year:04d}-{week_end.isocalendar().week:02d}"
            for c_idx, contract in enumerate(contracts):
                rows.append(
                    {
                        "week": week,
                        "contract": contract,
                        "group": "All",
                        "net_pos": int(1000 + c_idx * 100 + w_idx * 5),
                        "zscore": float((w_idx - 10) / 5.0 + c_idx * 0.1),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _policy(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        dates = pd.bdate_range(start, end)
        rows = []
        for i, dt in enumerate(dates):
            cut3 = 0.45 + 0.05 * np.sin(i / 8)
            cut6 = 0.5 + 0.07 * np.sin(i / 13)
            rows.append(
                {
                    "dt": dt.date().isoformat(),
                    "implied_cut_prob_3m": float(np.clip(cut3, 0, 1)),
                    "implied_cut_prob_6m": float(np.clip(cut6, 0, 1)),
                    "term_front_slope": float(0.01 * np.sin(i / 5)),
                    "terminal_shift_30d": float(0.02 * np.cos(i / 7)),
                    "source": "mock",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _macro(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        months = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")
        rows: list[dict] = []
        for i, month in enumerate(months):
            month_end = month.to_timestamp(how="end").normalize()
            rows.append(
                {
                    "month_end": month_end.date().isoformat(),
                    "CPI_US": float(250 + i * 0.3),
                    "PAYROLLS_US": float(150 + i * 0.5),
                    "CPI_US__vintage_id": f"mock_{month.strftime('%Y%m')}",
                    "PAYROLLS_US__vintage_id": f"mock_{month.strftime('%Y%m')}",
                }
            )
        return pd.DataFrame(rows)


def provider_from_env(dataset: str | None = None) -> Provider:
    dataset_key = ""
    if dataset:
        dataset_key = "".join(ch if ch.isalnum() else "_" for ch in dataset.upper())
    provider_spec = (
        os.getenv(f"ATHENA_DATA_PROVIDER_{dataset_key}") if dataset_key else None
    ) or os.getenv("ATHENA_DATA_PROVIDER", "mock")

    if provider_spec == "mock":
        return MockProvider()

    if ":" not in provider_spec:
        raise ValueError(
            f"Invalid provider spec '{provider_spec}'. Use 'mock' or 'module.path:ClassName'."
        )

    module_name, class_name = provider_spec.split(":", 1)
    module = importlib.import_module(module_name)
    provider_cls = getattr(module, class_name)
    provider = provider_cls()
    if not isinstance(provider, Provider):
        if not hasattr(provider, "fetch"):
            raise TypeError(f"Provider '{provider_spec}' must implement fetch(dataset, start, end).")
    return provider


