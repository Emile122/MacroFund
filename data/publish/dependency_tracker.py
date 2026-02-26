from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def _parse_date_key(partition_key: str, prefix: str = "dt") -> datetime:
    _, value = partition_key.split("=", 1)
    return datetime.strptime(value, "%Y-%m-%d")


def _to_date_key(value: datetime, prefix: str = "dt") -> str:
    return f"{prefix}={value.strftime('%Y-%m-%d')}"


def compute_minimal_recompute(
    changed: dict[str, list[str]],
    cot_lookback_weeks: int = 260,
    policy_depends_on_prices: bool = False,
) -> dict[str, list[str]]:
    """Compute impacted gold partitions from changed silver partitions."""
    out: dict[str, set[str]] = {
        "gold.returns_daily": set(),
        "gold.macro_asof_monthly": set(),
        "gold.cot_z_weekly": set(),
        "gold.policy_features": set(),
    }

    for key in changed.get("silver.market_prices", []):
        dt = _parse_date_key(key)
        out["gold.returns_daily"].add(_to_date_key(dt))
        out["gold.returns_daily"].add(_to_date_key(dt + timedelta(days=1)))
        if policy_depends_on_prices:
            out["gold.policy_features"].add(_to_date_key(dt))

    for key in changed.get("silver.macro_releases", []):
        part = key.split("=", 1)[1]
        month = part[:7]
        out["gold.macro_asof_monthly"].add(f"month={month}")

    cot_keys = changed.get("silver.cot_weekly", [])
    if cot_keys:
        weeks = sorted(k.split("=", 1)[1] for k in cot_keys)
        latest = weeks[-1]
        year, week = latest.split("-")
        base = datetime.strptime(f"{year}-W{week}-1", "%Y-W%W-%w")
        for i in range(cot_lookback_weeks):
            point = base - timedelta(weeks=i)
            out["gold.cot_z_weekly"].add(f"week={point.strftime('%Y-%W')}")

    for key in changed.get("silver.policy_curve", []):
        out["gold.policy_features"].add(key)

    return {k: sorted(v) for k, v in out.items() if v}
