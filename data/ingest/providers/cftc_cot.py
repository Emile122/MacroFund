from __future__ import annotations

import csv
import time

import pandas as pd
import requests


_FIN_FUT_COLUMNS = [
    "Market_and_Exchange_Names",
    "As_of_Date_In_Form_YYMMDD",
    "Report_Date_as_YYYY-MM-DD",
    "CFTC_Contract_Market_Code",
    "CFTC_Market_Code",
    "CFTC_Region_Code",
    "CFTC_Commodity_Code",
    "Open_Interest_All",
    "Dealer_Positions_Long_All",
    "Dealer_Positions_Short_All",
    "Dealer_Positions_Spread_All",
    "Asset_Mgr_Positions_Long_All",
    "Asset_Mgr_Positions_Short_All",
    "Asset_Mgr_Positions_Spread_All",
    "Lev_Money_Positions_Long_All",
    "Lev_Money_Positions_Short_All",
    "Lev_Money_Positions_Spread_All",
    "Other_Rept_Positions_Long_All",
    "Other_Rept_Positions_Short_All",
    "Other_Rept_Positions_Spread_All",
    "Tot_Rept_Positions_Long_All",
    "Tot_Rept_Positions_Short_All",
    "NonRept_Positions_Long_All",
    "NonRept_Positions_Short_All",
    "Change_in_Open_Interest_All",
    "Change_in_Dealer_Long_All",
    "Change_in_Dealer_Short_All",
    "Change_in_Dealer_Spread_All",
    "Change_in_Asset_Mgr_Long_All",
    "Change_in_Asset_Mgr_Short_All",
    "Change_in_Asset_Mgr_Spread_All",
    "Change_in_Lev_Money_Long_All",
    "Change_in_Lev_Money_Short_All",
    "Change_in_Lev_Money_Spread_All",
    "Change_in_Other_Rept_Long_All",
    "Change_in_Other_Rept_Short_All",
    "Change_in_Other_Rept_Spread_All",
    "Change_in_Tot_Rept_Long_All",
    "Change_in_Tot_Rept_Short_All",
    "Change_in_NonRept_Long_All",
    "Change_in_NonRept_Short_All",
    "Pct_of_Open_Interest_All",
    "Pct_of_OI_Dealer_Long_All",
    "Pct_of_OI_Dealer_Short_All",
    "Pct_of_OI_Dealer_Spread_All",
    "Pct_of_OI_Asset_Mgr_Long_All",
    "Pct_of_OI_Asset_Mgr_Short_All",
    "Pct_of_OI_Asset_Mgr_Spread_All",
    "Pct_of_OI_Lev_Money_Long_All",
    "Pct_of_OI_Lev_Money_Short_All",
    "Pct_of_OI_Lev_Money_Spread_All",
    "Pct_of_OI_Other_Rept_Long_All",
    "Pct_of_OI_Other_Rept_Short_All",
    "Pct_of_OI_Other_Rept_Spread_All",
    "Pct_of_OI_Tot_Rept_Long_All",
    "Pct_of_OI_Tot_Rept_Short_All",
    "Pct_of_OI_NonRept_Long_All",
    "Pct_of_OI_NonRept_Short_All",
    "Traders_Tot_All",
    "Traders_Dealer_Long_All",
    "Traders_Dealer_Short_All",
    "Traders_Dealer_Spread_All",
    "Traders_Asset_Mgr_Long_All",
    "Traders_Asset_Mgr_Short_All",
    "Traders_Asset_Mgr_Spread_All",
    "Traders_Lev_Money_Long_All",
    "Traders_Lev_Money_Short_All",
    "Traders_Lev_Money_Spread_All",
    "Traders_Other_Rept_Long_All",
    "Traders_Other_Rept_Short_All",
    "Traders_Other_Rept_Spread_All",
    "Traders_Tot_Rept_Long_All",
    "Traders_Tot_Rept_Short_All",
    "Conc_Gross_LE_4_TDR_Long_All",
    "Conc_Gross_LE_4_TDR_Short_All",
    "Conc_Gross_LE_8_TDR_Long_All",
    "Conc_Gross_LE_8_TDR_Short_All",
    "Conc_Net_LE_4_TDR_Long_All",
    "Conc_Net_LE_4_TDR_Short_All",
    "Conc_Net_LE_8_TDR_Long_All",
    "Conc_Net_LE_8_TDR_Short_All",
    "Contract_Units",
    "CFTC_Contract_Market_Code_Quotes",
    "CFTC_Market_Code_Quotes",
    "CFTC_Commodity_Code_Quotes",
    "CFTC_SubGroup_Code",
    "FutOnly_or_Combined",
]

_LEGACY_REQUIRED = {
    "Market_and_Exchange_Names",
    "Report_Date_as_YYYY-MM-DD",
    "Noncommercial_Positions_Long_All",
    "Noncommercial_Positions_Short_All",
    "Open_Interest_All",
}


def _adapt_fin_fut_to_legacy(frame: pd.DataFrame) -> pd.DataFrame:
    # Approximate legacy "non-commercial" from disaggregated financial futures buckets.
    noncommercial_long = (
        pd.to_numeric(frame["Asset_Mgr_Positions_Long_All"], errors="coerce")
        + pd.to_numeric(frame["Lev_Money_Positions_Long_All"], errors="coerce")
        + pd.to_numeric(frame["Other_Rept_Positions_Long_All"], errors="coerce")
    )
    noncommercial_short = (
        pd.to_numeric(frame["Asset_Mgr_Positions_Short_All"], errors="coerce")
        + pd.to_numeric(frame["Lev_Money_Positions_Short_All"], errors="coerce")
        + pd.to_numeric(frame["Other_Rept_Positions_Short_All"], errors="coerce")
    )
    out = pd.DataFrame(
        {
            "Market_and_Exchange_Names": frame["Market_and_Exchange_Names"].astype(str),
            "Report_Date_as_YYYY-MM-DD": frame["Report_Date_as_YYYY-MM-DD"],
            "Noncommercial_Positions_Long_All": noncommercial_long,
            "Noncommercial_Positions_Short_All": noncommercial_short,
            "Open_Interest_All": pd.to_numeric(frame["Open_Interest_All"], errors="coerce"),
        }
    )
    return out.dropna(subset=["Report_Date_as_YYYY-MM-DD", "Market_and_Exchange_Names"])


def fetch_weekly_csv(url: str) -> pd.DataFrame:
    resp = _request_with_backoff(url)
    text = resp.text
    if not text.strip():
        return pd.DataFrame(columns=sorted(_LEGACY_REQUIRED))

    first_row = next(csv.reader([text.splitlines()[0]]), [])
    has_header = bool(first_row and "Report_Date_as_YYYY-MM-DD" in first_row)

    if has_header:
        frame = pd.read_csv(pd.io.common.StringIO(text))
    else:
        frame = pd.read_csv(pd.io.common.StringIO(text), header=None, names=_FIN_FUT_COLUMNS)

    if _LEGACY_REQUIRED.issubset(frame.columns):
        return frame

    needed_fin_fut = {
        "Market_and_Exchange_Names",
        "Report_Date_as_YYYY-MM-DD",
        "Open_Interest_All",
        "Asset_Mgr_Positions_Long_All",
        "Asset_Mgr_Positions_Short_All",
        "Lev_Money_Positions_Long_All",
        "Lev_Money_Positions_Short_All",
        "Other_Rept_Positions_Long_All",
        "Other_Rept_Positions_Short_All",
    }
    if needed_fin_fut.issubset(frame.columns):
        return _adapt_fin_fut_to_legacy(frame)

    return frame


def _request_with_backoff(url: str, max_retries: int = 3) -> requests.Response:
    delay = 1.0
    for i in range(max_retries):
        resp = requests.get(url, timeout=60)
        if resp.status_code < 400:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504) and i < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
    raise RuntimeError("unreachable")
