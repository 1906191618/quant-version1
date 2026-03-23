from __future__ import annotations

import pandas as pd
import requests

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_universe() -> pd.DataFrame:
    """
    Returns a DataFrame with at least:
      - ticker (Yahoo format)
      - name
      - sector
      - sub_industry

    Source: Wikipedia table (good enough for a screener / interview demo).
    """
    tables = pd.read_html(WIKI_SP500_URL)
    if not tables:
        raise RuntimeError("Failed to read S&P 500 table from Wikipedia.")

    df = tables[0].copy()

    rename = {
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    }
    for k in rename:
        if k not in df.columns:
            raise RuntimeError(f"Unexpected Wikipedia table schema, missing column: {k}")

    df = df.rename(columns=rename)

    # Yahoo uses '-' instead of '.' for tickers like BRK.B -> BRK-B
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()

    df = df[["ticker", "name", "sector", "sub_industry"]].dropna()
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    # Optional: fail-fast if network is blocked
    try:
        requests.get("https://query1.finance.yahoo.com", timeout=5)
    except Exception:
        pass

    return df
