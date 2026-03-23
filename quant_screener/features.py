from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(series: pd.Series) -> float:
    """
    Max drawdown on a price series.
    Returns a negative number (e.g., -0.25).
    """
    if series is None or len(series) < 2:
        return np.nan
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    running_max = s.cummax()
    dd = (s / running_max) - 1.0
    return float(dd.min())


def compute_features_for_ticker(px: pd.DataFrame) -> dict:
    """
    px: DataFrame with columns: Open, High, Low, Close, Volume (from yfinance)
    Returns a dict of common factors computed on the last available date.
    """
    close = px["Close"].astype(float)
    vol = px["Volume"].astype(float)

    rets = close.pct_change()

    def mom(n: int) -> float:
        if len(close) < n + 1:
            return np.nan
        return float(close.iloc[-1] / close.iloc[-(n + 1)] - 1.0)

    def sma(n: int) -> float:
        if len(close) < n:
            return np.nan
        return float(close.rolling(n).mean().iloc[-1])

    def ann_vol(n: int) -> float:
        if len(rets) < n + 1:
            return np.nan
        return float(rets.rolling(n).std().iloc[-1] * np.sqrt(252.0))

    dollar_vol = close * vol
    adv20 = float(dollar_vol.rolling(20).mean().iloc[-1]) if len(dollar_vol) >= 20 else np.nan

    ma50 = sma(50)
    ma200 = sma(200)
    last_close = float(close.iloc[-1])

    feat = {
        "last_close": last_close,
        "adv20_dollars": adv20,
        "mom20": mom(20),
        "mom60": mom(60),
        "mom120": mom(120),
        "vol20": ann_vol(20),
        "ma50": ma50,
        "ma200": ma200,
        "trend_200": (last_close / ma200 - 1.0) if (ma200 and not np.isnan(ma200) and ma200 != 0) else np.nan,
        "mdd120": _max_drawdown(close.tail(120)),
        "mdd252": _max_drawdown(close.tail(252)),
    }

    if ma50 and ma200 and not (np.isnan(ma50) or np.isnan(ma200)) and ma200 != 0:
        feat["ma50_over_ma200"] = float(ma50 / ma200 - 1.0)
    else:
        feat["ma50_over_ma200"] = np.nan

    return feat


def compute_label_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """
    Forward return over horizon trading days aligned to each date.
    """
    close = close.astype(float)
    return close.shift(-horizon) / close - 1.0
