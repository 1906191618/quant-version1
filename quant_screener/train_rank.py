from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from quant_screener.features import compute_features_for_ticker, compute_label_forward_return
from quant_screener.universe_sp500 import get_sp500_universe

FEATURE_COLS = [
    "mom20",
    "mom60",
    "mom120",
    "vol20",
    "trend_200",
    "mdd120",
    "ma50_over_ma200",
    "adv20_dollars",
]


@dataclass
class ScreenConfig:
    top_n: int = 30
    sector_cap: int = 5
    lookback: int = 252
    horizon: int = 20
    min_price: float = 5.0
    min_adv_dollars: float = 10_000_000.0
    require_uptrend: bool = False
    seed: int = 42


def _download_history(ticker: str) -> pd.DataFrame:
    # Use 2y to ensure enough data for MA200 and labels
    return yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )


def _build_training_rows(px: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = px.copy()
    df = df.dropna(subset=["Close", "Volume"])
    if len(df) < 260:
        return pd.DataFrame()

    close = df["Close"].astype(float)
    rets = close.pct_change()

    df["mom20"] = close / close.shift(20) - 1.0
    df["mom60"] = close / close.shift(60) - 1.0
    df["mom120"] = close / close.shift(120) - 1.0
    df["vol20"] = rets.rolling(20).std() * np.sqrt(252.0)

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    df["trend_200"] = close / ma200 - 1.0
    df["ma50_over_ma200"] = ma50 / ma200 - 1.0

    dollar_vol = close * df["Volume"].astype(float)
    df["adv20_dollars"] = dollar_vol.rolling(20).mean()

    rolling_max_120 = close.rolling(120).max()
    df["mdd120"] = close / rolling_max_120 - 1.0

    df["y"] = compute_label_forward_return(close, horizon)
    df = df.dropna(subset=FEATURE_COLS + ["y"])
    return df[FEATURE_COLS + ["y"]]


def _fit_model(train_df: pd.DataFrame, seed: int) -> Pipeline:
    model = GradientBoostingRegressor(random_state=seed)
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )
    X = train_df[FEATURE_COLS]
    y = train_df["y"]
    pipe.fit(X, y)
    return pipe


def _score_universe(model: Pipeline, feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    df["pred_fwd_ret"] = model.predict(df[FEATURE_COLS])
    return df


def _apply_filters(df: pd.DataFrame, cfg: ScreenConfig) -> pd.DataFrame:
    out = df.copy()

    out = out.dropna(subset=["last_close", "adv20_dollars"])
    out = out[out["last_close"] >= cfg.min_price]
    out = out[out["adv20_dollars"] >= cfg.min_adv_dollars]

    if cfg.require_uptrend:
        out = out.dropna(subset=["ma200"])
        out = out[out["last_close"] > out["ma200"]]

    return out


def _sector_diversified_top(df: pd.DataFrame, top_n: int, sector_cap: int) -> pd.DataFrame:
    picked = []
    counts: Dict[str, int] = {}

    for _, row in df.sort_values("pred_fwd_ret", ascending=False).iterrows():
        sec = row.get("sector", "Unknown")
        c = counts.get(sec, 0)
        if c >= sector_cap:
            continue
        picked.append(row)
        counts[sec] = c + 1
        if len(picked) >= top_n:
            break

    if not picked:
        return df.head(0)

    return pd.DataFrame(picked).reset_index(drop=True)


def run_screen(
    top_n: int = 30,
    sector_cap: int = 5,
    lookback: int = 252,  # kept for CLI compatibility; 2y download is used
    horizon: int = 20,
    min_price: float = 5.0,
    min_adv_dollars: float = 10_000_000.0,
    require_uptrend: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str]:
    cfg = ScreenConfig(
        top_n=top_n,
        sector_cap=sector_cap,
        lookback=lookback,
        horizon=horizon,
        min_price=min_price,
        min_adv_dollars=min_adv_dollars,
        require_uptrend=require_uptrend,
        seed=seed,
    )

    universe = get_sp500_universe()
    tickers = universe["ticker"].tolist()

    # Train on a subset for speed in Codespaces
    sample_n = min(120, len(tickers))
    sample = universe.sample(n=sample_n, random_state=seed)["ticker"].tolist()

    train_rows = []
    for t in sample:
        try:
            px = _download_history(t)
            if px is None or px.empty:
                continue
            td = _build_training_rows(px, horizon=cfg.horizon)
            if not td.empty:
                train_rows.append(td)
        except Exception:
            continue

    if not train_rows:
        raise RuntimeError("No training data built. Check network access or yfinance availability.")

    train_df = pd.concat(train_rows, axis=0).reset_index(drop=True)
    model = _fit_model(train_df, seed=cfg.seed)

    # Latest features for full universe
    feats = []
    for _, r in universe.iterrows():
        t = r["ticker"]
        try:
            px = _download_history(t)
            if px is None or px.empty or len(px) < 210:
                continue
            f = compute_features_for_ticker(px)
            f["ticker"] = t
            f["name"] = r["name"]
            f["sector"] = r["sector"]
            f["sub_industry"] = r["sub_industry"]
            feats.append(f)
        except Exception:
            continue

    if not feats:
        raise RuntimeError("No features computed. Check yfinance availability.")

    feature_df = pd.DataFrame(feats)
    for c in FEATURE_COLS:
        if c not in feature_df.columns:
            feature_df[c] = np.nan

    filtered = _apply_filters(feature_df, cfg)
    scored = _score_universe(model, filtered)
    final_df = _sector_diversified_top(scored, top_n=cfg.top_n, sector_cap=cfg.sector_cap)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "candidates.csv")

    cols = [
        "ticker",
        "name",
        "sector",
        "sub_industry",
        "pred_fwd_ret",
        "last_close",
        "adv20_dollars",
        "mom20",
        "mom60",
        "mom120",
        "vol20",
        "trend_200",
        "mdd120",
        "mdd252",
    ]
    cols = [c for c in cols if c in final_df.columns]
    final_df[cols].to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(final_df[cols].head(cfg.top_n).to_string(index=False))
    return final_df, out_path
