"""
Data loading and management for quantitative trading.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


class DataLoader:
    """Load and manage OHLCV (Open, High, Low, Close, Volume) market data."""

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    def __init__(self) -> None:
        self._data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load_csv(self, filepath: str, date_column: str = "date") -> pd.DataFrame:
        """Load OHLCV data from a CSV file.

        The CSV must contain at minimum a date column and a ``close`` column.
        Column names are normalised to lower-case.

        Args:
            filepath: Path to the CSV file.
            date_column: Name of the column containing date/datetime values.

        Returns:
            A ``pandas.DataFrame`` indexed by date with OHLCV columns.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, parse_dates=[date_column])
        df.columns = [c.lower().strip() for c in df.columns]
        date_col = date_column.lower().strip()

        if date_col not in df.columns:
            raise ValueError(
                f"Date column '{date_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.set_index(date_col).sort_index()

        if "close" not in df.columns:
            raise ValueError("CSV must contain a 'close' price column.")

        # Fill optional OHLCV columns with the close price / 0 if absent
        for col in ("open", "high", "low"):
            if col not in df.columns:
                df[col] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0.0

        self._data = df[self.REQUIRED_COLUMNS].astype(float)
        return self._data

    @staticmethod
    def generate_synthetic(
        n_periods: int = 252,
        start_price: float = 100.0,
        mu: float = 0.0005,
        sigma: float = 0.015,
        start_date: str = "2020-01-01",
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data using geometric Brownian motion.

        Args:
            n_periods: Number of daily bars to generate.
            start_price: Starting close price.
            mu: Daily drift (log-return mean).
            sigma: Daily volatility (log-return standard deviation).
            start_date: ISO date string for the first bar.
            seed: Random seed for reproducibility.

        Returns:
            A ``pandas.DataFrame`` with DatetimeIndex and OHLCV columns.
        """
        rng = np.random.default_rng(seed)
        log_returns = rng.normal(mu, sigma, n_periods)
        close = start_price * np.exp(np.cumsum(log_returns))
        close = np.concatenate([[start_price], close[:-1]])

        noise = rng.uniform(0.001, 0.015, n_periods)
        high = close * (1 + noise)
        low = close * (1 - noise)
        open_ = low + rng.uniform(0, 1, n_periods) * (high - low)
        volume = rng.integers(100_000, 1_000_000, n_periods).astype(float)

        dates = pd.date_range(start=start_date, periods=n_periods, freq="B")

        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

    # ------------------------------------------------------------------
    # Indicator helpers (shared across strategies)
    # ------------------------------------------------------------------

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple moving average."""
        return series.rolling(window=window).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential moving average."""
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """MACD indicator.

        Returns:
            DataFrame with columns ``macd``, ``signal``, and ``histogram``.
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram},
            index=series.index,
        )

    @staticmethod
    def bollinger_bands(
        series: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        """Bollinger Bands.

        Returns:
            DataFrame with columns ``middle``, ``upper``, and ``lower``.
        """
        middle = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return pd.DataFrame(
            {
                "middle": middle,
                "upper": middle + num_std * std,
                "lower": middle - num_std * std,
            },
            index=series.index,
        )
