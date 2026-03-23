"""
Trading strategies for quantitative trading.

Each strategy implements ``generate_signals(data)`` which returns a
``pandas.Series`` of integer signals:
    +1  = long (buy)
     0  = flat (no position)
    -1  = short (sell)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from .data import DataLoader


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with DatetimeIndex and OHLCV columns.

        Returns:
            Integer Series aligned to ``data.index`` with values in
            {-1, 0, +1}.
        """

    def _close(self, data: pd.DataFrame) -> pd.Series:
        """Return the close price series."""
        return data["close"]


class MovingAverageCrossover(BaseStrategy):
    """Moving Average Crossover strategy.

    Generates a long signal when the fast SMA crosses above the slow SMA and a
    short (or flat, configurable) signal when it crosses below.

    Args:
        fast_window: Lookback period for the fast moving average.
        slow_window: Lookback period for the slow moving average.
        allow_short: If True, generate -1 signals when fast < slow;
                     otherwise generate 0 (flat).
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
        allow_short: bool = False,
    ) -> None:
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be less than "
                f"slow_window ({slow_window})."
            )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._close(data)
        fast = DataLoader.sma(close, self.fast_window)
        slow = DataLoader.sma(close, self.slow_window)

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[fast > slow] = 1
        if self.allow_short:
            signals[fast < slow] = -1

        return signals.fillna(0).astype(int)


class RSIStrategy(BaseStrategy):
    """RSI mean-reversion strategy.

    Buys when RSI drops below ``oversold`` and sells when RSI rises above
    ``overbought``.

    Args:
        period: RSI calculation period.
        oversold: RSI level below which a long signal is generated.
        overbought: RSI level above which a short (or flat) signal is generated.
        allow_short: If True, generate -1 signals above overbought;
                     otherwise generate 0.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        allow_short: bool = False,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._close(data)
        rsi = DataLoader.rsi(close, self.period)

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[rsi < self.oversold] = 1
        if self.allow_short:
            signals[rsi > self.overbought] = -1

        return signals.fillna(0).astype(int)


class MACDStrategy(BaseStrategy):
    """MACD signal-line crossover strategy.

    Generates a long signal when the MACD line crosses above the signal line
    and a short (or flat) signal when it crosses below.

    Args:
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal EMA span.
        allow_short: If True, generate -1 signals when MACD < signal line.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        allow_short: bool = False,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._close(data)
        macd_df = DataLoader.macd(close, self.fast, self.slow, self.signal)

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[macd_df["macd"] > macd_df["signal"]] = 1
        if self.allow_short:
            signals[macd_df["macd"] < macd_df["signal"]] = -1

        return signals.fillna(0).astype(int)


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean-reversion strategy.

    Buys when price falls below the lower band and sells (or shorts) when
    price rises above the upper band.

    Args:
        window: Rolling window for the middle band (SMA).
        num_std: Number of standard deviations for the bands.
        allow_short: If True, generate -1 signals above the upper band.
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        allow_short: bool = False,
    ) -> None:
        self.window = window
        self.num_std = num_std
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = self._close(data)
        bands = DataLoader.bollinger_bands(close, self.window, self.num_std)

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[close < bands["lower"]] = 1
        if self.allow_short:
            signals[close > bands["upper"]] = -1

        return signals.fillna(0).astype(int)
