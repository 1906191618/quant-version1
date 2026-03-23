"""
Backtesting engine for quantitative trading strategies.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .data import DataLoader
from .metrics import PerformanceMetrics
from .portfolio import Portfolio
from .strategy import BaseStrategy


class Backtester:
    """Event-driven backtester.

    Iterates bar-by-bar through historical data, generates signals using the
    supplied strategy and executes trades via a ``Portfolio`` instance.

    Args:
        strategy: Any ``BaseStrategy`` subclass.
        initial_capital: Starting cash in currency units.
        commission: Per-trade commission as a fraction of trade value.
        allow_short: Whether to take short positions on -1 signals.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        allow_short: bool = False,
    ) -> None:
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.allow_short = allow_short

        self._portfolio: Optional[Portfolio] = None
        self._result: Optional[BacktestResult] = None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame) -> "BacktestResult":
        """Execute the backtest.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.  Must contain at least a
                  ``close`` column.

        Returns:
            A :class:`BacktestResult` containing the equity curve, trades and
            performance metrics.
        """
        if data.empty:
            raise ValueError("data must not be empty.")
        if "close" not in data.columns:
            raise ValueError("data must contain a 'close' column.")

        portfolio = Portfolio(self.initial_capital, self.commission)
        signals = self.strategy.generate_signals(data)

        prev_signal = 0
        dates = data.index.tolist()
        closes = data["close"].tolist()

        for i, (date, close) in enumerate(zip(dates, closes)):
            signal = signals.iloc[i]

            # ---- position management ----
            if prev_signal != 0 and signal != prev_signal:
                # Close existing position before flipping or going flat
                if not portfolio.is_flat:
                    portfolio.close_position(date, close)

            if signal == 1 and portfolio.is_flat:
                portfolio.buy(date, close)
            elif signal == -1 and self.allow_short and portfolio.is_flat:
                portfolio.sell_short(date, close)

            portfolio.record_equity(close)
            prev_signal = signal

        # Close any open position at the last bar
        if not portfolio.is_flat:
            portfolio.close_position(dates[-1], closes[-1])

        self._portfolio = portfolio
        self._result = BacktestResult(
            data=data,
            signals=signals,
            portfolio=portfolio,
            periods_per_year=252,
        )
        return self._result


class BacktestResult:
    """Container for backtest output.

    Attributes:
        data: The original OHLCV data used in the backtest.
        signals: Strategy signals aligned to ``data``.
        portfolio: The ``Portfolio`` instance after the backtest.
        metrics: Computed :class:`PerformanceMetrics`.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        portfolio: Portfolio,
        periods_per_year: int = 252,
    ) -> None:
        self.data = data
        self.signals = signals
        self.portfolio = portfolio

        equity = portfolio.equity_series()
        self.metrics = PerformanceMetrics(equity, periods_per_year)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def equity_curve(self) -> pd.Series:
        """Portfolio equity curve as a Series."""
        return self.portfolio.equity_series()

    @property
    def trades(self) -> pd.DataFrame:
        """Completed trades as a DataFrame."""
        return self.portfolio.trade_summary()

    def summary(self) -> dict:
        """Return a performance summary dict."""
        return self.metrics.summary(self.trades)

    def print_summary(self) -> None:
        """Print a formatted performance summary to stdout."""
        s = self.summary()
        print("=" * 50)
        print("Backtest Performance Summary")
        print("=" * 50)
        print(f"  Total Return       : {s['total_return']:>10.2%}")
        print(f"  Annualized Return  : {s['annualized_return']:>10.2%}")
        print(f"  Sharpe Ratio       : {s['sharpe_ratio']:>10.4f}")
        print(f"  Sortino Ratio      : {s['sortino_ratio']:>10.4f}")
        print(f"  Calmar Ratio       : {s['calmar_ratio']:>10.4f}")
        print(f"  Max Drawdown       : {s['max_drawdown']:>10.2%}")
        if "num_trades" in s:
            print(f"  Num Trades         : {s['num_trades']:>10d}")
            print(f"  Win Rate           : {s['win_rate']:>10.2%}")
            print(f"  Profit Factor      : {s['profit_factor']:>10.4f}")
            print(f"  Avg Trade PnL      : {s['average_trade_pnl']:>10.2f}")
        print("=" * 50)
