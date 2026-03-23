"""
Performance metrics for quantitative trading backtests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Compute standard performance metrics from an equity curve.

    Args:
        equity_curve: Series of portfolio values over time.
        periods_per_year: Trading periods in a year used for annualisation
                          (default 252 for daily bars).
        risk_free_rate: Annualised risk-free rate as a decimal (default 0).
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.equity_curve = equity_curve.reset_index(drop=True).astype(float)
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        """Total return of the strategy as a decimal."""
        if len(self.equity_curve) < 2 or self.equity_curve.iloc[0] == 0:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1.0

    def annualized_return(self) -> float:
        """Compound annual growth rate (CAGR)."""
        n = len(self.equity_curve) - 1
        if n <= 0:
            return 0.0
        years = n / self.periods_per_year
        total = self.total_return()
        if total <= -1.0:
            return -1.0
        return (1.0 + total) ** (1.0 / years) - 1.0

    def daily_returns(self) -> pd.Series:
        """Period-over-period returns."""
        return self.equity_curve.pct_change().dropna()

    # ------------------------------------------------------------------
    # Risk-adjusted metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio."""
        dr = self.daily_returns()
        if dr.empty or dr.std() == 0:
            return 0.0
        rfr_daily = (1 + self.risk_free_rate) ** (1 / self.periods_per_year) - 1
        excess = dr - rfr_daily
        return float(excess.mean() / excess.std() * np.sqrt(self.periods_per_year))

    def sortino_ratio(self) -> float:
        """Annualised Sortino ratio (penalises only downside volatility)."""
        dr = self.daily_returns()
        if dr.empty:
            return 0.0
        rfr_daily = (1 + self.risk_free_rate) ** (1 / self.periods_per_year) - 1
        excess = dr - rfr_daily
        downside = excess[excess < 0]
        if downside.empty or downside.std() == 0:
            return 0.0
        return float(excess.mean() / downside.std() * np.sqrt(self.periods_per_year))

    def calmar_ratio(self) -> float:
        """Calmar ratio: annualised return / max drawdown (absolute value)."""
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return 0.0
        return self.annualized_return() / mdd

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def max_drawdown(self) -> float:
        """Maximum drawdown as a decimal (negative value)."""
        if len(self.equity_curve) < 2:
            return 0.0
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        return float(drawdown.min())

    def drawdown_series(self) -> pd.Series:
        """Full drawdown series."""
        rolling_max = self.equity_curve.cummax()
        return (self.equity_curve - rolling_max) / rolling_max

    # ------------------------------------------------------------------
    # Trade-level statistics
    # ------------------------------------------------------------------

    def win_rate(self, trades: pd.DataFrame) -> float:
        """Fraction of winning trades (pnl > 0)."""
        if trades.empty:
            return 0.0
        return float((trades["pnl"] > 0).mean())

    def profit_factor(self, trades: pd.DataFrame) -> float:
        """Ratio of gross profit to gross loss."""
        if trades.empty:
            return 0.0
        gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        gross_loss = trades.loc[trades["pnl"] < 0, "pnl"].abs().sum()
        if gross_loss == 0:
            return float("inf")
        return float(gross_profit / gross_loss)

    def average_trade(self, trades: pd.DataFrame) -> float:
        """Average P&L per trade."""
        if trades.empty:
            return 0.0
        return float(trades["pnl"].mean())

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary(self, trades: Optional[pd.DataFrame] = None) -> dict:
        """Return a dict with all key metrics."""
        result = {
            "total_return": self.total_return(),
            "annualized_return": self.annualized_return(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": self.max_drawdown(),
        }
        if trades is not None:
            result.update(
                {
                    "num_trades": len(trades),
                    "win_rate": self.win_rate(trades),
                    "profit_factor": self.profit_factor(trades),
                    "average_trade_pnl": self.average_trade(trades),
                }
            )
        return result
