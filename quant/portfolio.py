"""
Portfolio and position tracking for quantitative trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


@dataclass
class Trade:
    """Record of a completed trade (round-trip)."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    shares: float
    pnl: float

    @property
    def return_pct(self) -> float:
        """Percentage return of the trade."""
        if self.entry_price == 0:
            return 0.0
        return self.pnl / (self.entry_price * self.shares)


class Portfolio:
    """Track cash, holdings and open positions during a backtest.

    Args:
        initial_capital: Starting cash balance in currency units.
        commission: Per-trade commission as a fraction of trade value
                    (e.g., 0.001 = 0.1 %).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission = commission

        # Running state
        self._cash: float = initial_capital
        self._shares: float = 0.0          # positive = long, negative = short
        self._entry_price: float = 0.0
        self._entry_date: Optional[pd.Timestamp] = None
        self._side: int = 0                 # +1 long, -1 short, 0 flat

        # History
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def shares(self) -> float:
        return self._shares

    @property
    def is_flat(self) -> bool:
        return self._side == 0

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def buy(self, date: pd.Timestamp, price: float) -> None:
        """Open a long position using all available cash."""
        if not self.is_flat:
            raise RuntimeError("Cannot open a new position while in a trade.")
        cost = self._cash
        fee = cost * self.commission
        self._shares = (cost - fee) / price
        self._cash = 0.0
        self._entry_price = price
        self._entry_date = date
        self._side = 1

    def sell_short(self, date: pd.Timestamp, price: float) -> None:
        """Open a short position (proceeds added to cash)."""
        if not self.is_flat:
            raise RuntimeError("Cannot open a new position while in a trade.")
        # Borrow and sell `shares` at current price
        self._shares = self._cash / price
        proceeds = self._shares * price
        fee = proceeds * self.commission
        self._cash = proceeds - fee
        self._entry_price = price
        self._entry_date = date
        self._side = -1

    def close_position(self, date: pd.Timestamp, price: float) -> Trade:
        """Close the current open position and record the trade."""
        if self.is_flat:
            raise RuntimeError("No open position to close.")

        if self._side == 1:
            # Close long
            gross = self._shares * price
            fee = gross * self.commission
            pnl = gross - fee - (self._entry_price * self._shares)
            self._cash = gross - fee
        else:
            # Close short: we buy back at current price
            buy_cost = self._shares * price
            fee = buy_cost * self.commission
            pnl = (self._entry_price * self._shares) - buy_cost - fee
            self._cash = self._cash + pnl

        trade = Trade(
            entry_date=self._entry_date,  # type: ignore[arg-type]
            exit_date=date,
            side=self._side,
            entry_price=self._entry_price,
            exit_price=price,
            shares=self._shares,
            pnl=pnl,
        )
        self.trades.append(trade)

        # Reset position state
        self._shares = 0.0
        self._entry_price = 0.0
        self._entry_date = None
        self._side = 0

        return trade

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def market_value(self, current_price: float) -> float:
        """Total portfolio value at the current market price."""
        if self._side == 1:
            return self._cash + self._shares * current_price
        if self._side == -1:
            unrealized = (self._entry_price - current_price) * self._shares
            return self._cash + unrealized
        return self._cash

    def record_equity(self, current_price: float) -> float:
        """Record and return the current equity value."""
        equity = self.market_value(current_price)
        self.equity_curve.append(equity)
        return equity

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def equity_series(self) -> pd.Series:
        """Return the equity curve as a pandas Series."""
        return pd.Series(self.equity_curve, dtype=float)

    def trade_summary(self) -> pd.DataFrame:
        """Return all completed trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "entry_date",
                    "exit_date",
                    "side",
                    "entry_price",
                    "exit_price",
                    "shares",
                    "pnl",
                    "return_pct",
                ]
            )
        records = [
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "shares": t.shares,
                "pnl": t.pnl,
                "return_pct": t.return_pct,
            }
            for t in self.trades
        ]
        return pd.DataFrame(records)
