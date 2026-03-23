"""Tests for trading strategies and the backtesting engine."""

import math
import unittest

import numpy as np
import pandas as pd

from quant.data import DataLoader
from quant.strategy import (
    BollingerBandsStrategy,
    MACDStrategy,
    MovingAverageCrossover,
    RSIStrategy,
)
from quant.backtest import Backtester
from quant.portfolio import Portfolio
from quant.metrics import PerformanceMetrics


def _make_data(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Return a small synthetic OHLCV DataFrame."""
    return DataLoader.generate_synthetic(n_periods=n, seed=seed)


class TestDataLoader(unittest.TestCase):
    def test_generate_synthetic_shape(self):
        data = DataLoader.generate_synthetic(n_periods=100, seed=1)
        self.assertEqual(len(data), 100)
        for col in ("open", "high", "low", "close", "volume"):
            self.assertIn(col, data.columns)

    def test_generate_synthetic_prices_positive(self):
        data = DataLoader.generate_synthetic(n_periods=50, seed=2)
        self.assertTrue((data["close"] > 0).all())
        self.assertTrue((data["high"] >= data["low"]).all())

    def test_sma(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = DataLoader.sma(s, 3)
        self.assertAlmostEqual(sma.iloc[2], 2.0)
        self.assertAlmostEqual(sma.iloc[4], 4.0)

    def test_rsi_range(self):
        data = DataLoader.generate_synthetic(n_periods=200, seed=3)
        rsi = DataLoader.rsi(data["close"], 14)
        valid = rsi.dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())

    def test_macd_columns(self):
        data = DataLoader.generate_synthetic(n_periods=200, seed=4)
        macd_df = DataLoader.macd(data["close"])
        for col in ("macd", "signal", "histogram"):
            self.assertIn(col, macd_df.columns)

    def test_bollinger_bands_columns(self):
        data = DataLoader.generate_synthetic(n_periods=100, seed=5)
        bb = DataLoader.bollinger_bands(data["close"])
        for col in ("middle", "upper", "lower"):
            self.assertIn(col, bb.columns)
        # Upper must be >= lower (drop NaN warmup rows)
        valid = bb.dropna()
        self.assertTrue((valid["upper"] >= valid["lower"]).all())


class TestMovingAverageCrossover(unittest.TestCase):
    def test_signal_values(self):
        data = _make_data()
        strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
        signals = strategy.generate_signals(data)
        self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))
        self.assertEqual(len(signals), len(data))

    def test_invalid_windows(self):
        with self.assertRaises(ValueError):
            MovingAverageCrossover(fast_window=50, slow_window=20)

    def test_no_short_by_default(self):
        data = _make_data()
        strategy = MovingAverageCrossover(20, 50, allow_short=False)
        signals = strategy.generate_signals(data)
        self.assertNotIn(-1, signals.values)

    def test_short_allowed(self):
        data = _make_data(seed=99)
        strategy = MovingAverageCrossover(10, 30, allow_short=True)
        signals = strategy.generate_signals(data)
        # With enough data there should be both +1 and -1 signals
        self.assertIn(1, signals.values)


class TestRSIStrategy(unittest.TestCase):
    def test_signal_values(self):
        data = _make_data()
        strategy = RSIStrategy()
        signals = strategy.generate_signals(data)
        self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))

    def test_no_short_by_default(self):
        data = _make_data()
        signals = RSIStrategy().generate_signals(data)
        self.assertNotIn(-1, signals.values)


class TestMACDStrategy(unittest.TestCase):
    def test_signal_values(self):
        data = _make_data()
        strategy = MACDStrategy()
        signals = strategy.generate_signals(data)
        self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))


class TestBollingerBandsStrategy(unittest.TestCase):
    def test_signal_values(self):
        data = _make_data()
        strategy = BollingerBandsStrategy()
        signals = strategy.generate_signals(data)
        self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))


class TestPortfolio(unittest.TestCase):
    def _ts(self, date_str: str) -> pd.Timestamp:
        return pd.Timestamp(date_str)

    def test_buy_and_close_long(self):
        p = Portfolio(initial_capital=10_000, commission=0.0)
        p.buy(self._ts("2023-01-01"), price=100.0)
        self.assertFalse(p.is_flat)
        trade = p.close_position(self._ts("2023-01-10"), price=110.0)
        self.assertTrue(p.is_flat)
        self.assertAlmostEqual(trade.pnl, 1_000.0, places=4)

    def test_sell_short_and_close(self):
        p = Portfolio(initial_capital=10_000, commission=0.0)
        p.sell_short(self._ts("2023-01-01"), price=100.0)
        trade = p.close_position(self._ts("2023-01-10"), price=90.0)
        self.assertGreater(trade.pnl, 0)

    def test_commission_reduces_pnl(self):
        p_no_fee = Portfolio(10_000, commission=0.0)
        p_fee = Portfolio(10_000, commission=0.01)
        ts = self._ts
        for p in (p_no_fee, p_fee):
            p.buy(ts("2023-01-01"), 100.0)
            p.close_position(ts("2023-01-02"), 100.0)
        self.assertGreater(p_no_fee.trades[0].pnl, p_fee.trades[0].pnl)

    def test_market_value_long(self):
        p = Portfolio(10_000, commission=0.0)
        p.buy(pd.Timestamp("2023-01-01"), 100.0)
        mv = p.market_value(120.0)
        self.assertAlmostEqual(mv, 12_000.0, places=2)

    def test_cannot_open_twice(self):
        p = Portfolio(10_000)
        p.buy(pd.Timestamp("2023-01-01"), 100.0)
        with self.assertRaises(RuntimeError):
            p.buy(pd.Timestamp("2023-01-02"), 110.0)


class TestPerformanceMetrics(unittest.TestCase):
    def _flat_equity(self, n: int = 100, value: float = 100_000.0) -> pd.Series:
        return pd.Series([value] * n)

    def _growing_equity(self, n: int = 252) -> pd.Series:
        return pd.Series([100_000 * (1.001 ** i) for i in range(n)])

    def test_total_return_flat(self):
        pm = PerformanceMetrics(self._flat_equity())
        self.assertAlmostEqual(pm.total_return(), 0.0)

    def test_total_return_growing(self):
        pm = PerformanceMetrics(self._growing_equity())
        self.assertGreater(pm.total_return(), 0.0)

    def test_max_drawdown_flat(self):
        pm = PerformanceMetrics(self._flat_equity())
        self.assertAlmostEqual(pm.max_drawdown(), 0.0)

    def test_max_drawdown_negative(self):
        equity = pd.Series([100, 90, 80, 85, 95, 100])
        pm = PerformanceMetrics(equity, periods_per_year=252)
        self.assertLess(pm.max_drawdown(), 0.0)

    def test_sharpe_ratio_zero_vol(self):
        pm = PerformanceMetrics(self._flat_equity())
        self.assertEqual(pm.sharpe_ratio(), 0.0)

    def test_summary_keys(self):
        pm = PerformanceMetrics(self._growing_equity())
        s = pm.summary()
        for key in ("total_return", "annualized_return", "sharpe_ratio",
                    "max_drawdown"):
            self.assertIn(key, s)


class TestBacktester(unittest.TestCase):
    def test_run_returns_result(self):
        data = _make_data(300)
        strategy = MovingAverageCrossover(10, 30)
        bt = Backtester(strategy, initial_capital=10_000, commission=0.001)
        result = bt.run(data)
        self.assertEqual(len(result.equity_curve), len(data))

    def test_equity_curve_starts_near_initial_capital(self):
        data = _make_data(300)
        strategy = MovingAverageCrossover(10, 30)
        bt = Backtester(strategy, initial_capital=10_000, commission=0.0)
        result = bt.run(data)
        # First value should be close to initial capital (no trade yet at bar 0)
        self.assertAlmostEqual(result.equity_curve.iloc[0], 10_000, delta=500)

    def test_summary_dict_keys(self):
        data = _make_data(300)
        strategy = MovingAverageCrossover(10, 30)
        bt = Backtester(strategy)
        result = bt.run(data)
        summary = result.summary()
        for key in ("total_return", "sharpe_ratio", "max_drawdown", "num_trades"):
            self.assertIn(key, summary)

    def test_empty_data_raises(self):
        strategy = MovingAverageCrossover(10, 30)
        bt = Backtester(strategy)
        with self.assertRaises(ValueError):
            bt.run(pd.DataFrame())

    def test_all_strategies_run(self):
        data = _make_data(300)
        for strategy in [
            MovingAverageCrossover(10, 30),
            RSIStrategy(),
            MACDStrategy(),
            BollingerBandsStrategy(),
        ]:
            bt = Backtester(strategy, initial_capital=10_000)
            result = bt.run(data)
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
