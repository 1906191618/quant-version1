# quant-version1

A Python quantitative trading library with a modular backtesting engine, multiple built-in trading strategies, and performance metrics.

## Features

- **Data loading** – Load OHLCV data from CSV files or generate synthetic data using Geometric Brownian Motion
- **Technical indicators** – SMA, EMA, RSI, MACD, Bollinger Bands
- **Trading strategies** – Moving Average Crossover, RSI mean-reversion, MACD signal-line crossover, Bollinger Bands mean-reversion
- **Backtesting engine** – Bar-by-bar event-driven backtester with configurable commission and short-selling
- **Portfolio tracking** – Cash, position and trade management with unrealized P&L
- **Performance metrics** – Total return, CAGR, Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown, win rate, profit factor

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from quant import DataLoader, MovingAverageCrossover, Backtester

# Generate 2 years of synthetic daily data
data = DataLoader.generate_synthetic(n_periods=504, seed=42)

# Create a strategy and run a backtest
strategy = MovingAverageCrossover(fast_window=20, slow_window=50)
bt = Backtester(strategy, initial_capital=100_000, commission=0.001)
result = bt.run(data)

# Print a performance summary
result.print_summary()
```

### Sample output

```
==================================================
Backtest Performance Summary
==================================================
  Total Return       :     -7.35%
  Annualised Return  :     -3.75%
  Sharpe Ratio       :    -0.2479
  Max Drawdown       :    -15.25%
  Num Trades         :          7
  Win Rate           :     14.29%
==================================================
```

## Project Structure

```
quant/
├── data.py       – DataLoader: CSV loading, synthetic data, indicator helpers
├── strategy.py   – BaseStrategy and concrete strategy implementations
├── backtest.py   – Backtester and BacktestResult
├── portfolio.py  – Portfolio: cash/position/trade management
└── metrics.py    – PerformanceMetrics: return, risk, drawdown calculations
tests/
└── test_quant.py – Unit tests for all modules
examples/
└── example_backtest.py – Compare four strategies on synthetic data
```

## Available Strategies

| Class | Description | Key parameters |
|---|---|---|
| `MovingAverageCrossover` | Fast/slow SMA crossover | `fast_window`, `slow_window` |
| `RSIStrategy` | RSI mean-reversion | `period`, `oversold`, `overbought` |
| `MACDStrategy` | MACD signal-line crossover | `fast`, `slow`, `signal` |
| `BollingerBandsStrategy` | Price/band mean-reversion | `window`, `num_std` |

All strategies accept an `allow_short=True` parameter to enable short selling.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Running the Example

```bash
PYTHONPATH=. python examples/example_backtest.py
```
