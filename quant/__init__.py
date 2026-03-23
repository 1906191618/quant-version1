"""
quant - A Python quantitative trading library.

Modules:
    data      - Data loading and management
    strategy  - Built-in trading strategies
    backtest  - Backtesting engine
    portfolio - Portfolio and position tracking
    metrics   - Performance metrics
"""

from .data import DataLoader
from .strategy import (
    MovingAverageCrossover,
    RSIStrategy,
    MACDStrategy,
    BollingerBandsStrategy,
)
from .backtest import Backtester
from .portfolio import Portfolio
from .metrics import PerformanceMetrics

__all__ = [
    "DataLoader",
    "MovingAverageCrossover",
    "RSIStrategy",
    "MACDStrategy",
    "BollingerBandsStrategy",
    "Backtester",
    "Portfolio",
    "PerformanceMetrics",
]
