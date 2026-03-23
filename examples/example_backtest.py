"""
Example: Run multiple strategies on synthetic data and compare results.
"""

import matplotlib.pyplot as plt
import pandas as pd

from quant import (
    BollingerBandsStrategy,
    Backtester,
    DataLoader,
    MACDStrategy,
    MovingAverageCrossover,
    RSIStrategy,
)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Generate synthetic daily OHLCV data (2 years)
    # ------------------------------------------------------------------ #
    data = DataLoader.generate_synthetic(
        n_periods=504,
        start_price=100.0,
        mu=0.0003,
        sigma=0.012,
        start_date="2022-01-03",
        seed=42,
    )
    print(f"Generated {len(data)} bars from {data.index[0].date()} "
          f"to {data.index[-1].date()}\n")

    # ------------------------------------------------------------------ #
    # 2. Define strategies
    # ------------------------------------------------------------------ #
    strategies = {
        "MA Crossover (20/50)": MovingAverageCrossover(20, 50),
        "RSI (14, 30/70)": RSIStrategy(14, 30, 70),
        "MACD (12/26/9)": MACDStrategy(12, 26, 9),
        "Bollinger Bands (20, 2σ)": BollingerBandsStrategy(20, 2.0),
    }

    # ------------------------------------------------------------------ #
    # 3. Backtest each strategy
    # ------------------------------------------------------------------ #
    results = {}
    for name, strategy in strategies.items():
        bt = Backtester(strategy, initial_capital=100_000, commission=0.001)
        result = bt.run(data)
        results[name] = result
        print(f"--- {name} ---")
        result.print_summary()
        print()

    # ------------------------------------------------------------------ #
    # 4. Plot equity curves
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, result in results.items():
        equity = result.equity_curve
        ax.plot(data.index[: len(equity)], equity, label=name)

    # Buy-and-hold benchmark
    bnh = 100_000 * (data["close"] / data["close"].iloc[0])
    ax.plot(data.index, bnh, label="Buy & Hold", linestyle="--", color="gray")

    ax.set_title("Strategy Equity Curves vs Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curves.png", dpi=150)
    print("Equity curve chart saved to equity_curves.png")


if __name__ == "__main__":
    main()
