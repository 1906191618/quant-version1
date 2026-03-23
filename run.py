import argparse
from quant_screener.train_rank import run_screen


def main():
    p = argparse.ArgumentParser(description="S&P500 screener (yfinance + sklearn) with sector diversification.")
    p.add_argument("--top", type=int, default=30, help="How many tickers to output")
    p.add_argument("--sector-cap", type=int, default=5, help="Max names per sector in final list")
    p.add_argument("--lookback", type=int, default=252, help="Lookback trading days for features (approx)")
    p.add_argument("--horizon", type=int, default=20, help="Prediction horizon in trading days (label)")
    p.add_argument("--min-price", type=float, default=5.0, help="Filter out stocks with last close < min-price")
    p.add_argument("--min-adv", type=float, default=10_000_000.0, help="Min 20d average dollar volume (ADV$)")
    p.add_argument("--require-uptrend", action="store_true", help="Require close > MA200 trend filter")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    run_screen(
        top_n=args.top,
        sector_cap=args.sector_cap,
        lookback=args.lookback,
        horizon=args.horizon,
        min_price=args.min_price,
        min_adv_dollars=args.min_adv,
        require_uptrend=args.require_uptrend,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
