"""
main.py — CLI entry point.

Example:
    python main.py --ticker SPY --strategy sma_crossover --start 2018-01-01

Compares the chosen strategy against a buy-and-hold benchmark on the same
ticker/period and prints a metrics table + saves an equity curve chart.
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_price_data
from strategies import STRATEGY_REGISTRY
from backtester import run_backtest
from metrics import summarize


def main():
    parser = argparse.ArgumentParser(description="Backtest a strategy on historical US stock/ETF data.")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol, e.g. SPY, AAPL")
    parser.add_argument("--strategy", type=str, default="sma_crossover", choices=STRATEGY_REGISTRY.keys())
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--commission_bps", type=float, default=5)
    parser.add_argument("--slippage_bps", type=float, default=5)
    parser.add_argument("--output", type=str, default="backtest_result.png")
    args = parser.parse_args()

    print(f"Fetching {args.ticker} data from {args.start} to {args.end or 'today'}...")
    df = get_price_data(args.ticker, args.start, args.end)

    strategy_fn = STRATEGY_REGISTRY[args.strategy]
    signal = strategy_fn(df)

    result = run_backtest(
        df, signal,
        initial_capital=args.capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )

    # Benchmark: buy and hold, same costs applied (a single entry cost)
    benchmark_signal = STRATEGY_REGISTRY["buy_and_hold"](df)
    benchmark = run_backtest(
        df, benchmark_signal,
        initial_capital=args.capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )

    strat_metrics = summarize(result.equity_curve, result.returns)
    bench_metrics = summarize(benchmark.equity_curve, benchmark.returns)

    print(f"\n=== {args.strategy} vs Buy & Hold — {args.ticker} ===")
    print(f"{'Metric':<24}{'Strategy':>14}{'Buy & Hold':>14}")
    for key in strat_metrics:
        print(f"{key:<24}{strat_metrics[key]:>14}{bench_metrics[key]:>14}")

    print(f"\nTotal trades executed: {len(result.trades)}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.equity_curve.index, result.equity_curve.values, label=args.strategy)
    ax.plot(benchmark.equity_curve.index, benchmark.equity_curve.values, label="Buy & Hold", alpha=0.7)
    ax.set_title(f"{args.ticker} — {args.strategy} vs Buy & Hold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"\nChart saved to {args.output}")


if __name__ == "__main__":
    main()
