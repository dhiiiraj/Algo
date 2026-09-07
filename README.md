# Algo Backtester

A simple, honest backtesting system for US stocks/ETFs. Vectorized, includes
transaction costs and slippage, avoids lookahead bias (signals computed on
close, executed on next open), and always compares your strategy against a
buy-and-hold benchmark so you can tell if it's actually adding value.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --ticker SPY --strategy sma_crossover --start 2015-01-01
```

Options:
- `--ticker` — any US stock/ETF symbol (default: SPY)
- `--strategy` — one of `sma_crossover`, `momentum`, `rsi_mean_reversion`, `buy_and_hold`
- `--start` / `--end` — date range, YYYY-MM-DD (end defaults to today)
- `--capital` — starting capital (default: $100,000)
- `--commission_bps` / `--slippage_bps` — cost assumptions in basis points per trade
- `--output` — filename for the saved equity curve chart

Output: a metrics table (return, Sharpe, Sortino, max drawdown, Calmar,
win rate) printed to console, plus a PNG chart of your strategy vs
buy-and-hold.

## Project structure

- `data.py` — fetches & caches historical OHLCV data via yfinance
- `strategies.py` — signal-generating strategies (add your own here)
- `backtester.py` — vectorized backtest engine with costs/slippage
- `metrics.py` — performance metrics
- `main.py` — CLI to run it all

## Adding your own strategy

Write a function that takes a DataFrame with a `Close` column and returns a
`pandas.Series` of target positions (1 = long, 0 = flat, -1 = short), then
register it in `STRATEGY_REGISTRY` in `strategies.py`.

```python
def my_strategy(df, param=20):
    signal = ...
    return signal

STRATEGY_REGISTRY["my_strategy"] = my_strategy
```

## Honest limitations

- **This is a backtester, not a live trading system.** No broker connection,
  no real order execution.
- **Daily bars only.** Not suitable for intraday/HFT strategies.
- **Backtests overstate real-world performance by default.** Watch for
  overfitting — if you tune parameters until Sharpe looks great, you've
  probably just fit noise. Test out-of-sample before trusting any result.
- **Survivorship bias**: pulling data for tickers that still exist today
  ignores delisted/bankrupt companies, which inflates historical returns.
  Matters most for stock-picking strategies; less so for index ETFs like SPY.
- Costs are simplified (flat bps per trade) — real slippage varies with
  order size and liquidity.
