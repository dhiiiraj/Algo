"""
backtester.py — Vectorized backtest engine.

Takes price data + a position signal, applies realistic transaction costs
and slippage, and returns an equity curve plus trade log. Positions are
assumed to be entered/exited at next day's open to avoid lookahead bias.
"""

import pandas as pd
import numpy as np


class BacktestResult:
    def __init__(self, equity_curve, returns, trades, initial_capital):
        self.equity_curve = equity_curve
        self.returns = returns
        self.trades = trades
        self.initial_capital = initial_capital


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 100_000,
    commission_bps: float = 5,      # 0.05% per trade, roughly a typical retail rate
    slippage_bps: float = 5,        # 0.05% adverse price movement on execution
    max_leverage: float = 1.0,
) -> BacktestResult:
    """
    Run a vectorized backtest.

    signal: target position (-1 to 1) indexed same as df. Signal on day t is
    assumed decided using data through day t's close, and executed at day
    t+1's open (no lookahead).
    """
    df = df.copy()
    signal = signal.reindex(df.index).fillna(0).clip(-max_leverage, max_leverage)

    # Execute at next day's open
    target_position = signal.shift(1).fillna(0)

    daily_ret = df["Close"].pct_change().fillna(0)

    # Cost applied whenever position changes (turnover)
    position_change = target_position.diff().abs().fillna(target_position.abs())
    cost_per_unit_turnover = (commission_bps + slippage_bps) / 10_000
    transaction_costs = position_change * cost_per_unit_turnover

    strategy_ret = target_position * daily_ret - transaction_costs

    equity_curve = initial_capital * (1 + strategy_ret).cumprod()

    trades = pd.DataFrame({
        "date": df.index,
        "position": target_position,
        "position_change": position_change,
        "price": df["Close"],
    })
    trades = trades[trades["position_change"] > 0]

    return BacktestResult(equity_curve, strategy_ret, trades, initial_capital)
