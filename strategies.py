"""
strategies.py — Signal-generating strategies.

Each strategy function takes a price DataFrame (with a 'Close' column) and
returns a Series of target positions: 1 = fully long, 0 = flat, -1 = short.
Position sizing/leverage is handled by the backtester, not here.
"""

import pandas as pd
import numpy as np


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    """Classic moving-average crossover: long when fast MA > slow MA, else flat."""
    fast_ma = df["Close"].rolling(fast).mean()
    slow_ma = df["Close"].rolling(slow).mean()
    signal = pd.Series(0, index=df.index)
    signal[fast_ma > slow_ma] = 1
    return signal


def momentum(df: pd.DataFrame, lookback: int = 90, threshold: float = 0.0) -> pd.Series:
    """Time-series momentum: long if trailing return over `lookback` days beats threshold."""
    ret = df["Close"].pct_change(lookback)
    signal = pd.Series(0, index=df.index)
    signal[ret > threshold] = 1
    return signal


def rsi_mean_reversion(df: pd.DataFrame, period: int = 14, buy_below: int = 30, sell_above: int = 70) -> pd.Series:
    """Buy oversold (RSI < buy_below), exit when RSI > sell_above. Long-only."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    signal = pd.Series(np.nan, index=df.index)
    signal[rsi < buy_below] = 1
    signal[rsi > sell_above] = 0
    signal = signal.ffill().fillna(0)
    return signal


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Baseline benchmark: always long."""
    return pd.Series(1, index=df.index)


STRATEGY_REGISTRY = {
    "sma_crossover": sma_crossover,
    "momentum": momentum,
    "rsi_mean_reversion": rsi_mean_reversion,
    "buy_and_hold": buy_and_hold,
}
