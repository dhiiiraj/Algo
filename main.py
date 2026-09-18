import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ==========================================
# 1. API SETUP
# ==========================================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("ERROR: Missing Alpaca credentials.")
    sys.exit(1)

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

INDICATOR_SYMBOL = "QQQ"
TRADE_SYMBOL = "TQQQ"


# ==========================================
# 2. DATA FETCHING
# ==========================================
def fetch_weekly_bars(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetches weekly bars for a symbol and returns a clean, symbol-indexed DataFrame."""
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Week,
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX
        )
        bars = data_client.get_stock_bars(request)
    except Exception as e:
        print(f"ERROR: Failed to fetch {symbol} data from Alpaca API: {e}")
        sys.exit(1)

    if symbol not in bars.df.index.get_level_values(0):
        print(f"ERROR: No data returned for {symbol}.")
        sys.exit(1)

    return bars.df.loc[symbol].copy()


# ==========================================
# 3. INDICATOR LOGIC (Wilder's Smoothing RSI, matches live script)
# ==========================================
def calc_wilder_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    # Zero-loss protection: avg_loss == 0 means RSI should be 100, not NaN
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi


# ==========================================
# 4. SIGNAL LOGIC
# NOTE: this must exactly match the live script's buy_cond / sell_cond.
# If you change the live logic, update this function too -- a backtest
# is only meaningful if it evaluates the strategy that's actually trading.
# ==========================================
def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI"] = calc_wilder_rsi(df["close"], window=14)

    macd_bullish = df["MACD"] > df["MACD_Signal"]
    trend_bullish = df["EMA12"] > df["EMA26"]
    overbought = df["RSI"] >= 70

    # Mutually exclusive, matches the patched live script:
    #   buy  = momentum bullish AND trend bullish
    #   sell = trend flips bearish, OR momentum rolls over while overbought
    buy_cond = macd_bullish & trend_bullish
    sell_cond = (~trend_bullish) | (overbought & ~macd_bullish)

    df["signal"] = np.nan
    df.loc[buy_cond, "signal"] = 1
    df.loc[sell_cond, "signal"] = 0

    # Shift by 1 bar: a signal computed from this week's close can only be
    # acted on starting next week, avoiding lookahead bias.
    df["position"] = df["signal"].ffill().shift(1).fillna(0)
    return df


# ==========================================
# 5. PERFORMANCE STATS
# ==========================================
def compute_stats(returns: pd.Series) -> dict:
    """Computes cumulative return and max drawdown for a return series."""
    returns = returns.dropna()
    if returns.empty:
        return {"total_return": np.nan, "max_drawdown": np.nan}

    equity_curve = (1 + returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1

    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1
    max_drawdown = drawdown.min()

    return {"total_return": total_return, "max_drawdown": max_drawdown}


# ==========================================
# 6. MAIN BACKTEST RUNNER
# ==========================================
def run_backtest():
    start_dt = datetime(2018, 1, 1)
    end_dt = datetime.now()

    print("Fetching historical market data for backtest...")
    qqq_df = fetch_weekly_bars(INDICATOR_SYMBOL, start_dt, end_dt)
    tqqq_df = fetch_weekly_bars(TRADE_SYMBOL, start_dt, end_dt)

    MIN_BARS = 104  # ~2 years of weekly bars, enough for EMA26/RSI14 to converge
    if len(qqq_df) < MIN_BARS:
        print(f"ERROR: Insufficient {INDICATOR_SYMBOL} history ({len(qqq_df)} rows). Need at least {MIN_BARS}.")
        sys.exit(1)

    qqq_df = generate_signal(qqq_df)

    # Trade-side returns
    tqqq_df["pct_change"] = tqqq_df["close"].pct_change()

    # Merge strategy position onto TQQQ returns
    combined = tqqq_df.join(qqq_df[["position"]])

    # Guard against silent misalignment between the two symbols' bars.
    # A plain .prod() on a series with NaNs silently skips them, which
    # would understate how many weeks actually went into the result
    # without any warning.
    n_before = len(combined)
    combined = combined.dropna(subset=["position", "pct_change"])
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        print(f"WARNING: dropped {n_dropped} misaligned/NaN row(s) between {INDICATOR_SYMBOL} and {TRADE_SYMBOL} before computing returns.")

    combined["strat_return"] = combined["position"] * combined["pct_change"]

    # Turnover: count how many times position actually changed (entries + exits)
    position_changes = combined["position"].diff().fillna(0) != 0
    total_trades = int(position_changes.sum())

    # ==========================================
    # 7. WINDOWED PERFORMANCE: YTD and Since-2020
    # NOTE: figures below are frictionless (no spread, slippage, or
    # commission costs applied) -- treat them as a best-case upper bound,
    # not a realistic forward expectation.
    # ==========================================
    current_year = datetime.now().year
    windows = {
        f"YTD ({current_year})": combined[combined.index.year == current_year],
        "2020 to Present": combined[combined.index.year >= 2020],
    }

    print("\n================ BACKTEST RESULTS ================")
    print("(Frictionless: excludes spread, slippage, and commissions)\n")

    for label, window_df in windows.items():
        strat_stats = compute_stats(window_df["strat_return"])
        bench_stats = compute_stats(window_df["pct_change"])
        window_trades = int((window_df["position"].diff().fillna(0) != 0).sum())

        print(f"--- {label} ---")
        print(f"  Strategy Return:        {strat_stats['total_return'] * 100:6.2f}%")
        print(f"  Strategy Max Drawdown:  {strat_stats['max_drawdown'] * 100:6.2f}%")
        print(f"  Buy & Hold {TRADE_SYMBOL} Return: {bench_stats['total_return'] * 100:6.2f}%")
        print(f"  Buy & Hold Max Drawdown:{bench_stats['max_drawdown'] * 100:6.2f}%")
        print(f"  Position Changes:       {window_trades}")
        print()

    print(f"Total position changes over full backtest window: {total_trades}")
    print("==================================================")


if __name__ == "__main__":
    run_backtest()
