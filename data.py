"""
data.py — Historical price data loading with local CSV caching.

Uses yfinance to pull daily OHLCV data for US stocks/ETFs. Caches to
./data_cache/ so repeated backtests don't re-hit the network.
"""

import os
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")


def get_price_data(ticker: str, start: str, end: str = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a ticker between start and end dates (YYYY-MM-DD).
    Returns a DataFrame indexed by date with columns: Open, High, Low, Close, Volume.
    Adjusted close is used for Close so splits/dividends are handled.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end or 'latest'}.csv")

    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol and date range.")

    # yfinance sometimes returns MultiIndex columns for single tickers; flatten if so
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    return df


def get_multi_ticker_data(tickers: list, start: str, end: str = None) -> dict:
    """Fetch data for multiple tickers, returns {ticker: DataFrame}."""
    return {t: get_price_data(t, start, end) for t in tickers}
