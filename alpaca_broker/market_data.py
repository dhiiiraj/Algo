import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from common.errors import StrategyError

log = logging.getLogger("algo.alpaca.market_data")


def fetch_weekly_bars(
    data_client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    feed: DataFeed,
) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Week,
        start=start,
        end=end,
        feed=feed,
        adjustment=Adjustment.ALL,
    )
    try:
        bars = data_client.get_stock_bars(request)
    except APIError as e:
        raise StrategyError(f"API error fetching {symbol} bars: {e}") from e

    df = bars.df
    if df.empty or symbol not in df.index.get_level_values(0):
        raise StrategyError(f"No data returned for {symbol}.")

    return df.loc[symbol].sort_index().copy()


def fetch_daily_closes(
    data_client: StockHistoricalDataClient,
    tickers: list[str],
    feed: DataFeed,
    min_rows: int,
    lookback_days: int = 400,
) -> pd.DataFrame:
    """Daily close prices, one column per ticker, aligned on a common start date."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            feed=feed,
        )
        barset = data_client.get_stock_bars(request_params)
        bars = barset.df
    except APIError as e:
        raise StrategyError(f"API error fetching market data: {e}") from e

    if bars.empty:
        raise StrategyError("No market data returned for any ticker.")

    bars_reset = bars.reset_index()
    bars_reset["date"] = pd.to_datetime(bars_reset["timestamp"]).dt.date

    log.info("Per-symbol bar coverage in window:")
    coverage = bars_reset.groupby("symbol")["date"].agg(
        count="count", first="min", last="max"
    )
    for symbol, row in coverage.iterrows():
        log.info("  %s: %d bars, %s to %s", symbol, row["count"], row["first"], row["last"])

    missing = [t for t in tickers if t not in coverage.index]
    if missing:
        raise StrategyError(f"No data returned at all for ticker(s): {missing}")

    df = bars_reset.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    )

    first_valid_per_symbol = df.apply(lambda col: col.first_valid_index())
    if first_valid_per_symbol.isna().any():
        bad = first_valid_per_symbol[first_valid_per_symbol.isna()].index.tolist()
        raise StrategyError(f"Ticker(s) with zero valid close prices: {bad}")

    common_start = max(first_valid_per_symbol)
    df = df.loc[df.index >= common_start].ffill(limit=3)

    remaining_nan_cols = df.columns[df.isna().any()].tolist()
    if remaining_nan_cols:
        raise StrategyError(
            f"Gaps longer than 3 days remain in: {remaining_nan_cols}. Refusing to proceed."
        )

    if len(df) < min_rows:
        raise StrategyError(
            f"Insufficient price history ({len(df)} rows). Need at least {min_rows}."
        )

    return df
