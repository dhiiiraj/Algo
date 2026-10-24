import os
from typing import Optional

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient


def get_credentials() -> tuple[Optional[str], Optional[str]]:
    return os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")


def get_feed() -> DataFeed:
    """Reads ALPACA_FEED ('iex' default, or 'sip'). Raises ValueError if invalid."""
    return DataFeed(os.getenv("ALPACA_FEED", "iex").lower())


def create_clients(
    api_key: str, secret_key: str, paper: bool
) -> tuple[StockHistoricalDataClient, TradingClient]:
    data_client = StockHistoricalDataClient(api_key, secret_key)
    trading_client = TradingClient(api_key, secret_key, paper=paper)
    return data_client, trading_client
