"""TEMPLATE: copy this file to start a new algorithm.

    1. cp algorithm/_template.py algorithm/my_strategy.py
    2. Fill in the CONFIGURATION and decide() below.
    3. Register it in algorithm/__init__.py:
           from algorithm.my_strategy import MyStrategy
           STRATEGIES["my_strategy"] = MyStrategy()
    4. Try it:  python run.py my_strategy --dry-run

You only decide WHAT to hold. The engine handles selling what's not in the
target, buying what's missing, dry-run, Google Sheets logging, and email.
"""

import logging

import pandas as pd

from algorithm.base import Decision, MarketContext, Strategy
from alpaca_broker.market_data import fetch_daily_closes  # or fetch_weekly_bars
from common.errors import StrategyError

# ==========================================
# 1. CONFIGURATION
# ==========================================
STRATEGY_ID = "MY_STRATEGY_V1"  # Unique; tags sheet rows + orders. Never change once live.
TICKERS = ["SPY", "BIL"]  # Every symbol this strategy may hold
DEFAULT_QTY = 10.0  # Override with env MY_STRATEGY_V1_QTY
MIN_TRADE_QTY = 1.0
MIN_ROWS = 200  # Minimum daily bars needed for your indicators

log = logging.getLogger("algo.my_strategy")


# ==========================================
# 2. SIGNALS (pure functions: easy to test/backtest)
# ==========================================
def evaluate_signal(prices: pd.DataFrame) -> tuple[str, str]:
    """Return (symbol_to_hold, reason)."""
    spy = prices["SPY"]
    if spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]:
        return "SPY", "SPY above 200SMA"
    return "BIL", "SPY below 200SMA -> cash proxy"


# ==========================================
# 3. STRATEGY
# ==========================================
class MyStrategy(Strategy):
    strategy_id = STRATEGY_ID
    description = "Describe the strategy in one line."
    universe = TICKERS
    default_qty = DEFAULT_QTY
    min_trade_qty = MIN_TRADE_QTY

    def decide(self, market: MarketContext) -> Decision:
        prices = fetch_daily_closes(market.data_client, TICKERS, market.feed, MIN_ROWS)
        if prices.empty:
            raise StrategyError("No prices.")  # StrategyError aborts the run + emails you

        symbol, reason = evaluate_signal(prices)
        log.info("Signal: %s (%s)", symbol, reason)

        return Decision(
            target={symbol: self.target_qty()},  # {} means "hold nothing"
            reason=reason,
            email_subject=symbol,
            email_details=f"Target Symbol: {symbol}",
        )
