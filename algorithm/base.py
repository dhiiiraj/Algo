"""What every algorithm implements.

An algorithm only DECIDES: given market data and its current positions, it
returns the positions it wants to hold. The engine (engine/executor.py) does
the selling, buying, dry-run handling, and Google Sheets logging.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient


@dataclass
class MarketContext:
    """Inputs handed to ``Strategy.decide``."""

    data_client: StockHistoricalDataClient  # pass to alpaca_broker.market_data helpers
    feed: DataFeed
    positions: dict[str, float]  # shares this strategy currently holds (from the sheet)


@dataclass
class Decision:
    """What the strategy wants to hold after this run.

    target:        {symbol: shares}. Symbols not listed are sold; an empty dict
                   means "be fully out". Symbols already held are kept as-is.
    reason:        Short reason written to the Google Sheet for each trade.
    email_subject: Text after "[STRATEGY_ID] " in the email subject.
    email_details: Indicator/branch summary included in the email body.
    """

    target: dict[str, float]
    reason: str
    email_subject: str
    email_details: str = ""
    extra: dict = field(default_factory=dict)


class Strategy(ABC):
    # ---- Override these in each algorithm ----
    strategy_id: str  # unique tag in the sheet + order IDs; never change once trading
    description: str
    universe: list[str]  # every symbol this strategy may ever hold
    default_qty: float  # shares per position if no env override is set
    min_trade_qty: float = 1.0
    dust_threshold: float = 0.001
    log_file_name: str = ""  # defaults to "<strategy_id>.log"

    @property
    def qty_env_var(self) -> str:
        return f"{self.strategy_id.upper()}_QTY"

    def target_qty(self) -> float:
        """Shares per position.

        Order of precedence: <STRATEGY_ID>_QTY, then the legacy shared
        STRATEGY_QTY / STRATEGY_CAPITAL, then ``default_qty``.
        """
        for name in (self.qty_env_var, "STRATEGY_QTY", "STRATEGY_CAPITAL"):
            value = os.getenv(name)
            if value:
                return float(value)
        return float(self.default_qty)

    def get_log_file_name(self) -> str:
        return self.log_file_name or f"{self.strategy_id.lower()}.log"

    @abstractmethod
    def decide(self, market: MarketContext) -> Decision:
        """Fetch data, compute signals, return the desired positions."""
