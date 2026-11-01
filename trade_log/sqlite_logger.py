import logging
import os
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger("algo.trade_log")

DEFAULT_DUST_THRESHOLD = 0.001
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trade_log.db"
)


class SQLiteExecutionLogger:
    """Logs trade executions and maintains strategy-level virtual positions.

    Drop-in replacement for GoogleSheetExecutionLogger: same method names and
    signatures, backed by a local SQLite file instead of a remote sheet. Each
    row is tagged with ``strategy_id`` so several strategies can share one
    Alpaca account and still compute their own positions independently.
    """

    def __init__(
        self, strategy_id: str, dust_threshold: float = DEFAULT_DUST_THRESHOLD
    ):
        self.strategy_id = strategy_id
        self.dust_threshold = dust_threshold
        self.db_path = os.getenv("TRADE_LOG_DB_PATH", DEFAULT_DB_PATH)

        try:
            self.conn = sqlite3.connect(self.db_path)
            self._ensure_schema()
        except sqlite3.Error as e:
            log.error("Failed to open local trade log DB %s: %s", self.db_path, e)
            self.conn = None

    def _ensure_schema(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                broker_order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                notional_usd REAL,
                qty_shares REAL,
                status TEXT,
                dry_run INTEGER,
                reason TEXT
            )
            """
        )
        self.conn.commit()

    def log_execution(
        self,
        broker_order_id: str,
        symbol: str,
        action: str,
        notional_usd: float,
        qty_shares: float,
        status: str,
        dry_run: bool,
        reason: str,
    ):
        if not self.conn:
            log.warning("Skipping trade log entry (local DB not connected).")
            return

        try:
            self.conn.execute(
                """
                INSERT INTO executions (
                    timestamp_utc, strategy_id, broker_order_id, symbol, action,
                    notional_usd, qty_shares, status, dry_run, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    self.strategy_id,
                    broker_order_id,
                    symbol,
                    action.upper(),
                    round(notional_usd, 2) if notional_usd else 0.0,
                    round(qty_shares, 4) if qty_shares else 0.0,
                    str(status).upper(),
                    int(dry_run),
                    reason,
                ),
            )
            self.conn.commit()
            log.info(
                "Successfully logged trade execution [Strategy: %s | OrderID: %s] to"
                " local trade log.",
                self.strategy_id,
                broker_order_id,
            )
        except sqlite3.Error as e:
            log.error("Failed to append trade record to local trade log: %s", e)

    def get_strategy_position(self, symbol: str) -> float:
        """Calculates net position for THIS strategy using filled logs in the DB."""
        if not self.conn:
            log.warning(
                "Trade log DB not connected; cannot deduce virtual position."
                " Returning 0.0."
            )
            return 0.0

        try:
            rows = self.conn.execute(
                """
                SELECT action, qty_shares, status FROM executions
                WHERE strategy_id = ? AND symbol = ?
                """,
                (self.strategy_id, symbol),
            ).fetchall()

            net_qty = 0.0
            for action, qty_shares, status in rows:
                row_status = str(status or "").strip().upper()
                is_filled = "FILLED" in row_status or row_status in {"DONE", "EXECUTED"}
                if not is_filled:
                    continue

                shares = float(qty_shares or 0.0)
                if action == "BUY":
                    net_qty += shares
                elif action == "SELL":
                    net_qty -= shares

            if 0 < net_qty < self.dust_threshold:
                return 0.0
            return max(net_qty, 0.0)

        except sqlite3.Error as e:
            log.error("Error reading virtual position from local trade log: %s", e)
            return 0.0

    def get_all_strategy_positions(self, tickers: list[str]) -> dict[str, float]:
        """Returns virtual positions for all strategy tickers."""
        return {t: self.get_strategy_position(t) for t in tickers}
