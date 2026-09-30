#!/usr/bin/env python3
"""Daily QQQ-signal / TQQQ-trade bot (Alpaca) with Google Sheets Execution Logging & Virtual Strategy Tracking.

Signal (computed on COMPLETED daily QQQ bars only):
    BUY  state : Close > SMA20  AND  SMA20 > SMA50
    SELL state : Close <= SMA50
    Otherwise  : keep the previous state (hysteresis).

Usage:
    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...
    export GOOGLE_SHEET_ID=...
    export GOOGLE_CREDENTIALS_B64='ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIs...'
    export STRATEGY_CAPITAL=5000.00          # Dedicated capital for this strategy

    python tqqq_daily_bot.py --dry-run      # paper account, log decision only
    python tqqq_daily_bot.py                # paper account, places orders
    python tqqq_daily_bot.py --live         # REAL MONEY, places orders
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

import exchange_calendars as xcal
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import pandas as pd

from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

# ==========================================
# 1. CONFIGURATION
# ==========================================
STRATEGY_ID = "TQQQ_DAILY_SMA_V1"  # Unique Strategy ID for trade tracking

INDICATOR_SYMBOL = "QQQ"
TRADE_SYMBOL = "TQQQ"

HISTORY_DAYS = 365  # daily history to pull for indicator warm-up
MIN_BARS = 60  # minimum completed daily bars required

MIN_TRADE_AMOUNT = 9000.0  # self-imposed floor in USD

# Hardcoded Strategy Capital (overridden by env var STRATEGY_CAPITAL if set)
STRATEGY_CAPITAL_ALLOCATION = float(os.getenv("STRATEGY_CAPITAL", "5000.00"))

POSITION_DUST_THRESHOLD = 0.001

ENTER_ONLY_ON_FRESH_BUY = False

ET = ZoneInfo("America/New_York")
MARKET_CLOSE_TIME = dtime(16, 0)

ORDER_POLL_TIMEOUT_SEC = 20

TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
    OrderStatus.DONE_FOR_DAY,
}
FAILED_STATUSES = {
    OrderStatus.REJECTED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
}

log = logging.getLogger("tqqq_daily_bot")


class StrategyError(Exception):
    """Expected, non-recoverable problem (bad data, blocked account, etc.)."""


# ==========================================
# 1b. GOOGLE SHEETS EXECUTION LOGGER
# ==========================================
class GoogleSheetExecutionLogger:
    """Logs trade executions and maintains strategy-level virtual positions."""

    def __init__(self):
        self.sheet_id = os.getenv("GOOGLE_SHEET_ID")
        self.creds_b64 = os.getenv("GOOGLE_CREDENTIALS_B64")
        self.worksheet = None

        if not self.sheet_id or not self.creds_b64:
            log.warning(
                "GOOGLE_SHEET_ID or GOOGLE_CREDENTIALS_B64 not set. Google"
                " Sheets logging disabled."
            )
            return

        try:
            raw_b64 = self.creds_b64.strip()

            if (raw_b64.startswith('"') and raw_b64.endswith('"')) or (
                raw_b64.startswith("'") and raw_b64.endswith("'")
            ):
                raw_b64 = raw_b64[1:-1].strip()

            missing_padding = len(raw_b64) % 4
            if missing_padding:
                raw_b64 += "=" * (4 - missing_padding)

            decoded_json = base64.b64decode(raw_b64).decode("utf-8").strip()
            creds_dict = json.loads(decoded_json)

            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            client = gspread.authorize(creds)
            self.worksheet = client.open_by_key(self.sheet_id).sheet1
            self._ensure_headers()
        except Exception as e:
            log.error("Failed to connect to Google Sheets: %s", e)
            self.worksheet = None

    def _ensure_headers(self):
        try:
            if not self.worksheet.get_all_values():
                headers = [
                    "timestamp_utc",
                    "strategy_id",
                    "broker_order_id",
                    "symbol",
                    "action",
                    "notional_usd",
                    "qty_shares",
                    "status",
                    "dry_run",
                    "reason",
                ]
                self.worksheet.append_row(headers)
        except Exception as e:
            log.warning("Could not verify or write Google Sheet headers: %s", e)

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
        if not self.worksheet:
            log.warning("Skipping Google Sheets log entry (client not connected).")
            return

        try:
            row = [
                datetime.now(timezone.utc).isoformat(),
                STRATEGY_ID,
                broker_order_id,
                symbol,
                action.upper(),
                round(notional_usd, 2) if notional_usd else 0.0,
                round(qty_shares, 4) if qty_shares else 0.0,
                status,
                dry_run,
                reason,
            ]
            self.worksheet.append_row(row)
            log.info(
                "Successfully logged trade execution [Strategy: %s | OrderID: %s] to"
                " Google Sheets.",
                STRATEGY_ID,
                broker_order_id,
            )
        except Exception as e:
            log.error("Failed to append trade record to Google Sheet: %s", e)

    def get_strategy_position(self, symbol: str) -> float:
        """Calculates net position for THIS strategy using filled logs from Google Sheets."""
        if not self.worksheet:
            log.warning("Sheet logging disabled; cannot deduce virtual position. Returning 0.0.")
            return 0.0

        try:
            records = self.worksheet.get_all_records()
            net_qty = 0.0
            for r in records:
                row_strat = str(r.get("strategy_id", "")).strip()
                row_symbol = str(r.get("symbol", "")).strip()
                row_status = str(r.get("status", "")).upper()
                row_action = str(r.get("action", "")).upper()

                if row_strat == STRATEGY_ID and row_symbol == symbol and row_status == "FILLED":
                    try:
                        shares = float(r.get("qty_shares", 0.0))
                    except (ValueError, TypeError):
                        shares = 0.0

                    if row_action == "BUY":
                        net_qty += shares
                    elif row_action == "SELL":
                        net_qty -= shares

            if 0 < net_qty < POSITION_DUST_THRESHOLD:
                return 0.0
            return max(net_qty, 0.0)

        except Exception as e:
            log.error("Error reading virtual position from Google Sheets: %s", e)
            return 0.0


# ==========================================
# 1c. MARKET HOLIDAY GUARD
# ==========================================
def is_market_open_today() -> bool:
    """Checks if NYSE/NASDAQ is scheduled for an active trading session today."""
    try:
        nyse = xcal.get_calendar("XNYS")
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        return nyse.is_session(today_str)
    except Exception as e:
        log.warning("Could not check market calendar (%s); proceeding as open.", e)
        return True


# ==========================================
# 2. DATA FETCHING
# ==========================================
def fetch_daily_bars(
    data_client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
    feed: DataFeed,
) -> pd.DataFrame:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
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


# ==========================================
# 3. INDICATORS & SIGNALS
# ==========================================
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["close"].rolling(window=20).mean()
    df["SMA50"] = df["close"].rolling(window=50).mean()

    buy_cond = (df["close"] > df["SMA20"]) & (df["SMA20"] > df["SMA50"])
    sell_cond = df["close"] <= df["SMA50"]

    df["signal"] = np.nan
    df.loc[buy_cond, "signal"] = 1.0
    df.loc[sell_cond, "signal"] = 0.0

    df["state"] = df["signal"].ffill()
    return df


# ==========================================
# 4. BROKER HELPERS
# ==========================================
def _status(order) -> str:
    return str(getattr(order.status, "value", order.status))


def wait_for_order(client: TradingClient, order):
    latest = order
    deadline = time.monotonic() + ORDER_POLL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        latest = client.get_order_by_id(order.id)
        if latest.status in TERMINAL_STATUSES:
            break
        time.sleep(2)

    if latest.status not in TERMINAL_STATUSES:
        log.warning(
            "Order %s did not reach a terminal state within %ds (status: %s).",
            order.id,
            ORDER_POLL_TIMEOUT_SEC,
            _status(latest),
        )
    return latest


def assert_order_ok(order, label: str) -> None:
    if order.status in FAILED_STATUSES:
        raise StrategyError(f"{label} order {order.id} ended as {_status(order)}")


# ==========================================
# 5. EXECUTION
# ==========================================
def execute(
    trading_client: TradingClient,
    target_long: bool,
    fresh_buy: bool,
    dry_run: bool,
    exec_logger: GoogleSheetExecutionLogger,
) -> str:
    log.info(
        "================ EXECUTION [Strategy ID: %s] ================",
        STRATEGY_ID,
    )

    account = trading_client.get_account()
    if account.trading_blocked or account.account_blocked:
        raise StrategyError("Account is blocked from trading; aborting.")

    asset = trading_client.get_asset(TRADE_SYMBOL)
    if not asset.tradable:
        raise StrategyError(f"{TRADE_SYMBOL} is not tradable on this account.")
    if not asset.fractionable:
        raise StrategyError(
            f"{TRADE_SYMBOL} is not fractionable; notional orders unsupported."
        )

    market_open = True
    try:
        market_open = bool(trading_client.get_clock().is_open)
    except Exception as e:
        log.warning("Could not check market clock: %s", e)
    if not market_open:
        log.warning("Market is CLOSED. Day orders will queue for the next open.")

    # 1. Check virtual position tracking specifically for STRATEGY_ID
    qty = exec_logger.get_strategy_position(TRADE_SYMBOL)
    log.info("Strategy [%s] Tracked Position: %s shares of %s", STRATEGY_ID, qty, TRADE_SYMBOL)

    open_orders = trading_client.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=[TRADE_SYMBOL]
        )
    )
    # Check if open order belongs to this strategy via client_order_id prefix
    strat_open_orders = [
        o for o in open_orders if str(o.client_order_id).startswith(STRATEGY_ID)
    ]
    if strat_open_orders:
        msg = (
            f"{len(strat_open_orders)} open {TRADE_SYMBOL} order(s) already pending for "
            f"strategy {STRATEGY_ID}; skipping to avoid duplicates."
        )
        log.warning(msg)
        return msg

    log.info(
        "Target state: %s | fresh buy: %s",
        "LONG" if target_long else "FLAT",
        fresh_buy,
    )

    # ---- BUY ----
    if target_long and qty == 0:
        if ENTER_ONLY_ON_FRESH_BUY and not fresh_buy:
            msg = (
                "NO ACTION: flat, state is long but not a fresh buy"
                " (ENTER_ONLY_ON_FRESH_BUY)."
            )
            log.info(msg)
            return msg

        # Trade amount set directly to hardcoded strategy capital allocation
        amount = round(STRATEGY_CAPITAL_ALLOCATION, 2)
        log.info(
            "BUY signal. Deploying full strategy allocation: $%.2f",
            amount,
        )

        if amount < MIN_TRADE_AMOUNT:
            msg = (
                f"Trade amount ${amount:.2f} below minimum ${MIN_TRADE_AMOUNT:.2f};"
                " not buying."
            )
            log.error(msg)
            return msg

        client_order_id = f"{STRATEGY_ID}_BUY_{uuid.uuid4().hex[:8]}"

        if dry_run:
            msg = f"[DRY RUN] Would BUY ${amount:.2f} of {TRADE_SYMBOL}."
            log.info("%s | Client Order ID: %s", msg, client_order_id)
            exec_logger.log_execution(
                broker_order_id=client_order_id,
                symbol=TRADE_SYMBOL,
                action="BUY",
                notional_usd=amount,
                qty_shares=0.0,
                status="DRY_RUN",
                dry_run=True,
                reason="Daily SMA20/50 Bullish Trend Signal",
            )
            return msg

        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=TRADE_SYMBOL,
                notional=amount,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
        )
        log.info(
            "Submitted BUY $%.2f %s | client_id=%s | order_id=%s | status=%s",
            amount,
            TRADE_SYMBOL,
            client_order_id,
            order.id,
            _status(order),
        )
        if market_open:
            order = wait_for_order(trading_client, order)
            log.info(
                "BUY order status: %s (filled_qty=%s)",
                _status(order),
                order.filled_qty,
            )

        assert_order_ok(order, "BUY")

        exec_logger.log_execution(
            broker_order_id=str(order.id),
            symbol=TRADE_SYMBOL,
            action="BUY",
            notional_usd=amount,
            qty_shares=float(order.filled_qty or 0.0),
            status=_status(order),
            dry_run=False,
            reason="Daily SMA20/50 Bullish Trend Signal",
        )
        return (
            f"BUY ${amount:.2f} of {TRADE_SYMBOL} | Order ID: {order.id} | Status:"
            f" {_status(order)}"
        )

    # ---- SELL ----
    elif not target_long and qty > 0:
        log.info(
            "SELL signal. Closing strategy position (%s shares of %s).", qty, TRADE_SYMBOL
        )
        client_order_id = f"{STRATEGY_ID}_SELL_{uuid.uuid4().hex[:8]}"

        if dry_run:
            msg = f"[DRY RUN] Would SELL {qty} shares of {TRADE_SYMBOL}."
            log.info("%s | Client Order ID: %s", msg, client_order_id)
            exec_logger.log_execution(
                broker_order_id=client_order_id,
                symbol=TRADE_SYMBOL,
                action="SELL",
                notional_usd=0.0,
                qty_shares=qty,
                status="DRY_RUN",
                dry_run=True,
                reason="Daily SMA50 Breakdown Signal",
            )
            return msg

        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=TRADE_SYMBOL,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
        )
        log.info(
            "Submitted SELL %s shares %s | client_id=%s | order_id=%s | status=%s",
            qty,
            TRADE_SYMBOL,
            client_order_id,
            order.id,
            _status(order),
        )
        if market_open:
            order = wait_for_order(trading_client, order)
            log.info(
                "SELL order status: %s (filled_qty=%s)",
                _status(order),
                order.filled_qty,
            )

        assert_order_ok(order, "SELL")

        exec_logger.log_execution(
            broker_order_id=str(order.id),
            symbol=TRADE_SYMBOL,
            action="SELL",
            notional_usd=0.0,
            qty_shares=float(order.filled_qty or qty),
            status=_status(order),
            dry_run=False,
            reason="Daily SMA50 Breakdown Signal",
        )
        return (
            f"SOLD strategy position of {qty} shares in {TRADE_SYMBOL} | Order ID:"
            f" {order.id} | Status: {_status(order)}"
        )

    # ---- HOLD / NO ACTION ----
    else:
        msg = (
            f"NO ACTION: target is {'LONG' if target_long else 'FLAT'} and strategy position"
            f" is {qty} shares."
        )
        log.info(msg)

        hold_id = f"HOLD_{uuid.uuid4().hex[:8]}"
        exec_logger.log_execution(
            broker_order_id=hold_id,
            symbol=TRADE_SYMBOL,
            action="HOLD",
            notional_usd=0.0,
            qty_shares=qty,
            status="NO_ACTION",
            dry_run=dry_run,
            reason=msg,
        )
        return msg


# ==========================================
# 6. EMAIL NOTIFICATION
# ==========================================
def send_email_report(subject: str, body: str) -> None:
    from email.mime.text import MIMEText
    import smtplib

    gmail_address = os.getenv("GMAIL_ADDRESS")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
    email_to = os.getenv("EMAIL_TO", gmail_address)

    if not gmail_address or not gmail_app_password:
        log.warning(
            "GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set; skipping email report."
        )
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = email_to

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(gmail_address, gmail_app_password)
            server.sendmail(
                gmail_address,
                [a.strip() for a in email_to.split(",")],
                msg.as_string(),
            )
        log.info("Email report sent to %s", email_to)
    except Exception as e:
        log.warning("Failed to send email report: %s", e)


# ==========================================
# 7. MAIN
# ==========================================
def setup_logging() -> None:
    log_file = os.getenv(
        "LOG_FILE",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tqqq_daily_bot.log"
        ),
    )
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    log.addHandler(stream)

    try:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError as e:
        log.warning("Could not open log file %s: %s", log_file, e)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily QQQ-signal / TQQQ-trade bot (Alpaca)."
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Trade the LIVE account (real money). Default is paper.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log the decision but place no orders.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    setup_logging()

    # ---- MARKET HOLIDAY GUARD ----
    if not is_market_open_today():
        today_stamp = datetime.now(ET).strftime("%Y-%m-%d")
        log.info(
            "Market is CLOSED today (%s). Exiting job safely with success code.",
            today_stamp,
        )
        return 0

    exec_logger = GoogleSheetExecutionLogger()
    mode = "PAPER" if not args.live else "*** LIVE ***"
    run_stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M %Z")

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        log.error(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables."
        )
        send_email_report(
            f"[{STRATEGY_ID}] ERROR ({run_stamp})",
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.",
        )
        return 1

    paper = not args.live
    log.info(
        "Starting Strategy [%s] | Capital=$%.2f | mode=%s | dry_run=%s",
        STRATEGY_ID,
        STRATEGY_CAPITAL_ALLOCATION,
        mode,
        args.dry_run,
    )

    try:
        feed = DataFeed(os.getenv("ALPACA_FEED", "iex").lower())
    except ValueError:
        log.error("Invalid ALPACA_FEED (use 'iex' or 'sip').")
        send_email_report(
            f"[{STRATEGY_ID}] ERROR ({run_stamp})",
            "Invalid ALPACA_FEED env var (use 'sip' or 'iex').",
        )
        return 1

    data_client = StockHistoricalDataClient(api_key, secret_key)
    trading_client = TradingClient(api_key, secret_key, paper=paper)

    try:
        now_utc = datetime.now(timezone.utc)
        start = now_utc - timedelta(days=HISTORY_DAYS)

        log.info(
            "Fetching %s daily bars (%s feed)...", INDICATOR_SYMBOL, feed.value
        )
        df = fetch_daily_bars(data_client, INDICATOR_SYMBOL, start, now_utc, feed)

        if len(df) < MIN_BARS:
            raise StrategyError(
                f"Only {len(df)} completed daily bars; need at least {MIN_BARS}."
            )

        df = compute_signals(df)
        state = df["state"].dropna()
        if state.empty:
            raise StrategyError("No valid signal generated from indicators.")

        latest_state = float(state.iloc[-1])
        prev_state = float(state.iloc[-2]) if len(state) > 1 else np.nan
        fresh_buy = latest_state == 1.0 and prev_state != 1.0

        last = df.iloc[-1]
        date_str = str(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1])
        state_str = "LONG (BUY)" if latest_state == 1.0 else "FLAT (SELL/OUT)"

        indicator_summary = (
            f"Strategy ID: {STRATEGY_ID}\n"
            f"Strategy Allocation: ${STRATEGY_CAPITAL_ALLOCATION:,.2f}\n"
            f"Date: {date_str}\n"
            f"  Close: {last['close']:.2f}\n"
            f"  SMA20: {last['SMA20']:.2f}\n"
            f"  SMA50: {last['SMA50']:.2f}\n"
            f"  Signal state: {state_str}\n"
            f"  Fresh flip today: {fresh_buy}"
        )
        log.info(
            "Date %s | close=%.2f SMA20=%.2f SMA50=%.2f | state=%s",
            date_str,
            last["close"],
            last["SMA20"],
            last["SMA50"],
            state_str,
        )

        outcome = execute(
            trading_client=trading_client,
            target_long=(latest_state == 1.0),
            fresh_buy=fresh_buy,
            dry_run=args.dry_run,
            exec_logger=exec_logger,
        )

        body = (
            f"Run time: {run_stamp}\n"
            f"Strategy ID: {STRATEGY_ID}\n"
            f"Mode: {mode}{' (dry-run)' if args.dry_run else ''}\n\n"
            f"{indicator_summary}\n\n"
            f"Execution outcome:\n  {outcome}\n"
        )
        send_email_report(
            f"[{STRATEGY_ID}] {state_str.split()[0]} - {date_str}", body
        )
        return 0

    except StrategyError as e:
        log.error("ABORTED: %s", e)
        send_email_report(
            f"[{STRATEGY_ID}] ABORTED ({run_stamp})",
            f"Run time: {run_stamp}\nMode: {mode}\n\nABORTED: {e}",
        )
        return 1
    except Exception as e:
        log.exception("Unhandled error")
        send_email_report(
            f"[{STRATEGY_ID}] UNHANDLED ERROR ({run_stamp})",
            f"Run time: {run_stamp}\nMode: {mode}\n\nUnhandled error: {e}",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
