#!/usr/bin/env python3
"""Weekly QQQ-signal / TQQQ-trade bot (Alpaca) with Google Sheets Execution Logging.

Signal (computed on COMPLETED weekly QQQ bars only):
    BUY  state : MACD > MACD signal  AND  EMA12 > EMA26
    SELL state : EMA12 <= EMA26  OR  (RSI >= 70 AND MACD <= MACD signal)
    Otherwise  : keep the previous state (hysteresis).

Usage:
    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...
    export GOOGLE_SHEET_ID=...
    export GOOGLE_CREDENTIALS_B64='ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIs...'

    python tqqq_weekly_bot.py --dry-run      # paper account, log decision only
    python tqqq_weekly_bot.py                # paper account, places orders
    python tqqq_weekly_bot.py --live         # REAL MONEY, places orders
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
STRATEGY_ID = "QQQ_MACD_WEEKLY_V1"  # Unique Strategy ID for trade tracking

INDICATOR_SYMBOL = "QQQ"
TRADE_SYMBOL = "TQQQ"

HISTORY_YEARS = 8  # weekly history to pull (more = better indicator warm-up)
MIN_BARS = 104  # minimum completed weekly bars required

CASH_DEPLOYMENT_PCT = 0.95  # fraction of available cash to deploy on a BUY
MIN_TRADE_AMOUNT = 10.0  # self-imposed floor in USD

POSITION_DUST_THRESHOLD = 0.001

ENTER_ONLY_ON_FRESH_BUY = False

ET = ZoneInfo("America/New_York")
WEEK_CLOSE_TIME = dtime(16, 30)

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

log = logging.getLogger("tqqq_bot")


class StrategyError(Exception):
    """Expected, non-recoverable problem (bad data, blocked account, etc.)."""


# ==========================================
# 1b. GOOGLE SHEETS EXECUTION LOGGER
# ==========================================
class GoogleSheetExecutionLogger:
    """Logs trade executions directly to a live Google Sheet using gspread."""

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

            # Strip outer quotes if passed with enclosing quotes
            if (raw_b64.startswith('"') and raw_b64.endswith('"')) or (
                raw_b64.startswith("'") and raw_b64.endswith("'")
            ):
                raw_b64 = raw_b64[1:-1].strip()

            # Fix Base64 padding if length is not a multiple of 4
            missing_padding = len(raw_b64) % 4
            if missing_padding:
                raw_b64 += "=" * (4 - missing_padding)

            # Decode Base64 string to raw JSON
            decoded_json = base64.b64decode(raw_b64).decode("utf-8").strip()

            # Parse JSON dictionary
            creds_dict = json.loads(decoded_json)

            # Fix escaped newlines in private key if present
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


# ==========================================
# 3. COMPLETED-WEEK LOGIC
# ==========================================
def week_monday(ts: pd.Timestamp) -> date:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_et = (ts + pd.Timedelta(hours=12)).tz_convert(ET)
    return ts_et.date() - timedelta(days=ts_et.weekday())


def last_complete_week_monday(now_et: datetime) -> date:
    this_monday = now_et.date() - timedelta(days=now_et.weekday())
    this_week_close = datetime.combine(
        this_monday + timedelta(days=4), WEEK_CLOSE_TIME, tzinfo=ET
    )
    return (
        this_monday if now_et >= this_week_close else this_monday - timedelta(days=7)
    )


def keep_completed_weeks(df: pd.DataFrame, now_et: datetime) -> pd.DataFrame:
    expected = last_complete_week_monday(now_et)
    mask = np.array([week_monday(ts) <= expected for ts in df.index])
    df = df.loc[mask].copy()

    if df.empty:
        raise StrategyError("No completed weekly bars available.")

    newest = week_monday(df.index[-1])
    if newest != expected:
        raise StrategyError(
            f"Stale data: newest completed bar is week of {newest}, expected week"
            f" of {expected}. Feed may be lagging; refusing to trade."
        )
    return df


# ==========================================
# 4. INDICATORS & SIGNALS
# ==========================================
def calc_wilder_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI"] = calc_wilder_rsi(df["close"], window=14)

    macd_bullish = df["MACD"] > df["MACD_Signal"]
    trend_bullish = df["EMA12"] > df["EMA26"]
    overbought = df["RSI"] >= 70

    buy_cond = macd_bullish & trend_bullish
    sell_cond = (~trend_bullish) | (overbought & ~macd_bullish)

    df["signal"] = np.nan
    df.loc[buy_cond, "signal"] = 1.0
    df.loc[sell_cond, "signal"] = 0.0

    df["state"] = df["signal"].ffill()
    return df


# ==========================================
# 5. BROKER HELPERS
# ==========================================
def get_position_qty(client: TradingClient, symbol: str) -> float:
    try:
        qty = float(client.get_open_position(symbol).qty)
    except APIError as e:
        if getattr(e, "status_code", None) == 404:
            return 0.0
        raise
    if 0 < qty < POSITION_DUST_THRESHOLD:
        log.warning("Ignoring residual fractional position dust (%s shares)", qty)
        return 0.0
    return qty


def deployable_cash(account) -> float:
    cash = float(account.cash)
    nmbp = getattr(account, "non_marginable_buying_power", None)
    available = cash if nmbp is None else min(cash, float(nmbp))
    return max(available, 0.0) * CASH_DEPLOYMENT_PCT


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
# 6. EXECUTION
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

    qty = get_position_qty(trading_client, TRADE_SYMBOL)
    log.info("Position: %s shares of %s", qty, TRADE_SYMBOL)
    if qty < 0:
        raise StrategyError(
            f"Unexpected SHORT position ({qty}) in {TRADE_SYMBOL}; aborting."
        )

    open_orders = trading_client.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=[TRADE_SYMBOL]
        )
    )
    if open_orders:
        msg = (
            f"{len(open_orders)} open {TRADE_SYMBOL} order(s) already pending; "
            "skipping to avoid duplicates."
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

        amount = round(deployable_cash(account), 2)
        log.info(
            "BUY signal. Deployable cash (%.0f%%): $%.2f",
            CASH_DEPLOYMENT_PCT * 100,
            amount,
        )

        if amount < MIN_TRADE_AMOUNT:
            msg = (
                f"Trade amount ${amount:.2f} below minimum ${MIN_TRADE_AMOUNT:.2f};"
                " not buying."
            )
            log.error(msg)
            return msg

        if dry_run:
            dry_id = f"DRY_RUN_ORD_{uuid.uuid4().hex[:8]}"
            msg = f"[DRY RUN] Would BUY ${amount:.2f} of {TRADE_SYMBOL}."
            log.info("%s | Simulated Order ID: %s", msg, dry_id)
            exec_logger.log_execution(
                broker_order_id=dry_id,
                symbol=TRADE_SYMBOL,
                action="BUY",
                notional_usd=amount,
                qty_shares=0.0,
                status="DRY_RUN",
                dry_run=True,
                reason="Weekly MACD/EMA Bullish Signal",
            )
            return msg

        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=TRADE_SYMBOL,
                notional=amount,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        log.info(
            "Submitted BUY $%.2f %s | id=%s status=%s",
            amount,
            TRADE_SYMBOL,
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
            reason="Weekly MACD/EMA Bullish Signal",
        )
        return (
            f"BUY ${amount:.2f} of {TRADE_SYMBOL} | Order ID: {order.id} | Status:"
            f" {_status(order)}"
        )

    # ---- SELL ----
    elif not target_long and qty > 0:
        log.info(
            "SELL signal. Closing entire %s position (%s shares).", TRADE_SYMBOL, qty
        )

        if dry_run:
            dry_id = f"DRY_RUN_ORD_{uuid.uuid4().hex[:8]}"
            msg = f"[DRY RUN] Would close position in {TRADE_SYMBOL} ({qty} shares)."
            log.info("%s | Simulated Order ID: %s", msg, dry_id)
            exec_logger.log_execution(
                broker_order_id=dry_id,
                symbol=TRADE_SYMBOL,
                action="SELL",
                notional_usd=0.0,
                qty_shares=qty,
                status="DRY_RUN",
                dry_run=True,
                reason="Weekly Trend Breakdown / Overbought Signal",
            )
            return msg

        order = trading_client.close_position(TRADE_SYMBOL)
        log.info(
            "Submitted CLOSE %s | id=%s status=%s",
            TRADE_SYMBOL,
            order.id,
            _status(order),
        )
        if market_open:
            order = wait_for_order(trading_client, order)
            log.info(
                "CLOSE order status: %s (filled_qty=%s)",
                _status(order),
                order.filled_qty,
            )

        assert_order_ok(order, "CLOSE")

        exec_logger.log_execution(
            broker_order_id=str(order.id),
            symbol=TRADE_SYMBOL,
            action="SELL",
            notional_usd=0.0,
            qty_shares=qty,
            status=_status(order),
            dry_run=False,
            reason="Weekly Trend Breakdown / Overbought Signal",
        )
        return (
            f"SOLD entire {TRADE_SYMBOL} position ({qty} shares) | Order ID:"
            f" {order.id} | Status: {_status(order)}"
        )

    # ---- HOLD / NO ACTION ----
    else:
        msg = (
            f"NO ACTION: target is {'LONG' if target_long else 'FLAT'} and position"
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
# 6b. EMAIL NOTIFICATION
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
            os.path.dirname(os.path.abspath(__file__)), "tqqq_weekly_bot.log"
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
        description="Weekly QQQ-signal / TQQQ-trade bot (Alpaca)."
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
        "Starting Strategy [%s] | mode=%s | dry_run=%s",
        STRATEGY_ID,
        mode,
        args.dry_run,
    )

    try:
        feed = DataFeed(os.getenv("ALPACA_FEED", "iex").lower())
    except ValueError:
        log.error("Invalid ALPACA_FEED (use 'iex' or 'sip').")
        send_email_report(
            f"[{STRATEGY_ID}] ERROR ({run_stamp})",
            "Invalid ALPACA_FEED env var (use 'iex' or 'sip').",
        )
        return 1

    data_client = StockHistoricalDataClient(api_key, secret_key)
    trading_client = TradingClient(api_key, secret_key, paper=paper)

    try:
        now_utc = datetime.now(timezone.utc)
        start = now_utc - timedelta(days=365 * HISTORY_YEARS)

        log.info(
            "Fetching %s weekly bars (%s feed)...", INDICATOR_SYMBOL, feed.value
        )
        df = fetch_weekly_bars(data_client, INDICATOR_SYMBOL, start, now_utc, feed)
        df = keep_completed_weeks(df, now_utc.astimezone(ET))

        if len(df) < MIN_BARS:
            raise StrategyError(
                f"Only {len(df)} completed bars; need at least {MIN_BARS}."
            )

        df = compute_signals(df)
        state = df["state"].dropna()
        if state.empty:
            raise StrategyError("No valid signal generated from indicators.")

        latest_state = float(state.iloc[-1])
        prev_state = float(state.iloc[-2]) if len(state) > 1 else np.nan
        fresh_buy = latest_state == 1.0 and prev_state != 1.0

        last = df.iloc[-1]
        week_str = str(week_monday(df.index[-1]))
        state_str = "LONG (BUY)" if latest_state == 1.0 else "FLAT (SELL/OUT)"

        indicator_summary = (
            f"Strategy ID: {STRATEGY_ID}\n"
            f"Week of {week_str}\n"
            f"  Close:         {last['close']:.2f}\n"
            f"  EMA12:         {last['EMA12']:.2f}\n"
            f"  EMA26:         {last['EMA26']:.2f}\n"
            f"  MACD:          {last['MACD']:.3f}\n"
            f"  MACD Signal:   {last['MACD_Signal']:.3f}\n"
            f"  RSI(14):       {last['RSI']:.1f}\n"
            f"  Signal state: {state_str}\n"
            f"  Fresh flip this week: {fresh_buy}"
        )
        log.info(
            "Week of %s | close=%.2f EMA12=%.2f EMA26=%.2f MACD=%.3f Sig=%.3f"
            " RSI=%.1f | state=%s",
            week_str,
            last["close"],
            last["EMA12"],
            last["EMA26"],
            last["MACD"],
            last["MACD_Signal"],
            last["RSI"],
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
            f"[{STRATEGY_ID}] {state_str.split()[0]} - {week_str}", body
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
