#!/usr/bin/env python3
"""Multi-Asset Rotation Bot (Alpaca) with Virtual Strategy Tracking & Quantity-Based Execution.

Strategy:
    Evaluates Composer-style 200SMA, 20SMA, and 10-period RSI signals across
    TQQQ, SOXL, SQQQ, BSV, UVXY, and TECL to select a single target ticker.
    Liquidates out-of-strategy positions and enters the target asset up to
    STRATEGY_QTY_ALLOCATION.

Usage:
    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...
    export GOOGLE_SHEET_ID=...
    export GOOGLE_CREDENTIALS_B64='ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIs...'
    export STRATEGY_QTY=10.0                 # Target share quantity for rotation

    python rotation_bot.py --dry-run          # Paper account, log decision only
    python rotation_bot.py                    # Paper account, places orders
    python rotation_bot.py --live             # REAL MONEY, places orders
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import exchange_calendars as xcal
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import pandas as pd

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
STRATEGY_ID = "COMPOSER_ROTATION_V1"  # Unique Strategy ID for trade tracking

TICKERS = ["TQQQ", "SOXL", "SQQQ", "BSV", "UVXY", "TECL"]

MIN_ROWS = 200  # Minimum daily history bars required for 200-day SMA
MIN_TRADE_QTY = 10  # Self-imposed minimum share quantity floor
POSITION_DUST_THRESHOLD = 0.001  # Ignore fractional share residuals

# Target share quantity (overridden by env var STRATEGY_QTY or STRATEGY_CAPITAL if set)
STRATEGY_QTY_ALLOCATION = float(
    os.getenv("STRATEGY_QTY", os.getenv("STRATEGY_CAPITAL", "15.0"))
)

ET = ZoneInfo("America/New_York")
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

log = logging.getLogger("rotation_bot")


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
                creds_dict["private_key"] = creds_dict["private_key"].replace(
                    "\\n", "\n"
                )

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = Credentials.from_service_account_info(
                creds_dict, scopes=scopes
            )
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
            log.warning(
                "Sheet logging disabled; cannot deduce virtual position. Returning 0.0."
            )
            return 0.0

        try:
            records = self.worksheet.get_all_records()
            net_qty = 0.0
            for r in records:
                row_strat = str(r.get("strategy_id", "")).strip()
                row_symbol = str(r.get("symbol", "")).strip()
                row_status = str(r.get("status", "")).upper()
                row_action = str(r.get("action", "")).upper()

                if (
                    row_strat == STRATEGY_ID
                    and row_symbol == symbol
                    and row_status == "FILLED"
                ):
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

    def get_all_strategy_positions(self, tickers: list[str]) -> dict[str, float]:
        """Returns virtual positions for all strategy tickers."""
        return {t: self.get_strategy_position(t) for t in tickers}


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
# 2. INDICATOR LOGIC & MARKET DATA
# ==========================================
def calc_composer_rsi10(series: pd.Series, window: int = 10) -> pd.Series:
    """Calculates RSI using a Simple Moving Average (SMA) of gains/losses."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi


def get_market_data(
    data_client: StockHistoricalDataClient, tickers: list[str], feed: DataFeed
) -> pd.DataFrame:
    """Fetches historical daily close prices required for 200 SMA calculation."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=400)

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

    if len(df) < MIN_ROWS:
        raise StrategyError(
            f"Insufficient price history ({len(df)} rows). Need at least {MIN_ROWS}."
        )

    return df


# ==========================================
# 3. COMPOSER SIGNAL ENGINE
# ==========================================
def evaluate_signal(df: pd.DataFrame) -> tuple[str, str]:
    """Evaluates the strategy logic using the latest daily price data."""
    df_calc = df.copy()
    df_calc["TQQQ_200SMA"] = df_calc["TQQQ"].rolling(200).mean()
    df_calc["TQQQ_20SMA"] = df_calc["TQQQ"].rolling(20).mean()

    df_calc["TQQQ_RSI10"] = calc_composer_rsi10(df_calc["TQQQ"], 10)
    df_calc["SOXL_RSI10"] = calc_composer_rsi10(df_calc["SOXL"], 10)
    df_calc["SQQQ_RSI10"] = calc_composer_rsi10(df_calc["SQQQ"], 10)
    df_calc["BSV_RSI10"] = calc_composer_rsi10(df_calc["BSV"], 10)

    latest = df_calc.iloc[-1]

    required_fields = [
        "TQQQ",
        "TQQQ_200SMA",
        "TQQQ_20SMA",
        "TQQQ_RSI10",
        "SOXL_RSI10",
        "SQQQ_RSI10",
        "BSV_RSI10",
    ]
    nan_fields = [f for f in required_fields if pd.isna(latest[f])]
    if nan_fields:
        raise StrategyError(f"Cannot evaluate signal, NaN values present in: {nan_fields}")

    tqqq_price = latest["TQQQ"]
    tqqq_200sma = latest["TQQQ_200SMA"]
    tqqq_rsi10 = latest["TQQQ_RSI10"]

    if tqqq_price > tqqq_200sma:
        if tqqq_rsi10 > 79:
            reason = (
                f"TQQQ > 200SMA and RSI10 ({tqqq_rsi10:.1f}) > 79 -> overbought -> UVXY"
            )
            log.info("BRANCH: %s", reason)
            return "UVXY", reason
        else:
            reason = (
                f"TQQQ > 200SMA and RSI10 ({tqqq_rsi10:.1f}) <= 79 -> uptrend intact -> TQQQ"
            )
            log.info("BRANCH: %s", reason)
            return "TQQQ", reason
    else:
        if tqqq_rsi10 < 31:
            reason = (
                f"TQQQ <= 200SMA and RSI10 ({tqqq_rsi10:.1f}) < 31 -> oversold bounce -> TECL"
            )
            log.info("BRANCH: %s", reason)
            return "TECL", reason
        else:
            if latest["SOXL_RSI10"] < 30:
                reason = (
                    f"downtrend, SOXL RSI10 ({latest['SOXL_RSI10']:.1f}) < 30 -> SOXL"
                )
                log.info("BRANCH: %s", reason)
                return "SOXL", reason
            else:
                if tqqq_price < latest["TQQQ_20SMA"]:
                    if latest["SQQQ_RSI10"] >= latest["BSV_RSI10"]:
                        reason = (
                            f"downtrend, TQQQ < 20SMA, SQQQ RSI10 ({latest['SQQQ_RSI10']:.1f})"
                            f" >= BSV RSI10 ({latest['BSV_RSI10']:.1f}) -> SQQQ"
                        )
                        log.info("BRANCH: %s", reason)
                        return "SQQQ", reason
                    else:
                        reason = (
                            f"downtrend, TQQQ < 20SMA, BSV RSI10 ({latest['BSV_RSI10']:.1f})"
                            f" > SQQQ RSI10 ({latest['SQQQ_RSI10']:.1f}) -> BSV"
                        )
                        log.info("BRANCH: %s", reason)
                        return "BSV", reason
                else:
                    reason = "downtrend on 200SMA but above 20SMA -> stay long TQQQ"
                    log.info("BRANCH: %s", reason)
                    return "TQQQ", reason


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
            "Order %s did not reach terminal state within %ds (status: %s).",
            order.id,
            ORDER_POLL_TIMEOUT_SEC,
            _status(latest),
        )
    return latest


def assert_order_ok(order, label: str) -> None:
    if order.status in FAILED_STATUSES:
        raise StrategyError(f"{label} order {order.id} ended as {_status(order)}")


# ==========================================
# 5. REBALANCING ENGINE (QUANTITY BASED)
# ==========================================
def rebalance(
    trading_client: TradingClient,
    target_ticker: str,
    reason: str,
    dry_run: bool,
    exec_logger: GoogleSheetExecutionLogger,
) -> str:
    """Liquidates out-of-strategy positions and enters target position by share quantity."""
    log.info("================ REBALANCE RUN STARTED ================")
    log.info("Strategy ID: %s | Target Symbol: %s", STRATEGY_ID, target_ticker)

    account = trading_client.get_account()
    if account.trading_blocked or account.account_blocked:
        raise StrategyError("Account is blocked from trading; aborting.")

    market_open = True
    try:
        market_open = bool(trading_client.get_clock().is_open)
    except Exception as e:
        log.warning("Could not check market clock: %s", e)
    if not market_open:
        log.warning("Market is CLOSED. Day orders will queue for next open.")

    # 1. Fetch virtual positions tracked for THIS strategy
    virtual_positions = exec_logger.get_all_strategy_positions(TICKERS)
    active_positions = {
        sym: qty for sym, qty in virtual_positions.items() if qty > POSITION_DUST_THRESHOLD
    }
    log.info("Strategy [%s] Tracked Positions: %s", STRATEGY_ID, active_positions)

    # 2. Check for pending strategy open orders
    open_orders = trading_client.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=TICKERS
        )
    )
    strat_open_orders = [
        o for o in open_orders if str(o.client_order_id).startswith(STRATEGY_ID)
    ]
    if strat_open_orders:
        msg = f"SKIP: {len(strat_open_orders)} open order(s) already pending for {STRATEGY_ID}."
        log.warning(msg)
        return msg

# 3. Check if already properly allocated
    current_target_qty = active_positions.get(target_ticker, 0.0)
    if current_target_qty > POSITION_DUST_THRESHOLD and len(active_positions) == 1:
        # Changed message here:
        msg = "No action taken as trend intact"
        log.info(msg)
        hold_id = f"HOLD_{uuid.uuid4().hex[:8]}"
        exec_logger.log_execution(
            broker_order_id=hold_id,
            symbol=target_ticker,
            action="HOLD",
            notional_usd=0.0,
            qty_shares=current_target_qty,
            status="NO_ACTION",
            dry_run=dry_run,
            reason=msg,
        )
        return msg

    # 4. Liquidate out-of-strategy holdings
    for symbol, qty in active_positions.items():
        if symbol != target_ticker:
            log.info("Liquidating non-target position: %s (%s shares)...", symbol, qty)
            client_order_id = f"{STRATEGY_ID}_SELL_{uuid.uuid4().hex[:8]}"

            if dry_run:
                log.info("[DRY RUN] Would SELL %s shares of %s", qty, symbol)
                exec_logger.log_execution(
                    broker_order_id=client_order_id,
                    symbol=symbol,
                    action="SELL",
                    notional_usd=0.0,
                    qty_shares=qty,
                    status="DRY_RUN",
                    dry_run=True,
                    reason=f"Rotation away from {symbol} to {target_ticker}",
                )
                continue

            order = trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=client_order_id,
                )
            )
            if market_open:
                order = wait_for_order(trading_client, order)

            assert_order_ok(order, f"SELL {symbol}")

            filled_qty = float(order.filled_qty or qty)
            filled_price = float(order.filled_avg_price or 0.0)
            est_notional = round(filled_qty * filled_price, 2) if filled_price else 0.0

            exec_logger.log_execution(
                broker_order_id=str(order.id),
                symbol=symbol,
                action="SELL",
                notional_usd=est_notional,
                qty_shares=filled_qty,
                status=_status(order),
                dry_run=False,
                reason=f"Rotation away from {symbol} to {target_ticker}",
            )

    # 5. Buy Target Position by Quantity
    buy_qty = round(STRATEGY_QTY_ALLOCATION, 4)
    if buy_qty < MIN_TRADE_QTY:
        msg = f"Target quantity {buy_qty} below minimum threshold {MIN_TRADE_QTY}; skipping buy."
        log.error(msg)
        return msg

    client_order_id = f"{STRATEGY_ID}_BUY_{uuid.uuid4().hex[:8]}"
    if dry_run:
        msg = f"[DRY RUN] Would BUY {buy_qty} shares of {target_ticker}."
        log.info(msg)
        exec_logger.log_execution(
            broker_order_id=client_order_id,
            symbol=target_ticker,
            action="BUY",
            notional_usd=0.0,
            qty_shares=buy_qty,
            status="DRY_RUN",
            dry_run=True,
            reason=reason,
        )
        return msg

    order = trading_client.submit_order(
        MarketOrderRequest(
            symbol=target_ticker,
            qty=buy_qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
    )
    log.info("Submitted BUY %s shares of %s | Order ID: %s", buy_qty, target_ticker, order.id)

    if market_open:
        order = wait_for_order(trading_client, order)

    assert_order_ok(order, f"BUY {target_ticker}")

    filled_qty = float(order.filled_qty or buy_qty)
    filled_price = float(order.filled_avg_price or 0.0)
    est_notional = round(filled_qty * filled_price, 2) if filled_price else 0.0

    exec_logger.log_execution(
        broker_order_id=str(order.id),
        symbol=target_ticker,
        action="BUY",
        notional_usd=est_notional,
        qty_shares=filled_qty,
        status=_status(order),
        dry_run=False,
        reason=reason,
    )

    msg = f"Rotated into {filled_qty} shares of {target_ticker} (~${est_notional:.2f}) | Order ID: {order.id}"
    log.info(msg)
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
        log.warning("GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set; skipping email report.")
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
# 7. MAIN RUNNER
# ==========================================
def setup_logging() -> None:
    log_file = os.getenv(
        "LOG_FILE",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "rotation_bot.log"
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
    p = argparse.ArgumentParser(description="Multi-Asset Rotation Bot (Alpaca)")
    p.add_argument("--live", action="store_true", help="Trade LIVE account (real money).")
    p.add_argument("--dry-run", action="store_true", help="Compute decision, place no orders.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    setup_logging()

    if not is_market_open_today():
        today_stamp = datetime.now(ET).strftime("%Y-%m-%d")
        log.info("Market is CLOSED today (%s). Exiting job safely.", today_stamp)
        return 0

    exec_logger = GoogleSheetExecutionLogger()
    mode = "PAPER" if not args.live else "*** LIVE ***"
    run_stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M %Z")

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        log.error("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.")
        send_email_report(
            f"[{STRATEGY_ID}] ERROR ({run_stamp})",
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.",
        )
        return 1

    paper = not args.live
    log.info(
        "Starting Strategy [%s] | Target Qty=%s shares | mode=%s | dry_run=%s",
        STRATEGY_ID,
        STRATEGY_QTY_ALLOCATION,
        mode,
        args.dry_run,
    )

    try:
        feed = DataFeed(os.getenv("ALPACA_FEED", "iex").lower())
    except ValueError:
        log.error("Invalid ALPACA_FEED (use 'iex' or 'sip').")
        return 1

    data_client = StockHistoricalDataClient(api_key, secret_key)
    trading_client = TradingClient(api_key, secret_key, paper=paper)

    try:
        market_data = get_market_data(data_client, TICKERS, feed)
        target_symbol, branch_reason = evaluate_signal(market_data)

        outcome = rebalance(
            trading_client=trading_client,
            target_ticker=target_symbol,
            reason=branch_reason,
            dry_run=args.dry_run,
            exec_logger=exec_logger,
        )

        body = (
            f"Run time: {run_stamp}\n"
            f"Strategy ID: {STRATEGY_ID}\n"
            f"Target Share Quantity: {STRATEGY_QTY_ALLOCATION} shares\n"
            f"Mode: {mode}{' (dry-run)' if args.dry_run else ''}\n\n"
            f"Target Symbol: {target_symbol}\n"
            f"Reason: {branch_reason}\n\n"
            f"Rebalance Outcome:\n  {outcome}\n"
        )
        send_email_report(f"[{STRATEGY_ID}] {target_symbol} - {run_stamp}", body)
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
