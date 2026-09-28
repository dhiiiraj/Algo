#!/usr/bin/env python3
"""Multi-Asset Rotation Bot (Alpaca) with Google Sheets Execution Logging.

Strategy:
    Evaluates Composer-style 200SMA, 20SMA, and 10-period RSI signals across
    TQQQ, SOXL, SQQQ, BSV, UVXY, and TECL to select a single target ticker.
    Liquidates out-of-strategy positions and enters the target asset up to
    MAX_BUDGET.

Usage:
    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...
    export GOOGLE_SHEET_ID=...
    export GOOGLE_CREDENTIALS_B64='ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIs...'

    python rotation_bot.py           # Runs in PAPER mode
    python rotation_bot.py --live    # Runs in LIVE mode
"""

import argparse
import base64
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import exchange_calendars as xcal
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import pandas as pd

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
STRATEGY_ID = "COMPOSER_ROTATION_V1"  # Unique Strategy ID for trade tracking

# Collects human-readable lines during this run for the end-of-run email report.
RUN_LOG: list[str] = []


def _log(line: str) -> None:
    """Print + remember a line so it can go into the email report too."""
    print(line)
    RUN_LOG.append(line)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Asset Rotation Bot")
    parser.add_argument("--live", action="store_true", help="Run in LIVE trading mode")
    return parser.parse_args()


args = parse_args()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    _log("ERROR: Missing Alpaca API credentials.")
    _log("Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in the environment.")
    sys.exit(1)

# Set to False when going live with real money via command line flag
PAPER = not args.live

# SET YOUR HARDCODED CASH ALLOCATION LIMIT HERE (in USD)
MAX_BUDGET = 100.00

# Minimum rows of daily history required before trusting the 200-day SMA
MIN_ROWS = 200

TICKERS = ["TQQQ", "SOXL", "SQQQ", "BSV", "UVXY", "TECL"]

ET = ZoneInfo("America/New_York")

# Initialize Alpaca API Clients
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


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
            _log(
                "WARNING: GOOGLE_SHEET_ID or GOOGLE_CREDENTIALS_B64 not set."
                " Google Sheets logging disabled."
            )
            return

        try:
            raw_b64 = self.creds_b64.strip()

            # Strip outer quotes if passed with enclosing quotes
            if (raw_b64.startswith('"') and raw_b64.endswith('"')) or (
                raw_b64.startswith("'") and raw_b64.endswith("'")
            ):
                raw_b64 = raw_b64[1:-1].strip()

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
            _log(f"ERROR: Failed to connect to Google Sheets: {e}")
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
            _log(f"WARNING: Could not verify or write Google Sheet headers: {e}")

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
            _log("WARNING: Skipping Google Sheets log entry (client not connected).")
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
            _log(
                f"Successfully logged trade execution [Strategy: {STRATEGY_ID} |"
                f" OrderID: {broker_order_id}] to Google Sheets."
            )
        except Exception as e:
            _log(f"ERROR: Failed to append trade record to Google Sheet: {e}")


# Initialize Execution Logger
exec_logger = GoogleSheetExecutionLogger()


# ==========================================
# 1c. MARKET HOLIDAY GUARD
# ==========================================
def is_market_open_today() -> bool:
    """Checks if the NYSE/NASDAQ is open for trading today."""
    try:
        nyse = xcal.get_calendar("XNYS")
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        return nyse.is_session(today_str)
    except Exception as e:
        _log(f"WARNING: Could not check market calendar ({e}); proceeding.")
        return True


# ==========================================
# 2. INDICATOR LOGIC
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


# ==========================================
# 3. MARKET DATA FETCHING
# ==========================================
def get_market_data(tickers: list[str]) -> pd.DataFrame:
    """Fetches historical daily close prices required for 200 SMA calculation."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=400)

    try:
        feed_type = DataFeed(os.getenv("ALPACA_FEED", "iex").lower())
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            feed=feed_type,
        )
        barset = data_client.get_stock_bars(request_params)
        bars = barset.df
    except Exception as e:
        _log(f"ERROR: Failed to fetch market data from Alpaca API: {e}")
        sys.exit(1)

    if bars.empty:
        _log("ERROR: No market data returned for any ticker.")
        sys.exit(1)

    bars_reset = bars.reset_index()
    bars_reset["date"] = pd.to_datetime(bars_reset["timestamp"]).dt.date

    _log("Per-symbol bar coverage in this window:")
    coverage = bars_reset.groupby("symbol")["date"].agg(
        count="count", first="min", last="max"
    )
    for symbol, row in coverage.iterrows():
        _log(f"  {symbol}: {row['count']} bars, {row['first']} to {row['last']}")

    missing = [t for t in tickers if t not in coverage.index]
    if missing:
        _log(f"ERROR: No data returned at all for ticker(s): {missing}")
        sys.exit(1)

    df = bars_reset.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    )

    first_valid_per_symbol = df.apply(lambda col: col.first_valid_index())
    if first_valid_per_symbol.isna().any():
        bad = first_valid_per_symbol[first_valid_per_symbol.isna()].index.tolist()
        _log(f"ERROR: Ticker(s) with zero valid close prices in this window: {bad}")
        sys.exit(1)

    common_start = max(first_valid_per_symbol)
    df = df.loc[df.index >= common_start]

    df = df.ffill(limit=3)

    missing = [t for t in tickers if t not in df.columns]
    if missing:
        _log(f"ERROR: No data returned for ticker(s): {missing}")
        sys.exit(1)

    remaining_nan_cols = df.columns[df.isna().any()].tolist()
    if remaining_nan_cols:
        _log(
            f"ERROR: Gaps longer than 3 days remain in: {remaining_nan_cols}."
            " Refusing to proceed on stale/interpolated data."
        )
        sys.exit(1)

    if len(df) < MIN_ROWS:
        _log(
            f"ERROR: Insufficient price history returned ({len(df)} rows). Need at"
            f" least {MIN_ROWS}."
        )
        sys.exit(1)

    return df


# ==========================================
# 4. COMPOSER SIGNAL ENGINE
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
        _log(f"ERROR: Cannot evaluate signal, NaN values present in: {nan_fields}")
        sys.exit(1)

    tqqq_price = latest["TQQQ"]
    tqqq_200sma = latest["TQQQ_200SMA"]
    tqqq_rsi10 = latest["TQQQ_RSI10"]

    if tqqq_price > tqqq_200sma:
        if tqqq_rsi10 > 79:
            reason = (
                f"TQQQ > 200SMA and RSI10 ({tqqq_rsi10:.1f}) > 79 -> extremely"
                " overbought -> UVXY"
            )
            _log(f"BRANCH: {reason}")
            return "UVXY", reason
        else:
            reason = (
                f"TQQQ > 200SMA and RSI10 ({tqqq_rsi10:.1f}) <= 79 -> uptrend intact"
                " -> TQQQ"
            )
            _log(f"BRANCH: {reason}")
            return "TQQQ", reason
    else:
        if tqqq_rsi10 < 31:
            reason = (
                f"TQQQ <= 200SMA and RSI10 ({tqqq_rsi10:.1f}) < 31 -> oversold bounce"
                " -> TECL"
            )
            _log(f"BRANCH: {reason}")
            return "TECL", reason
        else:
            if latest["SOXL_RSI10"] < 30:
                reason = (
                    "downtrend, SOXL RSI10"
                    f" ({latest['SOXL_RSI10']:.1f}) < 30 -> semis oversold -> SOXL"
                )
                _log(f"BRANCH: {reason}")
                return "SOXL", reason
            else:
                if tqqq_price < latest["TQQQ_20SMA"]:
                    if latest["SQQQ_RSI10"] >= latest["BSV_RSI10"]:
                        reason = (
                            "downtrend, TQQQ < 20SMA, SQQQ RSI10 "
                            f"({latest['SQQQ_RSI10']:.1f}) >= BSV RSI10"
                            f" ({latest['BSV_RSI10']:.1f}) -> SQQQ"
                        )
                        _log(f"BRANCH: {reason}")
                        return "SQQQ", reason
                    else:
                        reason = (
                            "downtrend, TQQQ < 20SMA, BSV RSI10 "
                            f"({latest['BSV_RSI10']:.1f}) > SQQQ RSI10 "
                            f"({latest['SQQQ_RSI10']:.1f}) -> BSV"
                        )
                        _log(f"BRANCH: {reason}")
                        return "BSV", reason
                else:
                    reason = (
                        "downtrend on 200SMA but TQQQ still above 20SMA -> stay long ->"
                        " TQQQ"
                    )
                    _log(f"BRANCH: {reason}")
                    return "TQQQ", reason


# ==========================================
# 5. EXECUTION & ORDER SYNCHRONIZATION HELPERS
# ==========================================
def wait_for_liquidation(
    symbol: str, timeout: int = 15, poll_interval: float = 0.5
) -> bool:
    """Polls until the position is fully closed, or timeout (seconds) elapses."""
    elapsed = 0.0
    while elapsed < timeout:
        try:
            positions = {
                p.symbol: float(p.qty) for p in trading_client.get_all_positions()
            }
        except Exception as e:
            _log(f"WARNING: Failed to poll positions for {symbol}: {e}")
            return False
        if symbol not in positions:
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
    _log(f"WARNING: Liquidation of {symbol} did not confirm within {timeout}s.")
    return False


# ==========================================
# 6. REBALANCING ENGINE
# ==========================================
def rebalance(target_ticker: str, reason: str) -> str:
    """Liquidates out-of-strategy positions and enters target position up to MAX_BUDGET."""
    try:
        positions = {
            p.symbol: float(p.qty) for p in trading_client.get_all_positions()
        }
    except Exception as e:
        _log(f"ERROR: Failed to retrieve account positions: {e}")
        sys.exit(1)

    _log("--- REBALANCE RUN STARTED ---")
    _log(f"Strategy ID: {STRATEGY_ID}")
    _log(f"Mode: {'PAPER' if PAPER else 'LIVE'}")
    _log(f"Target Symbol: {target_ticker}")
    _log(f"Current Portfolio Positions: {positions}")

    target_qty = positions.get(target_ticker, 0)
    if target_qty > 0 and all(
        symbol == target_ticker for symbol in positions.keys()
    ):
        msg = f"NO ACTION: Already holding target asset ({target_ticker}) exclusively."
        _log(msg)

        hold_id = f"HOLD_{uuid.uuid4().hex[:8]}"
        exec_logger.log_execution(
            broker_order_id=hold_id,
            symbol=target_ticker,
            action="HOLD",
            notional_usd=0.0,
            qty_shares=target_qty,
            status="NO_ACTION",
            dry_run=False,
            reason=msg,
        )
        return msg

    try:
        symbols_in_play = list(set(list(positions.keys()) + [target_ticker]))
        open_orders_request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN, symbols=symbols_in_play
        )
        open_orders = trading_client.get_orders(filter=open_orders_request)
        if open_orders:
            msg = (
                f"SKIP: {len(open_orders)} open order(s) already exist on"
                f" {symbols_in_play}. Not rebalancing."
            )
            _log(msg)
            return msg
    except Exception as e:
        _log(f"ERROR: Failed to check open orders: {e}")
        sys.exit(1)

    try:
        account = trading_client.get_account()
        cash_before = float(account.cash)
    except Exception as e:
        _log(f"ERROR: Failed to read pre-liquidation cash balance: {e}")
        sys.exit(1)

    had_liquidations = any(symbol != target_ticker for symbol in positions.keys())

    liquidation_failed = False
    for symbol, qty in positions.items():
        if symbol != target_ticker:
            _log(f"Liquidating position: {symbol} ({qty} shares)...")
            try:
                close_order = trading_client.close_position(symbol)
                order_id = str(close_order.id)
                status = str(getattr(close_order.status, "value", close_order.status))

                exec_logger.log_execution(
                    broker_order_id=order_id,
                    symbol=symbol,
                    action="SELL",
                    notional_usd=0.0,
                    qty_shares=qty,
                    status=status,
                    dry_run=False,
                    reason=f"Rotation away from {symbol} to {target_ticker}",
                )
            except Exception as e:
                _log(f"ERROR: Failed to submit close position order for {symbol}: {e}")
                liquidation_failed = True
                continue

            confirmed = wait_for_liquidation(symbol, timeout=15)
            if not confirmed:
                liquidation_failed = True

    if liquidation_failed:
        msg = (
            "ERROR: One or more liquidations failed or timed out. Aborting before"
            " entering a new position."
        )
        _log(msg)
        sys.exit(1)

    available_cash = cash_before
    if had_liquidations:
        for attempt in range(10):
            try:
                acct = trading_client.get_account()
                available_cash = float(acct.cash)
                if available_cash > cash_before:
                    break
            except Exception as e:
                _log(f"WARNING: Failed to read account cash balance: {e}")
            time.sleep(1)
        else:
            _log(
                "WARNING: Cash balance did not rise above pre-liquidation baseline"
                f" (${cash_before:.2f}) after 10s; proceeding with last known value"
                f" ${available_cash:.2f}."
            )

    target_spend = min(MAX_BUDGET, available_cash)
    notional_amount = round(target_spend * 0.995, 2)
    min_order_limit = 10.00  # Alpaca dollar limit for fractional orders

    if notional_amount >= min_order_limit:
        market_order_data = MarketOrderRequest(
            symbol=target_ticker,
            notional=notional_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        try:
            buy_order = trading_client.submit_order(order_data=market_order_data)
            order_id = str(buy_order.id)
            status = str(getattr(buy_order.status, "value", buy_order.status))

            exec_logger.log_execution(
                broker_order_id=order_id,
                symbol=target_ticker,
                action="BUY",
                notional_usd=notional_amount,
                qty_shares=0.0,
                status=status,
                dry_run=False,
                reason=reason,
            )

            msg = (
                f"Successfully submitted Market BUY order: ${notional_amount} of"
                f" {target_ticker} | Order ID: {order_id}"
            )
            _log(msg)
            return msg
        except Exception as e:
            _log(f"ERROR: Failed to submit order for {target_ticker}: {e}")
            sys.exit(1)
    else:
        msg = (
            f"Execution skipped: Calculated spend (${notional_amount}) is below"
            f" Alpaca's ${min_order_limit} minimum requirement."
        )
        _log(msg)
        return msg


# ==========================================
# 6b. EMAIL NOTIFICATION
# ==========================================
def send_email_report(subject: str, body: str) -> None:
    """Sends a plain-text email via Gmail SMTP using an app password."""
    from email.mime.text import MIMEText
    import smtplib

    gmail_address = os.getenv("GMAIL_ADDRESS")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
    email_to = os.getenv("EMAIL_TO", gmail_address)

    if not gmail_address or not gmail_app_password:
        _log("WARNING: GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set; skipping email report.")
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
        print(f"Email report sent to {email_to}")
    except Exception as e:
        print(f"WARNING: Failed to send email report: {e}")


# ==========================================
# 7. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    run_stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M %Z")

    if not is_market_open_today():
        today_stamp = datetime.now(ET).strftime("%Y-%m-%d")
        _log(
            f"Market is CLOSED today ({today_stamp}). Exiting job safely with"
            " success code."
        )
        sys.exit(0)

    try:
        market_data = get_market_data(TICKERS)
        target_symbol, branch_reason = evaluate_signal(market_data)
        outcome = rebalance(target_symbol, branch_reason)

        body = (
            f"Run time: {run_stamp}\n"
            f"Strategy ID: {STRATEGY_ID}\n"
            f"Mode: {'PAPER' if PAPER else 'LIVE'}\n\n"
            f"Signal: {target_symbol}\n"
            f"Reason: {branch_reason}\n\n"
            f"Rebalance outcome:\n  {outcome}\n\n"
            f"--- Full run log ---\n"
            + "\n".join(RUN_LOG)
        )
        send_email_report(f"[{STRATEGY_ID}] {target_symbol} - {run_stamp}", body)

    except SystemExit as e:
        if e.code != 0:
            body = (
                f"Run time: {run_stamp}\n"
                f"Strategy ID: {STRATEGY_ID}\n"
                f"Mode: {'PAPER' if PAPER else 'LIVE'}\n\n"
                f"Run FAILED before completing. See log below.\n\n"
                f"--- Full run log ---\n"
                + "\n".join(RUN_LOG)
            )
            send_email_report(f"[{STRATEGY_ID}] ERROR - {run_stamp}", body)
        raise
