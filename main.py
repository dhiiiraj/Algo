#!/usr/bin/env python3
"""
Weekly QQQ-signal / TQQQ-trade bot (Alpaca).

Signal (computed on COMPLETED weekly QQQ bars only):
    BUY  state : MACD > MACD signal  AND  EMA12 > EMA26
    SELL state : EMA12 <= EMA26  OR  (RSI >= 70 AND MACD <= MACD signal)
    Otherwise  : keep the previous state (hysteresis).

Usage:
    export ALPACA_API_KEY=...
    export ALPACA_SECRET_KEY=...

    python tqqq_weekly_bot.py --dry-run      # paper account, log decision only
    python tqqq_weekly_bot.py                # paper account, places orders
    python tqqq_weekly_bot.py --live         # REAL MONEY, places orders

Best run after Friday's close (or on the weekend / before Monday's open).

Optional env vars:
    ALPACA_FEED  iex (default) | sip
    LOG_FILE     path to log file (default: tqqq_weekly_bot.log next to script)
"""

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from typing import Optional
from zoneinfo import ZoneInfo

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
INDICATOR_SYMBOL = "QQQ"
TRADE_SYMBOL = "TQQQ"

HISTORY_YEARS = 8            # weekly history to pull (more = better indicator warm-up)
MIN_BARS = 104               # minimum completed weekly bars required

CASH_DEPLOYMENT_PCT = 0.95   # fraction of available cash to deploy on a BUY
MIN_TRADE_AMOUNT = 10.0      # self-imposed floor in USD (Alpaca's own minimum is lower)

# Ignore residual long fractional shares below this threshold to avoid false "held" states
POSITION_DUST_THRESHOLD = 0.001

# False: buy whenever the current state is "long" and you are flat (matches original behavior).
# True : only buy when the state has just flipped to long on the latest completed week.
ENTER_ONLY_ON_FRESH_BUY = False

ET = ZoneInfo("America/New_York")
# A weekly bar is treated as complete once Friday 16:30 ET has passed.
WEEK_CLOSE_TIME = dtime(16, 30)

ORDER_POLL_TIMEOUT_SEC = 20

TERMINAL_STATUSES = {
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
    OrderStatus.DONE_FOR_DAY,
}
FAILED_STATUSES = {OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED}

log = logging.getLogger("tqqq_bot")


class StrategyError(Exception):
    """Expected, non-recoverable problem (bad data, blocked account, etc.)."""


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
    """Fetch weekly bars for one symbol; returns a DataFrame indexed by timestamp."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Week,
        start=start,
        end=end,
        feed=feed,
        adjustment=Adjustment.ALL,  # split/dividend adjusted for indicator continuity
    )
    try:
        bars = data_client.get_stock_bars(request)
    except APIError as e:
        raise StrategyError(f"API error fetching {symbol} bars: {e}") from e

    df = bars.df  # build once
    if df.empty or symbol not in df.index.get_level_values(0):
        raise StrategyError(f"No data returned for {symbol}.")

    return df.loc[symbol].sort_index().copy()


# ==========================================
# 3. COMPLETED-WEEK LOGIC
# ==========================================
def week_monday(ts: pd.Timestamp) -> date:
    """Monday (as a date) of the week a weekly bar belongs to.

    Shifts +12h before converting to ET so the result is the same whether the feed
    stamps weekly bars at Monday 00:00 UTC or Monday 00:00 ET.
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts_et = (ts + pd.Timedelta(hours=12)).tz_convert(ET)
    return ts_et.date() - timedelta(days=ts_et.weekday())


def last_complete_week_monday(now_et: datetime) -> date:
    """Monday of the most recent weekly bar that should be fully closed right now."""
    this_monday = now_et.date() - timedelta(days=now_et.weekday())
    this_week_close = datetime.combine(
        this_monday + timedelta(days=4), WEEK_CLOSE_TIME, tzinfo=ET
    )
    return this_monday if now_et >= this_week_close else this_monday - timedelta(days=7)


def keep_completed_weeks(df: pd.DataFrame, now_et: datetime) -> pd.DataFrame:
    """Drop any in-progress week and verify the newest bar is the week we expect."""
    expected = last_complete_week_monday(now_et)
    mask = np.array([week_monday(ts) <= expected for ts in df.index])
    df = df.loc[mask].copy()

    if df.empty:
        raise StrategyError("No completed weekly bars available.")

    newest = week_monday(df.index[-1])
    if newest != expected:
        raise StrategyError(
            f"Stale data: newest completed bar is week of {newest}, expected week of "
            f"{expected}. Feed may be lagging; refusing to trade."
        )
    return df


# ==========================================
# 4. INDICATORS & SIGNALS
# ==========================================
def calc_wilder_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)      # keeps NaN for the first row
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    # x/0 -> inf -> RSI = 100 exactly; 0/0 -> NaN (undefined, treated as "not overbought")
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
    df.loc[sell_cond, "signal"] = 0.0   # sell wins if both ever fired (they are exclusive)

    # Persistent state: stays long/flat until the opposite condition fires.
    df["state"] = df["signal"].ffill()
    return df


# ==========================================
# 5. BROKER HELPERS
# ==========================================
def get_position_qty(client: TradingClient, symbol: str) -> float:
    """Return share qty held (negative = short). Only a 404 means 'no position'."""
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
    """Cash we are willing to deploy: never uses margin, never negative."""
    cash = float(account.cash)
    nmbp = getattr(account, "non_marginable_buying_power", None)
    available = cash if nmbp is None else min(cash, float(nmbp))
    return max(available, 0.0) * CASH_DEPLOYMENT_PCT


def _status(order) -> str:
    return str(getattr(order.status, "value", order.status))


def wait_for_order(client: TradingClient, order):
    """Poll briefly for a terminal status. Returns the latest order object."""
    latest = order
    deadline = time.monotonic() + ORDER_POLL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        latest = client.get_order_by_id(order.id)
        if latest.status in TERMINAL_STATUSES:
            break
        time.sleep(2)

    if latest.status not in TERMINAL_STATUSES:
        log.warning(
            "Order %s did not reach a terminal state within %ds (status: %s). "
            "The next run's pending-order check will skip until it resolves.",
            order.id, ORDER_POLL_TIMEOUT_SEC, _status(latest),
        )
    return latest


def assert_order_ok(order, label: str) -> None:
    """Raise (-> non-zero exit) if the order was rejected, canceled, or expired."""
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
) -> None:
    log.info("================ EXECUTION ================")

    account = trading_client.get_account()
    if account.trading_blocked or account.account_blocked:
        raise StrategyError("Account is blocked from trading; aborting.")

    asset = trading_client.get_asset(TRADE_SYMBOL)
    if not asset.tradable:
        raise StrategyError(f"{TRADE_SYMBOL} is not tradable on this account.")
    if not asset.fractionable:
        raise StrategyError(f"{TRADE_SYMBOL} is not fractionable; notional orders unsupported.")

    market_open = True
    try:
        market_open = bool(trading_client.get_clock().is_open)
    except Exception as e:  # clock is informational only
        log.warning("Could not check market clock: %s", e)
    if not market_open:
        log.warning("Market is CLOSED. Day orders will queue for the next open.")

    qty = get_position_qty(trading_client, TRADE_SYMBOL)
    log.info("Position: %s shares of %s", qty, TRADE_SYMBOL)
    if qty < 0:
        raise StrategyError(f"Unexpected SHORT position ({qty}) in {TRADE_SYMBOL}; aborting.")

    open_orders = trading_client.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[TRADE_SYMBOL])
    )
    if open_orders:
        log.warning(
            "%d open %s order(s) already pending; skipping to avoid duplicates.",
            len(open_orders), TRADE_SYMBOL,
        )
        return

    log.info("Target state: %s | fresh buy: %s", "LONG" if target_long else "FLAT", fresh_buy)

    # ---- BUY ----
    if target_long and qty == 0:
        if ENTER_ONLY_ON_FRESH_BUY and not fresh_buy:
            log.info("NO ACTION: flat, state is long but not a fresh buy (ENTER_ONLY_ON_FRESH_BUY).")
            return

        amount = round(deployable_cash(account), 2)
        log.info("BUY signal. Deployable cash (%.0f%%): $%.2f", CASH_DEPLOYMENT_PCT * 100, amount)

        if amount < MIN_TRADE_AMOUNT:
            log.error("Trade amount $%.2f below minimum $%.2f; not buying.", amount, MIN_TRADE_AMOUNT)
            return
        if dry_run:
            log.info("[DRY RUN] Would BUY $%.2f of %s.", amount, TRADE_SYMBOL)
            return

        order = trading_client.submit_order(
            MarketOrderRequest(
                symbol=TRADE_SYMBOL,
                notional=amount,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,  # notional/fractional orders require DAY
            )
        )
        log.info("Submitted BUY $%.2f %s | id=%s status=%s", amount, TRADE_SYMBOL, order.id, _status(order))
        if market_open:
            order = wait_for_order(trading_client, order)
            log.info("BUY order status: %s (filled_qty=%s)", _status(order), order.filled_qty)
        assert_order_ok(order, "BUY")

    # ---- SELL ----
    elif not target_long and qty > 0:
        log.info("SELL signal. Closing entire %s position (%s shares).", TRADE_SYMBOL, qty)
        if dry_run:
            log.info("[DRY RUN] Would close position in %s.", TRADE_SYMBOL)
            return

        order = trading_client.close_position(TRADE_SYMBOL)
        log.info("Submitted CLOSE %s | id=%s status=%s", TRADE_SYMBOL, order.id, _status(order))
        if market_open:
            order = wait_for_order(trading_client, order)
            log.info("CLOSE order status: %s (filled_qty=%s)", _status(order), order.filled_qty)
        assert_order_ok(order, "CLOSE")

    # ---- HOLD ----
    else:
        log.info(
            "NO ACTION: target is %s and position is %s shares.",
            "LONG" if target_long else "FLAT", qty,
        )


# ==========================================
# 7. MAIN
# ==========================================
def setup_logging() -> None:
    log_file = os.getenv(
        "LOG_FILE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tqqq_weekly_bot.log"),
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
    p = argparse.ArgumentParser(description="Weekly QQQ-signal / TQQQ-trade bot (Alpaca).")
    p.add_argument("--live", action="store_true", help="Trade the LIVE account (real money). Default is paper.")
    p.add_argument("--dry-run", action="store_true", help="Compute and log the decision but place no orders.")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    setup_logging()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        log.error("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY environment variables.")
        return 1

    paper = not args.live
    mode = "PAPER" if paper else "*** LIVE ***"
    log.info("Starting | mode=%s | dry_run=%s", mode, args.dry_run)

    try:
        feed = DataFeed(os.getenv("ALPACA_FEED", "iex").lower())
    except ValueError:
        log.error("Invalid ALPACA_FEED (use 'iex' or 'sip').")
        return 1

    data_client = StockHistoricalDataClient(api_key, secret_key)
    trading_client = TradingClient(api_key, secret_key, paper=paper)

    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=365 * HISTORY_YEARS)

    log.info("Fetching %s weekly bars (%s feed)...", INDICATOR_SYMBOL, feed.value)
    df = fetch_weekly_bars(data_client, INDICATOR_SYMBOL, start, now_utc, feed)
    df = keep_completed_weeks(df, now_utc.astimezone(ET))

    if len(df) < MIN_BARS:
        raise StrategyError(f"Only {len(df)} completed bars; need at least {MIN_BARS}.")

    df = compute_signals(df)
    state = df["state"].dropna()
    if state.empty:
        raise StrategyError("No valid signal generated from indicators.")

    latest_state = float(state.iloc[-1])
    prev_state = float(state.iloc[-2]) if len(state) > 1 else np.nan
    fresh_buy = latest_state == 1.0 and prev_state != 1.0

    last = df.iloc[-1]
    log.info(
        "Week of %s | close=%.2f EMA12=%.2f EMA26=%.2f MACD=%.3f Sig=%.3f RSI=%.1f | state=%s",
        week_monday(df.index[-1]), last["close"], last["EMA12"], last["EMA26"],
        last["MACD"], last["MACD_Signal"], last["RSI"],
        "LONG" if latest_state == 1.0 else "FLAT",
    )

    execute(trading_client, target_long=(latest_state == 1.0), fresh_buy=fresh_buy, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StrategyError as e:
        log.error("ABORTED: %s", e)
        sys.exit(1)
    except Exception:
        log.exception("Unhandled error")
        sys.exit(1)
