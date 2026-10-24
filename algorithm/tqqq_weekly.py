"""Weekly QQQ-signal / TQQQ-trade strategy.

Signal (computed on COMPLETED weekly QQQ bars only):
    BUY  state : MACD > MACD signal  AND  EMA12 > EMA26
    SELL state : EMA12 <= EMA26  OR  (RSI >= 70 AND MACD <= MACD signal)
    Otherwise  : keep the previous state (hysteresis).

Target:
    LONG -> hold DEFAULT_QTY shares of TQQQ; FLAT -> hold nothing.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime

import numpy as np
import pandas as pd

from algorithm.base import Decision, MarketContext, Strategy
from alpaca_broker.market_data import fetch_weekly_bars
from common.errors import StrategyError
from common.market_calendar import ET

# ==========================================
# 1. CONFIGURATION
# ==========================================
STRATEGY_ID = "TQQQ_Dhiraj_V1"  # Unique Strategy ID for trade tracking

INDICATOR_SYMBOL = "QQQ"
TRADE_SYMBOL = "TQQQ"

HISTORY_YEARS = 8  # weekly history to pull (more = better indicator warm-up)
MIN_BARS = 104  # minimum completed weekly bars required

MIN_TRADE_QTY = 1.0  # Self-imposed minimum share quantity floor

# Default share quantity (override with env TQQQ_DHIRAJ_V1_QTY, or legacy STRATEGY_QTY)
DEFAULT_QTY = 100.0

POSITION_DUST_THRESHOLD = 0.001

ENTER_ONLY_ON_FRESH_BUY = False

WEEK_CLOSE_TIME = dtime(16, 30)

log = logging.getLogger("algo.tqqq_weekly")


# ==========================================
# 2. COMPLETED-WEEK LOGIC
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
        this_monday
        if now_et >= this_week_close
        else this_monday - timedelta(days=7)
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
# 3. INDICATORS & SIGNALS
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
# 4. STRATEGY
# ==========================================
class TqqqWeekly(Strategy):
    strategy_id = STRATEGY_ID
    description = "Weekly QQQ-signal / TQQQ-trade bot (Alpaca)."
    universe = [TRADE_SYMBOL]
    default_qty = DEFAULT_QTY
    min_trade_qty = MIN_TRADE_QTY
    dust_threshold = POSITION_DUST_THRESHOLD
    log_file_name = "tqqq_weekly_bot.log"

    def decide(self, market: MarketContext) -> Decision:
        now_utc = datetime.now(timezone.utc)
        start = now_utc - timedelta(days=365 * HISTORY_YEARS)

        log.info("Fetching %s weekly bars (%s feed)...", INDICATOR_SYMBOL, market.feed.value)
        df = fetch_weekly_bars(market.data_client, INDICATOR_SYMBOL, start, now_utc, market.feed)
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
        target_long = latest_state == 1.0

        last = df.iloc[-1]
        week_str = str(week_monday(df.index[-1]))
        state_str = "LONG (BUY)" if target_long else "FLAT (SELL/OUT)"

        log.info(
            "Week of %s | close=%.2f EMA12=%.2f EMA26=%.2f MACD=%.3f Sig=%.3f"
            " RSI=%.1f | state=%s | fresh buy=%s",
            week_str,
            last["close"],
            last["EMA12"],
            last["EMA26"],
            last["MACD"],
            last["MACD_Signal"],
            last["RSI"],
            state_str,
            fresh_buy,
        )

        if target_long:
            reason = "Weekly MACD/EMA Bullish Signal"
            target = {TRADE_SYMBOL: self.target_qty()}
            held = market.positions.get(TRADE_SYMBOL, 0.0)
            if ENTER_ONLY_ON_FRESH_BUY and not fresh_buy and not held:
                reason = "State is long but not a fresh buy (ENTER_ONLY_ON_FRESH_BUY)"
                target = {}
        else:
            reason = "Weekly Trend Breakdown / Overbought Signal"
            target = {}

        details = (
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
        return Decision(
            target=target,
            reason=reason,
            email_subject=f"{state_str.split()[0]} - {week_str}",
            email_details=details,
        )
