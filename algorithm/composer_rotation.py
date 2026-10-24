"""Multi-Asset Rotation strategy (Composer-style).

Evaluates 200SMA, 20SMA, and 10-period RSI signals across TQQQ, SOXL, SQQQ,
BSV, UVXY, and TECL to select a single target ticker, then targets holding
DEFAULT_QTY shares of it (and nothing else).
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from algorithm.base import Decision, MarketContext, Strategy
from alpaca_broker.market_data import fetch_daily_closes
from common.errors import StrategyError
from common.market_calendar import ET

# ==========================================
# 1. CONFIGURATION
# ==========================================
STRATEGY_ID = "COMPOSER_V1"  # Unique Strategy ID for trade tracking

TICKERS = ["TQQQ", "SOXL", "SQQQ", "BSV", "UVXY", "TECL"]

MIN_ROWS = 200  # Minimum daily history bars required for 200-day SMA
MIN_TRADE_QTY = 10  # Self-imposed minimum share quantity floor
POSITION_DUST_THRESHOLD = 0.001  # Ignore fractional share residuals

# Default share quantity (override with env COMPOSER_V1_QTY, or legacy STRATEGY_QTY)
DEFAULT_QTY = 15.0

log = logging.getLogger("algo.composer_rotation")


# ==========================================
# 2. INDICATORS
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
# 4. STRATEGY
# ==========================================
class ComposerRotation(Strategy):
    strategy_id = STRATEGY_ID
    description = "Multi-Asset Rotation Bot (Alpaca)"
    universe = TICKERS
    default_qty = DEFAULT_QTY
    min_trade_qty = MIN_TRADE_QTY
    dust_threshold = POSITION_DUST_THRESHOLD
    log_file_name = "rotation_bot.log"

    def decide(self, market: MarketContext) -> Decision:
        prices = fetch_daily_closes(market.data_client, TICKERS, market.feed, MIN_ROWS)
        target_symbol, branch_reason = evaluate_signal(prices)

        run_stamp = datetime.now(ET).strftime("%Y-%m-%d %H:%M %Z")
        return Decision(
            target={target_symbol: self.target_qty()},
            reason=branch_reason,
            email_subject=f"{target_symbol} - {run_stamp}",
            email_details=f"Target Symbol: {target_symbol}",
        )
