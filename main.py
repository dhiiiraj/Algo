import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def run_strategy():
    # 1. Fetch API Credentials from Environment Variables
    API_KEY = os.environ.get("ALPACA_API_KEY")
    SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

    if not API_KEY or not SECRET_KEY:
        print("ERROR: Missing Alpaca API credentials.")
        print("Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in GitHub Secrets.")
        sys.exit(1)

    print("Authenticating with Alpaca API (Paper Environment)...")
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    # 2. Fetch Weekly QQQ Data for Signal Generation (1 Year History)
    INDICATOR_SYMBOL = "QQQ"
    TRADE_SYMBOL = "TQQQ"
    
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365)

    print(f"Fetching weekly {INDICATOR_SYMBOL} bar data for indicator generation...")
    request_params = StockBarsRequest(
        symbol_or_symbols=INDICATOR_SYMBOL,
        timeframe=TimeFrame.Week,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX  # Explicitly set free IEX data feed
    )

    try:
        bars = data_client.get_stock_bars(request_params)
        df = bars.df.loc[INDICATOR_SYMBOL].copy()
    except Exception as e:
        print(f"ERROR: Failed to fetch market data from Alpaca API: {e}")
        sys.exit(1)

    if df.empty or len(df) < 26:
        print(f"ERROR: Insufficient price history returned ({len(df)} rows). Need at least 26 weeks.")
        sys.exit(1)

    # 3. Calculate Indicators on QQQ using Standard Wilder's Smoothing
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Wilder's Smoothing RSI Calculation (Standard Charting Method)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Wilder's exponential moving average uses alpha = 1 / 14
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # 4. Determine Trade Signal from QQQ Indicators
    latest = df.iloc[-1]
    
    buy_cond = (latest["MACD"] > latest["MACD_Signal"]) or (latest["EMA12"] > latest["EMA26"])
    sell_cond = (latest["EMA26"] > latest["EMA12"]) or (
        (latest["RSI"] >= 70) and ((latest["MACD"] < latest["MACD_Signal"]) or (latest["EMA26"] > latest["EMA12"]))
    )

    print(f"Weekly QQQ Indicators -> MACD: {latest['MACD']:.2f} | Signal: {latest['MACD_Signal']:.2f} | RSI (Wilder): {latest['RSI']:.2f}")

    # 5. Check Portfolio Holdings & Execute Rebalance on TQQQ
    try:
        account = trading_client.get_account()
        positions = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
    except Exception as e:
        print(f"ERROR: Failed to retrieve account status or positions: {e}")
        sys.exit(1)

    tqqq_qty = positions.get(TRADE_SYMBOL, 0)
    print(f"Current Portfolio State: {tqqq_qty} shares of {TRADE_SYMBOL}")

    if buy_cond and tqqq_qty == 0:
        buying_power = float(account.buying_power)
        trade_amount = round(buying_power * 0.98, 2)  # 98% cash allocation
        print(f"SIGNAL: BUY -> Placing order for ${trade_amount:.2f} of {TRADE_SYMBOL}")
        
        order_data = MarketOrderRequest(
            symbol=TRADE_SYMBOL,
            notional=trade_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(order_data)

    elif sell_cond and tqqq_qty > 0:
        print(f"SIGNAL: SELL -> Liquidating all {tqqq_qty} shares of {TRADE_SYMBOL}")
        trading_client.close_position(TRADE_SYMBOL)

    else:
        print("SIGNAL: HOLD -> Portfolio state already matches target signal.")


if __name__ == "__main__":
    run_strategy()
