import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Data imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Trading imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ==========================================
# 1. API & STRATEGY CONFIGURATION
# ==========================================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

PAPER_TRADING = True 

# Set your fixed starting cash investment here:
INITIAL_INVESTMENT = 100.0  # Dollars

if not API_KEY or not SECRET_KEY:
    print("ERROR: Missing Alpaca credentials.")
    sys.exit(1)

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_TRADING)

INDICATOR_SYMBOL = "QQQ"
TRADE_SYMBOL = "TQQQ"


# ==========================================
# 2. DATA FETCHING
# ==========================================
def fetch_weekly_bars(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetches weekly bars for a symbol and returns a clean, symbol-indexed DataFrame."""
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Week,
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX
        )
        bars = data_client.get_stock_bars(request)
    except Exception as e:
        print(f"ERROR: Failed to fetch {symbol} data from Alpaca API: {e}")
        sys.exit(1)

    if symbol not in bars.df.index.get_level_values(0):
        print(f"ERROR: No data returned for {symbol}.")
        sys.exit(1)

    return bars.df.loc[symbol].copy()


# ==========================================
# 3. INDICATOR LOGIC
# ==========================================
def calc_wilder_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi


# ==========================================
# 4. SIGNAL LOGIC
# ==========================================
def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
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
    df.loc[buy_cond, "signal"] = 1
    df.loc[sell_cond, "signal"] = 0

    return df


# ==========================================
# 5. LIVE ORDER EXECUTION LOGIC
# ==========================================
def execute_live_trade(latest_signal: float):
    print("\n================ LIVE EXECUTION ================")
    
    # 1. Fetch current open position for TRADE_SYMBOL
    current_qty = 0
    try:
        position = trading_client.get_open_position(TRADE_SYMBOL)
        current_qty = float(position.qty)
        print(f"Current Position: {current_qty} shares of {TRADE_SYMBOL}")
    except Exception:
        print(f"Current Position: 0 shares of {TRADE_SYMBOL}")

    # 2. Check current account buying power
    account = trading_client.get_account()
    cash_available = float(account.cash)
    print(f"Available Cash: ${cash_available:,.2f}")

    # 3. Decision Logic
    if latest_signal == 1 and current_qty == 0:
        print(f"SIGNAL: BUY detected for {TRADE_SYMBOL}.")
        
        # LOGIC FOR INITIAL vs REINVESTMENT:
        # If available cash is greater than or equal to INITIAL_INVESTMENT, 
        # use the INITIAL_INVESTMENT amount.
        # If available cash is less (e.g., after a previous cycle or lower balance), 
        # invest all available cash (with 95% safety buffer).
        if cash_available >= INITIAL_INVESTMENT:
            trade_amount = INITIAL_INVESTMENT
            print(f"Using initial configured amount: ${trade_amount:,.2f}")
        else:
            trade_amount = cash_available * 0.95
            print(f"Reinvesting accumulated cash: ${trade_amount:,.2f}")

        if trade_amount >= 10:  # Minimum execution threshold
            order_data = MarketOrderRequest(
                symbol=TRADE_SYMBOL,
                notional=round(trade_amount, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            order = trading_client.submit_order(order_data)
            print(f"SUCCESS: Submitted BUY order for ${trade_amount:,.2f} of {TRADE_SYMBOL}. Order ID: {order.id}")
        else:
            print("ERROR: Insufficient cash balance to place buy order.")

    elif latest_signal == 0 and current_qty > 0:
        print(f"SIGNAL: SELL detected for {TRADE_SYMBOL}.")
        # Liquidate full position; initial capital + profits convert back to cash
        trading_client.close_position(TRADE_SYMBOL)
        print(f"SUCCESS: Liquidated all shares of {TRADE_SYMBOL}. Capital + returns saved to cash for reinvestment.")

    else:
        print(f"NO ACTION: Signal is {int(latest_signal) if not np.isnan(latest_signal) else 0} and current position is {current_qty} shares.")

    print("==================================================")


# ==========================================
# 6. MAIN RUNNER
# ==========================================
def main():
    start_dt = datetime(2018, 1, 1)
    end_dt = datetime.now()

    print("Fetching historical market data...")
    qqq_df = fetch_weekly_bars(INDICATOR_SYMBOL, start_dt, end_dt)

    if len(qqq_df) < 104:
        print("ERROR: Insufficient historical data.")
        sys.exit(1)

    qqq_df = generate_signal(qqq_df)
    latest_signal = qqq_df["signal"].dropna().iloc[-1]

    print(f"Latest signal for {INDICATOR_SYMBOL}: {latest_signal}")
    execute_live_trade(latest_signal)


if __name__ == "__main__":
    main()
