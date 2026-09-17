import os
import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# SET TO False WHEN GOING LIVE WITH REAL MONEY
PAPER = True  

# SET YOUR HARDCODED CASH ALLOCATION LIMIT HERE (in USD)
MAX_BUDGET = 100.00  

TICKERS = ["TQQQ", "SOXL", "SQQQ", "BSV", "UVXY", "TECL"]

# Initialize Alpaca API Clients
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


# ==========================================
# 2. INDICATOR LOGIC
# ==========================================
def calc_composer_rsi10(series: pd.Series, window: int = 10) -> pd.Series:
    """Calculates RSI using a Simple Moving Average (SMA) of gains/losses

    (Matches Composer's standard rolling window engine).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ==========================================
# 3. MARKET DATA FETCHING
# ==========================================
def get_market_data(tickers: list[str]) -> pd.DataFrame:
    """Fetches historical daily close prices required for 200 SMA calculation."""
    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        limit=250
    )
    bars = data_client.get_stock_bars(request_params).df
    
    # Pivot dataframe so columns are ticker close prices indexed by timestamp
    df = bars.reset_index().pivot(index='timestamp', columns='symbol', values='close')
    return df


# ==========================================
# 4. COMPOSER SIGNAL ENGINE
# ==========================================
def evaluate_signal(df: pd.DataFrame) -> str:
    """Evaluates the strategy logic using the latest daily price data."""
    # 1. Moving Averages
    df["TQQQ_200SMA"] = df["TQQQ"].rolling(200).mean()
    df["TQQQ_20SMA"] = df["TQQQ"].rolling(20).mean()

    # 2. 10-period RSIs
    df["TQQQ_RSI10"] = calc_composer_rsi10(df["TQQQ"], 10)
    df["SOXL_RSI10"] = calc_composer_rsi10(df["SOXL"], 10)
    df["SQQQ_RSI10"] = calc_composer_rsi10(df["SQQQ"], 10)
    df["BSV_RSI10"] = calc_composer_rsi10(df["BSV"], 10)

    # Use the most recent completed market candle
    latest = df.iloc[-1]
    
    tqqq_price = latest["TQQQ"]
    tqqq_200sma = latest["TQQQ_200SMA"]
    tqqq_rsi10 = latest["TQQQ_RSI10"]

    # Strategy Decision Tree
    if tqqq_price > tqqq_200sma:
        if tqqq_rsi10 > 79:
            return "UVXY"
        else:
            return "TQQQ"
    else:
        if tqqq_rsi10 < 31:
            return "TECL"
        else:
            if latest["SOXL_RSI10"] < 30:
                return "SOXL"
            else:
                if tqqq_price < latest["TQQQ_20SMA"]:
                    if latest["SQQQ_RSI10"] >= latest["BSV_RSI10"]:
                        return "SQQQ"
                    else:
                        return "BSV"
                else:
                    return "TQQQ"


# ==========================================
# 5. REBALANCING & EXECUTION ENGINE
# ==========================================
def rebalance(target_ticker: str):
    """Liquidates out-of-strategy positions and enters the target position using the hardcoded budget cap."""
    # Fetch current positions
    positions = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
    
    print(f"--- REBALANCE RUN STARTED ---")
    print(f"Mode: {'PAPER' if PAPER else 'LIVE'}")
    print(f"Target Symbol: {target_ticker}")
    print(f"Current Portfolio Positions: {positions}")

    # Step 1: Liquidate non-target assets
    for symbol in list(positions.keys()):
        if symbol != target_ticker:
            print(f"Liquidating position: {symbol}...")
            trading_client.close_position(symbol)

    # Step 2: Check if target position is already held exclusively
    if target_ticker in positions and len(positions) == 1:
        print(f"Already holding target asset ({target_ticker}) exclusively. No new order needed.")
        return

    # Step 3: Fetch cash and determine buy allocation based on MAX_BUDGET
    account = trading_client.get_account()
    available_cash = float(account.cash)

    # Cap execution to whichever is smaller: your MAX_BUDGET setting or actual account cash
    target_spend = min(MAX_BUDGET, available_cash)
    
    # 0.5% buffer for market order price slippage
    notional_amount = round(target_spend * 0.995, 2)
    min_order_limit = 10.00  # Alpaca dollar limit for fractional orders

    if notional_amount >= min_order_limit:
        market_order_data = MarketOrderRequest(
            symbol=target_ticker,
            notional=notional_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order_data=market_order_data)
        print(f"Successfully submitted Market BUY order: ${notional_amount} of {target_ticker}")
    else:
        print(f"Execution skipped: Calculated spend (${notional_amount}) is below Alpaca's ${min_order_limit} minimum requirement.")


# ==========================================
# 6. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    market_data = get_market_data(TICKERS)
    target_symbol = evaluate_signal(market_data)
    rebalance(target_symbol)
