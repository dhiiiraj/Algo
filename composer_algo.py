import os
import sys
import time
import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ==========================================
# 1. CONFIGURATION & CREDENTIALS
# ==========================================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("ERROR: Missing Alpaca API credentials.")
    print("Ensure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in the environment.")
    sys.exit(1)

# SET TO False WHEN GOING LIVE WITH REAL MONEY
PAPER = True

# SET YOUR HARDCODED CASH ALLOCATION LIMIT HERE (in USD)
MAX_BUDGET = 100.00

# Minimum rows of daily history required before trusting the 200-day SMA
MIN_ROWS = 200

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

    # Zero-loss protection: avg_loss == 0 means RSI should be 100, not NaN
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100.0
    return rsi


# ==========================================
# 3. MARKET DATA FETCHING
# ==========================================
def get_market_data(tickers: list[str]) -> pd.DataFrame:
    """Fetches historical daily close prices required for 200 SMA calculation."""
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            limit=250
        )
        bars = data_client.get_stock_bars(request_params).df
    except Exception as e:
        print(f"ERROR: Failed to fetch market data from Alpaca API: {e}")
        sys.exit(1)

    if bars.empty:
        print("ERROR: No market data returned for any ticker.")
        sys.exit(1)

    # Pivot dataframe so columns are ticker close prices indexed by timestamp
    df = bars.reset_index().pivot(index='timestamp', columns='symbol', values='close')

    # Data integrity validation
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        print(f"ERROR: No data returned for ticker(s): {missing}")
        sys.exit(1)

    if len(df) < MIN_ROWS:
        print(f"ERROR: Insufficient price history returned ({len(df)} rows). Need at least {MIN_ROWS}.")
        sys.exit(1)

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

    # Validate no required values are NaN before evaluating decision tree
    required_fields = [
        "TQQQ", "TQQQ_200SMA", "TQQQ_20SMA", "TQQQ_RSI10",
        "SOXL_RSI10", "SQQQ_RSI10", "BSV_RSI10",
    ]
    nan_fields = [f for f in required_fields if pd.isna(latest[f])]
    if nan_fields:
        print(f"ERROR: Cannot evaluate signal, NaN values present in: {nan_fields}")
        sys.exit(1)

    tqqq_price = latest["TQQQ"]
    tqqq_200sma = latest["TQQQ_200SMA"]
    tqqq_rsi10 = latest["TQQQ_RSI10"]

    # Strategy Decision Tree
    if tqqq_price > tqqq_200sma:
        if tqqq_rsi10 > 79:
            print(f"BRANCH: TQQQ > 200SMA and RSI10 ({tqqq_rsi10:.1f}) > 79 -> extremely overbought -> UVXY")
            return "UVXY"
        else:
            print(f"BRANCH: TQQQ > 200SMA and RSI10 ({tqqq_rsi10:.1f}) <= 79 -> uptrend intact -> TQQQ")
            return "TQQQ"
    else:
        if tqqq_rsi10 < 31:
            print(f"BRANCH: TQQQ <= 200SMA and RSI10 ({tqqq_rsi10:.1f}) < 31 -> oversold bounce -> TECL")
            return "TECL"
        else:
            if latest["SOXL_RSI10"] < 30:
                print(f"BRANCH: downtrend, SOXL RSI10 ({latest['SOXL_RSI10']:.1f}) < 30 -> semis oversold -> SOXL")
                return "SOXL"
            else:
                if tqqq_price < latest["TQQQ_20SMA"]:
                    if latest["SQQQ_RSI10"] >= latest["BSV_RSI10"]:
                        print(
                            f"BRANCH: downtrend, TQQQ < 20SMA, SQQQ RSI10 "
                            f"({latest['SQQQ_RSI10']:.1f}) >= BSV RSI10 ({latest['BSV_RSI10']:.1f}) -> SQQQ"
                        )
                        return "SQQQ"
                    else:
                        print(
                            f"BRANCH: downtrend, TQQQ < 20SMA, BSV RSI10 "
                            f"({latest['BSV_RSI10']:.1f}) > SQQQ RSI10 ({latest['SQQQ_RSI10']:.1f}) -> BSV"
                        )
                        return "BSV"
                else:
                    print("BRANCH: downtrend on 200SMA but TQQQ still above 20SMA -> stay long -> TQQQ")
                    return "TQQQ"


# ==========================================
# 5. EXECUTION & ORDER SYNCHRONIZATION HELPERS
# ==========================================
def wait_for_liquidation(symbol: str, timeout: int = 15, poll_interval: float = 0.5) -> bool:
    """Polls until the position is fully closed, or timeout (seconds) elapses."""
    elapsed = 0.0
    while elapsed < timeout:
        try:
            positions = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
        except Exception as e:
            print(f"WARNING: Failed to poll positions for {symbol}: {e}")
            return False
        if symbol not in positions:
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
    print(f"WARNING: Liquidation of {symbol} did not confirm within {timeout}s.")
    return False


# ==========================================
# 6. REBALANCING ENGINE
# ==========================================
def rebalance(target_ticker: str):
    """Liquidates out-of-strategy positions and enters the target position using the hardcoded budget cap."""
    try:
        positions = {p.symbol: float(p.qty) for p in trading_client.get_all_positions()}
    except Exception as e:
        print(f"ERROR: Failed to retrieve account positions: {e}")
        sys.exit(1)

    print("--- REBALANCE RUN STARTED ---")
    print(f"Mode: {'PAPER' if PAPER else 'LIVE'}")
    print(f"Target Symbol: {target_ticker}")
    print(f"Current Portfolio Positions: {positions}")

    # Pre-check: Check if already correctly positioned before triggering liquidations
    if positions.get(target_ticker, 0) > 0 and all(
        symbol == target_ticker for symbol in positions.keys()
    ):
        print(f"Already holding target asset ({target_ticker}) exclusively. No action needed.")
        return

    # Check for pending open orders to prevent duplicate execution
    try:
        symbols_in_play = list(set(list(positions.keys()) + [target_ticker]))
        open_orders_request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=symbols_in_play)
        open_orders = trading_client.get_orders(filter=open_orders_request)
        if open_orders:
            print(f"SKIP: {len(open_orders)} open order(s) already exist on {symbols_in_play}. Not rebalancing.")
            return
    except Exception as e:
        print(f"ERROR: Failed to check open orders: {e}")
        sys.exit(1)

    # Capture cash BEFORE liquidating anything, so there's a real baseline
    # to compare against once liquidations settle. Without this baseline,
    # a polling loop that only checks "is cash > 0" can exit on the very
    # first check just because the account had a few stray dollars sitting
    # idle -- long before the sale proceeds actually land.
    try:
        cash_before = float(trading_client.get_account().cash)
    except Exception as e:
        print(f"ERROR: Failed to read pre-liquidation cash balance: {e}")
        sys.exit(1)

    had_liquidations = any(symbol != target_ticker for symbol in positions.keys())

    # Step 1: Liquidate non-target assets synchronously
    liquidation_failed = False
    for symbol in list(positions.keys()):
        if symbol != target_ticker:
            print(f"Liquidating position: {symbol}...")
            try:
                trading_client.close_position(symbol)
            except Exception as e:
                print(f"ERROR: Failed to submit close position order for {symbol}: {e}")
                liquidation_failed = True
                continue

            # Wait for liquidation to confirm on Alpaca's side
            confirmed = wait_for_liquidation(symbol, timeout=15)
            if not confirmed:
                liquidation_failed = True

    if liquidation_failed:
        print("ERROR: One or more liquidations failed or timed out. Aborting before entering a new position.")
        sys.exit(1)

    # Step 2: Poll cash balance until it actually reflects the liquidation
    # proceeds (i.e. has risen above the pre-liquidation baseline), not
    # just until it happens to be nonzero. If nothing was liquidated this
    # run, there's nothing to wait for -- skip straight to using cash_before.
    available_cash = cash_before
    if had_liquidations:
        for attempt in range(10):
            try:
                available_cash = float(trading_client.get_account().cash)
                if available_cash > cash_before:
                    break
            except Exception as e:
                print(f"WARNING: Failed to read account cash balance: {e}")
            time.sleep(1)
        else:
            print(
                f"WARNING: Cash balance did not rise above pre-liquidation baseline "
                f"(${cash_before:.2f}) after 10s; proceeding with last known value ${available_cash:.2f}."
            )

    # Cap execution to whichever is smaller: your MAX_BUDGET setting or actual account cash
    target_spend = min(MAX_BUDGET, available_cash)

    # 0.5% buffer for market order price slippage
    notional_amount = round(target_spend * 0.995, 2)
    min_order_limit = 10.00  # Alpaca dollar limit for fractional orders

    # Step 3: Execute Buy Order
    if notional_amount >= min_order_limit:
        market_order_data = MarketOrderRequest(
            symbol=target_ticker,
            notional=notional_amount,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        try:
            trading_client.submit_order(order_data=market_order_data)
            print(f"Successfully submitted Market BUY order: ${notional_amount} of {target_ticker}")
        except Exception as e:
            print(f"ERROR: Failed to submit order for {target_ticker}: {e}")
            sys.exit(1)
    else:
        print(f"Execution skipped: Calculated spend (${notional_amount}) is below Alpaca's ${min_order_limit} minimum requirement.")


# ==========================================
# 7. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    market_data = get_market_data(TICKERS)
    target_symbol = evaluate_signal(market_data)
    rebalance(target_symbol)
