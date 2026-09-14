from ib_async import IB, MarketOrder, Stock
import numpy as np
import pandas as pd

ib = IB()
# Connect to TWS or IB Gateway (Paper Trading Port = 7497, Live = 7496)
ib.connect("127.0.0.1", 7497, clientId=1)

# 1. Define Contracts
qqq_contract = Stock("QQQ", "SMART", "USD")
tqqq_contract = Stock("TQQQ", "SMART", "USD")
ib.qualifyContracts(qqq_contract, tqqq_contract)

# 2. Fetch Historical Weekly Data for QQQ
bars = ib.reqHistoricalData(
    qqq_contract,
    endDateTime="",
    durationStr="1 Y",
    barSizeSetting="1 week",
    whatToShow="MIDPOINT",
    useRTH=True,
)
df = pd.DataFrame(bars)

# 3. Calculate Indicators
df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

delta = df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df["RSI"] = 100 - (100 / (1 + (gain / loss)))

latest = df.iloc[-1]
buy_cond = (latest["MACD"] > latest["MACD_Signal"]) or (
    latest["EMA12"] > latest["EMA26"]
)
sell_cond = (latest["EMA26"] > latest["EMA12"]) or (
    (latest["RSI"] >= 70)
    and (
        (latest["MACD"] < latest["MACD_Signal"])
        or (latest["EMA26"] > latest["EMA12"])
    )
)

# 4. Check Current Portfolio Holdings
positions = ib.positions()
tqqq_pos = sum(p.position for p in positions if p.contract.symbol == "TQQQ")

# 5. Trade Execution
if buy_cond and tqqq_pos == 0:
    # Get available cash balance
    account_summary = ib.accountSummary()
    cash_avail = float(
        [
            item.value
            for item in account_summary
            if item.tag == "TotalCashValue"
        ][0]
    )

    # Get TQQQ market price
    ticker = ib.reqMktData(tqqq_contract)
    ib.sleep(2)
    price = ticker.marketPrice()

    shares_to_buy = int((cash_avail * 0.98) // price)
    if shares_to_buy > 0:
        order = MarketOrder("BUY", shares_to_buy)
        trade = ib.placeOrder(tqqq_contract, order)
        print(f"Placed BUY order for {shares_to_buy} shares of TQQQ.")

elif sell_cond and tqqq_pos > 0:
    order = MarketOrder("SELL", abs(tqqq_pos))
    trade = ib.placeOrder(tqqq_contract, order)
    print(f"Placed SELL order for {tqqq_pos} shares of TQQQ.")

else:
    print("Portfolio state matches weekly indicator signal.")

ib.disconnect()
