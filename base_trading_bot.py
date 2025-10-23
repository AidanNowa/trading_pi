import os
import sys
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import alpaca_trade_api as tradeapi
from datetime import datetime

load_dotenv()
api = tradeapi.REST(
    os.getenv('APCA_API_KEY_ID'),
    os.getenv('APCA_API_SECRET_KEY'),
    base_url=os.getenv('APCA_API_BASE_URL'),
    api_version='v2'
)

account = api.get_account()
print(f"Paper account status: {account.status}")
print(f"Cash: {account.cash}")

#example for how to get ticker data
#symbol = "AAPL"
#data = yf.download(symbol, start="2022-01-01", end="2022-12-31", auto_adjust=True)
#print(data.head())
#print(data.describe())
#data['Close'].plot(title='Apple Closing Prices')

def moving_average_strategy(data):
    #SMA is 'Simple Moving Average' - calculate two of them and binary indicator (Signal) when 50-day SMA
    #crosses the 200-day SMA
    #If signal changes from 0->1 buy, 1->0 sell
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Signal'] = 0
    data['Signal'][50:] = np.where(data['SMA_50'][50:] > data['SMA_200'][50:], 1, 0)
    data['Position'] = data['Signal'].diff()
    return data

def calculate_indicators(data):
    # Exponential Moving Averages
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    # Relative Strength Index
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

def generate_signals(data):
    # Buy and Sell signals based on EMA and RSI
    data['Signal'] = 0
    buy_signal = (data['EMA_12'] > data['EMA_26']) & (data['RSI'] < 30)
    sell_signal = (data['EMA_12'] < data['EMA_26']) & (data['RSI'] > 70)
    data.loc[buy_signal, 'Signal'] = 1
    data.loc[sell_signal, 'Signal'] = -1

    # Future-proof forward-fill
    data['Position'] = (
        data['Signal']
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
        .astype(int)
    )
    return data

def backtest(data, initial_balance=10000):
    balance = initial_balance
    position = 0.0
    stop_loss = 0.95
    take_profit = 1.10
    entry_price = 0.0
    buy_signal_count = 0
    sell_signal_count = 0
    stop_loss_count = 0
    take_profit_count = 0

    for i, row in data.iterrows():
        # Coerce Signal to a scalar int safely
        sig = row['Signal']
        if hasattr(sig, "item"):
            try:
                sig = sig.item()
            except Exception:
                pass
        if not np.isscalar(sig):
            # if duplicate columns ever slip in, take the first element
            try:
                sig = np.asarray(sig).ravel()[0]
            except Exception:
                sig = 0
        try:
            sig = int(sig)
        except Exception:
            sig = 0

        price = float(row['Close'].iloc[0]) #float(row['Close'])

        if sig == 1 and balance > 0:
            position = balance / price
            balance = 0.0
            entry_price = price
            logging.info(f"BUY at {price} on {i.date()}")
            buy_signal_count += 1
        elif sig == -1 and position > 0:
            balance = position * price
            position = 0.0
            entry_price = 0.0
            logging.info(f"SELL at {price} on {i.date()}")
            sell_signal_count += 1

        if position > 0:
            if price <= entry_price * stop_loss:
                balance = position * price
                position = 0.0
                entry_price = 0.0
                logging.info(f"STOP LOSS at {price} on {i.date()}")
                stop_loss_count += 1
            elif price >= entry_price * take_profit:
                balance = position * price
                position = 0.0
                entry_price = 0.0
                logging.info(f"TAKE PROFIT at {price} on {i.date()}")
                take_profit_count += 1

    if position > 0:
        balance = position * float(data.iloc[-1]['Close'])  # close at last price

    return balance, buy_signal_count, sell_signal_count, stop_loss_count, take_profit_count

for arg in sys.argv:
    symbol = arg
    data = yf.download(symbol, start="2024-10-22", end="2025-10-21", auto_adjust=True)
    data = calculate_indicators(data)
    data = generate_signals(data)
    data = data.loc[:, ~data.columns.duplicated()]

    logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    #print(data.head)
    #print(data.describe)
    final_balance, buy_count, sell_count, stop_loss_count, take_profit_count = backtest(data)
    print(f"Final Balance testing {symbol}: ${final_balance:.2f}, Times Bought: {buy_count}, Times Sold: {sell_count}, Stopped Loss: {stop_loss_count}, Took Profit: {take_profit_count}")
