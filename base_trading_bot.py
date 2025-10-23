import os
import sys
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import alpaca_trade_api as tradeapi
from datetime import datetime
import time
import argparse

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

# ---------- helpers: symbol utilities & safety ----------

def parse_args():
    p = argparse.ArgumentParser(description="Backtest and/or paper trade EMA/RSI strategy.")
    p.add_argument("tickers", nargs="+", help="Ticker symbols (e.g. AAPL MSFT SPY)")
    p.add_argument("--live", action="store_true", help="If set, place paper trades on Alpaca (requires TRADE_MODE=paper).")
    p.add_argument("--timeframe", default="1Day", choices=["1Day", "1Hour", "15Min", "5Min"], help="Bar timeframe for the latest signal check.")
    return p.parse_args()

def ok_to_trade():
    # Only trade when TRADE_MODE=paper and account not blocked
    if os.getenv("TRADE_MODE", "").lower() != "paper":
        logging.info("TRADE_MODE not 'paper' — skipping live trading.")
        return False
    try:
        acct = api.get_account()
        if getattr(acct, "trading_blocked", False):
            logging.warning("Account is trading_blocked — skipping live trading.")
            return False
        return True
    except Exception as e:
        logging.exception(f"Account check failed: {e}")
        return False

def get_cash():
    try:
        return float(api.get_account().cash)
    except Exception:
        return 0.0

def current_position_qty(symbol):
    try:
        pos = api.get_position(symbol)
        return float(pos.qty)
    except Exception:
        # not holding
        return 0.0

def cancel_open_orders(symbol):
    try:
        for o in api.list_orders(status="open"):
            if o.symbol == symbol:
                api.cancel_order(o.id)
                time.sleep(0.2)
    except Exception as e:
        logging.warning(f"cancel_open_orders({symbol}) error: {e}")

def round_qty(qty):
    # Simple round down to 3 decimals to be safe for fractional shares
    return max(0.0, float(np.floor(qty * 1000) / 1000.0))

# ---------- live signal → order logic ----------

def compute_last_signal_row(df):
    """
    Use your existing indicator & signal generator, then look at last two rows.
    Return a tuple (prev_sig, last_sig, last_close, last_index) or None if insufficient rows.
    """
    if df is None or df.empty or len(df) < 2:
        return None
    df = calculate_indicators(df)
    df = generate_signals(df)
    df = df.loc[:, ~df.columns.duplicated()]
    prev_sig = int(df['Signal'].iloc[-2])
    last_sig = int(df['Signal'].iloc[-1])

    # Safely extract last close
    last_close = df['Close'].iloc[-1]
    last_idx = df.index[-1]
    return prev_sig, last_sig, float(last_close.iloc[0]), last_idx

def place_bracket_order(symbol, side, qty, entry_price, tp_pct, sl_pct):
    """
    Place a market order with attached take-profit & stop-loss brackets.
    side: 'buy' or 'sell' (we only use 'buy' here; 'sell' is flatting a long)
    """
    qty = round_qty(qty)
    if qty <= 0:
        logging.info(f"Qty calculated as 0 for {symbol}; skip.")
        return None
    tp_price = round(entry_price * (1 + tp_pct), 2)
    sl_price = round(entry_price * (1 - sl_pct), 2)

    logging.info(f"Submitting {side.upper()} {qty} {symbol} @ MKT with TP {tp_price} SL {sl_price}")
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=str(qty),
            side=side,
            type="market",
            time_in_force="day",
            order_class="bracket",
            take_profit={"limit_price": str(tp_price)},
            stop_loss={"stop_price": str(sl_price)}
        )
        return order
    except Exception as e:
        logging.exception(f"submit_order failed for {symbol}: {e}")
        return None

def maybe_trade_symbol(symbol, timeframe="1Day"):
    """
    Pull recent prices (via yfinance for now), compute latest signal, and react:
    - 0 -> +1 : enter long (if flat)
    - +1 -> -1 : exit any long (market sell + cancel remaining OCO legs)
    """
    # You can switch to Alpaca market data if you have it; for now, use yfinance:
    # Use enough history to compute indicators robustly
    lookback_start = (pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=400)).date().isoformat()
    df = yf.download(symbol, start=lookback_start, end=None, auto_adjust=True, progress=False)
    sig_row = compute_last_signal_row(df)
    if not sig_row:
        logging.info(f"{symbol}: not enough data to compute signal.")
        return

    prev_sig, last_sig, last_close, last_idx = sig_row
    logging.info(f"{symbol} last bar {last_idx}: prev_sig={prev_sig}, last_sig={last_sig}, close={last_close}")

    qty_held = current_position_qty(symbol)

    # Determine risk-based quantity
    cash = get_cash()
    risk_pct = float(os.getenv("RISK_PCT", "0.01"))   # risk 1% of cash per trade
    sl_pct   = float(os.getenv("SL_PCT", "0.05"))     # stop 5% below
    tp_pct   = float(os.getenv("TP_PCT", "0.10"))     # take profit 10% above

    # Risk model: if SL is 5%, size so that a stop-out loses ~1% of cash.
    # risk_amount = cash * risk_pct = qty * entry_price * sl_pct => qty = risk_amount/(entry_price * sl_pct)
    risk_amount = cash * risk_pct
    qty_risk = risk_amount / max(0.01, last_close * sl_pct)
    # Also cap with what we can afford at market
    qty_afford = cash / max(0.01, last_close)
    qty = min(qty_risk, qty_afford)

    # Transitions
    entered_long = (prev_sig <= 0 and last_sig == 1)
    exit_long    = (prev_sig >= 0 and last_sig == -1)

    if entered_long:
        if qty_held > 0:
            logging.info(f"{symbol}: Long signal but already holding {qty_held}. Skipping new buy.")
            return
        # cancel any stale open orders just in case
        cancel_open_orders(symbol)
        place_bracket_order(symbol, "buy", qty, last_close, tp_pct, sl_pct)

    elif exit_long:
        if qty_held > 0:
            logging.info(f"{symbol}: Exit signal; selling {qty_held} to flatten.")
            cancel_open_orders(symbol)
            try:
                api.submit_order(symbol=symbol, qty=str(round_qty(qty_held)), side="sell", type="market", time_in_force="day")
            except Exception as e:
                logging.exception(f"Exit sell failed for {symbol}: {e}")
        else:
            logging.info(f"{symbol}: Exit signal but no position to close.")

# ---------- Backtest Logic ----------

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

# ---------- main driver ----------

if __name__ == "__main__":
    logging.basicConfig(
        filename='trading_bot.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    args = parse_args()
    tickers = [t.upper() for t in args.tickers if t and not t.startswith("-")]

    # Backtest each ticker over last ~year for quick sanity check
    for symbol in tickers:
        data = yf.download(symbol, start="2024-10-22", end="2025-10-21", auto_adjust=True, progress=False)
        data = calculate_indicators(data)
        data = generate_signals(data)
        data = data.loc[:, ~data.columns.duplicated()]
        final_balance, buy_count, sell_count, stop_loss_count, take_profit_count = backtest(data)
        print(f"Backtest {symbol}: ${final_balance:.2f} | BUYs:{buy_count} SELLs:{sell_count} SL:{stop_loss_count} TP:{take_profit_count}")

    # Optional live trading pass (single-shot). For a daemonized loop, see below.
    if args.live and ok_to_trade():
        # Optional market-hours check (paper will still accept after-hours, but you can gate with the clock)
        try:
            clock = api.get_clock()
            logging.info(f"Alpaca clock: is_open={clock.is_open} next_open={clock.next_open} next_close={clock.next_close}")
        except Exception as e:
            logging.warning(f"Clock check failed: {e}")

        for symbol in tickers:
            maybe_trade_symbol(symbol, timeframe=args.timeframe)
            time.sleep(0.5)  # be nice to APIs


'''
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
'''
