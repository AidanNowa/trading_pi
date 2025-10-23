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

def _ensure_single_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have a single-symbol, flat-column OHLCV DataFrame.
    If a multi-ticker DataFrame sneaks in, raise a clear error.
    """
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected single-symbol DataFrame with columns Open/High/Low/Close/Volume.")
    # Normalize capitalization just in case ("close" -> "Close")
    cols = {c: c.title() for c in df.columns}
    return df.rename(columns=cols)

def _scalar(x):
    """Return a Python scalar from Series/ndarray/scalar without FutureWarnings."""
    if hasattr(x, "iloc"):
        # 1-element Series
        return x.iloc[0]
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x).ravel()[0]
    return x


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

# ---------- Backtest and Strategy Helpers ----------

def ta_rsi(close: pd.Series, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def ta_atr(high: pd.Series, low: pd.Series, close: pd.Series, length=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    return atr


# ---------- Backtest and Strategy Logic ----------

#Faster EMA Crossover (with optional RSI filter) (much more acitve than orignal EMA
def signals_ema_cross(df, fast=8, slow=21, use_rsi=False, rsi_th=55):
    df = df.copy()
    df["EMA_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    df["Signal"] = 0
    cross_up   = (df["EMA_fast"] > df["EMA_slow"]) & (df["EMA_fast"].shift(1) <= df["EMA_slow"].shift(1))
    cross_down = (df["EMA_fast"] < df["EMA_slow"]) & (df["EMA_fast"].shift(1) >= df["EMA_slow"].shift(1))
    if use_rsi:
        rsi = ta_rsi(df["Close"], length=14)
        cross_up &= (rsi < rsi_th)  # e.g., buy when momentum resumes but not overbought
    df.loc[cross_up, "Signal"] = 1
    df.loc[cross_down, "Signal"] = -1
    return df

#MACD line cross (very common and moderately active)
def signals_macd(df, fast=12, slow=26, signal_len=9):
    df = df.copy()
    df = df.xs(symbol, axis=1, level='Ticker')
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_len, adjust=False).mean()
    df["Signal"] = 0
    df.loc[(macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1)), "Signal"] = 1
    df.loc[(macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1)), "Signal"] = -1
    return df

#Bollinger mean-reversion ("bounce" is very active, "breakout"is trend-following)
def signals_bollinger(symbol, df, length=20, mult=2.0, mode="bounce"):
    df = df.copy()
    df = df.xs(symbol, axis=1, level='Ticker')
    ma = df["Close"].rolling(length).mean()
    sd = df["Close"].rolling(length).std(ddof=0)
    upper = ma + mult * sd
    lower = ma - mult * sd
    df["Signal"] = 0

    if mode == "bounce":
        # contrarian: buy near lower band, sell near upper band
        buy  = (df["Close"] < lower) & (df["Close"].shift(1) >= lower.shift(1))
        sell = (df["Close"] > upper) & (df["Close"].shift(1) <= upper.shift(1))
    else:  # "breakout"
        buy  = (df["Close"] > upper) & (df["Close"].shift(1) <= upper.shift(1))
        sell = (df["Close"] < lower) & (df["Close"].shift(1) >= lower.shift(1))

    df.loc[buy, "Signal"] = 1
    df.loc[sell, "Signal"] = -1
    return df

#Donchain breakout + ATR filer (fires on highest-N/lowest-N breakouts, ATR to reduce chop)
def signals_donchian_atr(symbol, df, ch_len=20, atr_len=14, atr_min_mult=0.5):
    df = df.copy()
    df = df.xs(symbol, axis=1, level='Ticker')
    don_high = df["High"].rolling(ch_len).max()
    don_low  = df["Low"].rolling(ch_len).min()
    atr = ta_atr(df["High"], df["Low"], df["Close"], length=atr_len)
    atr_ok = atr > (atr.rolling(atr_len).mean() * atr_min_mult)

    df["Signal"] = 0
    buy  = (df["Close"] > don_high.shift(1)) & atr_ok
    sell = (df["Close"] < don_low.shift(1))  & atr_ok
    df.loc[buy, "Signal"] = 1
    df.loc[sell, "Signal"] = -1
    return df

#Stochastic RSI cross (very active, will generate lots of trades)
def signals_stoch_rsi(symbol, df, rsi_len=14, stoch_len=14, k=3, d=3, buy_th=0.2, sell_th=0.8):
    df = df.copy()
    df = df.xs(symbol, axis=1, level='Ticker')
    #print(df.head())
    rsi = ta_rsi(df["Close"], rsi_len)
    min_rsi = rsi.rolling(stoch_len).min()
    max_rsi = rsi.rolling(stoch_len).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-9)
    k_ = stoch.rolling(k).mean()
    d_ = k_.rolling(d).mean()

    df["Signal"] = 0
    buy  = (k_.shift(1) < buy_th)  & (k_ > buy_th)  & (k_ > d_)
    sell = (k_.shift(1) > sell_th) & (k_ < sell_th) & (k_ < d_)
    df.loc[buy, "Signal"] = 1
    df.loc[sell, "Signal"] = -1
    return df

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
    #print(data[['EMA_12','EMA_26','RSI','Signal']].tail(20))
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

        price = float(row['Close']) #float(row['Close'].iloc[0]) #float(row['Close'])

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
        #data = calculate_indicators(data)
        #data = generate_signals(data)

        # choose ONE:
        #data = signals_ema_cross(data, fast=8, slow=21, use_rsi=False)     # very active
       # data = signals_macd(symbol, data)                                          # moderate
        #data = signals_bollinger(symbol, data, mode="bounce")                      # contrarian
        #data = signals_bollinger(symbol, data, mode="breakout")                    # trend
        #data = signals_donchian_atr(symbol, data)                                  # trend + atr
        data = signals_stoch_rsi(symbol, data)                                     # very active

        #clean dup columns just in case they exist
        data = data.loc[:, ~data.columns.duplicated()]
        #print(data[['Signal']].tail(20))
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
