import ccxt
import pandas as pd
import numpy as np
from telegram import Bot
from datetime import datetime
import time

TOKEN = "8404086668:AAF3C8QP_sAAaqgwQubQf1lhOFCEqkx8GSg"
CHAT_ID = 5833369092

bot = Bot(TOKEN)
binance = ccxt.binance()

def fetch(symbol="XVG/USDT"):
    data = binance.fetch_ohlcv(symbol, "1h", limit=200)
    df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def indicators(df):
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["EMA26"] = df["close"].ewm(span=26).mean()

    delta = df["close"].diff()
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100/(1+rs))

    rsi = df["RSI"]
    low = rsi.rolling(14).min()
    high = rsi.rolling(14).max()
    df["StochRSI"] = (rsi - low) / (high - low) * 100

    return df

def analyze(symbol):
    df = indicators(fetch(symbol))
    last = df.iloc[-1]

    price = last["close"]
    rsi = last["RSI"]
    stoch = last["StochRSI"]
    ema12 = last["EMA12"]
    ema26 = last["EMA26"]

    buy = rsi < 35 and stoch < 20 and ema12 > ema26
    sell = rsi > 70 and stoch > 80 and ema12 < ema26

    if buy:
        return "BUY", price
    elif sell:
        return "SELL", price
    else:
        return "WAIT", price

def send(symbol):
    signal, price = analyze(symbol)

    bot.send_message(
        chat_id=CHAT_ID,
        text=f"🔔 {symbol}\nSignal: {signal}\nPrice: {price:.6f}\nTime: {datetime.now()}",
    )

while True:
    for coin in ["XVG/USDT", "ROSE/USDT", "GALA/USDT"]:
        send(coin)
    time.sleep(300)
