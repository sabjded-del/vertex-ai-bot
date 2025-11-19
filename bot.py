import os
import time
import requests
import pandas as pd
import numpy as np
from telegram import Bot
from datetime import datetime

# ========= إعداد التوكن والـ Chat ID من متغيرات البيئة =========
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise RuntimeError("الرجاء ضبط متغيرات البيئة TOKEN و CHAT_ID في Render")

bot = Bot(TOKEN)

# ========= قائمة العملات =========
# المفتاح = رمز العملة اللي تحبه
# القيمة = اسم العملة في موقع CoinGecko (coin id)
COINS = {
    "XVG": "verge",
    # أمثلة لعملاتك، عدّل و/or زِد براحتك:
    # "ROSE": "oasis-network",
    # "GALA": "gala",
    # "BLUR": "blur",
    # "KAIA": "kaia",
    # "FIL": "filecoin",
}


# ========= جلب بيانات من CoinGecko =========
def fetch_ohlc(coin_id: str) -> pd.DataFrame:
    """
    نجلب بيانات السعر لعملة معينة من CoinGecko
    coin_id هو الاسم الخاص بالعملة في CoinGecko مثل verge, oasis-network
    """
    url = "https://api.coingecko.com/api/v3/coins/" + coin_id + "/market_chart"
    params = {
        "vs_currency": "usd",   # تقريبًا تعادل USDT
        "days": 1,              # آخر 24 ساعة
        "interval": "hourly",   # شموع كل ساعة
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    prices = data.get("prices", [])
    if not prices:
        raise ValueError(f"لا توجد بيانات أسعار من CoinGecko للعملة {coin_id}")

    df = pd.DataFrame(prices, columns=["time", "close"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df


# ========= حساب المؤشرات الفنية =========
def add_indicators(df
