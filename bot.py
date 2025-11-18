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

bot = Bot(TOKEN)


# ========= جلب بيانات XVG من CoinGecko =========
def fetch_ohlc():
    """
    نجلب بيانات سعر XVG مقابل الدولار من CoinGecko
    ونستخدمها كبديل لـ Binance.
    """
    url = "https://api.coingecko.com/api/v3/coins/verge/market_chart"
    params = {
        "vs_currency": "usd",   # تقريبًا تعادل USDT
        "days": 1,              # آخر 24 ساعة
        "interval": "hourly"    # شموع كل ساعة
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    # CoinGecko يرجع: [timestamp, price] في القائمة "prices"
    prices = data.get("prices", [])
    if not prices:
        raise ValueError("لا توجد بيانات أسعار من CoinGecko")

    df = pd.DataFrame(prices, columns=["time", "close"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df


# ========= حساب المؤشرات الفنية =========
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMA 12 و EMA 26
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()

    rs = roll_up / roll_down
    df["RSI14"] = 100 - (100 / (1 + rs))

    return df


# ========= بناء الرسالة للبوت =========
def build_message(df: pd.DataFrame) -> str:
    last = df.iloc[-1]

    price = last["close"]
    ema12 = last["EMA12"]
    ema26 = last["EMA26"]
    rsi = last["RSI14"]

    trend = "🔼 ترند صاعد" if ema12 > ema26 else "🔽 ترند هابط"
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    msg = (
        "📊 إشارات XVG (من CoinGecko)\n"
        f"⏰ الوقت: {now}\n"
        f"💰 السعر التقريبي: {price:.6f} USD\n\n"
        f"📈 EMA12: {ema12:.6f}\n"
        f"📉 EMA26: {ema26:.6f}\n"
        f"💡 RSI14: {rsi:.2f}\n\n"
        f"{trend}\n"
    )

    # إضافة تفسير بسيط للـ RSI
    if rsi >= 70:
        msg += "⚠️ المنطقة: تشبع شرائي محتمل (Overbought)\n"
    elif rsi <= 30:
        msg += "✅ المنطقة: تشبع بيعي محتمل (Oversold)\n"
    else:
        msg += "ℹ️ المنطقة: حركة متوازنة تقريبًا.\n"

    return msg


# ========= الحلقة الرئيسية =========
def main():
    if not TOKEN or not CHAT_ID:
        raise RuntimeError("الرجاء التأكد من ضبط متغيرات البيئة TOKEN و CHAT_ID في Render")

    bot.send_message(chat_id=CHAT_ID, text="✅ بوت VertexSignalsAI تم تشغيله بنجاح (CoinGecko).")

    while True:
        try:
            df = fetch_ohlc()
            df = indicators(df)
            text = build_message(df)
            bot.send_message(chat_id=CHAT_ID, text=text)
        except Exception as e:
            # نرسل الخطأ للتليجرام ليسهل تتبعه
            try:
                bot.send_message(chat_id=CHAT_ID, text=f"❌ حدث خطأ في البوت:\n{e}")
            except Exception:
                # لو فشل الإرسال نتجاهل فقط
                pass

        # انتظر 30 دقيقة بين كل تحديث وآخر (يمكنك تعديلها)
        time.sleep(60 * 30)


if __name__ == "__main__":
    main()
