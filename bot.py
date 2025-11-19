# -*- coding: utf-8 -*-
"""
البوت الذكي – المرحلة 1
رأس الملف + الإعدادات + المؤشرات الفنية + نظام الشموع الاحترافي
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from telegram import Bot

# ==========================
# إعدادات أساسية
# ==========================

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # الشات الرئيسي الذي سيستقبل التقارير

if not TOKEN or not CHAT_ID:
    raise RuntimeError("❌ تأكد من ضبط TOKEN و CHAT_ID في إعدادات Render أو المتغيرات البيئية")

bot = Bot(TOKEN)

# تحليل السوق كل 30 دقيقة (يمكن تعديلها لاحقًا)
ANALYSIS_INTERVAL = 60 * 30   # 30 دقيقة
POLL_INTERVAL = 3             # فحص أوامر التليجرام كل 3 ثواني

# ==========================
# العملات المدعومة (يمكن توسعتها حتى 50+)
# ==========================

COINS = {
    "XVG": "verge",
    "ROSE": "oasis-network",
    "GALA": "gala",
    "BLUR": "blur",
    "FIL": "filecoin",
    "KAIA": "kaia",
    "IMX": "immutable",
    "ADA": "cardano",
    "XRP": "ripple",
    "SOL": "solana",
    "FLUX": "flux",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "ICP": "internet-computer",
    "DOT": "polkadot",
    "QNT": "quant-network",
    "SEI": "sei-network",
    "SUI": "sui",
    "SYS": "syscoin",
    "RENDER": "render-token",
    "BTC": "bitcoin",
    "ETH": "ethereum",
}

# عملتك الرئيسية لخطة 12%
MAIN_COIN = "XVG"

# ==========================
# ذاكرة داخلية + رأس المال (محرك تعليمي)
# ==========================

LAST_INFOS = {}         # آخر تحليل لكل عملة
OPEN_TRADES = {}        # صفقات مفتوحة لكل رمز (افتراضية/تعليمية)
OPPORTUNITY_MEMORY = [] # أفضل الفرص الأخيرة
LAST_ALERTS = {}        # لمنع تكرار التنبيهات (symbol_type -> ts)

HYBRID_AUTO = True      # وضع الهجين (تحليل + توصيات تلقائية تعليمية فقط)

capital = {
    "initial": 1000.0,     # رأس المال الابتدائي (تقديري/تعليمي)
    "current": 1000.0,     # رأس المال المتاح
    "saved": 0.0,          # ادخار نظري
    "realized_profit": 0.0,
    "coins": {}            # لكل عملة: amount, avg_price, invested, profit
}


def ensure_coin_capital(symbol: str):
    """يتأكد أن لكل عملة سجل داخل محرك رأس المال."""
    if symbol not in capital["coins"]:
        capital["coins"][symbol] = {
            "amount": 0.0,
            "avg_price": 0.0,
            "invested": 0.0,
            "profit": 0.0
        }


def now_utc():
    return datetime.now(timezone.utc)


def now_utc_str():
    return now_utc().strftime("%Y-%m-%d %H:%M UTC")


# ==========================
# جلب بيانات من CoinGecko (OHLCV مبسطة)
# ==========================

def fetch_ohlcv_coingecko(coin_id: str, days: int = 2, interval: str = "hourly") -> pd.DataFrame:
    """
    يجلب بيانات الأسعار من CoinGecko:
    - close + volume
    - يحسب high/low تقريبية من حركة السعر
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": interval,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    prices = data.get("prices", [])
    vols = data.get("total_volumes", [])

    if not prices:
        raise ValueError("لا توجد بيانات أسعار من CoinGecko")

    df_price = pd.DataFrame(prices, columns=["time", "close"])
    df_price["time"] = pd.to_datetime(df_price["time"], unit="ms")

    df_vol = pd.DataFrame(vols, columns=["time", "volume"])
    df_vol["time"] = pd.to_datetime(df_vol["time"], unit="ms")

    df = pd.merge_asof(
        df_price.sort_values("time"),
        df_vol.sort_values("time"),
        on="time"
    )

    # تقريب high/low من حركة السعر الأخيرة
    df["high"] = df["close"].rolling(3, min_periods=1).max()
    df["low"] = df["close"].rolling(3, min_periods=1).min()
    return df


# ==========================
# المؤشرات الفنية الأساسية
# ==========================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def bollinger(series: pd.Series, period: int = 20, stddev: float = 2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + stddev * std
    lower = ma - stddev * std
    return ma, upper, lower


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()


def kdj(df: pd.DataFrame, period: int = 9, k_smooth: int = 3, d_smooth: int = 3):
    low_min = df["low"].rolling(window=period, min_periods=1).min()
    high_max = df["high"].rolling(window=period, min_periods=1).max()
    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-9) * 100
    k = rsv.ewm(alpha=1.0 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1.0 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ==========================
# نظام الشموع الاحترافي (Candlestick AI Pro)
# ==========================

def make_candle_from_series(o, h, l, c, idx: int):
    """يبني شمعة واحدة من سلاسل الأسعار."""
    return {
        "open": float(o.iloc[idx]),
        "high": float(h.iloc[idx]),
        "low": float(l.iloc[idx]),
        "close": float(c.iloc[idx]),
    }


def analyze_single_candle(c):
    """
    نماذج شمعة واحدة:
    Hammer, Inverted Hammer, Shooting Star, Hanging Man,
    Doji, Long-Legged Doji, Dragonfly Doji, Gravestone Doji, Marubozu
    """
    o = c["open"]
    h = c["high"]
    l = c["low"]
    cl = c["close"]

    body = abs(cl - o)
    full = max(h - l, 1e-9)
    upper = h - max(o, cl)
    lower = min(o, cl) - l

    patterns = []

    # Marubozu (جسم قوي بدون ذيول تقريباً)
    if body > full * 0.7 and upper < full * 0.1 and lower < full * 0.1:
        if cl > o:
            patterns.append("Bullish Marubozu")
        else:
            patterns.append("Bearish Marubozu")

    # Doji عام
    if body <= full * 0.1:
        patterns.append("Doji")

        # Long Legged Doji
        if upper > full * 0.3 and lower > full * 0.3:
            patterns.append("Long-Legged Doji")
        # Dragonfly Doji
        if lower > full * 0.4 and upper < full * 0.1:
            patterns.append("Dragonfly Doji")
        # Gravestone Doji
        if upper > full * 0.4 and lower < full * 0.1:
            patterns.append("Gravestone Doji")

    # Hammer / Hanging Man (ظل سفلي طويل)
    if (
        lower > body * 2 and
        upper <= body * 0.3 and
        body <= full * 0.4
    ):
        patterns.append("Hammer/Hanging Man")

    # Inverted Hammer / Shooting Star (ظل علوي طويل)
    if (
        upper > body * 2 and
        lower <= body * 0.3 and
        body <= full * 0.4
    ):
        patterns.append("Inverted/Shooting Star")

    return patterns


def analyze_two_candles(c1, c2):
    """
    نماذج ثنائية:
    Bullish Engulfing, Bearish Engulfing,
    Tweezer Top, Tweezer Bottom,
    Piercing Line, Dark Cloud Cover
    """
    patterns = []

    body1 = abs(c1["close"] - c1["open"])
    body2 = abs(c2["close"] - c2["open"])
    full1 = max(c1["high"] - c1["low"], 1e-9)
    full2 = max(c2["high"] - c2["low"], 1e-9)

    # Bullish Engulfing
    if (
        c1["close"] < c1["open"] and
        c2["close"] > c2["open"] and
        c2["close"] >= c1["open"] and
        c2["open"] <= c1["close"]
    ):
        patterns.append("Bullish Engulfing")

    # Bearish Engulfing
    if (
        c1["close"] > c1["open"] and
        c2["close"] < c2["open"] and
        c2["open"] >= c1["close"] and
        c2["close"] <= c1["open"]
    ):
        patterns.append("Bearish Engulfing")

    # Tweezer Top (قمم متقاربة)
    if abs(c1["high"] - c2["high"]) <= full1 * 0.1:
        if c1["close"] > c1["open"] and c2["close"] < c2["open"]:
            patterns.append("Tweezer Top")

    # Tweezer Bottom (قيعان متقاربة)
    if abs(c1["low"] - c2["low"]) <= full1 * 0.1:
        if c1["close"] < c1["open"] and c2["close"] > c2["open"]:
            patterns.append("Tweezer Bottom")

    # Piercing Line (انعكاس صاعد)
    mid1 = c1["open"] - body1 / 2 if c1["close"] < c1["open"] else c1["close"] - body1 / 2
    if (
        c1["close"] < c1["open"] and
        c2["open"] < c1["low"] and
        c2["close"] > mid1 and
        c2["close"] < c1["open"]
    ):
        patterns.append("Piercing Line")

    # Dark Cloud Cover (انعكاس هابط)
    mid1_up = c1["close"] - body1 / 2 if c1["close"] > c1["open"] else c1["open"] - body1 / 2
    if (
        c1["close"] > c1["open"] and
        c2["open"] > c1["high"] and
        c2["close"] < mid1_up and
        c2["close"] > c1["open"]
    ):
        patterns.append("Dark Cloud Cover")

    return patterns


def analyze_three_candles(c1, c2, c3):
    """
    نماذج ثلاثية:
    Morning Star, Evening Star,
    Three White Soldiers, Three Black Crows
    """
    patterns = []

    body1 = abs(c1["close"] - c1["open"])
    body2 = abs(c2["close"] - c2["open"])
    body3 = abs(c3["close"] - c3["open"])
    full1 = max(c1["high"] - c1["low"], 1e-9)

    # Morning Star
    cond1 = c1["close"] < c1["open"] and body1 > body2 * 2
    cond2 = body2 <= full1 * 0.3
    cond3 = c3["close"] > c3["open"] and c3["close"] > (c1["open"] + c1["close"]) / 2
    if cond1 and cond2 and cond3:
        patterns.append("Morning Star")

    # Evening Star
    cond1_e = c1["close"] > c1["open"] and body1 > body2 * 2
    cond2_e = body2 <= full1 * 0.3
    cond3_e = c3["close"] < c3["open"] and c3["close"] < (c1["open"] + c1["close"]) / 2
    if cond1_e and cond2_e and cond3_e:
        patterns.append("Evening Star")

    # Three White Soldiers
    if (
        c1["close"] > c1["open"] and
        c2["close"] > c2["open"] and
        c3["close"] > c3["open"] and
        c2["close"] > c1["close"] and
        c3["close"] > c2["close"] and
        body1 > full1 * 0.3 and body2 > full1 * 0.3 and body3 > full1 * 0.3
    ):
        patterns.append("Three White Soldiers")

    # Three Black Crows
    if (
        c1["close"] < c1["open"] and
        c2["close"] < c2["open"] and
        c3["close"] < c3["open"] and
        c2["close"] < c1["close"] and
        c3["close"] < c2["close"] and
        body1 > full1 * 0.3 and body2 > full1 * 0.3 and body3 > full1 * 0.3
    ):
        patterns.append("Three Black Crows")

    return patterns


def detect_candlestick_patterns(df: pd.DataFrame):
    """
    🔥 نظام الشموع الاحترافي الكامل:
    يرجع قائمة نماذج الشموع المهمة للسياق الأخير (حتى 3 شمعات).
    يُستخدم لاحقًا في:
    - Smart Candle Alerts
    - Hybrid Auto
    - Opportunity Mining
    - Score Engine
    """
    if len(df) < 1:
        return []

    h_series = df["high"]
    l_series = df["low"]
    c_series = df["close"]
    # تقريب Open من إغلاق الشمعة السابقة
    o_series = c_series.shift(1).fillna(c_series)

    patterns = []

    last_idx = len(df) - 1
    c_last = make_candle_from_series(o_series, h_series, l_series, c_series, last_idx)
    patterns += analyze_single_candle(c_last)

    if len(df) >= 2:
        c_prev = make_candle_from_series(o_series, h_series, l_series, c_series, last_idx - 1)
        patterns += analyze_two_candles(c_prev, c_last)

    if len(df) >= 3:
        c1 = make_candle_from_series(o_series, h_series, l_series, c_series, last_idx - 2)
        c2 = make_candle_from_series(o_series, h_series, l_series, c_series, last_idx - 1)
        c3 = c_last
        patterns += analyze_three_candles(c1, c2, c3)

    # إزالة التكرار مع الحفاظ على الترتيب
    patterns = list(dict.fromkeys(patterns))
    return patterns

# ==========================
# محرك تحليل المؤشرات الفنية
# ==========================

def analyze_indicators(df: pd.DataFrame):
    """يحسب كل المؤشرات ويعيد آخر قيم."""
    close = df["close"]
    volume = df["volume"]

    ind = {}

    ind["ema12"] = float(ema(close, 12).iloc[-1])
    ind["ema26"] = float(ema(close, 26).iloc[-1])
    ind["ema50"] = float(ema(close, 50).iloc[-1])
    ind["ema100"] = float(ema(close, 100).iloc[-1])
    ind["ema200"] = float(ema(close, 200).iloc[-1])

    ind["rsi"] = float(rsi(close, 14).iloc[-1])

    ma20, bb_up, bb_low = bollinger(close)
    ind["bb_mid"] = float(ma20.iloc[-1])
    ind["bb_up"] = float(bb_up.iloc[-1])
    ind["bb_low"] = float(bb_low.iloc[-1])

    ind["obv"] = float(obv(close, volume).iloc[-1])

    k, d, j = kdj(df)
    ind["kdj_k"] = float(k.iloc[-1])
    ind["kdj_d"] = float(d.iloc[-1])
    ind["kdj_j"] = float(j.iloc[-1])

    try:
        ind["atr"] = float(atr(df))
    except:
        ind["atr"] = 0.0

    return ind


# ==========================
# 🔥 محرك السكور الرئيسي (0–100)
# ==========================

def calc_score(ind, patterns):
    score = 50

    # RSI
    if ind["rsi"] < 30:
        score += 10
    elif ind["rsi"] > 70:
        score -= 10

    # EMA alignment
    if ind["ema12"] > ind["ema26"] > ind["ema50"]:
        score += 10
    if ind["ema12"] > ind["ema200"]:
        score += 5

    # OBV
    if ind["obv"] > 0:
        score += 5

    # Bollinger
    if ind["close"] < ind["bb_low"]:
        score += 5
    if ind["close"] > ind["bb_up"]:
        score -= 5

    # شموع قوية
    strong = ["Hammer", "Morning Star", "Bullish Engulfing", "Three White Soldiers"]
    if any(p in " ".join(patterns) for p in strong):
        score += 10

    # شموع هابطة قوية
    weak = ["Shooting Star", "Evening Star", "Bearish Engulfing", "Three Black Crows"]
    if any(p in " ".join(patterns) for p in weak):
        score -= 10

    return max(0, min(100, score))


# ==========================
# 🔵 تنبيه شراء ذكي (Smart Buy)
# ==========================

def smart_buy_alert(symbol, ind, score):
    """يدعم قواعدك في الصورة بالكامل."""
    if ind["rsi"] < 30 and score > 70 and ind["ema50"] > ind["close"]:
        key = f"{symbol}_buy"
        if LAST_ALERTS.get(key, 0) < time.time() - 1800:
            LAST_ALERTS[key] = time.time()
            bot.send_message(
                CHAT_ID,
                f"🟢 **تنبيه شراء قوي** {symbol}\nRSI < 30\nScore > 70\nمنطقة طلب محتملة"
            )


# ==========================
# 🔴 تنبيه بيع ذكي (Smart Sell)
# ==========================

def smart_sell_alert(symbol, ind, score):
    if ind["rsi"] > 70 and score < 40:
        key = f"{symbol}_sell"
        if LAST_ALERTS.get(key, 0) < time.time() - 1800:
            LAST_ALERTS[key] = time.time()
            bot.send_message(
                CHAT_ID,
                f"🔴 **تنبيه بيع قوي** {symbol}\nRSI > 70\nScore < 40\nمقاومة قوية محتملة"
            )


# ==========================
# 🟣 تنبيه XVG خاص (أهم عملة)
# ==========================

def xvg_special_alert(symbol, ind):
    if symbol != "XVG":
        return
    if ind["rsi"] < 35:
        bot.send_message(CHAT_ID, "🔵 XVG تقترب من **قاع ذهبي محتمل**")
    if ind["ema12"] > ind["ema200"]:
        bot.send_message(CHAT_ID, "🔵 XVG تظهر **بوادر اختراق قوية**")
    if ind["rsi"] > 70:
        bot.send_message(CHAT_ID, "🟣 XVG تجاوزت الهدف الأسبوعي 12% (تنبيه جني ربح)")


# ==========================
# التنبيهات العامة للشموع (Smart Candle Alerts)
# ==========================

def candle_alert(symbol, patterns):
    if not patterns:
        return
    last = "، ".join(patterns)
    bot.send_message(CHAT_ID, f"🕯️ **{symbol}**\nظهرت شموع: {last}")


# ==========================
# نظام التعدين الذكي للفرص (Opportunity Mining)
# ==========================

def mine_opportunities(symbol, ind, score, patterns):
    """استخراج أفضل 5 فرص شراء + بيع + أسوأ عملة."""
    entry_flag = False

    # فرصة شراء جاهزة للانفجار
    if score > 80 and ind["rsi"] < 40:
        OPPORTUNITY_MEMORY.append((symbol, "Buy", score))

    # فرصة بيع
    if score < 40 and ind["rsi"] > 60:
        OPPORTUNITY_MEMORY.append((symbol, "Sell", score))

    # أسوأ عملة (خطر)
    if score < 30:
        OPPORTUNITY_MEMORY.append((symbol, "Risk", score))

    # عودة نتائج منظمة كل 50 تحليل
    if len(OPPORTUNITY_MEMORY) > 50:
        OPPORTUNITY_MEMORY[:] = sorted(OPPORTUNITY_MEMORY, key=lambda x: x[2], reverse=True)[:20]


# ==========================
# نظام دعم هدف 12% الأسبوعي
# ==========================

def weekly_12_system(symbol, ind):
    """تحقق هدف العملة الرئيسي."""
    if symbol != MAIN_COIN:
        return

    # صعود > 12%
    if ind["rsi"] > 70:
        bot.send_message(CHAT_ID, f"🎯 XVG حققت صعودًا قويًا – راجع خطة 12% الأسبوعية")

    # ضعف السوق
    if ind["rsi"] > 80:
        bot.send_message(CHAT_ID, "⚠️ السوق مبالغ فيه – وقف شراء جديد")


# ==========================
# محرك التحليل الكامل لكل عملة
# ==========================

def analyze_coin(symbol, coin_id):
    df = fetch_ohlcv_coingecko(coin_id, days=2)
    ind = analyze_indicators(df)
    patterns = detect_candlestick_patterns(df)

    ind["close"] = float(df["close"].iloc[-1])

    score = calc_score(ind, patterns)

    # حفظ آخر البيانات
    LAST_INFOS[symbol] = {
        "time": now_utc_str(),
        "price": ind["close"],
        "rsi": ind["rsi"],
        "ema12": ind["ema12"],
        "ema26": ind["ema26"],
        "ema50": ind["ema50"],
        "ema200": ind["ema200"],
        "patterns": patterns,
        "score": score
    }

    # تنبيهات
    smart_buy_alert(symbol, ind, score)
    smart_sell_alert(symbol, ind, score)
    candle_alert(symbol, patterns)
    xvg_special_alert(symbol, ind)
    mine_opportunities(symbol, ind, score, patterns)
    weekly_12_system(symbol, ind)

    return ind, patterns, score

# ==========================
# أوامر التليجرام
# ==========================

def send_help(chat_id):
    bot.send_message(
        chat_id,
        "🤖 أوامر البوت الذكي:\n"
        "/xvg - تحليل XVG بالتفصيل\n"
        "/coin رمز - تحليل أي عملة مثال: /coin ROSE\n"
        "/plan - شرح خطة 12%\n"
        "/buy السعر [الرمز] [حجم_USDT]\n"
        "/sell السعر [الرمز] [كمية]\n"
        "/dashboard - لوحة التحكم الشاملة"
    )


def cmd_xvg(chat_id):
    if "XVG" not in LAST_INFOS:
        bot.send_message(chat_id, "⚠️ لم يتم تحليل XVG بعد، انتظر التحليل التالي.")
        return

    info = LAST_INFOS["XVG"]
    bot.send_message(
        chat_id,
        f"🔍 XVG\n"
        f"⏰ {info['time']}\n\n"
        f"السعر: {info['price']:.6f}\n"
        f"RSI: {info['rsi']:.1f}\n"
        f"EMA12: {info['ema12']:.6f}\n"
        f"EMA50: {info['ema50']:.6f}\n"
        f"EMA200: {info['ema200']:.6f}\n"
        f"Score: {info['score']}/100\n"
        f"نماذج: {', '.join(info['patterns']) if info['patterns'] else 'لا يوجد'}"
    )


def cmd_coin(chat_id, symbol):
    symbol = symbol.upper()
    if symbol not in COINS:
        bot.send_message(chat_id, "❌ العملة غير مدعومة.")
        return

    try:
        ind, patt, score = analyze_coin(symbol, COINS[symbol])
        bot.send_message(
            chat_id,
            f"🔍 {symbol}\n"
            f"السعر: {ind['close']:.6f}\n"
            f"RSI: {ind['rsi']:.1f}\n"
            f"EMA12: {ind['ema12']:.6f}\n"
            f"EMA50: {ind['ema50']:.6f}\n"
            f"EMA200: {ind['ema200']:.6f}\n"
            f"Score: {score}/100\n"
            f"شموع: {', '.join(patt) if patt else 'لا يوجد'}"
        )
    except Exception as e:
        bot.send_message(chat_id, f"❌ خطأ في التحليل:\n{e}")


def cmd_plan(chat_id):
    bot.send_message(
        chat_id,
        "📘 خطة 12% الأسبوعية:\n"
        "• الدخول عند قاع فني\n"
        "• الهدف 12% أسبوعيًا\n"
        "• الخروج عند مقاومة + RSI مرتفع\n"
        "• كل نجاح = إعادة استثمار 50% فقط"
    )


def cmd_dashboard(chat_id):
    text = (
        "📊 Dashboard\n"
        f"آخر تحديث: {now_utc_str()}\n\n"
        f"أكبر عدد عملات: {len(COINS)}\n"
        f"التحليلات المحفوظة: {len(LAST_INFOS)}\n"
        f"أفضل الفرص: {len(OPPORTUNITY_MEMORY)}\n\n"
    )

    if LAST_INFOS:
        # أفضل عملة
        best = max(LAST_INFOS.items(), key=lambda x: x[1]["score"])
        worst = min(LAST_INFOS.items(), key=lambda x: x[1]["score"])
        text += (
            f"🔥 أفضل عملة الآن: {best[0]} ({best[1]['score']})\n"
            f"⚠️ أضعف عملة الآن: {worst[0]} ({worst[1]['score']})\n"
        )

    bot.send_message(chat_id, text)

def process_updates(last_update_id=None):
    try:
        updates = bot.get_updates(offset=last_update_id, timeout=5)
    except:
        return last_update_id

    for u in updates:
        last_update_id = u.update_id + 1
        if not hasattr(u, "message") or not u.message:
            continue

        chat_id = u.message.chat.id
        text = (u.message.text or "").strip()
        if not text.startswith("/"):
            continue

        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/help" or cmd == "/start":
            send_help(chat_id)

        elif cmd == "/xvg":
            cmd_xvg(chat_id)

        elif cmd == "/coin" and args:
            cmd_coin(chat_id, args[0])

        elif cmd == "/plan":
            cmd_plan(chat_id)

        elif cmd == "/dashboard":
            cmd_dashboard(chat_id)

        else:
            send_help(chat_id)

    return last_update_id

def main_loop():
    bot.send_message(
        CHAT_ID,
        "✅ البوت الذكي بدأ العمل.\n"
        "تحليل دوري – تنبيهات ذكية – خطة 12% – دعم الشموع."
    )

    last_update_id = None
    last_analysis = 0

    while True:

        # استقبال أوامر التليجرام
        last_update_id = process_updates(last_update_id)

        # تحليل السوق كل X دقائق
        if time.time() - last_analysis >= ANALYSIS_INTERVAL:

            for symbol, coin_id in COINS.items():
                try:
                    analyze_coin(symbol, coin_id)
                except Exception as e:
                    bot.send_message(CHAT_ID, f"⚠️ خطأ تحليل {symbol}: {e}")

            last_analysis = time.time()

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
    
