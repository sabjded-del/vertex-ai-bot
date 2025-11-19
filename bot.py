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
CHAT_ID = os.getenv("CHAT_ID")  # الشات الرئيسي

if not TOKEN or not CHAT_ID:
    raise RuntimeError("❌ تأكد من ضبط TOKEN و CHAT_ID في إعدادات Render")

bot = Bot(TOKEN)

ANALYSIS_INTERVAL = 60 * 15   # تحليل السوق كل 15 دقيقة
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

MAIN_COIN = "XVG"   # عملتك الرئيسية لخطة 12%

# ==========================
# ذاكرة داخلية + رأس المال
# ==========================

LAST_INFOS = {}         # آخر تحليل لكل عملة
OPEN_TRADES = {}        # صفقات مفتوحة لكل رمز
OPPORTUNITY_MEMORY = [] # أفضل الفرص الأخيرة
LAST_ALERTS = {}        # لمنع تكرار التنبيهات (symbol_type -> ts)

HYBRID_AUTO = True      # وضع الهجين

# محرك رأس المال الداخلي (افتراضي / تعليمي)
capital = {
    "initial": 1000.0,     # رأس المال الابتدائي (تقديري)
    "current": 1000.0,     # رأس المال المستخدم
    "saved": 0.0,          # ادخار (نظري)
    "realized_profit": 0.0,
    "coins": {}            # لكل عملة: amount, avg_price, invested, profit
}


def ensure_coin_capital(symbol: str):
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
# جلب بيانات من CoinGecko
# ==========================

def fetch_ohlcv_coingecko(coin_id: str, days: int = 2, interval: str = "hourly") -> pd.DataFrame:
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
# المؤشرات الفنية
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
# Candlestick AI Engine (متقدم)
# ==========================

def make_candle_from_series(o, h, l, c, idx: int):
    return {
        "open": float(o.iloc[idx]),
        "high": float(h.iloc[idx]),
        "low": float(l.iloc[idx]),
        "close": float(c.iloc[idx]),
    }


def analyze_single_candle(c):
    """
    نماذج شمعة واحدة:
    Hammer, Inverted Hammer, Shooting Star, Hanging Man, Doji, Long-Legged Doji,
    Dragonfly Doji, Gravestone Doji, Marubozu
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

    # Hammer / Hanging Man / Inverted Hammer / Shooting Star
    # Hammer / Hanging Man (ظل سفلي طويل)
    if (
        lower > body * 2 and
        upper <= body * 0.3 and
        body <= full * 0.4
    ):
        # الاتجاه السابق يحدد: Hammer / Hanging Man
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
    Bullish Engulfing, Bearish Engulfing, Tweezer Top, Tweezer Bottom,
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
    Morning Star, Evening Star, Three White Soldiers, Three Black Crows
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
    يرجع قائمة نماذج الشموع للسياق الأخير (3 شمعات)
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

    # إزالة التكرار
    patterns = list(dict.fromkeys(patterns))
    return patterns


# ==========================
# Score + تحليل شامل (Balanced v2)
# ==========================

def calc_score(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    close = df["close"]
    volume = df["volume"].fillna(0)

    # ===== المؤشرات الأساسية =====
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    ema50 = ema(close, 50)
    ema100 = ema(close, 100)
    ema200 = ema(close, 200)

    rsi6 = rsi(close, 6)
    rsi12 = rsi(close, 12)
    rsi24 = rsi(close, 24)

    bb_mid, bb_up, bb_low = bollinger(close, 20, 2)
    obv_series = obv(close, volume)
    k, d, j = kdj(df)
    atr_val = atr(df, 14)

    price = float(last["close"])
    ema50_last = float(ema50.iloc[-1])
    ema100_last = float(ema100.iloc[-1])
    ema200_last = float(ema200.iloc[-1])

    # ===== Trend score (الوزن الخام 0–25) =====
    trend_score_raw = 0
    above_50 = price > ema50_last
    above_100 = price > ema100_last
    above_200 = price > ema200_last
    bull_stack = ema12.iloc[-1] > ema26.iloc[-1] > ema50_last > ema100_last > ema200_last
    bear_stack = ema12.iloc[-1] < ema26.iloc[-1] < ema50_last < ema100_last < ema200_last

    if above_50:
        trend_score_raw += 5
    if above_100:
        trend_score_raw += 5
    if above_200:
        trend_score_raw += 5
    if bull_stack:
        trend_score_raw += 10
    elif bear_stack and not above_50:
        trend_score_raw += 0
    trend_score_raw = min(trend_score_raw, 25)

    if bull_stack and above_200:
        trend_label, trend_ar = "strong_bull", "صاعد قوي 🔥"
    elif (above_50 and above_100) and price > ema200_last:
        trend_label, trend_ar = "bull", "صاعد ✅"
    elif bear_stack and not above_50 and not above_100 and not above_200:
        trend_label, trend_ar = "strong_bear", "هابط قوي 🚨"
    elif bear_stack and not above_50:
        trend_label, trend_ar = "bear", "هابط ⚠️"
    else:
        trend_label, trend_ar = "sideways", "تذبذب ⚪"

    # ===== RSI score (الوزن الخام 0–30) =====
    def rsi_part(val):
        if val < 25:
            return 10
        elif val < 70:
            return 5
        else:
            return -10

    r6 = float(rsi6.iloc[-1])
    r12 = float(rsi12.iloc[-1])
    r24 = float(rsi24.iloc[-1])
    rsi_score_raw = rsi_part(r6) + rsi_part(r12) + rsi_part(r24)
    rsi_score_raw = max(0, min(30, rsi_score_raw + 15))

    # ===== Bollinger score (الوزن الخام 0–15) =====
    b_low = bb_low.iloc[-1]
    b_mid = bb_mid.iloc[-1]
    b_up = bb_up.iloc[-1]
    bb_score_raw = 0
    if not np.isnan(b_low) and not np.isnan(b_up):
        if price <= b_low:
            bb_score_raw += 15
        elif price < b_mid:
            bb_score_raw += 8
        elif price >= b_up:
            bb_score_raw -= 10
    bb_score_raw = max(0, min(15, bb_score_raw))

    # ===== OBV score (الوزن الخام 0–15) =====
    obv_score_raw = 0
    if len(obv_series) >= 10:
        obv_last = obv_series.iloc[-1]
        obv_prev = obv_series.iloc[-10]
        if obv_last > obv_prev:
            obv_score_raw += 10
        else:
            obv_score_raw -= 5
    obv_score_raw = max(0, min(15, obv_score_raw + 5))

    # ===== KDJ score (الوزن الخام 0–15) =====
    k_last = float(k.iloc[-1])
    d_last = float(d.iloc[-1])
    k_prev = float(k.iloc[-2]) if len(k) > 1 else k_last
    golden_cross = k_last > d_last and (len(d) > 1 and k_prev < d.iloc[-2])
    dead_cross = k_last < d_last and (len(d) > 1 and k_prev > d.iloc[-2])

    kdj_score_raw = 0
    if golden_cross and k_last < 30:
        kdj_score_raw += 15
    elif k_last < 20:
        kdj_score_raw += 8
    elif dead_cross and k_last > 70:
        kdj_score_raw -= 10
    kdj_score_raw = max(0, min(15, kdj_score_raw + 5))

    # ===== دعم / مقاومة تقريبية =====
    recent_lows = df["low"].tail(40)
    recent_highs = df["high"].tail(40)
    support_level = float(recent_lows.min())
    resistance_level = float(recent_highs.max())

    zone = "neutral"
    if price <= support_level * 1.03:
        zone = "demand"
    elif price >= resistance_level * 0.97:
        zone = "supply"

    boll_state = "middle"
    if not np.isnan(b_low) and price <= b_low:
        boll_state = "lower"
    elif not np.isnan(b_up) and price >= b_up:
        boll_state = "upper"

    # ===== نماذج الشموع (Candlestick AI) =====
    patterns = detect_candlestick_patterns(df)

    candle_score_raw = 0
    bullish_patterns = {
        "Hammer/Hanging Man", "Bullish Engulfing", "Piercing Line",
        "Morning Star", "Three White Soldiers", "Dragonfly Doji"
    }
    bearish_patterns = {
        "Bearish Engulfing", "Dark Cloud Cover", "Evening Star",
        "Three Black Crows", "Gravestone Doji", "Tweezer Top"
    }

    for p in patterns:
        if p in bullish_patterns and zone == "demand":
            candle_score_raw += 8
        elif p in bullish_patterns:
            candle_score_raw += 5

        if p in bearish_patterns and zone == "supply":
            candle_score_raw += 8
        elif p in bearish_patterns:
            candle_score_raw += 4

        if "Doji" in p and zone in ("demand", "supply"):
            candle_score_raw += 2

    if "Morning Star" in patterns and zone == "demand":
        candle_score_raw += 10
    if "Evening Star" in patterns and zone == "supply":
        candle_score_raw += 10
    if "Three White Soldiers" in patterns and zone == "demand":
        candle_score_raw += 8
    if "Three Black Crows" in patterns and zone == "supply":
        candle_score_raw += 8

    candle_score_raw = max(0, min(15, candle_score_raw + 5))

    # ==========================
    #  ✅ موازنة الأوزان (Balanced Model رقم 2)
    # ==========================
    # Trend:   من 0–25  → يُعاد توزيعها إلى 0–30
    # RSI:     من 0–30  → 0–25
    # Boll:    من 0–15  → 0–15 (نفسه)
    # OBV:     من 0–15  → 0–10
    # KDJ:     من 0–15  → 0–10
    # Candles: من 0–15  → 0–10

    def scale(value, old_max, new_max):
        if old_max <= 0:
            return 0.0
        v = max(0.0, min(float(value), float(old_max)))
        return (v / old_max) * new_max

    trend_score = scale(trend_score_raw, 25, 30)
    rsi_score = scale(rsi_score_raw, 30, 25)
    bb_score = scale(bb_score_raw, 15, 15)
    obv_score = scale(obv_score_raw, 15, 10)
    kdj_score = scale(kdj_score_raw, 15, 10)
    candle_score = scale(candle_score_raw, 15, 10)

    total = trend_score + rsi_score + bb_score + obv_score + kdj_score + candle_score
    total = max(0, min(int(round(total)), 100))

    dist_ema50 = (price / ema50_last - 1) * 100 if ema50_last else 0.0
    dist_ema200 = (price / ema200_last - 1) * 100 if ema200_last else 0.0

    return {
        "score": total,

        # القيم بعد الموازنة (المهمة للقرار)
        "trend_score": trend_score,
        "rsi_score": rsi_score,
        "bb_score": bb_score,
        "obv_score": obv_score,
        "kdj_score": kdj_score,
        "candle_score": candle_score,

        # النسخ الخام (لمن يحب التحليل التفصيلي لاحقاً)
        "trend_score_raw": trend_score_raw,
        "rsi_score_raw": rsi_score_raw,
        "bb_score_raw": bb_score_raw,
        "obv_score_raw": obv_score_raw,
        "kdj_score_raw": kdj_score_raw,
        "candle_score_raw": candle_score_raw,

        "last_close": price,
        "rsi6": r6,
        "rsi12": r12,
        "rsi24": r24,
        "ema50": ema50_last,
        "ema100": ema100_last,
        "ema200": ema200_last,
        "bb_low": float(b_low) if not np.isnan(b_low) else None,
        "bb_mid": float(b_mid) if not np.isnan(b_mid) else None,
        "bb_up": float(b_up) if not np.isnan(b_up) else None,
        "support": support_level,
        "resistance": resistance_level,
        "trend_label": trend_label,
        "trend_ar": trend_ar,
        "dist_ema50": dist_ema50,
        "dist_ema200": dist_ema200,
        "golden_kdj": golden_cross,
        "dead_kdj": dead_cross,
        "atr": atr_val,
        "patterns": patterns,
        "zone": zone,
        "boll_state": boll_state,
    }


def classify_state(info: dict) -> str:
    s = info["score"]
    rsi6 = info["rsi6"]
    price = info["last_close"]
    support = info["support"]
    resistance = info["resistance"]

    if s >= 80 and rsi6 < 35 and price <= support * 1.03:
        return "🟢 قاع قوي / فرصة شراء ممتازة"
    if s >= 60 and rsi6 < 50:
        return "🟡 وضع إيجابي / فرصة محتملة"
    if s <= 35 and rsi6 > 70 and price >= resistance * 0.97:
        return "🔴 قرب قمة / خطر هبوط / وقت مثالي لجني ربح"
    return "⚪ منطقة تذبذب / لا وضوح قوي"


# ==========================
# تنبيهات ذكية + أصوات
# ==========================

def send_sound_alert(text: str, sound_type: str | None = None):
    """تنبيه نصي + محاولة إرسال صوت (اختياري)"""
    try:
        bot.send_message(chat_id=CHAT_ID, text=text)
        if sound_type:
            path = f"sounds/{sound_type}.ogg"
            try:
                with open(path, "rb") as f:
                    bot.send_audio(chat_id=CHAT_ID, audio=f)
            except Exception:
                pass
    except Exception:
        pass


def smart_alerts(all_infos: dict):
    now_ts = time.time()

    for sym, info in all_infos.items():
        price = info["last_close"]
        rsi6 = info["rsi6"]
        score = info["score"]
        support = info["support"]
        resistance = info["resistance"]
        bb_low = info["bb_low"]
        bb_up = info["bb_up"]
        trend = info["trend_ar"]
        patterns = info.get("patterns", [])
        zone = info.get("zone", "neutral")

        patterns_str = ", ".join(patterns) if patterns else "لا يوجد نموذج مهم"

        # Strong Buy
        strong_buy = (
            rsi6 < 30 and
            bb_low is not None and price <= bb_low and
            price <= support * 1.03 and
            score >= 70
        )
        if strong_buy:
            key = f"{sym}_strong_buy"
            if now_ts - LAST_ALERTS.get(key, 0) > 60 * 15:
                txt = (
                    f"🟢💎 تنبيه شراء قوي على {sym}\n"
                    f"السعر: {price:.6f}\n"
                    f"RSI6: {rsi6:.1f}\n"
                    f"الدعم: {support:.6f}\n"
                    f"Score: {score}\n"
                    f"الاتجاه: {trend}\n"
                    f"المنطقة: {zone}\n"
                    f"نموذج الشموع: {patterns_str}"
                )
                send_sound_alert(txt, sound_type="buy")
                LAST_ALERTS[key] = now_ts

        # Strong Sell
        strong_sell = (
            rsi6 > 70 and
            bb_up is not None and price >= bb_up and
            resistance > 0 and price >= resistance * 0.97 and
            score <= 40
        )
        if strong_sell:
            key = f"{sym}_strong_sell"
            if now_ts - LAST_ALERTS.get(key, 0) > 60 * 15:
                txt = (
                    f"🔴🚨 تنبيه بيع قوي على {sym}\n"
                    f"السعر: {price:.6f}\n"
                    f"RSI6: {rsi6:.1f}\n"
                    f"المقاومة: {resistance:.6f}\n"
                    f"Score: {score}\n"
                    f"الاتجاه: {trend}\n"
                    f"المنطقة: {zone}\n"
                    f"نموذج الشموع: {patterns_str}"
                )
                send_sound_alert(txt, sound_type="sell")
                LAST_ALERTS[key] = now_ts

        # Potential Bottom
        if rsi6 < 35 and price <= support * 1.05:
            key = f"{sym}_bottom"
            if now_ts - LAST_ALERTS.get(key, 0) > 60 * 30:
                txt = (
                    f"🟡📉 قاع محتمل على {sym}\n"
                    f"السعر: {price:.6f}\n"
                    f"RSI6: {rsi6:.1f}\n"
                    f"الدعم: {support:.6f}\n"
                    f"المنطقة: {zone}\n"
                    f"نموذج الشموع: {patterns_str}"
                )
                send_sound_alert(txt, sound_type="bottom")
                LAST_ALERTS[key] = now_ts

        # Potential Top
        if rsi6 > 65 and price >= resistance * 0.95:
            key = f"{sym}_top"
            if now_ts - LAST_ALERTS.get(key, 0) > 60 * 30:
                txt = (
                    f"🟠📈 قمة محتملة على {sym}\n"
                    f"السعر: {price:.6f}\n"
                    f"RSI6: {rsi6:.1f}\n"
                    f"المقاومة: {resistance:.6f}\n"
                    f"المنطقة: {zone}\n"
                    f"نموذج الشموع: {patterns_str}"
                )
                send_sound_alert(txt, sound_type="top")
                LAST_ALERTS[key] = now_ts


# ==========================
# Opportunity Mining
# ==========================

def mine_opportunities(all_infos: dict, top_n: int = 3):
    candidates = [
        (sym, info) for sym, info in all_infos.items()
        if info["score"] >= 70 and info["rsi6"] < 60
    ]
    candidates.sort(key=lambda x: x[1]["score"], reverse=True)
    best = candidates[:top_n]

    OPPORTUNITY_MEMORY.clear()
    for sym, info in best:
        OPPORTUNITY_MEMORY.append({
            "symbol": sym,
            "price": info["last_close"],
            "score": info["score"],
            "rsi6": info["rsi6"],
            "time": now_utc_str(),
        })
    return best


# ==========================
# تقارير
# ==========================

def build_coin_report(symbol: str, info: dict, is_main: bool = False) -> str:
    state = classify_state(info)
    patterns = info.get("patterns", [])
    patterns_str = ", ".join(patterns) if patterns else "لا يوجد"

    line1 = f"• {symbol}: {info['last_close']:.6f} USD | Score: {info['score']}/100"
    line2 = (
        f"  RSI(6/12/24): {info['rsi6']:.1f} / {info['rsi12']:.1f} / {info['rsi24']:.1f} | "
        f"Trend: {info.get('trend_ar', '')}"
    )
    line3 = f"  دعم: {info['support']:.6f} | مقاومة: {info['resistance']:.6f}"
    line4 = f"  نماذج الشموع: {patterns_str}"
    line5 = f"  الحالة: {state}"
    if is_main:
        line1 = "⭐ " + line1
    return "\n".join([line1, line2, line3, line4, line5])


def build_full_report(all_infos: dict) -> str:
    now = now_utc_str()
    header = f"🤖 البوت الذكي – تقرير السوق\n⏰ {now}\n\n"

    lines = []
    if MAIN_COIN in all_infos:
        lines.append(build_coin_report(MAIN_COIN, all_infos[MAIN_COIN], is_main=True))
        lines.append("")

    for sym, info in all_infos.items():
        if sym == MAIN_COIN:
            continue
        lines.append(build_coin_report(sym, info))

    best = max(all_infos.items(), key=lambda x: x[1]["score"])
    worst = min(all_infos.items(), key=lambda x: x[1]["score"])

    lines.append("")
    lines.append(f"🔥 أفضل فرصة الآن: {best[0]} (Score {best[1]['score']}/100)")
    lines.append(f"⚠️ أضعف عملة الآن: {worst[0]} (Score {worst[1]['score']}/100)")

    return header + "\n".join(lines)


def analyze_market() -> dict:
    infos = {}
    for symbol, cg_id in COINS.items():
        try:
            df = fetch_ohlcv_coingecko(cg_id, days=2, interval="hourly")
            info = calc_score(df)
            infos[symbol] = info
        except Exception as e:
            bot.send_message(chat_id=CHAT_ID, text=f"❌ خطأ في تحليل {symbol}:\n{e}")
    return infos


# ==========================
# إدارة الصفقات + رأس المال + DCA + SL
# ==========================

def suggest_smart_stop(info: dict, entry: float) -> float:
    """اقتراح Stop Loss ذكي يعتمد على ATR + الدعم"""
    atr_val = info["atr"]
    support = info["support"]
    raw_sl = min(entry - 1.5 * atr_val, support * 0.99)
    return max(raw_sl, 0)


def register_manual_buy(symbol: str, price: float, usd_size: float | None = None):
    ensure_coin_capital(symbol)

    if usd_size is None:
        usd_size = max(capital["current"] * 0.1, 10.0)  # 10% أو 10$ كحد أدنى

    if usd_size > capital["current"]:
        usd_size = capital["current"]

    amount = usd_size / price if price > 0 else 0
    c = capital["coins"][symbol]

    total_cost_prev = c["avg_price"] * c["amount"]
    total_cost_new = total_cost_prev + usd_size
    new_amount = c["amount"] + amount

    c["amount"] = new_amount
    c["avg_price"] = total_cost_new / new_amount if new_amount > 0 else 0
    c["invested"] += usd_size

    capital["current"] -= usd_size

    OPEN_TRADES[symbol] = {
        "entry": c["avg_price"],
        "target_12": round(c["avg_price"] * 1.12, 6),
        "time": now_utc_str(),
        "auto": False,
        "amount": c["amount"],
    }


def register_auto_buy(symbol: str, price: float):
    ensure_coin_capital(symbol)
    usd_size = max(capital["current"] * 0.05, 10.0)  # 5% من رأس المال
    if usd_size > capital["current"]:
        usd_size = capital["current"]

    amount = usd_size / price if price > 0 else 0
    c = capital["coins"][symbol]

    total_cost_prev = c["avg_price"] * c["amount"]
    total_cost_new = total_cost_prev + usd_size
    new_amount = c["amount"] + amount

    c["amount"] = new_amount
    c["avg_price"] = total_cost_new / new_amount if new_amount > 0 else 0
    c["invested"] += usd_size

    capital["current"] -= usd_size

    OPEN_TRADES[symbol] = {
        "entry": c["avg_price"],
        "target_12": round(c["avg_price"] * 1.12, 6),
        "time": now_utc_str(),
        "auto": True,
        "amount": c["amount"],
    }


def auto_dca(symbol: str, info: dict):
    """شراء تدرّجي DCA عندما يكون السعر في قاع واضح"""
    if symbol not in OPEN_TRADES:
        return

    trade = OPEN_TRADES[symbol]
    entry = trade["entry"]
    price = info["last_close"]
    rsi6 = info["rsi6"]
    support = info["support"]

    if price < entry and price <= support * 1.02 and rsi6 < 35 and capital["current"] > 10:
        usd_size = max(capital["current"] * 0.1, 10.0)
        register_manual_buy(symbol, price, usd_size)
        bot.send_message(
            chat_id=CHAT_ID,
            text=(
                f"🟡 DCA على {symbol}\n"
                f"تعزيز بسعر: {price:.6f}\n"
                f"حجم نظري: {usd_size:.2f} USDT\n"
                f"Entry جديد تقريبي: {capital['coins'][symbol]['avg_price']:.6f}"
            )
        )


def check_plan_targets(all_infos: dict):
    to_close = []
    for sym, trade in OPEN_TRADES.items():
        if sym not in all_infos:
            continue
        info = all_infos[sym]
        price = info["last_close"]
        target = trade["target_12"]
        entry = trade["entry"]

        if price >= target:
            profit_pct = (price / entry - 1) * 100
            amount = trade.get("amount", 0)
            profit_usd = (price - entry) * amount

            capital["realized_profit"] += profit_usd
            capital["current"] += profit_usd * 0.5
            capital["saved"] += profit_usd * 0.5

            bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    f"🎯 هدف 12% تحقق على {sym}!\n"
                    f"Entry: {entry:.6f}\n"
                    f"Current: {price:.6f}\n"
                    f"Target: {target:.6f}\n"
                    f"الربح التقريبي: {profit_pct:.2f}% (~{profit_usd:.2f} USDT)\n"
                    "📤 تم افتراضياً إضافة 50% للرأس مال و50% للادخار.\n"
                    "هذه حسابات تعليمية داخلية فقط."
                )
            )
            to_close.append(sym)

    for sym in to_close:
        del OPEN_TRADES[sym]


# ==========================
# Hybrid Auto Mode
# ==========================

def hybrid_auto_trading(all_infos: dict):
    if not HYBRID_AUTO:
        return
    if MAIN_COIN not in all_infos:
        return

    info = all_infos[MAIN_COIN]
    price = info["last_close"]
    rsi6 = info["rsi6"]
    score = info["score"]
    trend = info["trend_ar"]
    support = info["support"]
    resistance = info["resistance"]
    patterns = info.get("patterns", [])
    zone = info.get("zone", "neutral")

    # لا يوجد صفقة → فرصة دخول آلي تعليمي
    if MAIN_COIN not in OPEN_TRADES:
        strong_buy = (
            score >= 80 and
            rsi6 < 35 and
            price <= support * 1.03 and
            ("Morning Star" in patterns or "Bullish Engulfing" in patterns or zone == "demand")
        )
        if strong_buy and capital["current"] > 10:
            register_auto_buy(MAIN_COIN, price)
            bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    f"🟢 Hybrid Auto: دخول افتراضي على {MAIN_COIN}\n"
                    f"السعر: {price:.6f}\n"
                    f"الاتجاه: {trend}\n"
                    f"المنطقة: {zone}\n"
                    f"نماذج: {', '.join(patterns) if patterns else 'بدون'}\n"
                    f"هدف 12%: {price * 1.12:.6f}\n"
                    "هذه إشارة تعليمية فقط وليست تنفيذ فعلي على منصة التداول."
                )
            )
    else:
        # يوجد صفقة → خروج ذكي
        trade = OPEN_TRADES[MAIN_COIN]
        entry = trade["entry"]
        amount = trade.get("amount", 0)
        profit_pct = (price / entry - 1) * 100

        strong_sell = (
            profit_pct >= 10 and
            rsi6 > 70 and
            price >= resistance * 0.97 and
            ("Evening Star" in patterns or "Bearish Engulfing" in patterns or zone == "supply")
        )
        if strong_sell:
            bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    f"🔴 Hybrid Auto: توصية خروج على {MAIN_COIN}\n"
                    f"Entry: {entry:.6f}\n"
                    f"Current: {price:.6f}\n"
                    f"ربح تقريبي: {profit_pct:.2f}% على كمية تقريبية {amount:.2f}\n"
                    "يُفضل جني الربح الآن وفق نظام 12% الأسبوعي."
                )
            )


# ==========================
# أوامر التليجرام
# ==========================

def send_help(chat_id: int):
    bot.send_message(
        chat_id=chat_id,
        text=(
            "🤖 أوامر البوت الذكي:\n"
            "/xvg - تحليل مفصل لعملة XVG\n"
            "/coin رمز - تحليل عملة معينة مثلاً /coin ROSE\n"
            "/plan - شرح خطة 12% الأسبوعية\n"
            "/buy السعر [الرمز] [حجم_USDT] - تسجيل شراء يدوي\n"
            "   مثال: /buy 0.0065 XVG 100\n"
            "/sell السعر [الرمز] [كمية] - حساب ربح صفقة\n"
            "/dashboard - لوحة تحكم شاملة\n"
        )
    )


def cmd_xvg(chat_id: int):
    global LAST_INFOS
    try:
        if MAIN_COIN not in LAST_INFOS:
            df = fetch_ohlcv_coingecko(COINS[MAIN_COIN], days=2, interval="hourly")
            LAST_INFOS[MAIN_COIN] = calc_score(df)
        info = LAST_INFOS[MAIN_COIN]
        state = classify_state(info)
        trade = OPEN_TRADES.get(MAIN_COIN)
        ensure_coin_capital(MAIN_COIN)
        c = capital["coins"][MAIN_COIN]
        patterns = info.get("patterns", [])
        patterns_str = ", ".join(patterns) if patterns else "لا يوجد"

        msg = (
            f"🔍 تحليل {MAIN_COIN}\n"
            f"⏰ {now_utc_str()}\n\n"
            f"💰 السعر: {info['last_close']:.6f} USD\n"
            f"RSI(6/12/24): {info['rsi6']:.1f} / {info['rsi12']:.1f} / {info['rsi24']:.1f}\n\n"
            f"EMA50 : {info['ema50']:.6f}\n"
            f"EMA100: {info['ema100']:.6f}\n"
            f"EMA200: {info['ema200']:.6f}\n"
            f"البعد عن EMA50: {info['dist_ema50']:+.2f}%\n"
            f"البعد عن EMA200: {info['dist_ema200']:+.2f}%\n\n"
            f"الاتجاه: {info['trend_ar']}\n"
            f"Score: {info['score']}/100\n"
            f"الدعم: {info['support']:.6f}\n"
            f"المقاومة: {info['resistance']:.6f}\n"
            f"نماذج الشموع: {patterns_str}\n\n"
            f"التقييم: {state}\n\n"
            f"📦 المركز النظري على {MAIN_COIN}:\n"
            f"الكمية: {c['amount']:.2f}\n"
            f"متوسط السعر: {c['avg_price']:.6f}\n"
            f"إجمالي استثمار: {c['invested']:.2f} USDT\n"
        )

        if trade:
            sl = suggest_smart_stop(info, trade["entry"])
            msg += (
                "\n📘 صفقة مفتوحة (خطة 12%):\n"
                f"Entry: {trade['entry']:.6f}\n"
                f"Target 12%: {trade['target_12']:.6f}\n"
                f"Stop Loss ذكي مقترح: {sl:.6f}\n"
            )

        bot.send_message(chat_id=chat_id, text=msg)

    except Exception as e:
        bot.send_message(chat_id=chat_id, text=f"❌ خطأ في تحليل {MAIN_COIN}:\n{e}")


def cmd_coin(chat_id: int, symbol: str):
    symbol = symbol.upper()
    if symbol not in COINS:
        bot.send_message(chat_id=chat_id, text=f"❌ العملة {symbol} غير مضافة للبوت.")
        return
    try:
        df = fetch_ohlcv_coingecko(COINS[symbol], days=2, interval="hourly")
        info = calc_score(df)
        LAST_INFOS[symbol] = info
        msg = build_coin_report(symbol, info, is_main=(symbol == MAIN_COIN))
        bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        bot.send_message(chat_id=chat_id, text=f"❌ خطأ في تحليل {symbol}:\n{e}")


def cmd_plan(chat_id: int):
    bot.send_message(
        chat_id=chat_id,
        text=(
            "📘 خطة 12% الأسبوعية (XVG):\n\n"
            "• الهدف: ربح 12% لكل دورة أسبوعية تقريبًا.\n"
            "• البوت يحسب هدف 12% لكل Entry.\n"
            "• عند وصول السعر للهدف → تنبيه 🎯.\n"
            "• تسجيل شراء يدوي:\n"
            "  /buy 0.0065 XVG 100\n"
            "  (سعر – رمز – حجم بالدولار)\n"
        )
    )


def cmd_buy(chat_id: int, args: list):
    if not args:
        bot.send_message(chat_id=chat_id, text="❌ استخدم: /buy السعر [الرمز] [حجم_USDT]\nمثال: /buy 0.0065 XVG 100")
        return

    try:
        price = float(args[0])
    except Exception:
        bot.send_message(chat_id=chat_id, text="❌ السعر غير صحيح. مثال: /buy 0.0065 XVG 100")
        return

    symbol = MAIN_COIN
    usd_size = None

    if len(args) >= 2:
        if args[1].upper() in COINS:
            symbol = args[1].upper()
            if len(args) >= 3:
                try:
                    usd_size = float(args[2])
                except Exception:
                    usd_size = None
        else:
            try:
                usd_size = float(args[1])
            except Exception:
                pass

    if symbol not in COINS:
        bot.send_message(chat_id=chat_id, text=f"❌ العملة {symbol} غير مدعومة.")
        return

    if capital["current"] <= 0:
        bot.send_message(chat_id=chat_id, text="⚠️ لا يوجد رأس مال متاح نظريًا لصفقات جديدة.")
        return

    register_manual_buy(symbol, price, usd_size)
    trade = OPEN_TRADES[symbol]
    bot.send_message(
        chat_id=chat_id,
        text=(
            f"📥 تم تسجيل صفقة شراء على {symbol}\n"
            f"Entry (متوسط): {trade['entry']:.6f}\n"
            f"Target 12%: {trade['target_12']:.6f}\n"
            f"رأس المال المتبقي (نظريًا): {capital['current']:.2f} USDT"
        )
    )


def cmd_sell(chat_id: int, args: list):
    if not args:
        bot.send_message(chat_id=chat_id, text="❌ استخدم: /sell السعر [الرمز] [كمية]\nمثال: /sell 0.0072 XVG 5000")
        return

    try:
        price = float(args[0])
    except Exception:
        bot.send_message(chat_id=chat_id, text="❌ السعر غير صحيح.")
        return

    symbol = MAIN_COIN
    amount = None

    if len(args) >= 2:
        if args[1].upper() in COINS:
            symbol = args[1].upper()
            if len(args) >= 3:
                try:
                    amount = float(args[2])
                except Exception:
                    amount = None
        else:
            try:
                amount = float(args[1])
            except Exception:
                pass

    ensure_coin_capital(symbol)
    c = capital["coins"][symbol]

    if amount is None or amount > c["amount"]:
        amount = c["amount"]

    if amount <= 0:
        bot.send_message(chat_id=chat_id, text=f"ℹ️ لا تملك كمية مسجلة لـ {symbol} في المحرك الداخلي.")
        return

    entry = c["avg_price"]
    profit_pct = (price / entry - 1) * 100
    profit_usd = (price - entry) * amount

    bot.send_message(
        chat_id=chat_id,
        text=(
            f"📤 صفقة {symbol} (حساب نظري):\n"
            f"Entry: {entry:.6f}\n"
            f"Exit: {price:.6f}\n"
            f"Quantity: {amount:.2f}\n"
            f"الربح التقريبي: {profit_pct:.2f}% (~{profit_usd:.2f} USDT)\n"
            "هذا الحساب داخلي فقط ولا يعني تنفيذ حقيقي على المنصة."
        )
    )

    c["amount"] -= amount
    c["invested"] -= min(c["invested"], entry * amount)
    capital["current"] += price * amount
    capital["realized_profit"] += profit_usd


def cmd_dashboard(chat_id: int):
    lines = []
    lines.append(f"📊 Dashboard – البوت الذكي\n⏰ {now_utc_str()}\n")
    lines.append(f"• العملات المراقبة: {len(COINS)}")
    lines.append(f"• صفقات مفتوحة: {len(OPEN_TRADES)}")
    lines.append(f"• رأس المال الابتدائي: {capital['initial']:.2f} USDT")
    lines.append(f"• رأس المال الحالي (نظري): {capital['current']:.2f} USDT")
    lines.append(f"• الأرباح المحققة نظرياً: {capital['realized_profit']:.2f} USDT")
    lines.append(f"• الادخار النظري: {capital['saved']:.2f} USDT")

    if LAST_INFOS:
        best = max(LAST_INFOS.items(), key=lambda x: x[1]["score"])
        worst = min(LAST_INFOS.items(), key=lambda x: x[1]["score"])
        lines.append(f"\n• أقوى عملة الآن: {best[0]} (Score {best[1]['score']})")
        lines.append(f"• أضعف عملة الآن: {worst[0]} (Score {worst[1]['score']})")

    if OPPORTUNITY_MEMORY:
        lines.append("\n🔥 أفضل الفرص المحفوظة:")
        for opp in OPPORTUNITY_MEMORY:
            lines.append(
                f"- {opp['symbol']} @ {opp['price']:.6f} | Score {opp['score']} | RSI6 {opp['rsi6']:.1f}"
            )

    if OPEN_TRADES:
        lines.append("\n📘 الصفقات المفتوحة (خطة 12%):")
        for sym, tr in OPEN_TRADES.items():
            lines.append(
                f"- {sym}: Entry {tr['entry']:.6f} | Target 12% {tr['target_12']:.6f} | Amount ~{tr.get('amount',0):.2f}"
            )

    bot.send_message(chat_id=chat_id, text="\n".join(lines))


# ==========================
# قراءة أوامر التليجرام (Polling)
# ==========================

def process_updates(last_update_id=None):
    try:
        updates = bot.get_updates(offset=last_update_id, timeout=5)
    except Exception:
        return last_update_id

    for u in updates:
        last_update_id = u.update_id + 1
        if not hasattr(u, "message") or u.message is None:
            continue
        chat_id = u.message.chat.id
        text = (u.message.text or "").strip()

        if not text or not text.startswith("/"):
            continue

        parts = text.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ["/start", "/help"]:
            send_help(chat_id)
        elif cmd == "/xvg":
            cmd_xvg(chat_id)
        elif cmd == "/coin" and args:
            cmd_coin(chat_id, args[0])
        elif cmd == "/plan":
            cmd_plan(chat_id)
        elif cmd == "/buy":
            cmd_buy(chat_id, args)
        elif cmd == "/sell":
            cmd_sell(chat_id, args)
        elif cmd == "/dashboard":
            cmd_dashboard(chat_id)
        else:
            send_help(chat_id)

    return last_update_id


# ==========================
# الحلقة الرئيسية
# ==========================

def main_loop():
    global LAST_INFOS

    bot.send_message(
        chat_id=CHAT_ID,
        text="✅ البوت الذكي تم تشغيله (Hybrid + 12% + Capital + Smart Alerts + Candlestick AI Pro + Balanced Score v2)."
    )

    last_analysis_time = 0
    last_update_id = None

    while True:
        # 1) أوامر التليجرام
        last_update_id = process_updates(last_update_id)

        # 2) تحليل السوق
        now_ts = time.time()
        if now_ts - last_analysis_time > ANALYSIS_INTERVAL:
            try:
                infos = analyze_market()
                if infos:
                    LAST_INFOS = infos

                    report = build_full_report(infos)
                    bot.send_message(chat_id=CHAT_ID, text=report)

                    # تنبيهات ذكية
                    smart_alerts(infos)

                    # أفضل الفرص
                    mine_opportunities(infos)

                    # Hybrid Auto
                    hybrid_auto_trading(infos)

                    # DCA على XVG
                    if MAIN_COIN in infos:
                        auto_dca(MAIN_COIN, infos[MAIN_COIN])

                    # فحص أهداف 12%
                    check_plan_targets(infos)

            except Exception as e:
                try:
                    bot.send_message(chat_id=CHAT_ID, text=f"❌ خطأ عام في الحلقة الرئيسية:\n{e}")
                except Exception:
                    pass

            last_analysis_time = now_ts

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main_loop()
