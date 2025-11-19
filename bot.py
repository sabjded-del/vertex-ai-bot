import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from telegram import Bot

# ============ الإعدادات الأساسية ============

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise RuntimeError("يجب ضبط متغيرات البيئة TOKEN و CHAT_ID في Render")

bot = Bot(TOKEN)

# قائمة العملات (تقدر تضيف/تحذف لاحقاً)
COINS = {
    "XVG": "verge",
    "ROSE": "oasis-network",
    "GALA": "gala",
    "BLUR": "blur",
    "FIL": "filecoin",
}

MAIN_COIN = "XVG"  # عملة خطتك الأساسية


# ============ أدوات مساعدة ============

def now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def safe_get(d, k, default=None):
    return d[k] if k in d else default


# ============ جلب البيانات من CoinGecko ============

def fetch_ohlcv_coingecko(coin_id: str, days: int = 2, interval: str = "hourly") -> pd.DataFrame:
    """
    نجلب بيانات من CoinGecko: الأسعار + الحجم ونحوّلها إلى DataFrame.
    CoinGecko يعطينا:
      - prices: [timestamp, price]
      - total_volumes: [timestamp, volume]
    نستخدمهم كـ Close + Volume.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": interval
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

    df = pd.merge_asof(df_price.sort_values("time"),
                       df_vol.sort_values("time"),
                       on="time")
    # تقريب high/low باستخدام Close (حل تقريبي)
    df["high"] = df["close"].rolling(3, min_periods=1).max()
    df["low"] = df["close"].rolling(3, min_periods=1).min()
    return df


# ============ حساب المؤشرات الفنية ============

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def bollinger(series: pd.Series, period: int = 20, stddev: float = 2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + stddev * std
    lower = ma - stddev * std
    return ma, upper, lower


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    obv_val = (direction * volume).fillna(0).cumsum()
    return obv_val


def kdj(df: pd.DataFrame, period: int = 9, k_smooth: int = 3, d_smooth: int = 3):
    """
    نحسب KDJ بشكل تقريبي بناءً على high/low/close التقريبية.
    """
    low_min = df["low"].rolling(window=period, min_periods=1).min()
    high_max = df["high"].rolling(window=period, min_periods=1).max()
    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-9) * 100

    k = rsv.ewm(alpha=1.0 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1.0 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


# ============ نظام الـ Score (0–100) ============

def calc_score(df: pd.DataFrame) -> dict:
    """
    يحسب Score نهائي بناءً على آخر شمعة في df.
    يرجع dict فيه:
      - score
      - تفاصيل جزئية
    """
    last = df.iloc[-1]

    close = df["close"]
    volume = df["volume"].fillna(0)

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    ema50 = ema(close, 50)
    ema100 = ema(close, 100)

    rsi6 = rsi(close, 6)
    rsi12 = rsi(close, 12)
    rsi24 = rsi(close, 24)

    bb_mid, bb_up, bb_low = bollinger(close, 20, 2)

    obv_series = obv(close, volume)
    k, d, j = kdj(df)

    # -------- 1) Trend Score (0–20) --------
    trend_score = 0
    if last["close"] > ema50.iloc[-1]:
        trend_score += 5
    if last["close"] > ema100.iloc[-1]:
        trend_score += 5
    if ema12.iloc[-1] > ema26.iloc[-1] > ema50.iloc[-1]:
        trend_score += 10

    # -------- 2) Zone Score (0–25) --------
    zone_score = 0
    # قرب Bollinger Lower = قاع محتمل
    if bb_low.iloc[-1] and not np.isnan(bb_low.iloc[-1]):
        dist_to_lower = (last["close"] - bb_low.iloc[-1]) / (bb_mid.iloc[-1] - bb_low.iloc[-1] + 1e-9)
        if dist_to_lower <= 0.3:  # قريب جدا من القاع
            zone_score += 10

    # دعم بسيط من اللوات السابقة
    recent_lows = df["low"].tail(30)
    support_level = recent_lows.min()
    if last["close"] <= support_level * 1.03:
        zone_score += 10

    # بعيد عن مقاومة تقريبية (أعلى هاي سابق)
    recent_highs = df["high"].tail(50)
    resistance_level = recent_highs.max()
    if resistance_level > 0 and (resistance_level - last["close"]) / resistance_level >= 0.05:
        zone_score += 5

    # -------- 3) Momentum Score (0–30) --------
    momentum_score = 0
    if rsi6.iloc[-1] < 30 and rsi12.iloc[-1] < 35:
        momentum_score += 10

    if k.iloc[-1] < 20 and d.iloc[-1] < 20 and j.iloc[-1] > k.iloc[-2]:
        momentum_score += 10

    if rsi24.iloc[-1] < 60:
        momentum_score += 10

    # -------- 4) Volume / OBV Score (0–15) --------
    volume_score = 0
    if len(volume) >= 21:
        vol_ma = volume.rolling(20).mean()
        if volume.iloc[-1] > vol_ma.iloc[-1] * 1.2:
            volume_score += 5

    # OBV يكسر ترند هابط ← تبسيط: آخر قيمة أعلى من متوسطه الأخير
    if len(obv_series) >= 10:
        if obv_series.iloc[-1] > obv_series.tail(10).mean():
            volume_score += 10

    # -------- 5) شموع (0–10) --------
    candle_score = 0
    # Hammer بسيط: جسم صغير وذيل سفلي طويل
    o = df["close"].shift(1).fillna(df["close"])
    h = df["high"]
    l = df["low"]
    c = df["close"]

    body = abs(c.iloc[-1] - o.iloc[-1])
    lower_wick = c.iloc[-1] - l.iloc[-1]
    upper_wick = h.iloc[-1] - c.iloc[-1]

    if body < (upper_wick + lower_wick) * 0.3 and lower_wick > body * 2:
        candle_score += 10

    total = trend_score + zone_score + momentum_score + volume_score + candle_score
    total = max(0, min(int(total), 100))

    return {
        "score": total,
        "trend_score": trend_score,
        "zone_score": zone_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "candle_score": candle_score,
        "last_close": float(last["close"]),
        "rsi6": float(rsi6.iloc[-1]),
        "rsi12": float(rsi12.iloc[-1]),
        "rsi24": float(rsi24.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "ema100": float(ema100.iloc[-1]),
        "bb_low": float(bb_low.iloc[-1]) if not np.isnan(bb_low.iloc[-1]) else None,
        "bb_mid": float(bb_mid.iloc[-1]) if not np.isnan(bb_mid.iloc[-1]) else None,
        "bb_up": float(bb_up.iloc[-1]) if not np.isnan(bb_up.iloc[-1]) else None,
        "k": float(k.iloc[-1]),
        "d": float(d.iloc[-1]),
        "j": float(j.iloc[-1]),
        "support": float(support_level),
        "resistance": float(resistance_level),
    }


# ============ منطق بسيط لتصنيف الحالة ============

def classify_state(info: dict) -> str:
    s = info["score"]
    rsi6 = info["rsi6"]
    price = info["last_close"]
    support = info["support"]
    resistance = info["resistance"]

    if s >= 80 and rsi6 < 35 and price <= support * 1.03:
        return "🟢 قاع قوي / فرصة شراء ممتازة"
    if s >= 60 and rsi6 < 50:
        return "🟡 وضع إيجابي / فرصة معقولة"
    if s <= 35 and rsi6 > 70 and price >= resistance * 0.97:
        return "🔴 قرب قمة / خطر هبوط"
    return "⚪ منطقة تذبذب / لا يوجد وضوح قوي"


# ============ بناء رسالة تقرير للعملة ============

def build_coin_report(symbol: str, info: dict, is_main: bool = False) -> str:
    state = classify_state(info)

    line1 = f"• {symbol}: {info['last_close']:.6f} USD  | Score: {info['score']}/100"
    line2 = f"  RSI(6/12/24): {info['rsi6']:.1f} / {info['rsi12']:.1f} / {info['rsi24']:.1f}"
    line3 = f"  دعم تقريبي: {info['support']:.6f}  | مقاومة تقريبية: {info['resistance']:.6f}"
    line4 = f"  الحالة: {state}"

    if is_main:
        line1 = "⭐ " + line1

    return "\n".join([line1, line2, line3, line4])


# ============ بناء التقرير الكامل لكل العملات ============

def build_full_report(all_infos: dict) -> str:
    """
    all_infos: dict { "XVG": info_dict, ... }
    """
    now = now_utc_str()
    header = f"🤖 البوت الذكي – تقرير السوق\n⏰ الوقت: {now}\n\n"

    # XVG أولاً إن وجدت
    lines = []
    if MAIN_COIN in all_infos:
        lines.append(build_coin_report(MAIN_COIN, all_infos[MAIN_COIN], is_main=True))
        lines.append("")

    # باقي العملات
    for sym, info in all_infos.items():
        if sym == MAIN_COIN:
            continue
        lines.append(build_coin_report(sym, info))

    # أفضل فرصة شراء / أسوأ عملة
    best_buy = max(all_infos.items(), key=lambda x: x[1]["score"])
    worst = min(all_infos.items(), key=lambda x: x[1]["score"])

    lines.append("")
    lines.append(f"🔥 أفضل فرصة حالياً: {best_buy[0]} (Score {best_buy[1]['score']}/100)")
    lines.append(f"⚠️ أضعف عملة حالياً: {worst[0]} (Score {worst[1]['score']}/100)")

    return header + "\n".join(lines)


# ============ الحلقة الرئيسية للبوت ============

def main_loop():
    bot.send_message(chat_id=CHAT_ID, text="✅ تم تشغيل البوت الذكي بنجاح.")

    while True:
        all_infos = {}
        try:
            for symbol, cg_id in COINS.items():
                try:
                    df = fetch_ohlcv_coingecko(cg_id, days=2, interval="hourly")
                    info = calc_score(df)
                    all_infos[symbol] = info
                except Exception as e:
                    bot.send_message(chat_id=CHAT_ID, text=f"❌ خطأ في جلب/تحليل {symbol}: {e}")
                    continue

            if all_infos:
                report = build_full_report(all_infos)
                bot.send_message(chat_id=CHAT_ID, text=report)

        except Exception as e:
            try:
                bot.send_message(chat_id=CHAT_ID, text=f"❌ خطأ عام في البوت الذكي:\n{e}")
            except Exception:
                pass

        # انتظر 15 دقيقة بين كل تقرير وآخر
        time.sleep(60 * 15)


if __name__ == "__main__":
    main_loop()
