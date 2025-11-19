import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from telegram import Bot

# ============ الإعدادات الأساسية ============

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # الشات الأساسي اللي يوصله التقرير الدوري

if not TOKEN or not CHAT_ID:
    raise RuntimeError("يجب ضبط متغيرات البيئة TOKEN و CHAT_ID في Render")

bot = Bot(TOKEN)

# قائمة العملات (تقدر تعدل/تزيد لاحقاً)
COINS = {
    "XVG": "verge",
    "ROSE": "oasis-network",
    "GALA": "gala",
    "BLUR": "blur",
    "FIL": "filecoin",
}

MAIN_COIN = "XVG"  # عملتك الأساسية للخطة

# تخزين آخر تحليل لكل العملات
LAST_INFOS = {}

ANALYSIS_INTERVAL = 60 * 15  # تحليل سوق كامل كل 15 دقيقة


# ============ أدوات مساعدة ============

def now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


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
    # تقريب high/low باستخدام Close (حل تقريبي بسيط)
    df["high"] = df["close"].rolling(3, min_periods=1).max()
    df["low"] = df["close"].rolling(3, min_periods=1).min()
    return df


# ============ المؤشرات ============

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
    low_min = df["low"].rolling(window=period, min_periods=1).min()
    high_max = df["high"].rolling(window=period, min_periods=1).max()
    rsv = (df["close"] - low_min) / (high_max - low_min + 1e-9) * 100

    k = rsv.ewm(alpha=1.0 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1.0 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


# ============ نظام الـ Score (0–100) مع تطوير الترند ============

def calc_score(df: pd.DataFrame) -> dict:
    """
    يحسب Score نهائي بناءً على آخر شمعة.
    ويضيف تحليل اتجاه احترافي (EMA50/100/200 + ترتيب المتوسطات).
    """
    last = df.iloc[-1]

    close = df["close"]
    volume = df["volume"].fillna(0)

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

    price = last["close"]
    ema50_last = ema50.iloc[-1]
    ema100_last = ema100.iloc[-1]
    ema200_last = ema200.iloc[-1]

    # -------- Trend Meta --------
    above_50 = price > ema50_last
    above_100 = price > ema100_last
    above_200 = price > ema200_last

    bull_stack = ema12.iloc[-1] > ema26.iloc[-1] > ema50_last > ema100_last > ema200_last
    bear_stack = ema12.iloc[-1] < ema26.iloc[-1] < ema50_last < ema100_last < ema200_last

    # -------- 1) Trend Score (0–20) --------
    trend_score = 0
    if above_50:
        trend_score += 4
    if above_100:
        trend_score += 4
    if above_200:
        trend_score += 4

    if bull_stack:
        trend_score += 8
    elif bear_stack and not above_50:
        # نعطيه 0 إضافية (الاتجاه هابط) ولا نضيف نقاط إيجابية
        trend_score += 0

    if trend_score > 20:
        trend_score = 20

    # تصنيف الاتجاه بالعربي
    if bull_stack and above_200:
        trend_label = "strong_bull"
        trend_ar = "صاعد قوي 🔥"
    elif (above_50 and above_100) and price > ema200_last:
        trend_label = "bull"
        trend_ar = "صاعد ✅"
    elif bear_stack and not above_50 and not above_100 and not above_200:
        trend_label = "strong_bear"
        trend_ar = "هابط قوي 🚨"
    elif bear_stack and not above_50:
        trend_label = "bear"
        trend_ar = "هابط ⚠️"
    else:
        trend_label = "sideways"
        trend_ar = "تذبذب ⚪"

    # -------- 2) Zone Score (0–25) --------
    zone_score = 0
    if bb_low.iloc[-1] and not np.isnan(bb_low.iloc[-1]):
        dist_to_lower = (price - bb_low.iloc[-1]) / (bb_mid.iloc[-1] - bb_low.iloc[-1] + 1e-9)
        if dist_to_lower <= 0.3:
            zone_score += 10

    recent_lows = df["low"].tail(30)
    support_level = recent_lows.min()
    if price <= support_level * 1.03:
        zone_score += 10

    recent_highs = df["high"].tail(50)
    resistance_level = recent_highs.max()
    if resistance_level > 0 and (resistance_level - price) / resistance_level >= 0.05:
        zone_score += 5

    # -------- 3) Momentum Score (0–30) --------
    momentum_score = 0
    if rsi6.iloc[-1] < 30 and rsi12.iloc[-1] < 35:
        momentum_score += 10

    if len(k) > 1 and k.iloc[-1] < 20 and d.iloc[-1] < 20 and j.iloc[-1] > k.iloc[-2]:
        momentum_score += 10

    if rsi24.iloc[-1] < 60:
        momentum_score += 10

    # -------- 4) Volume / OBV Score (0–15) --------
    volume_score = 0
    if len(volume) >= 21:
        vol_ma = volume.rolling(20).mean()
        if volume.iloc[-1] > vol_ma.iloc[-1] * 1.2:
            volume_score += 5

    if len(obv_series) >= 10:
        if obv_series.iloc[-1] > obv_series.tail(10).mean():
            volume_score += 10

    # -------- 5) شموع (0–10) --------
    candle_score = 0
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

    # مسافات نسبية عن المتوسطات (للاستخدام التحليلي لاحقاً)
    dist_ema50 = (price / ema50_last - 1) * 100 if ema50_last != 0 else 0
    dist_ema200 = (price / ema200_last - 1) * 100 if ema200_last != 0 else 0

    return {
        "score": total,
        "trend_score": trend_score,
        "zone_score": zone_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "candle_score": candle_score,
        "last_close": float(price),
        "rsi6": float(rsi6.iloc[-1]),
        "rsi12": float(rsi12.iloc[-1]),
        "rsi24": float(rsi24.iloc[-1]),
        "ema12": float(ema12.iloc[-1]),
        "ema26": float(ema26.iloc[-1]),
        "ema50": float(ema50_last),
        "ema100": float(ema100_last),
        "ema200": float(ema200_last),
        "bb_low": float(bb_low.iloc[-1]) if not np.isnan(bb_low.iloc[-1]) else None,
        "bb_mid": float(bb_mid.iloc[-1]) if not np.isnan(bb_mid.iloc[-1]) else None,
        "bb_up": float(bb_up.iloc[-1]) if not np.isnan(bb_up.iloc[-1]) else None,
        "support": float(support_level),
        "resistance": float(resistance_level),
        "trend_label": trend_label,
        "trend_ar": trend_ar,
        "dist_ema50": float(dist_ema50),
        "dist_ema200": float(dist_ema200),
        "bull_stack": bool(bull_stack),
        "bear_stack": bool(bear_stack),
    }


# ============ تصنيف الحالة العامة (قاع/قمة/تذبذب) ============

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


# ============ بناء الرسائل ============

def build_coin_report(symbol: str, info: dict, is_main: bool = False) -> str:
    state = classify_state(info)

    line1 = f"• {symbol}: {info['last_close']:.6f} USD  | Score: {info['score']}/100"
    line2 = (
        f"  RSI(6/12/24): {info['rsi6']:.1f} / {info['rsi12']:.1f} / {info['rsi24']:.1f} | "
        f"Trend: {info.get('trend_ar', '')}"
    )
    line3 = f"  دعم: {info['support']:.6f}  | مقاومة: {info['resistance']:.6f}"
    line4 = f"  الحالة: {state}"

    if is_main:
        line1 = "⭐ " + line1

    return "\n".join([line1, line2, line3, line4])


def build_full_report(all_infos: dict) -> str:
    now = now_utc_str()
    header = f"🤖 البوت الذكي – تقرير السوق\n⏰ الوقت: {now}\n\n"

    lines = []
    if MAIN_COIN in all_infos:
        lines.append(build_coin_report(MAIN_COIN, all_infos[MAIN_COIN], is_main=True))
        lines.append("")

    for sym, info in all_infos.items():
        if sym == MAIN_COIN:
            continue
        lines.append(build_coin_report(sym, info))

    best_buy = max(all_infos.items(), key=lambda x: x[1]["score"])
    worst = min(all_infos.items(), key=lambda x: x[1]["score"])

    lines.append("")
    lines.append(f"🔥 أفضل فرصة حالياً: {best_buy[0]} (Score {best_buy[1]['score']}/100)")
    lines.append(f"⚠️ أضعف عملة حالياً: {worst[0]} (Score {worst[1]['score']}/100)")

    return header + "\n".join(lines)


# ============ تحليل السوق لكل العملات ============

def analyze_market() -> dict:
    infos = {}
    for symbol, cg_id in COINS.items():
        df = fetch_ohlcv_coingecko(cg_id, days=2, interval="hourly")
        info = calc_score(df)
        infos[symbol] = info
    return infos


# ============ أوامر تيليجرام ============

def cmd_xvg(chat_id: int):
    """
    أمر /xvg → تحليل مفصل لعملة XVG فقط مع اتجاه EMA200
    """
    global LAST_INFOS
    try:
        if MAIN_COIN not in LAST_INFOS:
            df = fetch_ohlcv_coingecko(COINS[MAIN_COIN], days=2, interval="hourly")
            LAST_INFOS[MAIN_COIN] = calc_score(df)

        info = LAST_INFOS[MAIN_COIN]
        state = classify_state(info)

        trend_ar = info.get("trend_ar", "غير محدد")
        dist_ema50 = info.get("dist_ema50", 0.0)
        dist_ema200 = info.get("dist_ema200", 0.0)

        msg = (
            f"🔍 تحليل XVG (عملة الخطة الأساسية)\n"
            f"⏰ {now_utc_str()}\n\n"
            f"💰 السعر الحالي: {info['last_close']:.6f} USD\n"
            f"📊 RSI(6/12/24): {info['rsi6']:.1f} / {info['rsi12']:.1f} / {info['rsi24']:.1f}\n\n"
            f"📈 المتوسطات المتحركة:\n"
            f"  EMA50 : {info['ema50']:.6f}\n"
            f"  EMA100: {info['ema100']:.6f}\n"
            f"  EMA200: {info['ema200']:.6f}\n"
            f"  البعد عن EMA50: {dist_ema50:+.2f}%\n"
            f"  البعد عن EMA200: {dist_ema200:+.2f}%\n\n"
            f"📌 الاتجاه العام: {trend_ar}\n"
            f"⭐ Trend Score: {info['trend_score']}/20\n\n"
            f"📉 الدعم التقريبي: {info['support']:.6f}\n"
            f"📉 المقاومة التقريبية: {info['resistance']:.6f}\n\n"
            f"🧮 Score الإجمالي: {info['score']}/100\n"
            f"⚖️ التقييم النهائي: {state}\n\n"
            f"💡 التفسير:\n"
            f"- EMA200 يستخدم كخط فاصل بين ترند صاعد/هابط على المدى المتوسط.\n"
            f"- لو السعر فوق EMA200 والترند صاعد قوي → أفضل مناطق الشراء تكون عند الاقتراب من EMA50 أو مناطق الطلب.\n"
            f"- لو السعر تحت EMA200 والترند هابط → البوت يميل للتحذير أكثر من الشراء."
        )
        bot.send_message(chat_id=chat_id, text=msg)

    except Exception as e:
        bot.send_message(chat_id=chat_id, text=f"❌ حدث خطأ أثناء تحليل XVG:\n{e}")


def cmd_plan(chat_id: int):
    """
    أمر /plan → ملخص خطتك 12% أسبوعياً
    """
    msg = (
        "📘 خطة التداول الأسبوعية (XVG):\n\n"
        "• رأس المال الأولي: 1,000 دولار (قابل للتعديل مستقبلاً في البوت الذكي).\n"
        "• الهدف الأسبوعي: ربح 12% من رأس المال.\n"
        "• الأرباح داخل الشهر: تتراكم (بدون سحب أسبوعي).\n"
        "• نهاية كل شهر:\n"
        "   - 50% من الربح يضاف إلى رأس المال.\n"
        "   - 50% من الربح للادخار.\n"
        "• نهاية كل ربع سنة: إضافة 1,000 دولار من الادخار إلى رأس المال.\n\n"
        "🎯 دور البوت الذكي:\n"
        "• مراقبة XVG كعملة أساسية.\n"
        "• اقتناص أفضل مناطق القاع والقمة بناءً على المؤشرات والمتوسطات خاصة EMA200.\n"
        "• مساعدتك في الاقتراب من هدف 12% أسبوعياً مع تقليل المخاطرة.\n\n"
        "لاحقاً يمكن ربط الخطة بحساب فعلي للأداء الأسبوعي حسب الصفقات."
    )
    bot.send_message(chat_id=chat_id, text=msg)


def cmd_buy(chat_id: int):
    """
    أمر /buy → يعرض أفضل فرص الشراء حالياً من العملات المراقبة
    """
    global LAST_INFOS
    if not LAST_INFOS:
        bot.send_message(chat_id=chat_id, text="ℹ️ لا توجد بيانات حديثة بعد. انتظر حتى أول تقرير آلي أو نفّذ /xvg أولاً.")
        return

    candidates = []
    for sym, info in LAST_INFOS.items():
        if info["score"] >= 70 and info["rsi6"] < 40:
            candidates.append((sym, info))

    if not candidates:
        bot.send_message(chat_id=chat_id, text="ℹ️ حالياً لا توجد فرص شراء قوية حسب شروط البوت الذكي.\nالأفضل الانتظار.")
        return

    candidates.sort(key=lambda x: x[1]["score"], reverse=True)

    lines = ["🟢 أفضل فرص الشراء حالياً (حسب البوت الذكي):\n"]
    for sym, info in candidates:
        state = classify_state(info)
        lines.append(
            f"• {sym} | السعر: {info['last_close']:.6f} | Score: {info['score']}/100\n"
            f"  RSI6: {info['rsi6']:.1f} | دعم: {info['support']:.6f}\n"
            f"  Trend: {info.get('trend_ar','')}\n"
            f"  {state}"
        )

    bot.send_message(chat_id=chat_id, text="\n\n".join(lines))


def cmd_sell(chat_id: int):
    """
    أمر /sell → يعرض العملات الأقرب لمناطق قمة/خطر (مرشحة لجني ربح/تخفيف)
    """
    global LAST_INFOS
    if not LAST_INFOS:
        bot.send_message(chat_id=chat_id, text="ℹ️ لا توجد بيانات حديثة بعد. انتظر حتى أول تقرير آلي.")
        return

    candidates = []
    for sym, info in LAST_INFOS.items():
        price = info["last_close"]
        resistance = info["resistance"]
        rsi6 = info["rsi6"]
        if rsi6 >= 65 and resistance > 0 and price >= resistance * 0.97:
            candidates.append((sym, info))

    if not candidates:
        bot.send_message(chat_id=chat_id, text="ℹ️ لا توجد حالياً قمم قوية واضحة لجني ربح.\nالبوت لا يرى خطرًا عاليًا الآن.")
        return

    candidates.sort(key=lambda x: x[1]["rsi6"], reverse=True)

    lines = ["🔴 عملات في مناطق قمة/خطر (مرشحة لجني ربح):\n"]
    for sym, info in candidates:
        state = classify_state(info)
        lines.append(
            f"• {sym} | السعر: {info['last_close']:.6f}\n"
            f"  RSI6: {info['rsi6']:.1f} | مقاومة: {info['resistance']:.6f}\n"
            f"  Trend: {info.get('trend_ar','')}\n"
            f"  {state}"
        )

    bot.send_message(chat_id=chat_id, text="\n\n".join(lines))


# ============ معالجة أوامر تيليجرام (getUpdates) ============

def process_updates(last_update_id=None):
    """
    يجلب التحديثات الجديدة من تيليجرام ويعالج الأوامر:
    /xvg, /plan, /buy, /sell
    """
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

        if not text.startswith("/"):
            continue

        if text.startswith("/xvg"):
            cmd_xvg(chat_id)
        elif text.startswith("/plan"):
            cmd_plan(chat_id)
        elif text.startswith("/buy"):
            cmd_buy(chat_id)
        elif text.startswith("/sell"):
            cmd_sell(chat_id)
        else:
            bot.send_message(
                chat_id=chat_id,
                text="🤖 الأوامر المتاحة:\n/xvg\n/plan\n/buy\n/sell"
            )

    return last_update_id


# ============ الحلقة الرئيسية ============

def main_loop():
    global LAST_INFOS

    bot.send_message(chat_id=CHAT_ID, text="✅ البوت الذكي تم تشغيله بنجاح مع تطوير اتجاه EMA200.")

    last_analysis_time = 0
    last_update_id = None

    while True:
        # 1) معالجة أوامر التيليجرام باستمرار
        last_update_id = process_updates(last_update_id)

        # 2) تحليل السوق وإرسال تقرير كل 15 دقيقة تقريباً
        now_ts = time.time()
        if now_ts - last_analysis_time > ANALYSIS_INTERVAL:
            try:
                infos = analyze_market()
                LAST_INFOS = infos
                report = build_full_report(infos)
                bot.send_message(chat_id=CHAT_ID, text=report)
            except Exception as e:
                try:
                    bot.send_message(chat_id=CHAT_ID, text=f"❌ خطأ عام في تحليل السوق:\n{e}")
                except Exception:
                    pass
            last_analysis_time = now_ts

        time.sleep(3)


if __name__ == "__main__":
    main_loop()
