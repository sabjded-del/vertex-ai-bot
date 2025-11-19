import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from telegram import Bot

# ==========================
# إعداد الأساسيات
# ==========================

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # الشات الأساسي لاستلام التقارير

if not TOKEN or not CHAT_ID:
    raise RuntimeError("❌ تأكد من ضبط TOKEN و CHAT_ID في إعدادات Render")

bot = Bot(TOKEN)

# تحليل السوق كل X ثانية (تقدر تزيد أو تنقص)
ANALYSIS_INTERVAL = 60 * 15  # كل 15 دقيقة
POLL_INTERVAL = 3            # فترة فحص أوامر التليجرام بالثواني

# ==========================
# العملات المدعومة (يمكنك التوسعة حتى 50 / 100)
# مفتاح: رمز العملة في البوت   القيمة: ID في CoinGecko
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

# عملتك الأساسية للخطة 12% (يمكن تغييرها)
MAIN_COIN = "XVG"

# ذاكرة داخلية بسيطة
LAST_INFOS = {}       # آخر تحليل لكل عملة
OPEN_TRADES = {}      # الصفقات المفتوحة لكل عملة {symbol: {"entry":..., "target":...}}
OPPORTUNITY_MEMORY = []  # حفظ الفرص القوية
LAST_ALERTS = {}      # لتقليل تكرار نفس التنبيه (symbol:type -> timestamp)

# وضع Hybrid Auto Mode (تشغيل/إيقاف)
HYBRID_AUTO = True


# ==========================
# أدوات مساعدة
# ==========================

def now_utc():
    return datetime.now(timezone.utc)


def now_utc_str():
    return now_utc().strftime("%Y-%m-%d %H:%M UTC")


# ==========================
# جلب البيانات من CoinGecko
# ==========================

def fetch_ohlcv_coingecko(coin_id: str, days: int = 2, interval: str = "hourly") -> pd.DataFrame:
    """
    جلب بيانات من CoinGecko: الأسعار + الحجم
    ونحوّلها إلى DataFrame.
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

    df = pd.merge_asof(df_price.sort_values("time"),
                       df_vol.sort_values("time"),
                       on="time")

    # تقدير high/low بشكل مبسط
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


# ==========================
# نظام Score (0–100) + اتجاه احترافي
# ==========================

def calc_score(df: pd.DataFrame) -> dict:
    """
    يحسب Score نهائي + معلومات الاتجاه + الدعوم والمقاومات
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

    price = float(last["close"])
    ema50_last = float(ema50.iloc[-1])
    ema100_last = float(ema100.iloc[-1])
    ema200_last = float(ema200.iloc[-1])

    # --- Trend (0–25) ---
    trend_score = 0
    above_50 = price > ema50_last
    above_100 = price > ema100_last
    above_200 = price > ema200_last

    bull_stack = ema12.iloc[-1] > ema26.iloc[-1] > ema50_last > ema100_last > ema200_last
    bear_stack = ema12.iloc[-1] < ema26.iloc[-1] < ema50_last < ema100_last < ema200_last

    if above_50:
        trend_score += 5
    if above_100:
        trend_score += 5
    if above_200:
        trend_score += 5
    if bull_stack:
        trend_score += 10
    elif bear_stack and not above_50:
        trend_score += 0

    if trend_score > 25:
        trend_score = 25

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

    # --- RSI (0–30) ---
    r6 = float(rsi6.iloc[-1])
    r12 = float(rsi12.iloc[-1])
    r24 = float(rsi24.iloc[-1])

    def rsi_part(val):
        if val < 25:
            return 10
        elif val < 70:
            return 5
        else:
            return -10

    rsi_score = rsi_part(r6) + rsi_part(r12) + rsi_part(r24)
    rsi_score = max(0, min(30, rsi_score + 15))  # نعيد موازنة النتيجة

    # --- Bollinger (0–15) ---
    b_low = bb_low.iloc[-1]
    b_mid = bb_mid.iloc[-1]
    b_up = bb_up.iloc[-1]
    bb_score = 0
    if not np.isnan(b_low) and not np.isnan(b_up):
        if price <= b_low:
            bb_score += 15
        elif price < b_mid:
            bb_score += 8
        elif price >= b_up:
            bb_score -= 10
    bb_score = max(0, min(15, bb_score))

    # --- OBV (0–15) ---
    obv_score = 0
    if len(obv_series) >= 10:
        obv_last = obv_series.iloc[-1]
        obv_prev = obv_series.iloc[-10]
        if obv_last > obv_prev:
            obv_score += 10
        else:
            obv_score -= 5
    obv_score = max(0, min(15, obv_score + 5))

    # --- KDJ (0–15) ---
    k_last = float(k.iloc[-1])
    d_last = float(d.iloc[-1])
    j_last = float(j.iloc[-1]) if not np.isnan(j.iloc[-1]) else 50.0
    k_prev = float(k.iloc[-2]) if len(k) > 1 else k_last

    kdj_score = 0
    golden_cross = k_last > d_last and k_prev < d.iloc[-2] if len(d) > 1 else False
    dead_cross = k_last < d_last and k_prev > d.iloc[-2] if len(d) > 1 else False

    if golden_cross and k_last < 30:
        kdj_score += 15
    elif k_last < 20:
        kdj_score += 8
    elif dead_cross and k_last > 70:
        kdj_score -= 10

    kdj_score = max(0, min(15, kdj_score + 5))

    # --- شمعة انعكاس بسيطة (0–10) ---
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

    # --- دعم ومقاومة تقريبية ---
    recent_lows = df["low"].tail(40)
    recent_highs = df["high"].tail(40)
    support_level = float(recent_lows.min())
    resistance_level = float(recent_highs.max())

    # --- المجموع ---
    total = trend_score + rsi_score + bb_score + obv_score + kdj_score + candle_score
    total = max(0, min(int(total), 100))

    dist_ema50 = (price / ema50_last - 1) * 100 if ema50_last else 0.0
    dist_ema200 = (price / ema200_last - 1) * 100 if ema200_last else 0.0

    return {
        "score": total,
        "trend_score": trend_score,
        "rsi_score": rsi_score,
        "bb_score": bb_score,
        "obv_score": obv_score,
        "kdj_score": kdj_score,
        "candle_score": candle_score,
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
    }


# ==========================
# تصنيف الحالة (قاع / قمة / تذبذب)
# ==========================

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
# Smart Alerts – التنبيهات الذكية
# ==========================

def smart_alerts(all_infos: dict):
    alerts = []
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
                alerts.append(
                    f"🟢💎 تنبيه شراء قوي على {sym}\n"
                    f"السعر: {price:.6f}\n"
                    f"RSI6: {rsi6:.1f}\n"
                    f"الدعم: {support:.6f}\n"
                    f"Score: {score}\n"
                    f"الاتجاه: {trend}"
                )
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
                alerts.append(
                    f"🔴🚨 تنبيه بيع قوي على {sym}\n"
                    f"السعر: {price:.6f}\n"
                    f"RSI6: {rsi6:.1f}\n"
                    f"المقاومة: {resistance:.6f}\n"
                    f"Score: {score}\n"
                    f"الاتجاه: {trend}"
                )
                LAST_ALERTS[key] = now_ts

        # Potential Bottom
        if rsi6 < 35 and price <= support * 1.05:
            key = f"{sym}_bottom"
            if now_ts - LAST_ALERTS.get(key, 0) > 60 * 30:
                alerts.append(
                    f"🟡📉 قاع محتمل على {sym}\n"
                    f"السعر: {price:.6f}\n"
                    f"RSI6: {rsi6:.1f}\n"
                    f"الدعم: {support:.6f}"
                )
                LAST_ALERTS[key] = now_ts

        # Potential Top
        if rsi6 > 65 and price >= resistance * 0.95:
            key = f"{sym}_top"
            if now_ts - LAST_ALERTS.get(key, 0) > 60 * 30:
                alerts.append(
                    f"🟠📈 قمة محتملة على {sym}\n"
                    f"السعر: {price:.6f}\n"
                    f"RSI6: {rsi6:.1f}\n"
                    f"المقاومة: {resistance:.6f}"
                )
                LAST_ALERTS[key] = now_ts

    return alerts


# ==========================
# Opportunity Mining – البحث عن أفضل الفرص
# ==========================

def mine_opportunities(all_infos: dict, top_n: int = 3):
    # نختار العملات ذات Score عالي + RSI معتدل
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
    line1 = f"• {symbol}: {info['last_close']:.6f} USD | Score: {info['score']}/100"
    line2 = (
        f"  RSI(6/12/24): {info['rsi6']:.1f} / {info['rsi12']:.1f} / {info['rsi24']:.1f} | "
        f"Trend: {info.get('trend_ar', '')}"
    )
    line3 = f"  دعم: {info['support']:.6f} | مقاومة: {info['resistance']:.6f}"
    line4 = f"  الحالة: {state}"
    if is_main:
        line1 = "⭐ " + line1
    return "\n".join([line1, line2, line3, line4])


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

    # أفضل وأسوأ عملة
    best = max(all_infos.items(), key=lambda x: x[1]["score"])
    worst = min(all_infos.items(), key=lambda x: x[1]["score"])

    lines.append("")
    lines.append(f"🔥 أفضل فرصة الآن: {best[0]} (Score {best[1]['score']}/100)")
    lines.append(f"⚠️ أضعف عملة الآن: {worst[0]} (Score {worst[1]['score']}/100)")

    return header + "\n".join(lines)


# ==========================
# تحليل السوق لكل العملات
# ==========================

def analyze_market() -> dict:
    infos = {}
    for symbol, cg_id in COINS.items():
        try:
            df = fetch_ohlcv_coingecko(cg_id, days=2, interval="hourly")
            info = calc_score(df)
            infos[symbol] = info
        except Exception as e:
            # في حالة خطأ لعملة معينة، نتجاهلها
            bot.send_message(chat_id=CHAT_ID, text=f"❌ خطأ في تحليل {symbol}:\n{e}")
    return infos


# ==========================
# خطة 12% أسبوعية – إدارة الصفقات
# ==========================

def register_manual_buy(symbol: str, price: float):
    OPEN_TRADES[symbol] = {
        "entry": price,
        "target_12": round(price * 1.12, 6),
        "time": now_utc_str(),
        "auto": False,
    }


def register_auto_buy(symbol: str, price: float):
    OPEN_TRADES[symbol] = {
        "entry": price,
        "target_12": round(price * 1.12, 6),
        "time": now_utc_str(),
        "auto": True,
    }


def check_plan_targets(all_infos: dict):
    """
    فحص الصفقات المفتوحة: هل تحقق هدف 12%؟
    """
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
            bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    f"🎯 هدف 12% تحقق على {sym}!\n"
                    f"Entry: {entry:.6f}\n"
                    f"Current: {price:.6f}\n"
                    f"Target: {target:.6f}\n"
                    f"الربح التقريبي: {profit_pct:.2f}%\n"
                    "📤 يفضّل البيع الجزئي أو الكلي حسب خطتك."
                )
            )
            to_close.append(sym)

    for sym in to_close:
        del OPEN_TRADES[sym]


# ==========================
# Hybrid Auto Mode
# ==========================

def hybrid_auto_trading(all_infos: dict):
    """
    وضع هجين: إذا لا توجد صفقة لـ XVG
    وظهرت إشارة شراء قوية → يسجل دخول افتراضي.
    وإذا ظهرت قمة قوية + ربح عالي → يوصي بالخروج.
    """
    if not HYBRID_AUTO:
        return

    # نركز أساساً على XVG
    if MAIN_COIN not in all_infos:
        return

    info = all_infos[MAIN_COIN]
    price = info["last_close"]
    rsi6 = info["rsi6"]
    score = info["score"]
    trend = info["trend_ar"]
    support = info["support"]
    resistance = info["resistance"]

    # لا يوجد صفقة حالية → نبحث عن فرصة دخول قوية
    if MAIN_COIN not in OPEN_TRADES:
        strong_buy = (
            score >= 80 and
            rsi6 < 35 and
            price <= support * 1.03
        )
        if strong_buy:
            register_auto_buy(MAIN_COIN, price)
            bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    f"🟢 Hybrid Auto: دخول افتراضي على {MAIN_COIN}\n"
                    f"السعر: {price:.6f}\n"
                    f"الاتجاه: {trend}\n"
                    f"هدف 12%: {price * 1.12:.6f}\n"
                    "هذه إشارة تعليمية فقط وليست تنفيذ حقيقي على منصة تداول."
                )
            )
    else:
        # يوجد صفقة → نبحث عن خروج ذكي
        trade = OPEN_TRADES[MAIN_COIN]
        entry = trade["entry"]
        target = trade["target_12"]
        profit_pct = (price / entry - 1) * 100

        strong_sell = (
            profit_pct >= 10 and
            rsi6 > 70 and
            price >= resistance * 0.97
        )
        if strong_sell:
            bot.send_message(
                chat_id=CHAT_ID,
                text=(
                    f"🔴 Hybrid Auto: توصية خروج على {MAIN_COIN}\n"
                    f"Entry: {entry:.6f}\n"
                    f"Current: {price:.6f}\n"
                    f"ربح تقريبي: {profit_pct:.2f}%\n"
                    "يُفضل جني الربح الآن وفق نظام 12% الأسبوعي."
                )
            )
            # لا نحذف الصفقة تلقائياً، نترك لك القرار


# ==========================
# أوامر تيليجرام
# ==========================

def send_help(chat_id: int):
    bot.send_message(
        chat_id=chat_id,
        text=(
            "🤖 أوامر البوت الذكي:\n"
            "/xvg - تحليل مفصل لعملة XVG\n"
            "/coin رمز - تحليل عملة معينة مثلاً /coin ROSE\n"
            "/plan - عرض شرح خطة 12% الأسبوعية\n"
            "/buy السعر [الرمز] - تسجيل شراء يدوي\n"
            "/sell السعر [الرمز] - تسجيل بيع (معلومات فقط)\n"
            "/dashboard - عرض ملخص السوق والفرص\n"
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
            f"المقاومة: {info['resistance']:.6f}\n\n"
            f"التقييم: {state}\n"
        )

        if trade:
            msg += (
                "\n📘 صفقة مفتوحة (خطة 12%):\n"
                f"Entry: {trade['entry']:.6f}\n"
                f"Target 12%: {trade['target_12']:.6f}\n"
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
            "• الهدف: ربح 12% من كل دورة أسبوعية.\n"
            "• البوت يتابع الصفقات المفتوحة ويحسب هدف 12% لكل Entry.\n"
            "• عند وصول السعر للهدف، يرسل تنبيه 🎯.\n"
            "• يمكنك تسجيل صفقة شراء يدويًا بالأمر:\n"
            "  /buy 0.00650 XVG\n"
        )
    )


def cmd_buy(chat_id: int, args: list):
    if not args:
        bot.send_message(chat_id=chat_id, text="❌ استخدم: /buy السعر [الرمز]\nمثال: /buy 0.0065 XVG")
        return

    try:
        price = float(args[0])
    except Exception:
        bot.send_message(chat_id=chat_id, text="❌ السعر غير صحيح. مثال: /buy 0.0065 XVG")
        return

    symbol = MAIN_COIN
    if len(args) >= 2:
        symbol = args[1].upper()

    if symbol not in COINS:
        bot.send_message(chat_id=chat_id, text=f"❌ العملة {symbol} غير مدعومة.")
        return

    register_manual_buy(symbol, price)
    bot.send_message(
        chat_id=chat_id,
        text=(
            f"📥 تم تسجيل صفقة شراء على {symbol}\n"
            f"Entry: {price:.6f}\n"
            f"Target 12%: {price * 1.12:.6f}\n"
        )
    )


def cmd_sell(chat_id: int, args: list):
    if not args:
        bot.send_message(chat_id=chat_id, text="❌ استخدم: /sell السعر [الرمز]\nمثال: /sell 0.0072 XVG")
        return

    try:
        price = float(args[0])
    except Exception:
        bot.send_message(chat_id=chat_id, text="❌ السعر غير صحيح.")
        return

    symbol = MAIN_COIN
    if len(args) >= 2:
        symbol = args[1].upper()

    trade = OPEN_TRADES.get(symbol)
    if not trade:
        bot.send_message(chat_id=chat_id, text=f"ℹ️ لا توجد صفقة مسجلة لـ {symbol}.")
        return

    entry = trade["entry"]
    profit_pct = (price / entry - 1) * 100

    bot.send_message(
        chat_id=chat_id,
        text=(
            f"📤 صفقة {symbol}:\n"
            f"Entry: {entry:.6f}\n"
            f"Exit: {price:.6f}\n"
            f"الربح التقريبي: {profit_pct:.2f}%\n"
        )
    )

    # لا نحذف الصفقة تلقائيًا، نترك لك الحرية
    # يمكن إضافة حذف لو أردت:
    # del OPEN_TRADES[symbol]


def cmd_dashboard(chat_id: int):
    lines = []
    lines.append(f"📊 Dashboard – ملخص البوت الذكي\n⏰ {now_utc_str()}\n")
    lines.append(f"• العملات المراقبة: {len(COINS)}")
    lines.append(f"• صفقات مفتوحة: {len(OPEN_TRADES)}")

    if LAST_INFOS:
        best = max(LAST_INFOS.items(), key=lambda x: x[1]["score"])
        worst = min(LAST_INFOS.items(), key=lambda x: x[1]["score"])
        lines.append(f"• أقوى عملة الآن: {best[0]} (Score {best[1]['score']})")
        lines.append(f"• أضعف عملة الآن: {worst[0]} (Score {worst[1]['score']})")

    if OPPORTUNITY_MEMORY:
        lines.append("\n🔥 أفضل الفرص المحفوظة:")
        for opp in OPPORTUNITY_MEMORY:
            lines.append(
                f"- {opp['symbol']} @ {opp['price']:.6f} | Score {opp['score']} | RSI6 {opp['rsi6']:.1f}"
            )

    if OPEN_TRADES:
        lines.append("\n📘 الصفقات المفتوحة:")
        for sym, tr in OPEN_TRADES.items():
            lines.append(
                f"- {sym}: Entry {tr['entry']:.6f} | Target 12% {tr['target_12']:.6f}"
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

        if not text.startswith("/"):
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

    bot.send_message(chat_id=CHAT_ID, text="✅ البوت الذكي تم تشغيله بنجاح (Hybrid + 12% + Smart Alerts).")

    last_analysis_time = 0
    last_update_id = None

    while True:
        # 1) أوامر التليجرام
        last_update_id = process_updates(last_update_id)

        # 2) تحليل السوق + تقرير + تنبيهات + Hybrid + Plan
        now_ts = time.time()
        if now_ts - last_analysis_time > ANALYSIS_INTERVAL:
            try:
                infos = analyze_market()
                if infos:
                    LAST_INFOS = infos

                    # تقرير السوق
                    report = build_full_report(infos)
                    bot.send_message(chat_id=CHAT_ID, text=report)

                    # Smart Alerts
                    alerts = smart_alerts(infos)
                    for a in alerts:
                        bot.send_message(chat_id=CHAT_ID, text=a)

                    # Opportunity Mining
                    best = mine_opportunities(infos)

                    # Hybrid Auto Mode
                    hybrid_auto_trading(infos)

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
