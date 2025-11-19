import os
import time
import requests
import telebot

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TOKEN)

# آخر أسعار ناجحة (Cache)
LAST_PRICES = {}

COINS = {
    "xvg":  {"binance": "XVGUSDT",   "coingecko": "verge"},
    "rose": {"binance": "ROSEUSDT",  "coingecko": "oasis-network"},
    "gala": {"binance": "GALAUSDT",  "coingecko": "gala"},
    "blur": {"binance": "BLURUSDT",  "coingecko": "blur"},
    "fil":  {"binance": "FILUSDT",   "coingecko": "filecoin"},
}

# ========== Binance API ==========
def get_binance_price(symbol):
    if symbol is None:
        return None

    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"

    for _ in range(5):
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                return float(res.json()["price"])
        except:
            time.sleep(0.8)

    return None

# ========== CoinGecko API ==========
def get_coingecko_price(coin_id):
    if coin_id is None:
        return None

    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"

    for _ in range(5):
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                return float(res.json()[coin_id]["usd"])
        except:
            time.sleep(0.8)

    return None

# ========== Unified Price ==========
def get_price(name, info):
    bin_price = get_binance_price(info["binance"])
    if bin_price is not None:
        LAST_PRICES[name] = bin_price
        return bin_price

    geo_price = get_coingecko_price(info["coingecko"])
    if geo_price is not None:
        LAST_PRICES[name] = geo_price
        return geo_price

    # إذا فشل الكل → استخدم آخر سعر ناجح
    if name in LAST_PRICES:
        return LAST_PRICES[name]

    return None

# ========== Send Prices ==========
def send_prices():
    message = "🔥 تحديث الأسعار المباشر 🔥\n\n"

    for name, info in COINS.items():
        price = get_price(name, info)

        if price is None:
            message += f"• {name.upper()}: N/A USD\n"
        else:
            message += f"• {name.upper()}: {price:.8f} USD\n"

    bot.send_message(CHAT_ID, message)

# ========== Start ==========
bot.send_message(CHAT_ID, "🚀 تم تشغيل البوت بنجاح!")

while True:
    send_prices()
    time.sleep(8)   # تأخير أكبر يحل مشاكل N/A
