import os
import time
import requests
import telebot

# ===== الإعداد =====
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TOKEN)

# ===== قائمة العملات =====
COINS = {
    "xvg": "XVGUSDT",
    "rose": "ROSEUSDT",
    "gala": "GALAUSDT",
    "blur": "BLURUSDT",
    "fil": "FILUSDT",
    "kaia": "kaia"   # CoinGecko فقط
}

# ===== دالة جلب سعر Binance =====
def get_binance_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        r = requests.get(url, timeout=5).json()
        if "price" in r:
            return float(r["price"])
        else:
            return None
    except:
        return None

# ===== دالة جلب سعر CoinGecko =====
def get_coingecko_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        r = requests.get(url, timeout=5).json()
        if coin_id in r:
            return float(r[coin_id]["usd"])
        else:
            return None
    except:
        return None

# ===== إرسال الأسعار =====
def send_prices():
    msg = "🔥 تحديث الأسعار المباشر 🔥\n\n"

    for name, symbol in COINS.items():

        # KAIA – CoinGecko فقط
        if name == "kaia":
            price = get_coingecko_price("kaia")
        else:
            price = get_binance_price(symbol)

        # معالجة المفقود
        if price is None:
            msg += f"• {name.upper()}: N/A USD\n"
        else:
            msg += f"• {name.upper()}: {price} USD\n"

    bot.send_message(CHAT_ID, msg)

# ===== تشغيل مستمر =====
bot.send_message(CHAT_ID, "تم تشغيل البوت بنجاح! 🚀")

while True:
    send_prices()
    time.sleep(20)
