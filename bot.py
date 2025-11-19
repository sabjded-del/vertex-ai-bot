import os
import time
import requests
import telebot

# ===== إعداد المتغيرات =====
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TOKEN)

# ===== قائمة العملات =====
COINS = {
    "xvg": {
        "binance": "XVGUSDT",
        "coingecko": "verge"
    },

    "rose": {
        "binance": "ROSEUSDT",
        "coingecko": "oasis-network"
    },

    "gala": {
        "binance": "GALAUSDT",
        "coingecko": "gala"
    },

    "blur": {
        "binance": "BLURUSDT",
        "coingecko": "blur"
    },

    "fil": {
        "binance": "FILUSDT",
        "coingecko": "filecoin"
    },

    "kaia": {
        "binance": None,
        "coingecko": "kaia"
    }
}

# ===== Binance API =====
def get_binance_price(symbol):
    if symbol is None:
        return None
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        r = requests.get(url, timeout=5).json()
        return float(r["price"]) if "price" in r else None
    except:
        return None

# ===== CoinGecko API =====
def get_coingecko_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        r = requests.get(url, timeout=5).json()
        return float(r[coin_id]["usd"]) if coin_id in r else None
    except:
        return None

# ===== إرسال الأسعار =====
def send_prices():
    msg = "🔥 تحديث الأسعار المباشر 🔥\n\n"

    for name, data in COINS.items():

        # 1) نحاول من Binance
        price = get_binance_price(data["binance"])

        # 2) ولو Binance فشل نستخدم CoinGecko
        if price is None:
            price = get_coingecko_price(data["coingecko"])

        if price is None:
            msg += f"• {name.upper()}: N/A USD\n"
        else:
            msg += f"• {name.upper()}: {price} USD\n"

    bot.send_message(CHAT_ID, msg)

# ===== رسالة بدء التشغيل =====
bot.send_message(CHAT_ID, "تم تشغيل البوت بنجاح! 🚀")

# ===== حلقة التحديث =====
while True:
    send_prices()
    time.sleep(20)
