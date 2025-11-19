import os
import time
import requests
import telebot

# ===== المتغيرات =====
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TOKEN)

# ===== قائمة العملات مع معرف CoinGecko =====
COINS = {
    "xvg": "verge",
    "rose": "oasis-network",
    "gala": "gala",
    "blur": "blur",
    "fil": "filecoin",
}

# ===== جلب السعر من CoinGecko =====
def get_coingecko_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data.get(coin_id, {}).get("usd")
    except:
        return None

# ===== دالة إرسال الأسعار =====
def send_prices():
    message = "🔥 تحديث الأسعار المباشر 🔥\n\n"

    for coin, cg_id in COINS.items():
        price = get_coingecko_price(cg_id)

        if price is not None:
            message += f"• {coin.upper()}: {price} USD\n"
        else:
            message += f"• {coin.upper()}: N/A USD\n"

    bot.send_message(CHAT_ID, message)

# ===== التشغيل =====
bot.send_message(CHAT_ID, "🚀 تم تشغيل البوت بنجاح باستخدام CoinGecko فقط!")

while True:
    send_prices()
    time.sleep(15)   # انتظر 15 ثانية بين كل تحديث
