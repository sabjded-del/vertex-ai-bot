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
    "XVG": "verge",
    "ROSE": "oasis-network",
    "GALA": "gala",
    "BLUR": "blur",
    "FIL": "filecoin",
    "KAIA": "kaia"
}

# ===== جلب السعر من CoinGecko =====
def get_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    r = requests.get(url).json()

    if coin_id in r:
        return r[coin_id]["usd"]
    else:
        return None   # لا يرجع خطأ – فقط N/A

# ===== إرسال الأسعار =====
def send_prices():
    msg = "🔥 تحديث الأسعار المباشر 🔥\n\n"

    for symbol, coin_id in COINS.items():
        price = get_price(coin_id)

        if price is None:
            msg += f"• {symbol}: N/A USD\n"
        else:
            msg += f"• {symbol}: {price} USD\n"

    bot.send_message(CHAT_ID, msg)

# ===== تشغيل البوت =====
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "🚀 تم تشغيل البوت بنجاح!")

@bot.message_handler(func=lambda m: m.text == "اسعار")
def prices(message):
    send_prices()

# ===== حلقة التشغيل =====
while True:
    send_prices()
    time.sleep(300)
