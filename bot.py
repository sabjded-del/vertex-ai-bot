import os
import time
import requests
import telebot

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TOKEN)

COINS = {
    "xvg": "verge",
    "rose": "oasis-network",
    "gala": "gala",
    "blur": "blur",
    "fil": "filecoin",
    "kaia": "kaia"
}

# ==========================
# جلب السعر من Coingecko
# ==========================
def get_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    
    try:
        r = requests.get(url, timeout=10).json()

        # التحقق من وجود العملة حتى لا يحدث KeyError
        if coin_id not in r:
            return "N/A"

        return r[coin_id]["usd"]

    except Exception:
        return "N/A"

# ==========================
# إرسال الأسعار
# ==========================
def send_prices():
    msg = "🔥 **تحديث الأسعار المباشر** 🔥\n\n"

    for symbol, coin_id in COINS.items():
        price = get_price(coin_id)
        msg += f"• **{symbol.upper()}**: {price} USD\n"

    bot.send_message(CHAT_ID, msg, parse_mode="Markdown")


# ==========================
# بداية تشغيل البوت
# ==========================
bot.send_message(CHAT_ID, "🚀 تم تشغيل البوت بنجاح!")

# إرسال الأسعار كل 15 دقيقة
while True:
    send_prices()
    time.sleep(900)  # 900 ثانية = 15 دقيقة
