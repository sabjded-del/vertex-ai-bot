import os
import time
import threading
import requests
import telebot

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = telebot.TeleBot(TOKEN)

COINS = {
    "XVG": "verge",
    "ROSE": "oasis-network",
    "GALA": "gala",
    "BLUR": "blur",
    "FIL": "filecoin",
    "KAIA": "kaia"
}

def get_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        r = requests.get(url, timeout=10).json()
        if coin_id in r:
            return r[coin_id]["usd"]
        else:
            return None
    except:
        return None

def send_prices():
    msg = "🔥 تحديث الأسعار المباشر 🔥\n\n"
    for symbol, coin_id in COINS.items():
        price = get_price(coin_id)
        if price is None:
            msg += f"• {symbol}: N/A USD\n"
        else:
            msg += f"• {symbol}: {price} USD\n"
    bot.send_message(CHAT_ID, msg)

# ============== أوامر البوت ==============
@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(message.chat.id, "🚀 تم تشغيل البوت بنجاح!")

@bot.message_handler(func=lambda m: m.text == "اسعار")
def prices_now(message):
    send_prices()

# ============== مهمة الخلفية ==============
def background_task():
    while True:
        send_prices()
        time.sleep(300)   # كل 5 دقائق

# تشغيل التحديث في Thread مستقل
thread = threading.Thread(target=background_task)
thread.daemon = True
thread.start()

# تشغيل البوت
bot.polling(none_stop=True)
