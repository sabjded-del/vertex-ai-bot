import os
import time
import requests
import telebot

# ====== الإعداد ======
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TOKEN)

# ====== قائمة العملات على Binance ======
COINS = {
    "XVG": "XVGUSDT",
    "ROSE": "ROSEUSDT",
    "GALA": "GALAUSDT",
    "BLUR": "BLURUSDT",
    "FIL": "FILUSDT",
    "KAIA": "KAIAUSDT"
}

# ====== دالة جلب الأسعار من Binance ======
def get_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        r = requests.get(url, timeout=5).json()
        return float(r["price"])
    except:
        return None

# ====== إرسال رسالة الأسعار ======
def send_prices():
    message = "🔥 تحديث الأسعار المباشر 🔥\n\n"

    for name, symbol in COINS.items():
        price = get_price(symbol)
        if price is None:
            message += f"• {name}: N/A USD\n"
        else:
            message += f"• {name}: {price} USD\n"

    bot.send_message(CHAT_ID, message)

# ====== الرد على أمر: أسعار ======
@bot.message_handler(func=lambda m: m.text and m.text.strip() in ["اسعار", "الأسعار", "price", "prices"])
def manual_prices(message):
    send_prices()

# ====== إرسال رسالة تشغيل البوت ======
bot.send_message(CHAT_ID, "🚀 تم تشغيل البوت بنجاح!")

# ====== التحديث المستمر ======
while True:
    try:
        send_prices()
    except Exception as e:
        bot.send_message(CHAT_ID, f"⚠️ خطأ: {e}")
    time.sleep(60)  # تحديث كل دقيقة
