import os
import time
import requests
import telebot

# ====== إعداد المتغيرات ======
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TOKEN)

# ====== قائمة العملات ======
COINS = {
    "xvg": "verge",
    "rose": "oasis-network",
    "gala": "gala",
    "blur": "blur",
    "fil": "filecoin",
    "kaia": "kaia"
}

# ====== جلب سعر من CoinGecko مع إعادة محاولات ======
def get_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"

    for attempt in range(3):   # نجرب 3 مرات قبل الرجوع N/A
        try:
            r = requests.get(url, timeout=5).json()

            if coin_id in r and "usd" in r[coin_id]:
                return r[coin_id]["usd"]

        except:
            pass

        time.sleep(1)  # ننتظر 1 ثانية بين كل محاولة

    return None  # لو ما قدر يجيب السعر


# ====== إرسال الأسعار ======
def send_prices():
    msg = "🔥 تحديث الأسعار المباشر 🔥\n\n"

    for symbol, coin_id in COINS.items():

        price = get_price(coin_id)

        if price is None:
            msg += f"• {symbol.upper()}: N/A USD\n"
        else:
            msg += f"• {symbol.upper()}: {price} USD\n"

        time.sleep(0.5)  # نصف ثانية بين كل عملة لتجنب الحظر

    bot.send_message(CHAT_ID, msg)


# ====== أمر /start ======
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "🚀 تم تشغيل البوت بنجاح!")
    send_prices()


# ====== أمر /اسعار ======
@bot.message_handler(func=lambda msg: msg.text in ["اسعار", "price", "prices"])
def manual_prices(message):
    send_prices()


# ====== تشغيل البوت بدون توقف ======
while True:
    try:
        send_prices()
        time.sleep(60)  # يحدث كل دقيقة
    except Exception as e:
        print("Error:", e)
        time.sleep(5)
