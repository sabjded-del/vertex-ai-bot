import os
import requests
from telegram import Bot
import time

# ====== بيانات التليجرام ======
TOKEN = os.getenv("TOKEN")       # من Environment Variables
CHAT_ID = os.getenv("CHAT_ID")   # من Environment Variables

bot = Bot(token=TOKEN)

# ====== قائمة العملات ======
COINS = {
    "xvg": "verge",
    "rose": "oasis-network",
    "gala": "gala",
    "blur": "blur",
    "fil": "filecoin",
    "kaia": "kaia"
}

# ====== دالة لجلب السعر من CoinGecko ======
def get_price(coin_id):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    response = requests.get(url).json()
    return response[coin_id]["usd"]

# ====== دالة إرسال رسالة ======
def send_message(msg):
    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

# ====== إرسال رسالة تشغيل ======
send_message("🚀 تم تشغيل البوت بنجاح!")

# ====== التحديث المستمر ======
while True:
    try:
        msg = "📊 *أسعار العملات الآن:*\n\n"
        for symbol, coin_id in COINS.items():
            price = get_price(coin_id)
            msg += f"• *{symbol.upper()}*: ${price}\n"

        send_message(msg)

    except Exception as e:
        send_message(f"❌ خطأ: {e}")

    time.sleep(20)  # يحدث الأسعار كل 20 ثانية
