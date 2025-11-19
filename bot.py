import os
import requests
from telegram import Bot
import time

# ====== بيانات التليجرام ======
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TOKEN)

# ====== قائمة العملات ======
# تقدر تضيف أو تشيل براحتك
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
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        return data[coin_id]["usd"]
    except:
        return None

# ====== رسالة التنبيه ======
def alert_message():
    text = "📊 **تحديث أسعار العملات الآن:**\n\n"
    for symbol, coin_id in COINS.items():
        price = get_price(coin_id)
        if price:
            text += f"💠 `{symbol.upper()}`: ${price}\n"
        else:
            text += f"❌ `{symbol.upper()}`: تعذّر جلب السعر\n"
    return text

# ====== التشغيل المستمر ======
def start_bot():
    bot.send_message(chat_id=CHAT_ID, text="🚀 تم تشغيل البوت بنجاح!")

    while True:
        msg = alert_message()
        bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        time.sleep(300)  # كل 5 دقائق

if __name__ == "__main__":
    start_bot()
