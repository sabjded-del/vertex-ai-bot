import os
import time
import requests
import telebot

# ===== إعداد =====
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TOKEN)

# قائمة العملات + ID CoinGecko
COINS = {
    "xvg": "verge",
    "rose": "oasis-network",
    "gala": "gala",
    "blur": "blur",
    "fil": "filecoin",
}

# ===== جلب أسعار CoinGecko دفعة واحدة =====
def get_prices():
    ids = ",".join(COINS.values())
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"

    for _ in range(3):  # إعادة المحاولة 3 مرات
        try:
            response = requests.get(url, timeout=10)

            # إذا CoinGecko رفض الطلب Rate Limit
            if response.status_code == 429:
                time.sleep(2)
                continue

            data = response.json()
            return data

        except:
            time.sleep(1)

    return None


# ===== تنسيق رسالة الأسعار =====
def format_prices(data):
    message = "🔥 تحديث الأسعار المباشر 🔥\n\n"

    for symbol, gecko_id in COINS.items():
        if gecko_id in data and "usd" in data[gecko_id]:
            price = data[gecko_id]["usd"]
            message += f"• {symbol.upper()}: {price} USD\n"
        else:
            message += f"• {symbol.upper()}: N/A USD\n"

    return message


# ===== وظيفة الإرسال =====
def send_prices():
    data = get_prices()
    if not data:
        bot.send_message(CHAT_ID, "خطأ في جلب الأسعار ❌")
        return

    msg = format_prices(data)
    bot.send_message(CHAT_ID, msg)


# ===== تشغيل البوت =====
bot.send_message(CHAT_ID, "🚀 تم تشغيل البوت بنجاح!")

while True:
    send_prices()
    time.sleep(60)  # كل 60 ثانية
