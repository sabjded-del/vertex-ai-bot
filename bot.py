import os
import time
import requests
import telebot

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = telebot.TeleBot(TOKEN)

# رسالة تشغيل مرة واحدة فقط
bot.send_message(CHAT_ID, "🚀 تم تشغيل البوت بنجاح!")

COINS = {
    "xvg": "verge",
    "rose": "oasis-network",
    "gala": "gala",
    "blur": "blur",
    "fil": "filecoin",
    "kaia": "kaia"
}

def get_price(symbol, coin_id):
    # خاص لـ XVG لأن Coingecko لا يدعمها
    if symbol == "xvg":
        url = "https://api.coinpaprika.com/v1/tickers/xvg-verge"
        r = requests.get(url).json()
        return r["quotes"]["USD"]["price"]

    # باقي العملات من Coingecko
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    r = requests.get(url).json()
    return r[coin_id]["usd"]

def send_prices():
    msg = "📊 **أسعار العملات الآن:**\n\n"
    for symbol, coin_id in COINS.items():
        price = get_price(symbol, coin_id)
        msg += f"- {symbol.upper()}: {price}\n"
    bot.send_message(CHAT_ID, msg)

# يحدث كل 15 دقيقة
while True:
    send_prices()
    time.sleep(900)
