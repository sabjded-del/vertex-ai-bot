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

def get_price(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        r = requests.get(url).json()
        return r.get(coin_id, {}).get("usd", "N/A")
    except:
        return "N/A"

def send_prices():
    msg = "📊 **أسعار العملات الآن:** \n\n"
    for symbol, coin_id in COINS.items():
        price = get_price(coin_id)
        msg += f"• {symbol.upper()}: {price}$\n"
    bot.send_message(CHAT_ID, msg, parse_mode="Markdown")

# تشغيل البوت كل 15 دقيقة
while True:
    send_prices()
    time.sleep(900)
