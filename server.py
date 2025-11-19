# server.py
import os
from flask import Flask, request
from telegram import Update
from bot import (
    bot,
    send_help,
    cmd_xvg,
    cmd_coin,
    cmd_plan,
    cmd_buy,
    cmd_sell,
    cmd_dashboard,
)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "✅ Smart Bot Webhook is running.", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    """
    نقطة استقبال Webhook من تيليجرام.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return "no json", 400

    if not data:
        return "empty", 400

    # تحويل JSON إلى Update من مكتبة python-telegram-bot
    update = Update.de_json(data, bot)

    message = update.message
    if message is None or not message.text:
        # لا يوجد نص (مثلاً صورة، ملصق، ...)، نتجاهله
        return "ok", 200

    chat_id = message.chat.id
    text = message.text.strip()

    # نتعامل فقط مع الأوامر التي تبدأ بـ "/"
    if not text.startswith("/"):
        return "ok", 200

    parts = text.split()
    cmd = parts[0].lower()
    args = parts[1:]

    # نفس منطق process_updates الذي كنت تستخدمه
    if cmd in ["/start", "/help"]:
        send_help(chat_id)
    elif cmd == "/xvg":
        cmd_xvg(chat_id)
    elif cmd == "/coin" and args:
        cmd_coin(chat_id, args[0])
    elif cmd == "/plan":
        cmd_plan(chat_id)
    elif cmd == "/buy":
        cmd_buy(chat_id, args)
    elif cmd == "/sell":
        cmd_sell(chat_id, args)
    elif cmd == "/dashboard":
        cmd_dashboard(chat_id)
    else:
        send_help(chat_id)

    return "ok", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
