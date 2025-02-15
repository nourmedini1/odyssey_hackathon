import asyncio
import requests
import dotenv
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TOKEN = dotenv.get_key(".env", "TELEGRAM_TOKEN")
CHATBOT_API_URL = "http://20.199.80.240:5010/chat/"

subscribers = set()
last_bot_message = {}

async def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    subscribers.add(chat_id)
    message = (
        "🛡️ Welcome to CryptoGuardian!\n\n"
        "Stay ahead in the Ethereum market with real-time updates! 📈\n\n"
        "🔹 Latest Crypto News 📰\n"
        "🔹 Market Insights & Price Trends 📊\n"
        "🔹 Pump & Dump Warnings ⚠️\n\n"
        "I'll keep you informed about the latest crypto movements. Stay safe, stay smart!\n\n"
        "🔔 Expect updates soon!"
    )
    await update.message.reply_text(message)
    last_bot_message[chat_id] = message
    logging.info(f"Sent welcome message to chat_id: {chat_id}")

async def handle_message(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    user_message = update.message.text

    payload = {
        "user_question": user_message,
        "context": last_bot_message.get(chat_id, "No previous message"),
    }
    chatbot_response = "Sorry, I'm currently unavailable. Please try again later."
    try:
        response = requests.post(CHATBOT_API_URL, json=payload)
        llm_response = response.json()
        chatbot_response = llm_response.get("chatbot_response", chatbot_response)
    except Exception as e:
        logging.error(f"Error calling chatbot API for chat_id {chat_id}: {e}")
        chatbot_response = "Sorry, I'm currently unavailable. Please try again later."

    await update.message.reply_text(chatbot_response)
    last_bot_message[chat_id] = chatbot_response
    logging.info(f"Processed message from chat_id {chat_id} with response: {chatbot_response}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    tg_app = Application.builder().token(TOKEN).build()
    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))  
    await tg_app.initialize()
    await tg_app.start()
    await tg_app.updater.start_polling()   
    app.state.telegram_bot = tg_app.bot
    app.state.tg_app = tg_app
    logging.info("Telegram bot started and polling is active.")
    yield  
    await tg_app.updater.stop()
    await tg_app.stop()
    await tg_app.shutdown()
    logging.info("Telegram bot shutdown complete.")

app = FastAPI(lifespan=lifespan)

@app.post("/send-message/")
async def send_message(message: dict):
    if not subscribers:
        logging.info("Send-message API called but no subscribers found.")
        return {"status": "No users subscribed"}
    bot = app.state.telegram_bot
    logging.info(f"Send-message API called. Attempting to send blog post to {len(subscribers)} subscribers.")  
    for chat_id in subscribers:
        try:
            await bot.send_message(chat_id=chat_id, text=message["blog_text"])
            last_bot_message[chat_id] = message["blog_text"]
            logging.info(message["blog_text"])
            logging.info(f"Sent blog post to chat_id: {chat_id}")
        except Exception as e:
            logging.error(f"Failed to send blog post to chat_id {chat_id}: {e}")
    return {"status": "Message sent to all subscribers"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
