from collections import deque
import asyncio
import time
import dotenv
from telethon import TelegramClient, events
from fastapi import FastAPI
from mistralai import Mistral 
from uvicorn import Config, Server
import json
from fastapi.middleware.cors import CORSMiddleware

API_HASH = dotenv.get_key('.env', 'API_HASH')
API_ID = dotenv.get_key('.env', 'API_ID')
PHONE_NUMBER = dotenv.get_key('.env', 'PHONE_NUMBER')
MISTRAL_API_KEY = dotenv.get_key('.env', 'MISTRAL_API_KEY')

pnd_groups = [
    'https://t.me/sharks_pump',
    'https://t.me/cryptoclubpump',
    'https://t.me/VerifiedCryptoNews',
    'https://t.me/mega_pump_group',
    'https://t.me/mega_pump_group_signals',
    'https://t.me/cryptoflashsignals',
    'https://t.me/RocketWallet_Official',
]

news_groups = [
    'https://t.me/ethereumnews',
    'https://t.me/VerifiedCryptoNews',
    'https://t.me/coinlistofficialchannel',
    'https://t.me/testing_scraping'
]

model = "mistral-large-latest"
llm = Mistral(api_key=MISTRAL_API_KEY)


pnd_unsent_messages = deque(maxlen=20)  
news_unsent_messages = deque(maxlen=20) 

def get_telegram_messages_prompt(messages):
    return f"""
You are a sentiment analysis model and chatbot for cryptocurrency topics.
Your task is to analyze the following Telegram messages discussing potential pump and dump schemes.
For each message, determine:
1. Whether the message is discussing a pump or dump scheme (return a boolean).
2. The cryptocurrencies being discussed.
3. A summary paragraph of what the messages are about.

Messages:
{messages}

Return the result in the following JSON format:
{{
    "is_pump_and_dump": boolean,
    "cryptocurrencies": [list of cryptocurrencies],
    "summary": "summary paragraph"
}}
"""

def get_news_prompt(news): 
    return f"""
You are a sentiment analysis model and chatbot specialized in cryptocurrency news.
Analyze the following news headlines and classify them into:
1. Political sentiment about crypto,
2. Technical analysis of the market,
3. News about new coins or projects.

Return a JSON object with the following format:
{{
    "political_sentiment": {{
        "summary_paragraph": "summary paragraph",
        "news_related_to": [list of headlines]
    }},
    "technical_analysis": {{
        "summary_paragraph": "summary paragraph",
        "news_related_to": [list of headlines]
    }},
    "new_coins": {{
        "summary_paragraph": "summary paragraph",
        "news_related_to": [list of headlines]
    }}
}}

News:
{news}
"""

client = TelegramClient('sentiment_analysis_session', API_ID, API_HASH)
app = FastAPI(description="Crypto Sentiment Analysis API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

def make_event_handler(queue: deque):
    async def handler(event):
        print("Received message:", event.message.text)
        chat = await event.get_chat()
        message_data = {
            "group_id": chat.id,
            "group_name": chat.title,
            "message_id": event.message.id,
            "sender": event.message.sender_id,
            "text": event.message.text,
            "timestamp": event.message.date.strftime('%Y-%m-%d %H:%M:%S')
        }
        queue.append(message_data)  
        print(f"Queue size is now: {len(queue)}")
    return handler

async def monitor_groups():
    await client.start(phone=PHONE_NUMBER)
    links = pnd_groups + news_groups
    for link in links:
        try:
            group = await client.get_entity(link)
            print(f"Monitoring group: {group.title}")
            if link in pnd_groups:
                client.add_event_handler(make_event_handler(pnd_unsent_messages), events.NewMessage(chats=[group]))
            else:
                client.add_event_handler(make_event_handler(news_unsent_messages), events.NewMessage(chats=[group]))
        except Exception as e:
            print(f"Failed to monitor {link}: {e}")
    print("Listening for new messages...")
    await asyncio.Event().wait()

async def get_llm_sentiment_verdict(prompt):
    chat_response = llm.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return chat_response.choices[0].message

@app.get("/telegram/messages")
def get_messages():
    messages_to_send = list(pnd_unsent_messages)  
    analysis = asyncio.run(get_llm_sentiment_verdict(get_telegram_messages_prompt(messages_to_send)))
    content = analysis.content  

    try:
        content_json = content.split("```json")[-1].split("```")[0].strip()
        analysis_result = json.loads(content_json)
        fixed_analysis = {
            "is_pump_and_dump": bool(analysis_result.get("is_pump_and_dump", False)),
            "cryptocurrencies": analysis_result.get("cryptocurrencies", []),
            "summary": analysis_result.get("summary", "")
        }
        if not isinstance(fixed_analysis["cryptocurrencies"], list):
            fixed_analysis["cryptocurrencies"] = []
        else:
            fixed_analysis["cryptocurrencies"] = [str(crypto) for crypto in fixed_analysis["cryptocurrencies"]]

    except (ValueError, IndexError, json.JSONDecodeError) as e:
        print(f"Failed to parse JSON: {e}")
        fixed_analysis = {
            "is_pump_and_dump": False,
            "cryptocurrencies": [],
            "summary": "Failed to parse LLM response correctly."
        }
    
    return {
        "messages": messages_to_send,
        "count": len(messages_to_send),
        "analysis": fixed_analysis  
    }

@app.get("/news")
def get_news():
    messages_to_send = list(news_unsent_messages)  
    analysis = asyncio.run(get_llm_sentiment_verdict(get_news_prompt(messages_to_send)))
    content = analysis.content

    try:
        content_json = content.split("```json")[-1].split("```")[0].strip()
        analysis_result = json.loads(content_json)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        analysis_result = {}

    return {
        "news": messages_to_send,
        "count": len(messages_to_send),
        "analysis": analysis_result
    }

async def main():
    monitor_task = asyncio.create_task(monitor_groups())
    config = Config(app, host="0.0.0.0", port=5030, reload=False, loop="asyncio")
    server = Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.gather(monitor_task, server_task)

if __name__ == "__main__":
    asyncio.run(main())
