from pydantic import BaseModel

class TelegramMessagesLLMResponse(BaseModel):
    is_pump_and_dump: bool
    cryptocurrencies: list[str]
    summary: str

    