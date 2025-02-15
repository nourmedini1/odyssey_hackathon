from pydantic import BaseModel


class Details(BaseModel):
    summary_paragraph: str
    news_related_to: list[str]

class NewsLLMResponse(BaseModel):
    political_sentiment: Details
    technical_analysis: Details
    new_coins: Details



