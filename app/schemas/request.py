from pydantic import BaseModel, Field
from typing import Dict, List


class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text to be analyzed",
        examples=["Hello, I feel good today"]
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, I feel good today"
            }
        }


class SentimentResult(BaseModel):
    probability: float
    label: str


class PredictResponse(BaseModel):
    """
    Example response:
    {
        "sentiment": {
            "probability": 0.9,
            "label": "positive"
        },
        "emotions": {
            "anger": 0.1,
            "anticipation": 0.2,
            "disgust": 0.3,
            "fear": 0.4,
            "joy": 0.5,
            "sadness": 0.6,
            "surprise": 0.7,
            "trust": 0.8
        },
        "active_emotions": ["anger", "anticipation", "joy"]}
    """
    sentiment: SentimentResult
    emotions: Dict[str, float]
    active_emotions: List[str]


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
