from pydantic_settings import BaseSettings

from config import (
    BASE_DIR,
    DATA_DIR,
    PROCESSED_DATA,
    RAW_DATA,
    CHECKPOINT,
    EMOTION_THRESHOLD,
    TOKENIZER_PARAMETERS,
)
from utils import SENTIMENT_LABELS, ALLOWED_EMOTIONS

# Model
MODEL = CHECKPOINT / "best_model.pt"


class Settings(BaseSettings):

    api_title: str = "Sentiment Analysis API"
    api_version: str = "1.0.0"
    api_description: str = "API for sentiment analysis and emotion detection."

    min_input_length: int = 1
    max_input_length: int = 100

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    class Config:
        env_file = ".venv"


settings = Settings()
