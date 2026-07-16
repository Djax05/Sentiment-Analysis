from pydantic_settings import BaseSettings

from config import (
    BASE_DIR,
    DATA_DIR,
    PROCESSED_DATA,
    CHECKPOINT,
)
from utils import ALLOWED_EMOTIONS


class Settings(BaseSettings):

    api_title: str = "Sentiment Analysis API"
    api_version: str = "1.0.0"
    api_description: str = "API for sentiment analysis and emotion detection."

    class Config:
        env_file = ".venv"


settings = Settings()
