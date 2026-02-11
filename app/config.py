from pydantic_settings import BaseSettings
from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# ML directory
ML_DIR = BASE_DIR / "ml"

# Data Path
DATA_DIR = ML_DIR / "data"
PROCESSED_DATA = DATA_DIR / "processed"
RAW_DATA = DATA_DIR / "raw"

# Model Checkpoint
CHECKPOINT = ML_DIR / "checkpoints"

SENTIMENT_LABELS = {
    0: "negative",
    1: "positive"
}

# Allowed emotions
ALLOWED_EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "neutral"
    ]
# Threshold values
EMOTION_THRESHOLD = {
    "joy": 0.35,
    "sadness": 0.35,
    "anger": 0.35,
    "fear": 0.30,
    "surprise": 0.35,
    "neutral": 0.50,
}

# Tokenizer Parameters
TOKENIZER_PARAMETERS = {
    "pad_token": "<PAD>",
    "unk_token": "<UNK>",
    "max_vocab_size": 30000,
    "max_seq_len": 100,
    "lowercase": True
}

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
