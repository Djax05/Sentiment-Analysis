from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml"
DATA_DIR = MODEL_DIR / "data"
PROCESSED_DATA = DATA_DIR / "processed"
RAW_DATA = DATA_DIR / "raw"

CHECKPOINT = MODEL_DIR / "checkpoints"


EMOTION_THRESHOLD = {
    "joy": 0.35,
    "sadness": 0.35,  # sadness
    "anger": 0.35,  # anger
    "fear": 0.30,  # fear
    "surprise": 0.35,  # surprise
    "neutral": 0.50,  # neutral
}
