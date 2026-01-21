from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


EMOTION_THRESHOLD = {
    "joy": 0.35,
    "sadness": 0.35,  # sadness
    "anger": 0.35,  # anger
    "fear": 0.30,  # fear
    "surprise": 0.35,  # surprise
    "neutral": 0.50,  # neutral
}