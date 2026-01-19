from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


EMOTION_THRESHOLD = {
    0: 0.35,  # joy
    1: 0.35,  # sadness
    2: 0.35,  # anger
    3: 0.30,  # fear
    4: 0.35,  # surprise
    5: 0.50,  # neutral
}