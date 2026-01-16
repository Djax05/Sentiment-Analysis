from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"
PROCESSED_DATA = DATA / "processed"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
