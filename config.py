from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml"
DATA_DIR = MODEL_DIR / "data"
PROCESSED_DATA = DATA_DIR / "processed"

CHECKPOINT = MODEL_DIR / "checkpoints"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
