import pickle
from pathlib import Path

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

VOCAB_PATH = Path("ml/artifacts/vocab.pkl")


def load_vocab(path=VOCAB_PATH):
    if not path.exists():
        raise FileNotFoundError("Vocab not found. Run training first.")

    with open(path, "rb") as f:
        return pickle.load(f)
