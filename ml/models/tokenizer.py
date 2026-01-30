import pickle
from pathlib import Path
from collections import Counter


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
VOCAB_PATH = Path("ml/artifacts/vocab.pkl")


def tokenize(text: str):
    return text.lower().split()


def build_vocab(texts, max_size):
    counter = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        if text.strip() == "":
            continue
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)

    return vocab


def save_vocab(vocab, path=VOCAB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vocab, f)


def load_vocab(path=VOCAB_PATH):
    if not path.exists():
        raise FileNotFoundError(
            "Vocab not found. Run vocab building first."
        )

    with open(path, "rb") as f:
        return pickle.load(f)


def encode_text(text, vocab, max_len=100):

    tokens = tokenize(text)
    indices = [vocab.get(word, vocab[UNK_TOKEN]) for word in tokens]

    if len(indices) > max_len:
        indices = indices[:max_len]

    else:
        indices += [vocab[PAD_TOKEN]] * (max_len - len(indices))

    return indices
