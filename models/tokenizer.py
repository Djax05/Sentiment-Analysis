from collections import Counter
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(texts, max_size):
    counter = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        if text.strip() == "":
            continue
        counter.update(text.split())

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab):
    return [vocab.get(word, vocab[UNK_TOKEN]) for word in text.split()]
