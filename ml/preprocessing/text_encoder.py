from .vocab import PAD_TOKEN, UNK_TOKEN


MAX_SEQ_LEN = 100


def tokenize(text: str):
    return text.lower().split()


def encode_text(text: str, vocab: dict):
    if not isinstance(text, str) or text.split() == "":
        raise ValueError("Invalid Input text")

    tokens = tokenize(text)
    indices = [vocab.get(word, vocab[UNK_TOKEN]) for word in tokens]

    if len(indices) > MAX_SEQ_LEN:
        indices = indices[:MAX_SEQ_LEN]
    else:
        indices += [vocab[PAD_TOKEN]] * (MAX_SEQ_LEN - len(indices))

    return indices
