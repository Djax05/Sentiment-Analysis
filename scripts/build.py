import pandas as pd
from ..models.tokenizer import build_vocab, save_vocab
from ..models.training_utils import PROCESSED_DATA
from scripts.data_preprocessing import clean_text


def main():
    df1 = pd.read_csv(PROCESSED_DATA / "sentiment_train.csv")
    df2 = pd.read_csv(PROCESSED_DATA / "goemotions_train.csv")

    texts = (
        df1["sentence"].fillna("").astype(str).apply(clean_text).tolist()
        + df2["text"].fillna("").astype(str).apply(clean_text).tolist()
    )

    vocab = build_vocab(texts, max_size=30000)
    save_vocab(vocab)

    print(f"Vocab saved. Size: {len(vocab)}")


if __name__ == "__main__":
    main()
