import re
import pandas as pd


def clean_text(text: str) -> str:

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_sentiment(df) -> pd.DataFrame:
    df = df.copy()
    df = df["sentence"].apply(clean_text)
    return df


def preprocess_goemotions(df) -> pd.DataFrame:
    df = df.copy()
    allowed_emotions = ["joy", "sadness", "anger", "fear",
                        "surprise", "neutral"]
    df = df[df[allowed_emotions].sum(axis=1) > 0]
    df = df["text"].apply(clean_text)
    return df
