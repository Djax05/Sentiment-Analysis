import re
import pandas as pd
from utils import ALLOWED_EMOTIONS


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_sentiment(df) -> pd.DataFrame:
    df = df.copy()
    df["sentence"] = df["sentence"].apply(clean_text)
    return df


def preprocess_goemotions(df) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[["text"] + ALLOWED_EMOTIONS].copy()
    df = df[df[ALLOWED_EMOTIONS].sum(axis=1) > 0]
    return df
