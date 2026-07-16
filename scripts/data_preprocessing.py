import re
import pandas as pd
from utils import ALLOWED_EMOTIONS


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@\w+|https?://\S+|www\.\S+", "", str(text))
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def preprocess_goemotions(df) -> pd.DataFrame:
    df = df.copy()
    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .apply(clean_text))
    df = df[["text"] + ALLOWED_EMOTIONS].copy()
    df = df[df[ALLOWED_EMOTIONS].sum(axis=1) > 0]
    return df
