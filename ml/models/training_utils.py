import pandas as pd
import torch
from functools import partial
from config import (PROCESSED_DATA, BATCH_SIZE,
                    CHECKPOINT)
from utils import ALLOWED_EMOTIONS
from torch.utils.data import DataLoader
from .models import EmotionsSentimentModel

from ..datasets.emotion_dataset import EmotionDataset
from ..datasets.sentiment_dataset import SentimentDataset

from .tokenizer import load_vocab, encode_text


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def collate_fn(batch, vocab):
    texts = [str(item["text"]) for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    encoded = [encode_text(t, vocab) for t in texts]
    max_len = max(len(e) for e in encoded)

    padded = [
        e + [0] * (max_len - len(e))
        for e in encoded
    ]

    input_ids = torch.tensor(padded, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "labels": labels
    }


def load_validation_setup():
    emotion_val = pd.read_csv(PROCESSED_DATA / "goemotions_val.csv")
    sentiment_val = pd.read_csv(PROCESSED_DATA / "sentiment_val.csv")

    sentiment_text = sentiment_val["sentence"].values.tolist()
    sentiment_labels = torch.tensor(sentiment_val["sentiment"])

    emotion_text = emotion_val["text"].values.tolist()
    emotion_labels = emotion_val[ALLOWED_EMOTIONS].to_numpy().tolist()

    emotion_dataset = EmotionDataset(
        texts=emotion_text,
        labels=emotion_labels
    )

    sentiment_dataset = SentimentDataset(
        texts=sentiment_text,
        labels=sentiment_labels
    )

    vocab = load_vocab()
    collate = partial(collate_fn, vocab=vocab)

    val_sentiment_loader = DataLoader(
        sentiment_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate
    )

    val_emotion_loader = DataLoader(
        emotion_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate
    )

    device = get_device()

    model = EmotionsSentimentModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load(CHECKPOINT / "best_model.pt"))
    model.to(device)
    model.eval()

    return model, val_sentiment_loader, val_emotion_loader, device