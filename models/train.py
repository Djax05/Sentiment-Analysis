import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.sentiment_dataset import SentimentDataset
from datasets.emotion_dataset import EmotionDataset
from .models import EmotionsSentimentModel
from utils import ALLOWED_EMOTIONS
from .tokenizer import build_vocab, encode_text
from scripts.data_preprocessing import clean_text

# Defined Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"
PROCESSED_DATA = DATA / "processed"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3

# importing dataset
sentiment_df = pd.read_csv(PROCESSED_DATA / "sentiment_train.csv")
goemotions_df = pd.read_csv(PROCESSED_DATA / "goemotions_train.csv")

sentiment_df["sentence"] = (
    sentiment_df["sentence"]
    .fillna("")
    .astype(str)
    .apply(clean_text)
)

goemotions_df["text"] = (
    goemotions_df["text"]
    .fillna("")
    .astype(str)
    .apply(clean_text)
)

# Defining the columns to be used
sentiment_text = sentiment_df["sentence"].values.tolist()
sentiment_labels = torch.tensor(sentiment_df["sentiment"])


goemotions_text = goemotions_df["text"].values.tolist()
goemotions_labels = goemotions_df[ALLOWED_EMOTIONS].to_numpy().tolist()

texts = sentiment_text + goemotions_text


# Model Preparation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab = build_vocab(texts, 30000)

model = EmotionsSentimentModel(vocab_size=len(vocab))
model.to(device)

sentiment_loss_fn = torch.nn.BCEWithLogitsLoss()
emotion_loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


sentiment_dataset = SentimentDataset(
    texts=sentiment_text,
    labels=sentiment_labels,
)

emotion_dataset = EmotionDataset(
    texts=goemotions_text,
    labels=goemotions_labels,
)


def collate_fn(batch):
    texts = [item["text"] for item in batch]
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


sentiment_loader = DataLoader(
    sentiment_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

emotion_loader = DataLoader(
    emotion_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for (sent_batch, emo_batch) in zip(sentiment_loader, emotion_loader):
        optimizer.zero_grad()

        sent_input_ids = sent_batch["input_ids"].to(device)
        sent_labels = sent_batch["labels"].to(device)
        sent_labels = sent_labels.unsqueeze(1)

        sent_logits, _ = model(sent_input_ids)
        sent_loss = sentiment_loss_fn(sent_logits, sent_labels)

        emo_input_ids = emo_batch["input_ids"].to(device)
        emo_labels = emo_batch["labels"].to(device)

        _, emo_logits = model(emo_input_ids)
        emo_loss = emotion_loss_fn(emo_logits, emo_labels)

        loss = sent_loss + emo_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")
