import torch
from functools import partial
import pandas as pd
from torch.utils.data import DataLoader
from datasets.sentiment_dataset import SentimentDataset
from datasets.emotion_dataset import EmotionDataset
from .models import EmotionsSentimentModel
from utils import ALLOWED_EMOTIONS
from .training_utils import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    PROCESSED_DATA
    )
from .tokenizer import encode_text, load_vocab
from scripts.data_preprocessing import clean_text
from .evaluate import evaluate
from .focal_loss import FocalLoss
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)


# Defined Paths


# Hyperparameters


# importing dataset
def import_data(PROCESSED_DATA):

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
    return (
        sentiment_text,
        sentiment_labels,
        goemotions_text,
        goemotions_labels,
    )


# Model Preparation
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def collate_fn(batch, vocab):
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


def load_dataset(sentiment_text, sentiment_labels,
                 goemotions_text, goemotions_labels, vocab):

    sentiment_dataset = SentimentDataset(
        texts=sentiment_text,
        labels=sentiment_labels,
    )

    emotion_dataset = EmotionDataset(
        texts=goemotions_text,
        labels=goemotions_labels,
    )

    collate = partial(collate_fn, vocab=vocab)

    sentiment_loader = DataLoader(
        sentiment_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate
    )

    emotion_loader = DataLoader(
        emotion_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate
    )

    return sentiment_loader, emotion_loader


def train_model(model, sentiment_loader, emotion_loader, device,
                optimizer, sentiment_loss_fn, emotion_loss_fn):
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        best_macro_f1 = 0.0
        for (sent_batch, emo_batch) in zip(sentiment_loader, emotion_loader):

            sent_input_ids = sent_batch["input_ids"].to(device)
            sent_labels = sent_batch["labels"].to(device)
            sent_labels = sent_labels.unsqueeze(1)

            emo_input_ids = emo_batch["input_ids"].to(device)
            emo_labels = emo_batch["labels"].to(device)

            optimizer.zero_grad()

            sent_logits = model(sent_input_ids, task="sentiment")
            emo_logits = model(emo_input_ids, task="emotion")

            sent_loss = sentiment_loss_fn(sent_logits, sent_labels)

            focal_loss = emotion_loss_fn(emo_logits, emo_labels)
            emo_loss = focal_loss.mean(dim=0).sum()

            loss = sent_loss + emo_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_outputs = evaluate(model,
                               sentiment_loader,
                               emotion_loader,
                               device)

        emo_probs = val_outputs["emotion_probability"]
        emo_targets = val_outputs["emotion_targets"]

        emo_threshold = 0.5
        emo_preds = (emo_probs >= emo_threshold).astype(int)

        sent_probs = val_outputs["sentiment_probability"]
        sent_targets = val_outputs["sentiment_targets"]

        sent_threshold = 0.5
        sent_preds = (sent_probs >= sent_threshold).astype(int)

        per_emotion_metrics = {}

        for i, emotion in enumerate(ALLOWED_EMOTIONS):
            y_true = emo_targets[:, i]
            y_pred = emo_preds[:, i]

            per_emotion_metrics[emotion] = {
                "precision": precision_score(y_true,
                                             y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0)
            }

        sentiment_metrics = {
            "accuracy": accuracy_score(sent_targets, sent_preds),
            "precision": precision_score(sent_targets,
                                         sent_preds, zero_division=0),
            "recall": recall_score(sent_targets, sent_preds,
                                   zero_division=0),
            "f1": f1_score(sent_targets, sent_preds,
                           zero_division=0)
        }

        emotion_metric = {
            "micro_f1": f1_score(emo_targets, emo_preds,
                                 average="micro", zero_division=0),
            "macro_f1": f1_score(emo_targets, emo_preds,
                                 average="macro", zero_division=0)
        }

        if emotion_metric["macro_f1"] > best_macro_f1:
            best_macro_f1 = emotion_metric["macro_f1"]
            torch.save(model.state_dict(), "checkpoints/best_model.pt")

        print("Validation Sentiment: ", sentiment_metrics)
        print("Validation emotion: ", emotion_metric)
        print("Validation per_emotion: ", per_emotion_metrics)
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    return model


def main():
    device = get_device()
    (
        sentiment_text,
        sentiment_labels,
        goemotions_text,
        goemotions_labels
        ) = import_data(PROCESSED_DATA)

    vocab = load_vocab()

    model = EmotionsSentimentModel(vocab_size=len(vocab))
    model.to(device)

    sentiment_loss_fn = torch.nn.BCEWithLogitsLoss()
    emotion_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    sentiment_loader, emotion_loader = load_dataset(sentiment_text,
                                                    sentiment_labels,
                                                    goemotions_text,
                                                    goemotions_labels,
                                                    vocab)

    model = train_model(model, sentiment_loader, emotion_loader, device,
                        optimizer, sentiment_loss_fn, emotion_loss_fn)

    torch.save(model.state_dict(), "checkpoints/last_model.pt")


if __name__ == "__main__":
    main()
