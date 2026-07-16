import torch
from functools import partial
import pandas as pd
from torch.utils.data import DataLoader
from ..datasets.sentiment_dataset import SentimentDataset
from ..datasets.emotion_dataset import EmotionDataset
from .models import EmotionsSentimentModel
from utils import ALLOWED_EMOTIONS
from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE
    )
from .training_utils import collate_fn, get_device
from .tokenizer import load_vocab
from .evaluate import evaluate
from .focal_loss import FocalLoss
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from config import PROCESSED_DATA, CHECKPOINT


def data_type(PROCESSED_DATA, split: str = "train"):

    sentiment_df = pd.read_csv(PROCESSED_DATA / f"sentiment_{split}.csv")
    goemotions_df = pd.read_csv(PROCESSED_DATA / f"goemotions_{split}.csv")
    return sentiment_df, goemotions_df


# importing dataset
def import_data(sentiment_df, goemotions_df):

    # Defining the columns to be used
    sentiment_text = sentiment_df["sentence"].values.tolist()
    sentiment_labels = torch.tensor(sentiment_df["sentiment"])

    goemotions_text = goemotions_df["text"].values.tolist()
    goemotions_labels = goemotions_df[ALLOWED_EMOTIONS].to_numpy().tolist()

    return (
        sentiment_text,
        sentiment_labels,
        goemotions_text,
        goemotions_labels,
    )


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


def metrics(sent_targets, sent_preds, emo_targets, emo_preds):
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

    return emotion_metric, sentiment_metrics, per_emotion_metrics


def train_model(model, sentiment_loader, emotion_loader, device,
                optimizer, sentiment_loss_fn, emotion_loss_fn, vocab):

    best_macro_f1 = 0.0
    sentiments_val, goemotions_val = data_type(PROCESSED_DATA, "val")
    (
        sent_val,
        sentiment_val_labels,
        go_val,
        goemotions__val_labels
        ) = import_data(sentiments_val, goemotions_val)

    sentiment_loader_val, emotion_loader_val = load_dataset(
        sent_val,
        sentiment_val_labels,
        go_val,
        goemotions__val_labels,
        vocab)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
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

        val_outputs = evaluate(
            model,
            sentiment_loader_val,
            emotion_loader_val,
            device)

        emo_probs = val_outputs["emotion_probability"]
        emo_targets = val_outputs["emotion_targets"]

        emo_threshold = 0.5
        emo_preds = (emo_probs >= emo_threshold).astype(int)

        sent_probs = val_outputs["sentiment_probability"]
        sent_targets = val_outputs["sentiment_targets"]

        sent_threshold = 0.5
        sent_preds = (sent_probs >= sent_threshold).astype(int)

        (emotion_metric,
         sentiment_metrics,
         per_emotion_metrics) = metrics(sent_targets,
                                        sent_preds,
                                        emo_targets,
                                        emo_preds)

        if emotion_metric["macro_f1"] > best_macro_f1:
            best_macro_f1 = emotion_metric["macro_f1"]
            torch.save(model.state_dict(), CHECKPOINT / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter == 3:
                print(f"Early stopping at epoch {epoch + 1}")
                return model

        print("Validation Sentiment: ", sentiment_metrics)
        print("Validation emotion: ", emotion_metric)
        print("Validation per_emotion: ", per_emotion_metrics)
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    return model


def main():
    device = get_device()
    sentiments_train, goemotions_train = data_type(PROCESSED_DATA)
    (
        sent_train,
        sentiment_train_labels,
        go_train,
        goemotions__train_labels
        ) = import_data(sentiments_train, goemotions_train)

    vocab = load_vocab()

    model = EmotionsSentimentModel(vocab_size=len(vocab))
    model.to(device)

    sentiment_loss_fn = torch.nn.BCEWithLogitsLoss()
    alpha = torch.tensor([1.85, 2.18, 1.86, 4.63, 2.63, 0.26])
    emotion_loss_fn = FocalLoss(alpha=alpha, gamma=2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    sentiment_loader, emotion_loader = load_dataset(sent_train,
                                                    sentiment_train_labels,
                                                    go_train,
                                                    goemotions__train_labels,
                                                    vocab)

    model = train_model(model, sentiment_loader, emotion_loader, device,
                        optimizer, sentiment_loss_fn, emotion_loss_fn, vocab)

    torch.save(model.state_dict(), CHECKPOINT / "last_model.pt")


if __name__ == "__main__":
    main()
