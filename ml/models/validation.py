import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from utils import ALLOWED_EMOTIONS
from torch.utils.data import DataLoader
from .train import (
    collate_fn,
    BATCH_SIZE,
    )
from ..datasets.emotion_dataset import EmotionDataset
from ..datasets.sentiment_dataset import SentimentDataset
from .evaluate import evaluate
from .tokenizer import load_vocab
from .models import EmotionsSentimentModel
from .train import (
    get_device,
    metrics
)
from functools import partial
from config import EMOTION_THRESHOLD, PROCESSED_DATA, CHECKPOINT


def tune_thresholds(probs, targets, thresholds=np.linspace(0.05, 0.95, 19)):
    best_threshold = {}
    best_f1s = {}

    num_emotions = probs.shape[1]

    for i in range(num_emotions):
        best_f1 = 0
        best_t = 0.5

        for t in thresholds:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(targets[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_threshold[i] = best_t
        best_f1s[i] = best_f1

    return best_threshold, best_f1s


def apply_thresholds(probs, threshold):
    preds = np.zeros_like(probs)

    for i, emotion in enumerate(ALLOWED_EMOTIONS):
        preds[:, i] = (probs[:, i] >= threshold[emotion]).astype(int)
    return preds


def collect_validation_prob(model, emotion_loader, device):
    all_probs = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch in emotion_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            emo_logits = model(input_ids, task="emotion")
            probs = torch.sigmoid(emo_logits)

            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.cpu())

    return (
        torch.cat(all_targets).numpy(),
        torch.cat(all_probs).numpy()
    )


def main():
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

    probs_targets = evaluate(model,
                             val_sentiment_loader,
                             val_emotion_loader,
                             device)

    emotion_probs = probs_targets["emotion_probability"]
    emotion_target = probs_targets["emotion_targets"]

    sentiment_probs = probs_targets["sentiment_probability"]
    sentiment_target = probs_targets["sentiment_targets"]

    emotion_preds = apply_thresholds(emotion_probs,
                                     EMOTION_THRESHOLD)
    sentiment_preds = (sentiment_probs >= 0.5).astype(int)

    emotion_metrics, sentiment_metrics, per_emotion_metrics = metrics(
        sentiment_target, sentiment_preds, emotion_target, emotion_preds
    )

    print("Validation Sentiment: ", sentiment_metrics)
    print("Validation emotion: ", emotion_metrics)
    print("Validation per_emotion: ", per_emotion_metrics)

    # y_true, y_prob = collect_validation_prob(model,
    #                                          val_emotion_loader, device)
    # best_t, best_f1 = tune_thresholds(y_prob, y_true)
    # print(best_t, best_f1)


if __name__ == "__main__":
    main()
