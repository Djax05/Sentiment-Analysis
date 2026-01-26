import torch
import pandas as pd
from .evaluate import evaluate
from .models import EmotionsSentimentModel
from .tokenizer import load_vocab
from .validation import apply_thresholds
from .train import (
    get_device,
    load_dataset,
    metrics
)
from utils import ALLOWED_EMOTIONS
from config import EMOTION_THRESHOLD, PROCESSED_DATA


def import_data(PROCESSED_DATA):

    sentiment_df = pd.read_csv(PROCESSED_DATA / "sentiment_test.csv")
    goemotions_df = pd.read_csv(PROCESSED_DATA / "goemotions_test.csv")

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


def main():

    (sentiment_text,
     sentiment_labels,
     goemotions_text,
     goemotions_labels) = import_data(PROCESSED_DATA)

    vocab = load_vocab()

    (test_sentiment_loader,
     test_emotion_loader) = load_dataset(sentiment_text,
                                         sentiment_labels,
                                         goemotions_text,
                                         goemotions_labels,
                                         vocab)

    model = EmotionsSentimentModel(vocab_size=len(vocab))

    model.load_state_dict(torch.load("ml/checkpoints/last_model.pt"))

    device = get_device()

    model.to(device)
    model.eval()

    probs_targets = evaluate(model,
                             test_sentiment_loader,
                             test_emotion_loader,
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


if __name__ == "__main__":
    main()
