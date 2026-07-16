import torch
import json
import numpy as np
from utils import ALLOWED_EMOTIONS
from .training_utils import load_validation_setup
from .evaluate import evaluate
from .train import metrics


def apply_thresholds(probs, threshold):
    preds = np.zeros_like(probs)

    for i, emotion in enumerate(ALLOWED_EMOTIONS):
        preds[:, i] = (probs[:, i] >= threshold[emotion]).astype(int)
    return preds


def main():
    model, val_sentiment_loader, val_emotion_loader, device = load_validation_setup()

    with open("ml/artifacts/thresholds.json") as f:
        EMOTION_THRESHOLD = json.load(f)

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


if __name__ == "__main__":
    main()
