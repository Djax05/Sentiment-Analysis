import torch
import json

from .loader import load_artifacts
from ml.models.tokenizer import encode_text
from app.config import ALLOWED_EMOTIONS
from scripts.data_preprocessing import clean_text

with open("ml/artifacts/thresholds.json") as f:
    EMOTION_THRESHOLD = json.load(f)


def predict_text(text: str):
    model, vocab = load_artifacts()
    text = clean_text(text)
    encoded = encode_text(text, vocab)
    input_tensor = torch.tensor(encoded).unsqueeze(0)

    with torch.no_grad():
        sentiment_logits, emotion_logits = model(input_tensor)

    return sentiment_logits, emotion_logits


def postprocess_outputs(text: str, sent_threshold=0.5):
    sentiment_logits, emotion_logits = predict_text(text)

    sentiment_prob = torch.sigmoid(sentiment_logits).item()
    emotion_probs = torch.sigmoid(emotion_logits).squeeze(0).tolist()

    sentiment_label = ("positive" if sentiment_prob > sent_threshold
                       else "negative")

    emotions = {}
    active_emotions = []
    for i, prob in enumerate(emotion_probs):
        emotion = ALLOWED_EMOTIONS[i]
        threshold = EMOTION_THRESHOLD.get(emotion, 0.5)

        emotions[emotion] = prob

        if prob > threshold:
            active_emotions.append(emotion)

    return {
        "sentiment": {
            "probability": sentiment_prob,
            "label": sentiment_label
        },
        "emotions": emotions,
        "active_emotions": active_emotions
    }

