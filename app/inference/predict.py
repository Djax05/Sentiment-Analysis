import torch

from .loader import load_artifacts
from ml.preprocessing.text_encoder import encode_text


def predict_text(text: str):
    model, vocab = load_artifacts()

    encoded = encode_text(text, vocab)
    input_tensor = torch.tensor(encoded).unsqueeze(0)

    with torch.no_grad():
        sentiment_logits, emotion_logits = model(input_tensor)

    return sentiment_logits, emotion_logits
