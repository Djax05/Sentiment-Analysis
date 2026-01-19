import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from utils import ALLOWED_EMOTIONS
from torch.utils.data import DataLoader
from models.train import (
    collate_fn,
    BATCH_SIZE,
    PROCESSED_DATA,
    )
from datasets.emotion_dataset import EmotionDataset
from .tokenizer import load_vocab
from scripts.data_preprocessing import clean_text
from .training_utils import BASE_DIR
from .models import EmotionsSentimentModel
from .train import (
    get_device,
)
from functools import partial


CHECKPOINT = BASE_DIR / "checkpoints"


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


emotion_val = pd.read_csv(PROCESSED_DATA / "goemotions_val.csv")

print(emotion_val.shape)

emotion_val["text"] = (
    emotion_val["text"]
    .fillna("")
    .astype(str)
    .apply(clean_text)
)

emotion_text = emotion_val["text"].values.tolist()
emotion_labels = emotion_val[ALLOWED_EMOTIONS].to_numpy().tolist()

emotion_dataset = EmotionDataset(
    texts=emotion_text,
    labels=emotion_labels
)

vocab = load_vocab()
collate = partial(collate_fn, vocab=vocab)

emotion_loader = DataLoader(
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

y_true, y_prob = collect_validation_prob(model, emotion_loader, device)
best_t, best_f1 = tune_thresholds(y_prob, y_true)
print(best_t, best_f1)
