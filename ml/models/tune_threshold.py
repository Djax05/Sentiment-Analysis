import torch
import numpy as np
import json

from utils import ALLOWED_EMOTIONS

from .training_utils import load_validation_setup
from sklearn.metrics import f1_score


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
            print(f"emotion {i} | t={t:.2f} | f1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_threshold[i] = best_t
        best_f1s[i] = best_f1

    return best_threshold, best_f1s


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
    (model, _,
     val_emotion_loader, device) = load_validation_setup()
    y_true, y_prob = collect_validation_prob(model,
                                             val_emotion_loader, device)

    best_t, _ = tune_thresholds(y_prob, y_true)

    named = {e: round(float(best_t[i]), 2) for i, e in enumerate(ALLOWED_EMOTIONS)}
    with open("ml/artifacts/thresholds.json", "w") as f:
        json.dump(named, f, indent=2)


if __name__ == "__main__":

    main()
