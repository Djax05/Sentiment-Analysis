import torch
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)


def evaluate(model, sentiment_dataloader, emotion_dataloader, device):
    model.eval()

    all_sent_preds = []
    all_sent_targets = []

    all_emotion_preds = []
    all_emotions_targets = []

    with torch.no_grad():
        for (sent_batch, emo_batch) in zip(sentiment_dataloader,
                                           emotion_dataloader):
            sent_input_ids = sent_batch["input_ids"].to(device)
            sent_targets = sent_batch["labels"]
            sent_targets = sent_targets.unsqueeze(1)

            emo_input_ids = emo_batch["input_ids"].to(device)
            emo_targets = emo_batch["labels"]

            sent_logits = model(sent_input_ids, task="sentiment")
            emo_logits = model(emo_input_ids, task="emotion")

            sent_probs = torch.sigmoid(sent_logits)
            sent_preds = (sent_probs > 0.5).long()

            emo_probs = torch.sigmoid(emo_logits)
            emo_preds = (emo_probs > 0.5).long()

            all_sent_preds.append(sent_preds.cpu())
            all_sent_targets.append(sent_targets.cpu())

            all_emotion_preds.append(emo_preds.cpu())
            all_emotions_targets.append(emo_targets.cpu())

    all_sent_preds = torch.cat(all_sent_preds).numpy()
    all_sent_targets = torch.cat(all_sent_targets).numpy()

    all_emotion_preds = torch.cat(all_emotion_preds).numpy()
    all_emotions_targets = torch.cat(all_emotions_targets).numpy()

    sentiment_metrics = {
        "accuracy": accuracy_score(all_sent_targets, all_sent_preds),
        "precision": precision_score(all_sent_targets,
                                     all_sent_preds, zero_division=0),
        "recall": recall_score(all_sent_targets, all_sent_preds,
                               zero_division=0),
        "f1": f1_score(all_sent_targets, all_sent_preds,
                       zero_division=0)
    }

    emotion_metric = {
        "micro_f1": f1_score(all_emotions_targets, all_emotion_preds,
                             average="micro", zero_division=0),
        "macro_f1": f1_score(all_emotions_targets, all_emotion_preds,
                             average="macro", zero_division=0)
    }
    return {
        "sentiment": sentiment_metrics,
        "emotion": emotion_metric
    }
