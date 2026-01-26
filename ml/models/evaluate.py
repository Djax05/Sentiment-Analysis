import torch


def evaluate(model, sentiment_dataloader, emotion_dataloader, device):
    model.eval()

    all_sent_probs = []
    all_sent_targets = []

    all_emotion_probs = []
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

            emo_probs = torch.sigmoid(emo_logits)

            all_sent_probs.append(sent_probs.cpu())
            all_sent_targets.append(sent_targets.cpu())

            all_emotion_probs.append(emo_probs.detach().cpu())
            all_emotions_targets.append(emo_targets.cpu())

    all_sent_probs = torch.cat(all_sent_probs).numpy()
    all_sent_targets = torch.cat(all_sent_targets).numpy()

    all_emotions_probs = torch.cat(all_emotion_probs).numpy()
    all_emotions_targets = torch.cat(all_emotions_targets).numpy()

    return {
        "sentiment_probability": all_sent_probs,
        "sentiment_targets": all_sent_targets,
        "emotion_probability": all_emotions_probs,
        "emotion_targets": all_emotions_targets,
    }
