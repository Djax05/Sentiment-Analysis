import torch.nn as nn
from .encoder import MeanPoolingEncoder


class EmotionsSentimentModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            num_emotions: int = 6
    ):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.encoder = MeanPoolingEncoder()

        self.sentiment_head = nn.Linear(embedding_dim, 1)
        self.emotion_head = nn.Linear(embedding_dim, num_emotions)

    def forward(self, input_ids, task=str):
        attention_mask = (input_ids != 0).float()

        embedded = self.embeddings(input_ids)

        sentence_rep = self.encoder(embedded, attention_mask)

        if task == "sentiment":
            return self.sentiment_head(sentence_rep)

        if task == "emotion":
            return self.emotion_head(sentence_rep)

        else:
            raise ValueError(f"Unknown task: {task}")
