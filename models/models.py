import torch.nn as nn


class EmotionsSentimentModel(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            num_emotions: int = 6
    ):
        super().__init__()

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.sentiment_head = nn.Linear(hidden_dim, 1)
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)

    def forward(self, input_ids):

        embedded = self.embeddings(input_ids)

        _, (hidden, _) = self.encoder(embedded)

        sentence_rep = hidden[-1]

        sentiment_logits = self.sentiment_head(sentence_rep)
        emotion_logits = self.emotion_head(sentence_rep)

        return sentiment_logits, emotion_logits
