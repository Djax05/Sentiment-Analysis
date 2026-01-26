import torch.nn as nn


class MeanPoolingEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, mask=None):

        if mask is None:
            return embeddings.mean(dim=1)

        mask = mask.unsqueeze(-1)
        masked_embeddings = embeddings * mask

        summed = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts
