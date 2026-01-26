import torch
from torch.utils.data import Dataset
from typing import List


class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: torch.Tensor):

        self.texts: List[str] = texts
        self.labels: torch.Tensor = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": self.labels[idx].float()
        }
