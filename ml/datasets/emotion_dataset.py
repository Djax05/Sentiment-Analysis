import torch
from torch.utils.data import Dataset
from typing import List


class EmotionDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[int]]):

        self.texts: List[str] = texts
        self.labels: List[List[int]] = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }
