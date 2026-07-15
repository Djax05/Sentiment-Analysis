import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=torch.tensor([1.85, 2.18, 1.86, 4.63, 2.63, 0.26]), gamma=2.0, reduction=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
            )

        probs = torch.sigmoid(logits)

        p_t = probs * targets + (1 - probs) * (1 - targets)

        focal_factor = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
