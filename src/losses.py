# src/losses.py — Focal Loss + Label Smoothing CE

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    gamma=0 reduces to standard cross-entropy.
    alpha: per-class weight tensor (inverse-frequency weighting recommended).
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.1,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha          # (C,) tensor or None
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C)  targets: (B,)
        ce_loss  = F.cross_entropy(
            logits, targets,
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt       = torch.exp(-ce_loss)
        focal    = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def compute_class_weights(samples, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights from sample list [(path, label_idx)]."""
    counts = torch.zeros(num_classes)
    for _, label in samples:
        counts[label] += 1
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * num_classes   # normalise
    return weights
