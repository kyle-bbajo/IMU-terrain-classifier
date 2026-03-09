"""src/losses.py — 손실 함수 모음."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss + Label Smoothing.

    val_loss가 노이즈하게 나오는 원인이므로 early stopping은
    반드시 val_acc 기준으로 사용할 것.
    """
    def __init__(self, gamma: float = 2.0, label_smooth: float = 0.1) -> None:
        super().__init__()
        self.gamma  = gamma
        self.smooth = label_smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)
        with torch.no_grad():
            td = torch.zeros_like(logits).fill_(self.smooth / max(n_cls - 1, 1))
            td.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth)
        logp = F.log_softmax(logits, dim=1)
        p    = torch.exp(logp)
        loss = -(td * (1.0 - p) ** self.gamma * logp).sum(dim=1)
        return loss.mean()


class LabelSmoothCE(nn.Module):
    def __init__(self, smooth: float = 0.1) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.smooth)


def build_loss(cfg) -> nn.Module:
    if cfg.use_focal:
        return FocalLoss(gamma=cfg.focal_gamma, label_smooth=cfg.label_smooth)
    return LabelSmoothCE(smooth=cfg.label_smooth)