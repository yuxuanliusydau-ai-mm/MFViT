from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute PSNR for tensors in [0, data_range].
    pred, target: (B,C,H,W)
    """
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1)  # per-image
    return 10.0 * torch.log10((data_range ** 2) / (mse + eps))


class PSNRLoss(torch.nn.Module):
    """
    Minimizing this is equivalent to maximizing PSNR.
    Uses: 10*log10(MSE + eps). (constant offset ignored)
    """
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target, reduction="mean")
        return 10.0 * torch.log10(mse + self.eps)
