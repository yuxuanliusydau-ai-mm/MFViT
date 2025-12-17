from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PairTransformConfig:
    resize_to: int = 512
    crop_size: int = 224
    random_flip: bool = True



class PairRandomCropFlip:
    """
    Simple paired augmentation: random crop to fixed size and random horizontal flip.
    Expects inputs in (C,H,W), float in [0,1].
    """
    def __init__(self, cfg: PairTransformConfig) -> None:
        self.cfg = cfg

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target = self.cfg.resize_to
        a = F.interpolate(a.unsqueeze(0), size=(target, target), mode="bilinear", align_corners=False).squeeze(0)
        b = F.interpolate(b.unsqueeze(0), size=(target, target), mode="bilinear", align_corners=False).squeeze(0)
        assert a.shape == b.shape, "Paired transforms require same shape."
        c, h, w = a.shape
        th = tw = self.cfg.crop_size

        if h < th or w < tw:
            # resize up if needed (keeps pair aligned)
            a = F.interpolate(a.unsqueeze(0), size=(max(h, th), max(w, tw)), mode="bilinear", align_corners=False).squeeze(0)
            b = F.interpolate(b.unsqueeze(0), size=(max(h, th), max(w, tw)), mode="bilinear", align_corners=False).squeeze(0)
            c, h, w = a.shape

        i = torch.randint(0, h - th + 1, (1,)).item()
        j = torch.randint(0, w - tw + 1, (1,)).item()

        a = a[:, i:i+th, j:j+tw]
        b = b[:, i:i+th, j:j+tw]

        if self.cfg.random_flip and torch.rand(1).item() < 0.5:
            a = torch.flip(a, dims=[2])
            b = torch.flip(b, dims=[2])

        return a, b
