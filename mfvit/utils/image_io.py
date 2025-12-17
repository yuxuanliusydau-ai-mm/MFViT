from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch


def load_image(path: str | Path) -> torch.Tensor:
    """
    Load an image as float tensor in [0,1] with shape (C,H,W).
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    # HWC -> CHW
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t


def save_image(t: torch.Tensor, path: str | Path) -> None:
    """
    Save (C,H,W) float tensor in [0,1] to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = t.detach().clamp(0.0, 1.0).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(str(path))
