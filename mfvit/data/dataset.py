from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset

from mfvit.utils.image_io import load_image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairedFolderConfig:
    root: str | Path
    split: str = "train"  # train/val/test
    snow_dirname: str = "snow"
    clean_dirname: str = "clean"


class PairedImageFolder(Dataset):
    """
    Paired dataset: {root}/{split}/snow/*.png and {root}/{split}/clean/*.png
    Filenames must match exactly.
    """
    def __init__(self, cfg: PairedFolderConfig, transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.root = Path(cfg.root)
        self.split = cfg.split
        self.snow_dir = self.root / cfg.split / cfg.snow_dirname
        self.clean_dir = self.root / cfg.split / cfg.clean_dirname
        self.transform = transform

        if not self.snow_dir.exists():
            raise FileNotFoundError(f"Snow directory not found: {self.snow_dir}")
        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean directory not found: {self.clean_dir}")

        self.items = self._index_pairs()

    def _index_pairs(self) -> List[Tuple[Path, Path]]:
        snow_files = sorted([p for p in self.snow_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
        pairs: List[Tuple[Path, Path]] = []
        for sp in snow_files:
            cp = self.clean_dir / sp.relative_to(self.snow_dir)
            if cp.exists():
                pairs.append((sp, cp))
        if not pairs:
            raise RuntimeError(
                f"No paired images found. Ensure filenames match between {self.snow_dir} and {self.clean_dir}."
            )
        return pairs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        snow_path, clean_path = self.items[idx]
        snow = load_image(snow_path)
        clean = load_image(clean_path)
        if self.transform is not None:
            snow, clean = self.transform(snow, clean)
        return snow, clean
