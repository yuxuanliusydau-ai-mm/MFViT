from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AverageMeter:
    name: str
    fmt: str = ":.4f"
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:
        return f"{self.name} {self.val{self.fmt}} (avg: {self.avg{self.fmt}})"
