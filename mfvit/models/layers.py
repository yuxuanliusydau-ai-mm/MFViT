from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """
    Stochastic depth (a.k.a DropPath).
    """
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast across non-batch dims
        rnd = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = (rnd < keep_prob).to(x.dtype)
        return x * mask / keep_prob


@dataclass(frozen=True)
class FractionalOpConfig:
    alpha: float
    eps: float = 1e-6
    use_angular_frequency: bool = True


class FractionalOrderDifferential(nn.Module):
    """
    Fractional-order differential operator implemented as a multiplier in the frequency domain.

    Given a sequence x[n] (along token axis), we compute:
      X(ω) = FFT(x)
      Y(ω) = |ω|^α * X(ω)
      y[n] = IFFT(Y)

    Paper note: up to a fixed 2π unit conversion, this corresponds to a Riesz-type fractional Laplacian
    (symmetric fractional derivative). We expose `use_angular_frequency` to toggle ω = 2π f.
    """
    def __init__(self, cfg: FractionalOpConfig) -> None:
        super().__init__()
        self.alpha = float(cfg.alpha)
        self.eps = float(cfg.eps)
        self.use_angular_frequency = bool(cfg.use_angular_frequency)

        # Cache weight for a given (N, device, dtype)
        self._cache: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _get_weight(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (n, device, dtype)
        if key in self._cache:
            return self._cache[key]

        freq = torch.fft.fftfreq(n, d=1.0, device=device)  # cycles/sample
        if self.use_angular_frequency:
            omega = 2.0 * math.pi * freq
        else:
            omega = freq

        w = (omega.abs() + self.eps) ** self.alpha
        # remove DC component to avoid bias amplification
        if n > 0:
            w[0] = 0.0
        w = w.to(dtype=torch.float32)  # multiplier used with complex FFT anyway

        self._cache[key] = w
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) real-valued
        returns: (B, N, D) real-valued
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B,N,D), got {tuple(x.shape)}")

        b, n, d = x.shape
        w = self._get_weight(n, x.device, x.dtype)  # (N,)
        # FFT along token axis
        x_fft = torch.fft.fft(x, dim=1)
        y_fft = x_fft * w.view(1, n, 1)
        y = torch.fft.ifft(y_fft, dim=1).real
        return y


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention over token dimension.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = float(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.proj_drop = float(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        # (3, B, heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, N, head_dim)

        # Prefer PyTorch 2.x fused attention when available
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
            )  # (B, heads, N, head_dim)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop, training=self.training)
            out = attn @ v

        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = F.dropout(out, p=self.proj_drop, training=self.training)
        return out


class MLP(nn.Module):
    """
    Feed-forward network used in Transformer blocks.
    """
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = float(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        return x
