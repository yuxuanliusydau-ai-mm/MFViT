from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from mfvit.models.layers import (
    DropPath,
    FractionalOpConfig,
    FractionalOrderDifferential,
    MultiHeadSelfAttention,
    MLP,
)


BlockStyle = Literal["paper", "vit"]


@dataclass(frozen=True)
class MFViTConfig:
    """
    Configuration for MF-ViT.

    Parameters derived from the paper:
    - base_channels C=16
    - downsample factor is 8 (H/8×W/8 tokens)
    - blocks and embedding dims depend on variant (S/M/L)
    - alpha_attn=1.2, alpha_ffn=0.8
    - MLP hidden dim is 2D (mlp_ratio=2)
    """
    variant: Literal["mfvit-s", "mfvit-m", "mfvit-l"] = "mfvit-m"
    in_chans: int = 3
    base_channels: int = 16
    embed_dim: int = 512
    num_heads: int = 4
    depth: int = 10
    mlp_ratio: float = 2.0

    alpha_attn: float = 1.2
    alpha_ffn: float = 0.8

    # regularization
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0

    # pool type in token generator
    pool: Literal["avg", "max"] = "avg"

    # how to wire residuals inside fractal blocks
    block_style: BlockStyle = "vit"

    # output behavior
    clamp_output: bool = True

    @staticmethod
    def from_variant(variant: str, **overrides) -> "MFViTConfig":
        v = variant.lower()
        if v in {"mfvit-s", "s", "mf-vit-s", "mf_vit_s"}:
            cfg = MFViTConfig(variant="mfvit-s", embed_dim=512, num_heads=4, depth=8)
        elif v in {"mfvit-m", "m", "mf-vit-m", "mf_vit_m"}:
            cfg = MFViTConfig(variant="mfvit-m", embed_dim=512, num_heads=4, depth=10)
        elif v in {"mfvit-l", "l", "mf-vit-l", "mf_vit_l"}:
            cfg = MFViTConfig(variant="mfvit-l", embed_dim=1024, num_heads=8, depth=12)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        return dataclass_replace(cfg, **overrides)


def dataclass_replace(cfg: MFViTConfig, **kwargs) -> MFViTConfig:
    data = cfg.__dict__.copy()
    data.update(kwargs)
    return MFViTConfig(**data)


class TokenGenerationBlock(nn.Module):
    """
    Token generation as described in Fig.3 of the paper:

    Input:  B×H×W×3
      Conv5×5 -> Pool -> B×H/2×W/2×C
      Conv3×3 -> Pool -> B×H/4×W/4×4C
      Conv3×3 -> Pool -> B×H/8×W/8×8C
      Flatten -> Linear embedding -> B×N×D (N = H/8 * W/8)

    See paper details for C=16 and N definition.
    """
    def __init__(self, in_chans: int, base_channels: int, embed_dim: int, pool: str = "avg") -> None:
        super().__init__()
        c = base_channels
        self.conv1 = nn.Conv2d(in_chans, c, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(c, 4 * c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4 * c, 8 * c, kernel_size=3, padding=1)
        self.act = nn.GELU()

        if pool == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unknown pool: {pool}")

        self.embed = nn.Linear(8 * c, embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        b, c_in, h, w = x.shape
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(f"Input H,W must be divisible by 8. Got H={h}, W={w}.")

        x = self.act(self.conv1(x))
        x = self.pool(x)  # H/2, W/2, C

        x = self.act(self.conv2(x))
        x = self.pool(x)  # H/4, W/4, 4C

        x = self.act(self.conv3(x))
        x = self.pool(x)  # H/8, W/8, 8C

        _, c_feat, h8, w8 = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h8 * w8, c_feat)  # B,N,8C
        x = self.embed(x)  # B,N,D
        return x, (h8, w8)


class ImageReconstructionBlock(nn.Module):
    """
    Image reconstruction as described in Fig.4 of the paper (reverse of token generation):

    Input: B×N×D
      Linear -> B×N×8C
      Reshape -> B×8C×H/8×W/8
      Conv3×3 -> 4C, Bilinear upsample -> H/4×W/4
      Conv3×3 -> C,  Bilinear upsample -> H/2×W/2
      Conv5×5 -> 3,  Bilinear upsample -> H×W
    """
    def __init__(self, embed_dim: int, base_channels: int, out_chans: int = 3, clamp_output: bool = True) -> None:
        super().__init__()
        c = base_channels
        self.proj = nn.Linear(embed_dim, 8 * c)
        self.conv1 = nn.Conv2d(8 * c, 4 * c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4 * c, c, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c, out_chans, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.clamp_output = bool(clamp_output)

    def forward(self, tokens: torch.Tensor, hw8: Tuple[int, int], out_hw: Tuple[int, int]) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected (B,N,D), got {tuple(tokens.shape)}")

        b, n, d = tokens.shape
        h8, w8 = hw8
        h, w = out_hw
        if n != h8 * w8:
            raise ValueError(f"N must equal h8*w8. Got N={n}, h8*w8={h8*w8}")

        x = self.proj(tokens)  # B,N,8C
        x = x.view(b, h8, w8, -1).permute(0, 3, 1, 2).contiguous()  # B,8C,h8,w8

        x = self.act(self.conv1(x))  # B,4C,h8,w8
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H/4

        x = self.act(self.conv2(x))  # B,C,H/4,W/4
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H/2

        x = self.conv3(x)  # B,3,H/2,W/2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # H

        if self.clamp_output:
            x = x.clamp(0.0, 1.0)
        return x


class GlobalFractalFeatureBlock(nn.Module):
    """
    Global fractal feature processing block.

    Two supported styles:
    - block_style="vit": standard ViT pre-norm with two residuals
    - block_style="paper": matches the residual wiring shown in Fig.5 (single residual at end)

    Fractional operators:
    - alpha_attn before attention
    - alpha_ffn before feedforward (after layer norm in vit style)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        alpha_attn: float,
        alpha_ffn: float,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        block_style: BlockStyle = "vit",
    ) -> None:
        super().__init__()
        self.block_style: BlockStyle = block_style

        self.frac_attn = FractionalOrderDifferential(FractionalOpConfig(alpha=alpha_attn))
        self.frac_ffn = FractionalOrderDifferential(FractionalOpConfig(alpha=alpha_ffn))

        self.attn = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_dim=hidden_dim, drop=drop)

        # norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_style == "paper":
            # Matches Fig.5 wiring:
            # x_att = Attn(FracOp(x))
            # z = LN(x_att)
            # z = MLP(FracOp(z))
            # z = LN(z)
            # out = x + z
            x_res = x
            x_att = self.attn(self.frac_attn(x))
            z = self.norm1(x_att)
            z = self.mlp(self.frac_ffn(z))
            z = self.norm2(z)
            out = x_res + self.drop_path(z)
            return out

        # Standard ViT pre-norm with two residuals:
        y = self.norm1(x)
        y = self.frac_attn(y)
        y = self.attn(y)
        x = x + self.drop_path(y)

        y = self.norm2(x)
        y = self.frac_ffn(y)
        y = self.mlp(y)
        x = x + self.drop_path(y)
        return x


class MFViT(nn.Module):
    """
    MF-ViT model:
      TokenGeneration -> [GlobalFractalFeatureBlock]×depth -> ImageReconstruction
    """
    def __init__(self, cfg: MFViTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tokenizer = TokenGenerationBlock(
            in_chans=cfg.in_chans,
            base_channels=cfg.base_channels,
            embed_dim=cfg.embed_dim,
            pool=cfg.pool,
        )

        # stochastic depth schedule
        dpr = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()

        self.blocks = nn.ModuleList([
            GlobalFractalFeatureBlock(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                alpha_attn=cfg.alpha_attn,
                alpha_ffn=cfg.alpha_ffn,
                drop=cfg.drop,
                attn_drop=cfg.attn_drop,
                drop_path=dpr[i],
                block_style=cfg.block_style,
            )
            for i in range(cfg.depth)
        ])

        self.reconstructor = ImageReconstructionBlock(
            embed_dim=cfg.embed_dim,
            base_channels=cfg.base_channels,
            out_chans=cfg.in_chans,
            clamp_output=cfg.clamp_output,
        )

    @torch.no_grad()
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens, hw8 = self.tokenizer(x)
        for blk in self.blocks:
            tokens = blk(tokens)
        out = self.reconstructor(tokens, hw8=hw8, out_hw=(h, w))
        return out
