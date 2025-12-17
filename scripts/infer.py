from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from mfvit.models.mf_vit import MFViT, MFViTConfig
from mfvit.utils.checkpoint import load_checkpoint
from mfvit.utils.image_io import load_image, save_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Inference with MF-ViT")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt).")
    p.add_argument("--input", type=str, required=True, help="Input image path or folder.")
    p.add_argument("--output", type=str, required=True, help="Output image path or folder.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resize-to", type=int, default=512)
    return p.parse_args()


def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
    # x: (1,C,H,W)
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def _unpad(x: torch.Tensor, pad: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad
    if pad_h == 0 and pad_w == 0:
        return x
    return x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise KeyError("Checkpoint does not contain cfg. Train with this repo to get cfg saved.")

    cfg = MFViTConfig(**cfg_dict)
    model = MFViT(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    in_path = Path(args.input)
    out_path = Path(args.output)

    def run_one(img_path: Path, save_path: Path) -> None:
        x = load_image(img_path).unsqueeze(0).to(device)  # 1,C,H,W

        x = F.interpolate(
            x,
            size=(args.resize_to, args.resize_to),
            mode="bilinear",
            align_corners=False
        )

        x, pad = _pad_to_multiple(x, multiple=8)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                y = model(x)
        y = _unpad(y, pad)

        save_image(y.squeeze(0), save_path)

    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        for p in sorted(in_path.rglob("*")):
            if p.suffix.lower() in exts:
                rel = p.relative_to(in_path)
                run_one(p, out_path / rel)
    else:
        if out_path.is_dir():
            out_path = out_path / in_path.name
        run_one(in_path, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
