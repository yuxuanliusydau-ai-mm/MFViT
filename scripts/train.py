from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mfvit.data.dataset import PairedFolderConfig, PairedImageFolder
from mfvit.data.transforms import PairTransformConfig, PairRandomCropFlip
from mfvit.models.mf_vit import MFViT, MFViTConfig
from mfvit.utils.seed import SeedConfig, seed_everything
from mfvit.utils.metrics import psnr, PSNRLoss
from mfvit.utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train MF-ViT on paired desnowing data (engineering implementation).")
    p.add_argument("--data-root", type=str, required=True, help="Dataset root (see README layout).")
    p.add_argument("--variant", type=str, default="mfvit-m", help="mfvit-s | mfvit-m | mfvit-l")
    p.add_argument("--config", type=str, default="", help="Optional JSON config (overrides defaults).")

    p.add_argument("--resize-to", type=int, default=512)
    p.add_argument("--crop-size", type=int, default=224) 

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.01)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-dir", type=str, default="runs/mfvit")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")
    p.add_argument("--save-every", type=int, default=1)

    return p.parse_args()


def build_cfg(variant: str, config_path: str) -> MFViTConfig:
    cfg = MFViTConfig.from_variant(variant)
    if config_path:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        # allow either specifying "variant" or not
        if "variant" in data:
            cfg = MFViTConfig.from_variant(data["variant"])
            del data["variant"]
        cfg = MFViTConfig(**{**cfg.__dict__, **data})
    return cfg


def evaluate(model: MFViT, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for snow, clean in loader:
            snow = snow.to(device)
            clean = clean.to(device)
            pred = model(snow)
            scores.append(psnr(pred, clean).mean().item())
    return float(sum(scores) / max(len(scores), 1))


def main() -> None:
    args = parse_args()
    seed_everything(SeedConfig(seed=args.seed, deterministic=args.deterministic))

    device = torch.device(args.device)

    cfg = build_cfg(args.variant, args.config)
    model = MFViT(cfg).to(device)

    print(f"Model: {cfg.variant} | params={model.count_params():,} | style={cfg.block_style}")

    # data
    train_tf = PairRandomCropFlip(
        PairTransformConfig(resize_to=args.resize_to, crop_size=args.crop_size, random_flip=True)
    )

    val_tf = None

    train_ds = PairedImageFolder(PairedFolderConfig(root=args.data_root, split="train"), transform=train_tf)
    val_ds = PairedImageFolder(PairedFolderConfig(root=args.data_root, split="val"), transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # optimizer / loss
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    criterion = PSNRLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 0
    best_psnr = -1e9

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        optim.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt and args.amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_psnr = float(ckpt.get("best_psnr", best_psnr))
        print(f"Resumed from {args.resume} at epoch={start_epoch}, best_psnr={best_psnr:.3f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs-1}")
        for step, (snow, clean) in pbar:
            snow = snow.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(snow)
                loss = criterion(pred, clean)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if step % args.log_every == 0:
                with torch.no_grad():
                    cur_psnr = psnr(pred.detach(), clean).mean().item()
                pbar.set_postfix(loss=float(loss.item()), psnr=float(cur_psnr))

        val_psnr = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: val_psnr={val_psnr:.3f}")

        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        if (epoch + 1) % args.save_every == 0 or is_best:
            ckpt_path = save_dir / ("best.pt" if is_best else f"epoch_{epoch:04d}.pt")
            save_checkpoint(
                ckpt_path,
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict() if args.amp else None,
                    "cfg": cfg.__dict__,
                    "best_psnr": best_psnr,
                },
            )

    print(f"Done. Best val PSNR: {best_psnr:.3f}")


if __name__ == "__main__":
    main()
