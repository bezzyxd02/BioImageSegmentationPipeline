from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .losses import bce_dice_loss, dice_score, iou_score


def train_epoch(model, loader: DataLoader, optimizer, scaler, device: str, amp: bool, grad_clip: float):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device if device == "cuda" else "cpu", enabled=amp):
            logits = model(x)
            loss = bce_dice_loss(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.item())
    return total / max(1, len(loader))


def validate_epoch(model, loader: DataLoader, device: str, amp: bool):
    model.eval()
    total, dsc, iou = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device if device == "cuda" else "cpu", enabled=amp):
                logits = model(x)
                loss = bce_dice_loss(logits, y)
            total += float(loss.item())
            dsc += dice_score(logits, y)
            iou += iou_score(logits, y)
    n = max(1, len(loader))
    return total / n, dsc / n, iou / n


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_score: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch, "best": best_score}, path)


def load_checkpoint(path: Path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    return ckpt.get("epoch", 0), ckpt.get("best", -1.0)


def fit(model, train_loader, val_loader, out_dir: Path, epochs: int, lr: float, wd: float, patience: int, device: str, amp: bool, grad_clip: float):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", patience=3, factor=0.5)
    scaler = torch.amp.GradScaler(device=device, enabled=amp and device == "cuda")
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    best, wait = -1.0, 0
    for epoch in range(1, epochs + 1):
        tl = train_epoch(model, train_loader, optim, scaler, device, amp, grad_clip)
        vl, vd, vi = validate_epoch(model, val_loader, device, amp)
        writer.add_scalar("loss/train", tl, epoch)
        writer.add_scalar("loss/val", vl, epoch)
        writer.add_scalar("metric/dice", vd, epoch)
        writer.add_scalar("metric/iou", vi, epoch)
        scheduler.step(vd)
        save_checkpoint(out_dir / "checkpoints" / "last.pt", model, optim, epoch, best)
        if vd > best:
            best, wait = vd, 0
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optim, epoch, best)
        else:
            wait += 1
            if wait >= patience:
                break
    writer.close()
