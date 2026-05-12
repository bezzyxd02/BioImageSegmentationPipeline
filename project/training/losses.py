from __future__ import annotations
import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * bce + 0.5 * dice_loss(logits, target)


def dice_score(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return float((2 * inter + eps) / (union + eps))


def iou_score(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum().item()
    union = (pred + target - pred * target).sum().item()
    return float((inter + eps) / (union + eps))
