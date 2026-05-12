from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class MDCKPatchDataset(Dataset):
    def __init__(self, frames: list[np.ndarray], masks: list[np.ndarray], patch_size: int = 256) -> None:
        self.frames = frames
        self.masks = masks
        self.patch_size = patch_size
        self.indexes: list[tuple[int, int, int]] = []
        for i, f in enumerate(frames):
            h, w = f.shape
            step = patch_size // 2
            for y in range(0, max(1, h - patch_size + 1), step):
                for x in range(0, max(1, w - patch_size + 1), step):
                    self.indexes.append((i, y, x))

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, idx: int):
        fi, y, x = self.indexes[idx]
        p = self.patch_size
        img = self.frames[fi][y:y+p, x:x+p]
        msk = self.masks[fi][y:y+p, x:x+p]
        if img.shape != (p, p):
            pad_h = p - img.shape[0]
            pad_w = p - img.shape[1]
            img = np.pad(img, ((0, pad_h), (0, pad_w)))
            msk = np.pad(msk, ((0, pad_h), (0, pad_w)))
        img_t = torch.from_numpy((img / 255.0).astype(np.float32)).unsqueeze(0)
        msk_t = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)
        return img_t, msk_t
