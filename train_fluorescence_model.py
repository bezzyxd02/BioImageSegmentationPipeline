from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.ndimage import binary_fill_holes
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ---------------- CONFIG / LOGGING / SYSTEM ----------------
def load_config() -> dict:
    cfg_file = Path("project/configs/config.yaml")
    if cfg_file.exists():
        return yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    return {
        "paths": {"ilastik_probability_dir": "", "output_dir": "outputs"},
        "training": {"epochs": 30, "batch_size": 8, "num_workers": 2, "learning_rate": 3e-4, "weight_decay": 1e-5, "early_stopping_patience": 8, "grad_clip": 1.0, "mixed_precision": True, "patch_size": 256},
        "model": {"in_channels": 1, "out_channels": 1, "base_channels": 64, "dropout": 0.2, "compile": True},
        "cellpose": {"enabled": True, "model_type": "cyto", "diameter": 0, "channels": [0, 0]},
        "ilastik": {"enabled": True},
        "inference": {"half_precision": True},
        "tracking": {"max_distance": 40.0},
    }


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mdck_onefile")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = RotatingFileHandler(log_dir / "pipeline.log", maxBytes=2_000_000, backupCount=4, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def detect_backends() -> dict:
    opencv_cuda = False
    try:
        opencv_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        opencv_cuda = False
    return {"torch_cuda": torch.cuda.is_available(), "opencv_cuda": opencv_cuda, "device": "cuda" if torch.cuda.is_available() else "cpu"}


# ---------------- PREPROCESS ----------------
def percentile_normalize(img: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    lo, hi = np.percentile(img, [low, high])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    x = np.clip((img - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def hot_pixel_removal(img: np.ndarray) -> np.ndarray:
    med = cv2.medianBlur(img, 3)
    diff = cv2.absdiff(img, med)
    out = img.copy()
    out[diff > 35] = med[diff > 35]
    return out


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    img = percentile_normalize(frame.astype(np.float32))
    img = hot_pixel_removal(img)
    bg = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
    img = cv2.subtract(img, bg)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.fastNlMeansDenoising(img, None, 7, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_batch(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [preprocess_frame(f) for f in frames]


# ---------------- HYBRID LABELS ----------------
def generate_cellpose_labels(frame: np.ndarray, model_type: str = "cyto", diameter: float = 0, channels: list[int] | tuple[int, int] = (0, 0), use_gpu: bool = True):
    from cellpose import models
    # Compatibilidade entre versões do Cellpose:
    # - versões antigas: models.Cellpose(...)
    # - versões novas: models.CellposeModel(...)
    cp = None
    if hasattr(models, "Cellpose"):
        cp = models.Cellpose(gpu=use_gpu, model_type=model_type)
        masks, flows, styles, diams = cp.eval(
            frame,
            diameter=diameter if diameter > 0 else None,
            channels=list(channels),
        )
        return masks, flows, styles, diams
    if hasattr(models, "CellposeModel"):
        cp = models.CellposeModel(gpu=use_gpu, model_type=model_type)
        masks, flows, styles = cp.eval(
            frame,
            diameter=diameter if diameter > 0 else None,
            channels=list(channels),
        )
        diams = float(diameter) if diameter and diameter > 0 else 0.0
        return masks, flows, styles, diams
    raise RuntimeError("Versão do Cellpose não suportada: nem Cellpose nem CellposeModel disponíveis.")


def load_ilastik_probability(path: Path) -> np.ndarray:
    p = tiff.imread(path)
    if p.ndim > 2:
        p = p[..., 0]
    p = p.astype(np.float32)
    if p.max() > 1:
        p /= 255.0
    return np.clip(p, 0, 1)


def combine_labels(cellpose_mask: np.ndarray, ilastik_probability: np.ndarray) -> np.ndarray:
    cp = (cellpose_mask > 0).astype(np.float32)
    il = (ilastik_probability > 0.5).astype(np.float32)
    return ((0.7 * cp + 0.3 * il) >= 0.5).astype(np.uint8)


# ---------------- MODEL ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 64, dropout: float = 0.2):
        super().__init__()
        self.e1 = ConvBlock(in_channels, base, dropout)
        self.e2 = ConvBlock(base, base * 2, dropout)
        self.e3 = ConvBlock(base * 2, base * 4, dropout)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(base * 4, base * 8, dropout)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4, dropout)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2, dropout)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base, dropout)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


# ---------------- DATASET ----------------
class MDCKPatchDataset(Dataset):
    def __init__(self, frames: list[np.ndarray], masks: list[np.ndarray], patch_size: int = 256):
        self.frames = frames
        self.masks = masks
        self.patch = patch_size
        self.idxs = []
        step = patch_size // 2
        for i, fr in enumerate(frames):
            h, w = fr.shape
            for y in range(0, max(1, h - patch_size + 1), step):
                for x in range(0, max(1, w - patch_size + 1), step):
                    self.idxs.append((i, y, x))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx: int):
        i, y, x = self.idxs[idx]
        p = self.patch
        img = self.frames[i][y:y+p, x:x+p]
        msk = self.masks[i][y:y+p, x:x+p]
        if img.shape != (p, p):
            ph, pw = p - img.shape[0], p - img.shape[1]
            img = np.pad(img, ((0, ph), (0, pw)))
            msk = np.pad(msk, ((0, ph), (0, pw)))
        return torch.from_numpy((img / 255.0).astype(np.float32))[None], torch.from_numpy(msk.astype(np.float32))[None]


# ---------------- LOSSES/METRICS ----------------
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target) + 0.5 * dice_loss(logits, target)


def dice_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum().item()
    den = pred.sum().item() + target.sum().item()
    return float((2 * inter + 1e-6) / (den + 1e-6))


# ---------------- TRAIN ----------------
def save_checkpoint(path: Path, model, opt, epoch: int, best: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "epoch": epoch, "best": best}, path)


def load_checkpoint(path: Path, model):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])


def fit(model, train_loader, val_loader, out_dir: Path, cfg: dict, device: str):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)
    scaler = torch.amp.GradScaler(device=device, enabled=cfg["training"]["mixed_precision"] and device == "cuda")
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    best, wait = -1.0, 0
    for ep in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        for x, y in tqdm(train_loader, desc=f"train ep{ep}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device if device == "cuda" else "cpu", enabled=cfg["training"]["mixed_precision"]):
                lg = model(x)
                loss = bce_dice(lg, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
        model.eval()
        vd = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                lg = model(x)
                vd += dice_score(lg, y)
        vd /= max(1, len(val_loader))
        writer.add_scalar("val/dice", vd, ep)
        sched.step(vd)
        save_checkpoint(out_dir / "checkpoints" / "last.pt", model, opt, ep, best)
        if vd > best:
            best, wait = vd, 0
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, opt, ep, best)
        else:
            wait += 1
            if wait >= cfg["training"]["early_stopping_patience"]:
                break
    writer.close()


# ---------------- MORPH/TRACK/EXPORT ----------------
def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return binary_fill_holes(m).astype(np.uint8)


def morphometry(mask: np.ndarray, image: np.ndarray) -> list[dict]:
    out = []
    for r in regionprops(label(mask > 0), intensity_image=image):
        per = r.perimeter if r.perimeter > 0 else 1.0
        out.append({"area": float(r.area), "perimeter": float(r.perimeter), "circularity": float(4 * np.pi * r.area / (per * per)), "eccentricity": float(r.eccentricity), "solidity": float(r.solidity), "aspect_ratio": float((r.major_axis_length + 1e-6) / (r.minor_axis_length + 1e-6)), "centroid_x": float(r.centroid[1]), "centroid_y": float(r.centroid[0]), "mean_intensity": float(r.mean_intensity)})
    return out


def track_cells(masks: list[np.ndarray], max_distance: float = 40.0):
    tracks, next_id, prev, active = [], 1, [], {}
    for t, m in enumerate(masks):
        ids = [i for i in np.unique(m) if i != 0]
        cur = []
        for i in ids:
            ys, xs = np.where(m == i)
            if len(xs):
                cur.append((float(xs.mean()), float(ys.mean())))
        if not prev:
            for x, y in cur:
                active[next_id] = (x, y)
                tracks.append((t, next_id, x, y))
                next_id += 1
            prev = cur
            continue
        if not prev or not cur:
            prev = cur
            continue
        pxy = np.array(prev)
        cxy = np.array(cur)
        d = ((pxy[:, None] - cxy[None]) ** 2).sum(-1) ** 0.5
        ri, ci = linear_sum_assignment(d)
        used, new_active = set(), {}
        prev_ids = list(active.keys())
        for r, c in zip(ri, ci):
            if d[r, c] <= max_distance and r < len(prev_ids):
                tid = prev_ids[r]
                x, y = map(float, cxy[c])
                new_active[tid] = (x, y)
                tracks.append((t, tid, x, y))
                used.add(c)
        for c, (x, y) in enumerate(cur):
            if c not in used:
                new_active[next_id] = (x, y)
                tracks.append((t, next_id, x, y))
                next_id += 1
        active, prev = new_active, cur
    return tracks


def export_results(out_dir: Path, frames: list[np.ndarray], pred_masks: list[np.ndarray], morph_rows: list[dict], tracks: list[tuple]):
    out_dir.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(out_dir / "masks.tif", np.stack(pred_masks).astype(np.uint8))
    for i, (fr, mk) in enumerate(zip(frames, pred_masks)):
        rgb = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        rgb[mk > 0] = (0, 255, 0)
        cv2.imwrite(str(out_dir / f"overlay_{i:04d}.png"), rgb)
    pd.DataFrame(morph_rows).to_csv(out_dir / "morphometry.csv", index=False)
    pd.DataFrame([{"frame": f, "track_id": tid, "x": x, "y": y} for f, tid, x, y in tracks]).to_csv(out_dir / "tracks.csv", index=False)


# ---------------- ORCHESTRATION ----------------
def select_base_dir() -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    p = filedialog.askdirectory(title="Selecione pasta base")
    root.destroy()
    if not p:
        raise RuntimeError("Pasta não selecionada")
    return Path(p)


def run_pipeline(base_dir: Path, log_q: queue.Queue[str]):
    cfg = load_config()
    out_dir = base_dir / cfg["paths"]["output_dir"]
    logger = setup_logger(out_dir / "logs")
    back = detect_backends()
    log_q.put(f"Backends: {back}")

    movies = sorted(base_dir.glob("*.tif"))
    if not movies:
        raise RuntimeError("Nenhum TIFF encontrado")

    frames, masks = [], []
    for mv in movies:
        stack = tiff.imread(mv)
        if stack.ndim == 2:
            stack = stack[None]
        pre = preprocess_batch([s.astype(np.float32) for s in stack])
        for i, fr in enumerate(pre):
            cp_mask, *_ = generate_cellpose_labels(fr, model_type=cfg["cellpose"]["model_type"], diameter=cfg["cellpose"]["diameter"], channels=cfg["cellpose"]["channels"], use_gpu=back["torch_cuda"])
            il_path = Path(cfg["paths"]["ilastik_probability_dir"]) / f"{mv.stem}_f{i:04d}.tif"
            if cfg["ilastik"]["enabled"] and il_path.exists():
                mk = combine_labels(cp_mask, load_ilastik_probability(il_path))
            else:
                mk = (cp_mask > 0).astype(np.uint8)
            frames.append(fr)
            masks.append(mk)

    ds = MDCKPatchDataset(frames, masks, patch_size=cfg["training"]["patch_size"])
    nv = max(1, int(len(ds) * 0.2))
    nt = max(1, len(ds) - nv)
    train_ds, val_ds = random_split(ds, [nt, nv])
    device = "cuda" if back["torch_cuda"] else "cpu"
    loader_tr = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], pin_memory=back["torch_cuda"])
    loader_val = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"]["num_workers"], pin_memory=back["torch_cuda"])

    model = UNet(cfg["model"]["in_channels"], cfg["model"]["out_channels"], cfg["model"]["base_channels"], cfg["model"]["dropout"]).to(device)
    if cfg["model"]["compile"] and hasattr(torch, "compile"):
        model = torch.compile(model)
    model = model.to(memory_format=torch.channels_last)
    fit(model, loader_tr, loader_val, out_dir, cfg, device)
    load_checkpoint(out_dir / "checkpoints" / "best.pt", model)

    model.eval()
    pred_masks, morph_rows = [], []
    with torch.no_grad():
        for fr in frames:
            x = torch.from_numpy((fr / 255.0).astype(np.float32))[None, None].to(device)
            with torch.autocast(device_type=device if device == "cuda" else "cpu", enabled=cfg["inference"]["half_precision"] and device == "cuda"):
                pr = torch.sigmoid(model(x)).squeeze().float().cpu().numpy()
            mk = postprocess_mask((pr > 0.5).astype(np.uint8))
            pred_masks.append(mk)
            morph_rows.extend(morphometry(mk, fr))

    tracks = track_cells(pred_masks, max_distance=cfg["tracking"]["max_distance"])
    export_results(out_dir, frames, pred_masks, morph_rows, tracks)
    logger.info("Pipeline concluído")
    log_q.put("Pipeline concluído")


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MDCK One-File Biomedical Segmentation")
        self.log_q: queue.Queue[str] = queue.Queue()
        self.text = tk.Text(self.root, width=120, height=32)
        self.text.pack(fill="both", expand=True)
        tk.Button(self.root, text="Selecionar pasta e executar", command=self.start).pack()
        self.root.after(200, self.poll)

    def start(self):
        try:
            base = select_base_dir()
        except Exception as e:
            messagebox.showerror("Erro", str(e))
            return

        def worker():
            try:
                run_pipeline(base, self.log_q)
            except Exception as e:
                self.log_q.put(f"Erro: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        while not self.log_q.empty():
            m = self.log_q.get_nowait()
            self.text.insert("end", m + "\n")
            self.text.see("end")
        self.root.after(200, self.poll)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
