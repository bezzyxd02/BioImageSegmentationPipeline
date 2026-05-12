from __future__ import annotations
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor


def percentile_normalize(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    x = np.clip((img - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def hot_pixel_removal(img: np.ndarray) -> np.ndarray:
    med = cv2.medianBlur(img, 3)
    diff = cv2.absdiff(img, med)
    mask = diff > 35
    out = img.copy()
    out[mask] = med[mask]
    return out


def preprocess_frame(frame: np.ndarray, rolling_ball_radius: int = 21, clahe_clip: float = 2.5, gaussian_kernel: int = 5) -> np.ndarray:
    img = percentile_normalize(frame.astype(np.float32))
    img = hot_pixel_removal(img)
    bg = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((rolling_ball_radius, rolling_ball_radius), np.uint8))
    img = cv2.subtract(img, bg)
    img = cv2.GaussianBlur(img, (gaussian_kernel, gaussian_kernel), 0)
    img = cv2.fastNlMeansDenoising(img, None, 7, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_batch(frames: list[np.ndarray], workers: int = 4) -> list[np.ndarray]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(preprocess_frame, frames))
