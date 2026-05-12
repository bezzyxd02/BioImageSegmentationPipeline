from __future__ import annotations
from pathlib import Path
import pandas as pd
import tifffile as tiff
import cv2
import numpy as np


def export_masks_tiff(path: Path, masks: list[np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, np.stack(masks).astype(np.uint8))


def export_overlay_png(path: Path, image: np.ndarray, mask: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    rgb[mask > 0] = (0, 255, 0)
    cv2.imwrite(str(path), rgb)


def export_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
