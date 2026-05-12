from __future__ import annotations
import numpy as np
import tifffile as tiff


def generate_cellpose_labels(frame: np.ndarray, model_type: str = "cyto", diameter: float = 0, channels: list[int] | tuple[int, int] = (0, 0), use_gpu: bool = True):
    from cellpose import models
    model = models.Cellpose(gpu=use_gpu, model_type=model_type)
    masks, flows, styles, diams = model.eval(frame, diameter=diameter if diameter > 0 else None, channels=list(channels))
    return masks, flows, styles, diams


def load_ilastik_probability(path: str) -> np.ndarray:
    prob = tiff.imread(path)
    if prob.ndim > 2:
        prob = prob[..., 0]
    prob = prob.astype(np.float32)
    if prob.max() > 1:
        prob /= 255.0
    return np.clip(prob, 0, 1)


def combine_labels(cellpose_mask: np.ndarray, ilastik_probability: np.ndarray, thr: float = 0.5) -> np.ndarray:
    cp = (cellpose_mask > 0).astype(np.float32)
    il = (ilastik_probability > thr).astype(np.float32)
    combo = 0.7 * cp + 0.3 * il
    return (combo >= 0.5).astype(np.uint8)
