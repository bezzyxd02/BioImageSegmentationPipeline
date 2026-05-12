from __future__ import annotations
import cv2
import numpy as np
from skimage.measure import regionprops, label
from scipy.ndimage import binary_fill_holes


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = binary_fill_holes(m).astype(np.uint8)
    return m


def morphometry(mask: np.ndarray, image: np.ndarray) -> list[dict]:
    lbl = label(mask > 0)
    rows = []
    for r in regionprops(lbl, intensity_image=image):
        per = r.perimeter if r.perimeter > 0 else 1.0
        rows.append({
            "area": float(r.area),
            "perimeter": float(r.perimeter),
            "circularity": float(4 * np.pi * r.area / (per * per)),
            "eccentricity": float(r.eccentricity),
            "solidity": float(r.solidity),
            "aspect_ratio": float((r.major_axis_length + 1e-6) / (r.minor_axis_length + 1e-6)),
            "centroid_x": float(r.centroid[1]),
            "centroid_y": float(r.centroid[0]),
            "mean_intensity": float(r.mean_intensity),
        })
    return rows
