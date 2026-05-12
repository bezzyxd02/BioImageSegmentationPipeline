from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def _centroids(mask: np.ndarray):
    ids = [i for i in np.unique(mask) if i != 0]
    out = []
    for i in ids:
        ys, xs = np.where(mask == i)
        if len(xs) == 0:
            continue
        out.append((i, float(xs.mean()), float(ys.mean())))
    return out


def track_cells(masks: list[np.ndarray], max_distance: float = 40.0):
    tracks = []
    next_id = 1
    prev = []
    active = {}
    for t, m in enumerate(masks):
        cur = _centroids(m)
        if not prev:
            for _, x, y in cur:
                active[next_id] = (x, y)
                tracks.append((t, next_id, x, y))
                next_id += 1
            prev = cur
            continue
        pxy = np.array([[x, y] for _, x, y in prev], dtype=np.float32)
        cxy = np.array([[x, y] for _, x, y in cur], dtype=np.float32)
        if len(pxy) == 0 or len(cxy) == 0:
            prev = cur
            continue
        d = ((pxy[:, None, :] - cxy[None, :, :]) ** 2).sum(-1) ** 0.5
        ri, ci = linear_sum_assignment(d)
        used = set()
        prev_ids = list(active.keys())
        new_active = {}
        for r, c in zip(ri, ci):
            if d[r, c] <= max_distance and r < len(prev_ids):
                tid = prev_ids[r]
                x, y = cxy[c]
                new_active[tid] = (float(x), float(y))
                tracks.append((t, tid, float(x), float(y)))
                used.add(c)
        for c, (_, x, y) in enumerate(cur):
            if c not in used:
                new_active[next_id] = (x, y)
                tracks.append((t, next_id, x, y))
                next_id += 1
        active = new_active
        prev = cur
    return tracks
