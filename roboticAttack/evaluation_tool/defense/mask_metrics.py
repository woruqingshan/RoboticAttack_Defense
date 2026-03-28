"""Common mask metrics for geometry-aware defense logging and evaluation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    denom = float(max(int(a_bool.sum()), 1))
    return float(np.logical_and(a_bool, b_bool).sum() / denom)


def summarize_mask(mask: Optional[np.ndarray]) -> Optional[Dict[str, float | int | tuple]]:
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    m = mask.astype(bool)
    area = int(m.sum())
    h, w = int(m.shape[0]), int(m.shape[1])
    box = mask_to_box(m)
    return {
        "area": area,
        "area_ratio": float(area / max(float(h * w), 1.0)),
        "shape": (h, w),
        "box": box,
    }
