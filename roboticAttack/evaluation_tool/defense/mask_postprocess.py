"""Mask post-processing utilities for conflict-aware refinement."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _shift_and(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    y0_src = max(0, -dy)
    y1_src = min(h, h - dy) if dy >= 0 else h
    x0_src = max(0, -dx)
    x1_src = min(w, w - dx) if dx >= 0 else w
    y0_dst = max(0, dy)
    y1_dst = y0_dst + (y1_src - y0_src)
    x0_dst = max(0, dx)
    x1_dst = x0_dst + (x1_src - x0_src)
    if y1_src > y0_src and x1_src > x0_src:
        out[y0_dst:y1_dst, x0_dst:x1_dst] = mask[y0_src:y1_src, x0_src:x1_src]
    return out


def erode_mask(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """Binary erosion using 3x3 neighborhood (numpy-only)."""
    out = mask.astype(bool)
    k = int(max(0, iters))
    if k == 0:
        return out
    for _ in range(k):
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nbrs.append(_shift_and(out, dy, dx))
        acc = nbrs[0]
        for n in nbrs[1:]:
            acc = np.logical_and(acc, n)
        out = acc
    return out


def dilate_mask(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """Binary dilation using 3x3 neighborhood (numpy-only)."""
    out = mask.astype(bool)
    k = int(max(0, iters))
    if k == 0:
        return out
    for _ in range(k):
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nbrs.append(_shift_and(out, dy, dx))
        acc = nbrs[0]
        for n in nbrs[1:]:
            acc = np.logical_or(acc, n)
        out = acc
    return out


def mask_to_tight_box(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
