# localizer.py
# -*- coding: utf-8 -*-
"""
ROI localizer for stable attention grids.

This module is decoupled and depends only on numpy.
It provides connected-component localization on a thresholded stable grid.

Main API:
- AttentionLocalizer.localize(stable_grid) -> Optional[LocalizeResult]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .temporal import GridBox


@dataclass
class LocalizeResult:
    roi: GridBox
    roi_mass: float
    area_ratio: float
    threshold: float


def _percentile_threshold(x: np.ndarray, top_p: float) -> float:
    """
    Compute a value threshold such that approximately top_p fraction of elements are kept.
    top_p in (0,1). Larger top_p keeps more pixels.
    """
    flat = x.reshape(-1)
    if flat.size == 0:
        return float("inf")
    # Keep top_p => drop (1-top_p)
    k = int(round((1.0 - float(top_p)) * float(flat.size)))
    k = max(0, min(k, flat.size - 1))
    return float(np.partition(flat, k)[k])


def _connected_components_4(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Minimal 4-neighborhood connected components labeling (BFS).
    Returns (num_labels, labels) with labels in {0..num}.
    """
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    label = 0

    for y in range(H):
        for x in range(W):
            if not bool(mask[y, x]) or labels[y, x] != 0:
                continue
            label += 1
            stack = [(y, x)]
            labels[y, x] = label
            while stack:
                cy, cx = stack.pop()
                # 4-neighbors
                if cy > 0 and mask[cy - 1, cx] and labels[cy - 1, cx] == 0:
                    labels[cy - 1, cx] = label
                    stack.append((cy - 1, cx))
                if cy + 1 < H and mask[cy + 1, cx] and labels[cy + 1, cx] == 0:
                    labels[cy + 1, cx] = label
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and labels[cy, cx - 1] == 0:
                    labels[cy, cx - 1] = label
                    stack.append((cy, cx - 1))
                if cx + 1 < W and mask[cy, cx + 1] and labels[cy, cx + 1] == 0:
                    labels[cy, cx + 1] = label
                    stack.append((cy, cx + 1))

    return label, labels


def _bbox_from_label(labels: np.ndarray, lab: int) -> Optional[GridBox]:
    ys, xs = np.where(labels == lab)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    return GridBox(gx0=x0, gy0=y0, gx1=x1, gy1=y1)


@dataclass
class AttentionLocalizer:
    """
    Localize suspicious ROI from a stable attention grid.

    Strategy:
    1) Normalize stable_grid (shift by min)
    2) threshold by top_p percentile
    3) CC labeling
    4) choose Top-K components with highest roi_mass under area constraints
    """
    top_p: float = 0.07
    min_area: float = 0.01   # relative to grid area
    max_area: float = 0.25
    eps: float = 1e-6

    def localize(self, stable_grid: np.ndarray, top_k: int = 1) -> List[LocalizeResult]:
        g = stable_grid.astype(np.float32)
        g = g - float(g.min())
        total_sum = float(g.sum())

        if total_sum <= self.eps:
            return []

        thr = _percentile_threshold(g, float(self.top_p))
        mask = g >= thr

        num, labels = _connected_components_4(mask)
        if num <= 0:
            return []

        H, W = g.shape
        total_area = float(H * W)

        candidates = []

        for lab in range(1, num + 1):
            box = _bbox_from_label(labels, lab)
            if box is None:
                continue
            # GridBox is a dataclass, access attributes instead of unpacking
            x0, y0, x1, y1 = box.gx0, box.gy0, box.gx1, box.gy1
            area = float((x1 - x0) * (y1 - y0))
            area_ratio = area / (total_area + self.eps)
            if area_ratio < float(self.min_area) or area_ratio > float(self.max_area):
                continue

            roi_sum = float(g[y0:y1, x0:x1].sum())
            roi_mass = roi_sum / (total_sum + self.eps)

            candidates.append(LocalizeResult(
                roi=box,
                roi_mass=float(roi_mass),
                area_ratio=float(area_ratio),
                threshold=float(thr),
            ))

        # Sort by density (mass/area) descending, prefer compact & peaky regions
        # Patches are typically small and dense (high density), while gripper/object
        # regions are often large with high mass but lower density
        # Use density as primary sort key, roi_mass as secondary tie-breaker
        candidates.sort(key=lambda x: (x.roi_mass / (x.area_ratio + self.eps), x.roi_mass), reverse=True)
        return candidates[:top_k]

