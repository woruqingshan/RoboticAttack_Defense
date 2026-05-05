"""Patch and ROI mask helpers for recovery metrics.

The first supported target is the common LIBERO simulation setting where the
patch is axis-aligned. For transformed patches, callers can still log a bounding
box fallback and mark that mode in the metrics row.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def infer_patch_hw(patch: Any) -> Tuple[int, int]:
    """Infer patch height/width from a torch or numpy patch tensor."""
    if patch is None:
        raise ValueError("patch is None")
    if hasattr(patch, "detach"):
        patch = patch.detach()
    shape = tuple(getattr(patch, "shape", ()))
    if len(shape) == 4:
        return int(shape[-2]), int(shape[-1])
    if len(shape) == 3:
        return int(shape[-2]), int(shape[-1])
    if len(shape) == 2:
        return int(shape[-2]), int(shape[-1])
    raise ValueError(f"Unsupported patch shape: {shape}")


def patch_box_xyxy(
    x: int,
    y: int,
    patch_w: int,
    patch_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    """Build a clipped xyxy patch box."""
    x0 = max(0, min(int(x), int(image_w)))
    y0 = max(0, min(int(y), int(image_h)))
    x1 = max(0, min(int(x) + int(patch_w), int(image_w)))
    y1 = max(0, min(int(y) + int(patch_h), int(image_h)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return int(x0), int(y0), int(x1), int(y1)


def box_to_mask(image_hw: Tuple[int, int], box: Optional[Any]) -> np.ndarray:
    """Rasterize an xyxy box or PatchBox-like object as a boolean mask."""
    h, w = int(image_hw[0]), int(image_hw[1])
    mask = np.zeros((h, w), dtype=bool)
    if box is None:
        return mask
    if isinstance(box, dict):
        vals = (box.get("x0"), box.get("y0"), box.get("x1"), box.get("y1"))
    elif isinstance(box, (tuple, list)) and len(box) == 4:
        vals = tuple(box)
    else:
        vals = (
            getattr(box, "x0", None),
            getattr(box, "y0", None),
            getattr(box, "x1", None),
            getattr(box, "y1", None),
        )
    if any(v is None for v in vals):
        return mask
    x0, y0, x1, y1 = patch_box_xyxy(
        int(vals[0]),
        int(vals[1]),
        int(vals[2]) - int(vals[0]),
        int(vals[3]) - int(vals[1]),
        image_w=w,
        image_h=h,
    )
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def build_axis_aligned_patch_mask(
    image_hw: Tuple[int, int],
    x: int,
    y: int,
    patch_w: int,
    patch_h: int,
) -> np.ndarray:
    """Build the ground-truth patch mask from known top-left placement."""
    h, w = int(image_hw[0]), int(image_hw[1])
    box = patch_box_xyxy(x, y, patch_w, patch_h, image_w=w, image_h=h)
    return box_to_mask((h, w), box)


def roi_to_mask(image_hw: Tuple[int, int], roi: Optional[Any]) -> np.ndarray:
    """Alias for box_to_mask used by rollout instrumentation."""
    return box_to_mask(image_hw, roi)


def mask_to_box_xyxy(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert a binary mask to a clipped xyxy box."""
    if mask is None or not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    ys, xs = np.nonzero(mask.astype(bool))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)

