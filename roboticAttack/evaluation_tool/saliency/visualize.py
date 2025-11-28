"""Helper functions for saving and overlaying saliency heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


def normalize_heatmap(heatmap: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize heatmap into [0, 1]."""

    heatmap = heatmap.astype(np.float32)
    min_val = float(heatmap.min())
    max_val = float(heatmap.max())
    denom = max(max_val - min_val, eps)
    return (heatmap - min_val) / denom


def _to_bgr(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image from {image}")
        return bgr
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected HxWx3 image array.")
        if image.dtype != np.uint8:
            img = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img = image
        # Assume incoming array is RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image type: {type(image)}")


def save_overlay_heatmap(
    image: Union[str, Path, np.ndarray],
    heatmap: np.ndarray,
    output_path: Union[str, Path],
    alpha: float = 0.45,
    colormap: str = "JET",
) -> None:
    """Overlay a heatmap on top of an RGB image and write to disk."""

    normalized = normalize_heatmap(heatmap)
    bgr_image = _to_bgr(image)
    heatmap_uint8 = np.uint8(normalized * 255)
    cmap_flag = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(heatmap_uint8, cmap_flag)
    overlay = cv2.addWeighted(colored, alpha, bgr_image, 1 - alpha, 0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)

