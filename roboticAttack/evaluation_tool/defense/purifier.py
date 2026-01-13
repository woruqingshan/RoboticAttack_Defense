"""Image purification utilities for online patch defense."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anomaly_detector import PatchBox


@dataclass
class ImagePurifier:
    """
    Purify an RGB image by modifying a known patch region.

    Supported strategies:
    - mask_mean: overwrite patch region with the image mean color.
    - mask_gray: overwrite patch region with a fixed gray value.
    """

    strategy: str = "mask_mean"
    pad: int = 0
    gray_value: int = 127

    def purify(self, image: np.ndarray, patch_box: PatchBox) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape={image.shape}")
        if image.dtype != np.uint8:
            img = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img = image

        h, w, _ = img.shape
        box = patch_box.pad(self.pad).clamp(width=w, height=h)
        if box.x1 <= box.x0 or box.y1 <= box.y0:
            return img

        out = img.copy()
        if self.strategy == "mask_mean":
            mean_color = out.mean(axis=(0, 1)).astype(np.uint8)
            out[box.y0 : box.y1, box.x0 : box.x1] = mean_color
            return out
        if self.strategy == "mask_gray":
            gv = np.uint8(int(np.clip(self.gray_value, 0, 255)))
            out[box.y0 : box.y1, box.x0 : box.x1] = np.array([gv, gv, gv], dtype=np.uint8)
            return out

        raise ValueError(f"Unsupported purifier strategy: {self.strategy}")


