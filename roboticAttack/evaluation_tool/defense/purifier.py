"""Image purification utilities for online patch defense."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .anomaly_detector import PatchBox

# Optional dependency; OpenCV is commonly available in this repo environment.
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class ImagePurifier:
    """
    Purify an RGB image by modifying a suspected patch region (ROI).

    Backward compatible:
      - purify(image, patch_box) still works exactly as before.

    Supported strategies:
      - mask_mean : overwrite ROI with global mean color (hard).
      - mask_gray : overwrite ROI with fixed gray value (hard).
      - blend_mean: blend ROI toward global mean color (soft, reversible).
      - blend_gray: blend ROI toward gray value (soft, reversible).
      - blur      : blur ROI (soft); can be blended with original by alpha/strength.
    """

    strategy: str = "mask_mean"
    pad: int = 0
    gray_value: int = 127

    # Soft purification controls
    alpha: float = 1.0  # default blend strength (0=no change, 1=full replace)
    blur_ksize: int = 7  # odd integer recommended
    blur_sigma: float = 2.0

    def purify(
        self,
        image: np.ndarray,
        patch_box: PatchBox,
        strength: Optional[float] = None,
    ) -> np.ndarray:
        """
        Args:
            image: HxWx3 RGB image (uint8 preferred).
            patch_box: ROI bbox in pixel coordinates.
            strength: Optional override for alpha (0..1). If None, uses self.alpha.

        Returns:
            A purified image (uint8).
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape={image.shape}")

        # Ensure uint8
        if image.dtype != np.uint8:
            img = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img = image

        h, w, _ = img.shape
        box = patch_box.pad(self.pad).clamp(width=w, height=h)
        if box.x1 <= box.x0 or box.y1 <= box.y0:
            return img

        # Strength override
        a = self.alpha if strength is None else float(strength)
        a = float(np.clip(a, 0.0, 1.0))

        out = img.copy()
        roi = out[box.y0 : box.y1, box.x0 : box.x1]

        if self.strategy == "mask_mean":
            mean_color = out.mean(axis=(0, 1)).astype(np.float32)
            if a >= 0.999:  # keep legacy hard mask
                out[box.y0 : box.y1, box.x0 : box.x1] = mean_color.astype(np.uint8)
                return out

            # NEW: soft mask (blend toward mean)
            tgt = np.broadcast_to(mean_color, roi.shape).astype(np.float32)
            roi_f = roi.astype(np.float32)
            out_roi = (1.0 - a) * roi_f + a * tgt
            out[box.y0 : box.y1, box.x0 : box.x1] = np.clip(out_roi, 0, 255).astype(np.uint8)
            return out

        if self.strategy == "mask_gray":
            gv = float(np.clip(self.gray_value, 0, 255))
            if a >= 0.999:  # legacy hard mask
                gvu = np.uint8(int(gv))
                out[box.y0 : box.y1, box.x0 : box.x1] = np.array([gvu, gvu, gvu], dtype=np.uint8)
                return out

            # NEW: soft mask (blend toward gray)
            tgt = np.full_like(roi, gv, dtype=np.float32)
            roi_f = roi.astype(np.float32)
            out_roi = (1.0 - a) * roi_f + a * tgt
            out[box.y0 : box.y1, box.x0 : box.x1] = np.clip(out_roi, 0, 255).astype(np.uint8)
            return out

        if self.strategy == "blend_mean":
            # Blend ROI toward global mean color (soft / less destructive)
            mean_color = out.mean(axis=(0, 1)).astype(np.float32)
            tgt = np.broadcast_to(mean_color, roi.shape).astype(np.float32)
            roi_f = roi.astype(np.float32)
            out_roi = (1.0 - a) * roi_f + a * tgt
            out[box.y0 : box.y1, box.x0 : box.x1] = np.clip(out_roi, 0, 255).astype(np.uint8)
            return out

        if self.strategy == "blend_gray":
            gv = float(np.clip(self.gray_value, 0, 255))
            tgt = np.full_like(roi, gv, dtype=np.float32)
            roi_f = roi.astype(np.float32)
            out_roi = (1.0 - a) * roi_f + a * tgt
            out[box.y0 : box.y1, box.x0 : box.x1] = np.clip(out_roi, 0, 255).astype(np.uint8)
            return out

        if self.strategy == "blur":
            # Gaussian blur ROI; optionally blend with original by alpha/strength.
            if cv2 is None:
                raise ValueError("purifier strategy 'blur' requires OpenCV (cv2) to be installed.")

            k = int(self.blur_ksize)
            if k <= 1:
                return out
            if k % 2 == 0:
                k += 1  # ensure odd kernel size

            blurred = cv2.GaussianBlur(roi, (k, k), sigmaX=float(self.blur_sigma), sigmaY=float(self.blur_sigma))
            if a >= 0.999:
                out[box.y0 : box.y1, box.x0 : box.x1] = blurred
            else:
                roi_f = roi.astype(np.float32)
                blr_f = blurred.astype(np.float32)
                out_roi = (1.0 - a) * roi_f + a * blr_f
                out[box.y0 : box.y1, box.x0 : box.x1] = np.clip(out_roi, 0, 255).astype(np.uint8)
            return out

        raise ValueError(f"Unsupported purifier strategy: {self.strategy}")
