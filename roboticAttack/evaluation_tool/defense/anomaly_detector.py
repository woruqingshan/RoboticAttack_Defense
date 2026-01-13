"""Anomaly detection utilities for attention-based patch defense."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PatchBox:
    """Axis-aligned patch bounding box in image pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    def clamp(self, width: int, height: int) -> "PatchBox":
        x0 = int(max(0, min(self.x0, width)))
        x1 = int(max(0, min(self.x1, width)))
        y0 = int(max(0, min(self.y0, height)))
        y1 = int(max(0, min(self.y1, height)))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return PatchBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def pad(self, pad: int) -> "PatchBox":
        p = int(max(0, pad))
        return PatchBox(x0=self.x0 - p, y0=self.y0 - p, x1=self.x1 + p, y1=self.y1 + p)


@dataclass(frozen=True)
class DetectionResult:
    """Detection decision for one frame/step."""

    is_anomaly: bool
    score: float
    patch_mass: float
    entropy: Optional[float] = None


def normalized_entropy(heatmap: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized entropy in [0, 1] for a non-negative heatmap."""

    h = heatmap.astype(np.float64)
    h = h - float(h.min())
    s = float(h.sum())
    if s <= eps:
        return 1.0
    p = h / (s + eps)
    ent = float(-(p * np.log(p + eps)).sum())
    max_ent = float(np.log(p.size + eps))
    return float(ent / (max_ent + eps))


class PatchAttentionAnomalyDetector:
    """
    Detect attention hijacking by measuring how much attention mass falls inside the patch region.

    Notes:
    - This detector assumes a known patch location (x, y) in the current debug stage.
    - When future geometry transforms (rotation/shear) are enabled, a box may be inaccurate and
      should be replaced by a mask-based region.
    """

    def __init__(
        self,
        patch_mass_threshold: float = 0.25,
        entropy_threshold: Optional[float] = None,
        use_entropy_gate: bool = False,
    ) -> None:
        self.patch_mass_threshold = float(patch_mass_threshold)
        self.entropy_threshold = float(entropy_threshold) if entropy_threshold is not None else None
        self.use_entropy_gate = bool(use_entropy_gate)

    def detect(self, heatmap: np.ndarray, patch_box: PatchBox) -> DetectionResult:
        if heatmap.ndim != 2:
            raise ValueError(f"Expected 2D heatmap, got shape={heatmap.shape}")

        h, w = heatmap.shape
        box = patch_box.clamp(width=w, height=h)
        if box.x1 <= box.x0 or box.y1 <= box.y0:
            return DetectionResult(is_anomaly=False, score=0.0, patch_mass=0.0, entropy=None)

        hm = heatmap.astype(np.float64)
        hm = hm - float(hm.min())
        total = float(hm.sum()) + 1e-12
        patch_sum = float(hm[box.y0 : box.y1, box.x0 : box.x1].sum())
        patch_mass = float(patch_sum / total)

        ent = normalized_entropy(hm)
        score = patch_mass

        is_anomaly = patch_mass >= self.patch_mass_threshold
        if self.use_entropy_gate and (self.entropy_threshold is not None):
            is_anomaly = is_anomaly and (ent <= self.entropy_threshold)

        return DetectionResult(is_anomaly=is_anomaly, score=score, patch_mass=patch_mass, entropy=ent)


