# verifier.py
# -*- coding: utf-8 -*-
"""
Counterfactual verification for candidate ROI.

This module is decoupled and depends only on numpy.
It uses user-provided callbacks:
- purify_fn(image, roi_box) -> purified_image
- forward_fn(purified_image) -> any (triggers model forward + hooks)
- heatmap_fn() -> np.ndarray heatmap (H x W), e.g., 224 x 224

Main API:
- CounterfactualVerifier.verify(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np

# Accept either (x0,y0,x1,y1) or an object with .x0,.y0,.x1,.y1
BoxLike = Union[Tuple[int, int, int, int], Any]


def _as_xyxy(box: BoxLike) -> Tuple[int, int, int, int]:
    if isinstance(box, tuple) or isinstance(box, list):
        x0, y0, x1, y1 = box
        return int(x0), int(y0), int(x1), int(y1)
    # object with attributes
    return int(box.x0), int(box.y0), int(box.x1), int(box.y1)


def normalized_entropy(hm: np.ndarray, eps: float = 1e-8) -> float:
    """
    Normalized entropy in [0,1]. High => diffuse attention; Low => concentrated.
    """
    x = hm.astype(np.float32)
    x = x - float(x.min())
    s = float(x.sum())
    if s <= eps:
        return 1.0
    p = x.reshape(-1) / (s + eps)
    ent = -float((p * np.log(p + eps)).sum())
    ent /= float(np.log(p.size + eps))
    return float(ent)


def roi_mass(hm: np.ndarray, roi_box: BoxLike, eps: float = 1e-8) -> float:
    """
    ROI mass = sum(hm in ROI) / sum(hm)
    """
    x = hm.astype(np.float32)
    x = x - float(x.min())
    total = float(x.sum()) + eps

    x0, y0, x1, y1 = _as_xyxy(roi_box)
    H, W = x.shape[:2]
    x0 = max(0, min(x0, W))
    x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H))
    y1 = max(0, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    return float(x[y0:y1, x0:x1].sum() / total)


@dataclass
class VerifyResult:
    verified: bool
    stats: Dict[str, float]


@dataclass
class CounterfactualVerifier:
    """
    One-step counterfactual verification:
    - measure (roi_mass, entropy) BEFORE
    - purify ROI once -> forward -> measure AFTER
    - verified if roi_mass drops enough AND entropy rises enough

    This reduces false positives (e.g., robot arm/object attention).
    """
    min_mass_drop_rel: float = 0.15   # require roi_mass_after <= (1 - rel)*before
    min_entropy_gain_abs: float = 0.02
    eps: float = 1e-8

    def verify(
        self,
        image: np.ndarray,
        roi_box: BoxLike,
        purify_fn: Callable[[np.ndarray, BoxLike], np.ndarray],
        forward_fn: Callable[[np.ndarray], Any],
        heatmap_fn: Callable[[], np.ndarray],
        *,
        hm_before: Optional[np.ndarray] = None,
    ) -> VerifyResult:
        """
        Args:
            image: original image (H,W,3) numpy
            roi_box: PatchBox or (x0,y0,x1,y1) in pixel coords
            purify_fn: function to mask ROI (should be cheap)
            forward_fn: runs model forward on given image (must trigger hooks)
            heatmap_fn: returns the current heatmap from hooks
            hm_before: optional precomputed heatmap for BEFORE

        Returns:
            VerifyResult with verified flag and diagnostic stats.
        """
        # BEFORE metrics
        if hm_before is None:
            hm0 = heatmap_fn()
        else:
            hm0 = hm_before

        m0 = roi_mass(hm0, roi_box, eps=self.eps)
        e0 = normalized_entropy(hm0, eps=self.eps)

        # Counterfactual: purify ROI and forward once
        img1 = purify_fn(image, roi_box)
        _ = forward_fn(img1)  # triggers hook update
        hm1 = heatmap_fn()

        m1 = roi_mass(hm1, roi_box, eps=self.eps)
        e1 = normalized_entropy(hm1, eps=self.eps)

        # Decision
        # Avoid division by zero
        m0_safe = max(m0, float(self.eps))
        rel_drop = (m0_safe - m1) / m0_safe
        ent_gain = e1 - e0

        verified = (rel_drop >= float(self.min_mass_drop_rel)) and (ent_gain >= float(self.min_entropy_gain_abs))

        stats = {
            "roi_mass_before": float(m0),
            "roi_mass_after": float(m1),
            "roi_mass_rel_drop": float(rel_drop),
            "entropy_before": float(e0),
            "entropy_after": float(e1),
            "entropy_gain": float(ent_gain),
        }
        return VerifyResult(verified=bool(verified), stats=stats)

