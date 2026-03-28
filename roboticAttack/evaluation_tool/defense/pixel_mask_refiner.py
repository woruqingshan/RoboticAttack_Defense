"""Pixel-level mask refinement under geometry safety constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _as_xyxy(box: Any) -> Tuple[int, int, int, int]:
    if isinstance(box, (tuple, list)) and len(box) == 4:
        return int(box[0]), int(box[1]), int(box[2]), int(box[3])
    return int(box.x0), int(box.y0), int(box.x1), int(box.y1)


def _clip_box(box: Tuple[int, int, int, int], hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    h, w = int(hw[0]), int(hw[1])
    x0 = max(0, min(int(x0), w))
    x1 = max(0, min(int(x1), w))
    y0 = max(0, min(int(y0), h))
    y1 = max(0, min(int(y1), h))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep largest 4-connected component in a binary mask."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    best_label = 0
    best_size = 0
    for yy in range(h):
        for xx in range(w):
            if not mask[yy, xx] or labels[yy, xx] != 0:
                continue
            current += 1
            stack = [(yy, xx)]
            labels[yy, xx] = current
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                if cy > 0 and mask[cy - 1, cx] and labels[cy - 1, cx] == 0:
                    labels[cy - 1, cx] = current
                    stack.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and labels[cy + 1, cx] == 0:
                    labels[cy + 1, cx] = current
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and labels[cy, cx - 1] == 0:
                    labels[cy, cx - 1] = current
                    stack.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and labels[cy, cx + 1] == 0:
                    labels[cy, cx + 1] = current
                    stack.append((cy, cx + 1))
            if size > best_size:
                best_size = size
                best_label = current
    if best_label == 0:
        return np.zeros_like(mask, dtype=bool)
    return labels == best_label


@dataclass
class PixelMaskRefinerConfig:
    lambda_safety: float = 0.75
    score_quantile: float = 0.65
    min_area_ratio: float = 0.08
    min_cover_ratio: float = 0.40
    keep_largest_component: bool = True
    hard_forbid_core: bool = True


@dataclass
class PixelMaskRefineResult:
    valid: bool
    mask_px: np.ndarray
    tight_box_xyxy: Optional[Tuple[int, int, int, int]]
    stats: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


class PixelMaskRefiner:
    """Refine a coarse ROI box into a pixel mask using heatmap-safety tradeoff."""

    def __init__(self, config: Optional[PixelMaskRefinerConfig] = None):
        self.config = config if config is not None else PixelMaskRefinerConfig()

    def refine(
        self,
        initial_roi_px: Any,
        heatmap: np.ndarray,
        safety_bundle: Optional[Any] = None,
    ) -> PixelMaskRefineResult:
        if heatmap.ndim != 2:
            raise ValueError(f"heatmap must be 2D, got shape={heatmap.shape}")

        h, w = int(heatmap.shape[0]), int(heatmap.shape[1])
        roi = _clip_box(_as_xyxy(initial_roi_px), (h, w))
        x0, y0, x1, y1 = roi
        if x1 <= x0 or y1 <= y0:
            return PixelMaskRefineResult(
                valid=False,
                mask_px=np.zeros((h, w), dtype=bool),
                tight_box_xyxy=None,
                reason="invalid_roi",
            )

        hm = heatmap.astype(np.float32)
        hm = hm - float(hm.min())
        hm = hm / (float(hm.max()) + 1e-6)

        penalty = np.zeros((h, w), dtype=np.float32)
        core_mask = np.zeros((h, w), dtype=bool)
        guard_mask = np.zeros((h, w), dtype=bool)
        if safety_bundle is not None:
            try:
                penalty = safety_bundle.penalty_map_px.astype(np.float32)
                if penalty.shape != (h, w):
                    penalty = np.zeros((h, w), dtype=np.float32)
            except Exception:
                penalty = np.zeros((h, w), dtype=np.float32)

            try:
                masks_px = getattr(safety_bundle, "masks_px", {})
                arm_core = masks_px.get("arm_core", np.zeros((h, w), dtype=bool))
                grip_core = masks_px.get("gripper_core", np.zeros((h, w), dtype=bool))
                arm_guard = masks_px.get("arm_guard", np.zeros((h, w), dtype=bool))
                grip_guard = masks_px.get("gripper_guard", np.zeros((h, w), dtype=bool))
                core_mask = np.logical_or(arm_core, grip_core)
                guard_mask = np.logical_or(arm_guard, grip_guard)
                if core_mask.shape != (h, w):
                    core_mask = np.zeros((h, w), dtype=bool)
                if guard_mask.shape != (h, w):
                    guard_mask = np.zeros((h, w), dtype=bool)
            except Exception:
                core_mask = np.zeros((h, w), dtype=bool)
                guard_mask = np.zeros((h, w), dtype=bool)

        score = hm - float(self.config.lambda_safety) * penalty
        roi_score = score[y0:y1, x0:x1]
        roi_hm = hm[y0:y1, x0:x1]
        roi_area = int((x1 - x0) * (y1 - y0))
        min_pixels = int(max(1, round(float(self.config.min_area_ratio) * float(roi_area))))

        q = float(np.clip(self.config.score_quantile, 0.0, 1.0))
        thr = float(np.quantile(roi_score, q))
        local_mask = roi_score >= thr

        if bool(self.config.hard_forbid_core):
            local_core = core_mask[y0:y1, x0:x1]
            local_mask = np.logical_and(local_mask, np.logical_not(local_core))

        selected = int(local_mask.sum())
        if selected < min_pixels:
            flat_score = roi_score.reshape(-1)
            if bool(self.config.hard_forbid_core):
                core_flat = core_mask[y0:y1, x0:x1].reshape(-1)
                flat_score = np.where(core_flat, -1e9, flat_score)
            k = int(min(min_pixels, flat_score.size))
            idx = np.argpartition(flat_score, -k)[-k:]
            local_mask = np.zeros(flat_score.size, dtype=bool)
            local_mask[idx] = True
            local_mask = local_mask.reshape(roi_score.shape)

        if bool(self.config.keep_largest_component):
            local_mask = _largest_connected_component(local_mask)

        # Enforce minimum heatmap coverage within ROI.
        roi_hm_sum = float(roi_hm.sum()) + 1e-8
        covered_hm = float(roi_hm[local_mask].sum())
        cover_ratio = covered_hm / roi_hm_sum
        if cover_ratio < float(self.config.min_cover_ratio):
            target = float(self.config.min_cover_ratio) * roi_hm_sum
            flat_hm = roi_hm.reshape(-1)
            if bool(self.config.hard_forbid_core):
                core_flat = core_mask[y0:y1, x0:x1].reshape(-1)
                flat_hm = np.where(core_flat, -1e9, flat_hm)
            idx_sorted = np.argsort(flat_hm)[::-1]
            add_mask = np.zeros(flat_hm.size, dtype=bool)
            run_sum = 0.0
            for idx in idx_sorted:
                if flat_hm[idx] <= -1e8:
                    continue
                add_mask[idx] = True
                run_sum += float(flat_hm[idx])
                if run_sum >= target:
                    break
            local_mask = np.logical_or(local_mask.reshape(-1), add_mask).reshape(local_mask.shape)

        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[y0:y1, x0:x1] = local_mask
        tight_box = _mask_to_box(full_mask)
        if tight_box is None:
            full_mask[y0:y1, x0:x1] = True
            tight_box = roi

        core_overlap = float(np.logical_and(full_mask, core_mask).sum() / max(float(full_mask.sum()), 1.0))
        guard_overlap = float(np.logical_and(full_mask, guard_mask).sum() / max(float(full_mask.sum()), 1.0))

        stats = {
            "selected_pixels": float(full_mask.sum()),
            "roi_area": float(roi_area),
            "selected_ratio": float(full_mask.sum() / max(float(roi_area), 1.0)),
            "cover_ratio": float((hm[full_mask].sum() / (hm[y0:y1, x0:x1].sum() + 1e-8))),
            "core_overlap_ratio": core_overlap,
            "guard_overlap_ratio": guard_overlap,
        }

        return PixelMaskRefineResult(
            valid=True,
            mask_px=full_mask,
            tight_box_xyxy=tight_box,
            stats=stats,
            reason="ok",
        )
