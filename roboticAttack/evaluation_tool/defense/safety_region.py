"""Safety region fusion for geometry-first mask control.

This module merges arm / gripper core+guard regions into a unified bundle that
can be consumed by selector/refiner/controller without depending on prior
internal details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np


@dataclass
class SafetyRegionConfig:
    """Weights used to build a pixel-level safety penalty map."""

    w_arm_core: float = 1.0
    w_arm_guard: float = 0.6
    w_gripper_core: float = 1.0
    w_gripper_guard: float = 0.7
    smooth_kernel: int = 0


@dataclass
class SafetyRegionBundle:
    """Unified safety region object shared by downstream modules."""

    valid: bool
    masks_px: Dict[str, np.ndarray]
    penalty_map_px: np.ndarray
    boxes_px: Dict[str, Optional[Tuple[int, int, int, int]]]
    masks_grid: Dict[str, Optional[np.ndarray]]


def _empty_bool(hw: Tuple[int, int]) -> np.ndarray:
    return np.zeros((int(hw[0]), int(hw[1])), dtype=bool)


def _mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))


def _ensure_bool_mask(mask: Optional[np.ndarray], hw: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return _empty_bool(hw)
    if not isinstance(mask, np.ndarray):
        return _empty_bool(hw)
    out = mask.astype(bool)
    if out.shape != (int(hw[0]), int(hw[1])):
        return _empty_bool(hw)
    return out


def _mask_to_grid_mask(mask: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    gh, gw = int(grid_shape[0]), int(grid_shape[1])
    h, w = int(mask.shape[0]), int(mask.shape[1])
    grid = np.zeros((gh, gw), dtype=bool)
    for gy in range(gh):
        y0 = int(np.floor(gy * h / gh))
        y1 = int(np.ceil((gy + 1) * h / gh))
        for gx in range(gw):
            x0 = int(np.floor(gx * w / gw))
            x1 = int(np.ceil((gx + 1) * w / gw))
            if y1 > y0 and x1 > x0 and bool(mask[y0:y1, x0:x1].any()):
                grid[gy, gx] = True
    return grid


class SafetyRegionBuilder:
    """Fuse arm/gripper geometry into a single safety bundle."""

    def __init__(self, config: Optional[SafetyRegionConfig] = None):
        self.config = config if config is not None else SafetyRegionConfig()

    def _smooth_penalty(self, penalty: np.ndarray) -> np.ndarray:
        k = int(self.config.smooth_kernel)
        if k <= 1:
            return penalty
        # Fallback-safe smoothing: use a mean blur implemented by 2D separable conv via numpy.
        # Keep dependencies minimal and avoid hard cv2 requirement in this module.
        k = int(max(1, k))
        if (k % 2) == 0:
            k += 1
        pad = k // 2
        x = np.pad(penalty, ((pad, pad), (pad, pad)), mode="edge")
        # Box filter (simple and stable for low-resolution penalty maps)
        out = np.zeros_like(penalty, dtype=np.float32)
        for yy in range(out.shape[0]):
            for xx in range(out.shape[1]):
                win = x[yy : yy + k, xx : xx + k]
                out[yy, xx] = float(win.mean())
        return out

    def build(
        self,
        policy_hw: Tuple[int, int],
        gripper_res: Optional[Any],
        arm_res: Optional[Any],
        grid_shape: Optional[Tuple[int, int]] = None,
    ) -> SafetyRegionBundle:
        """Build a unified safety bundle from prior results.

        Args:
            policy_hw: policy image shape (H, W).
            gripper_res: GripperPriorResult or None.
            arm_res: ArmSkeletonResult or None.
            grid_shape: optional low-res grid shape (Hgrid, Wgrid).
        """
        hw = (int(policy_hw[0]), int(policy_hw[1]))

        arm_core = _ensure_bool_mask(getattr(arm_res, "arm_core_mask_px", None), hw)
        arm_guard = _ensure_bool_mask(getattr(arm_res, "arm_guard_mask_px", None), hw)
        gripper_core = _ensure_bool_mask(getattr(gripper_res, "gripper_core_mask_px", None), hw)
        gripper_guard = _ensure_bool_mask(getattr(gripper_res, "gripper_guard_mask_px", None), hw)

        masks_px: Dict[str, np.ndarray] = {
            "arm_core": arm_core,
            "arm_guard": arm_guard,
            "gripper_core": gripper_core,
            "gripper_guard": gripper_guard,
        }

        # Guard is expected to include core; enforce monotonicity defensively.
        masks_px["arm_guard"] = np.logical_or(masks_px["arm_guard"], masks_px["arm_core"])
        masks_px["gripper_guard"] = np.logical_or(masks_px["gripper_guard"], masks_px["gripper_core"])

        penalty = np.zeros(hw, dtype=np.float32)
        penalty += float(self.config.w_arm_core) * masks_px["arm_core"].astype(np.float32)
        penalty += float(self.config.w_arm_guard) * masks_px["arm_guard"].astype(np.float32)
        penalty += float(self.config.w_gripper_core) * masks_px["gripper_core"].astype(np.float32)
        penalty += float(self.config.w_gripper_guard) * masks_px["gripper_guard"].astype(np.float32)
        penalty = self._smooth_penalty(penalty)

        boxes_px: Dict[str, Optional[Tuple[int, int, int, int]]] = {
            "arm_core": _mask_to_box(masks_px["arm_core"]),
            "arm_guard": _mask_to_box(masks_px["arm_guard"]),
            "gripper_core": _mask_to_box(masks_px["gripper_core"]),
            "gripper_guard": _mask_to_box(masks_px["gripper_guard"]),
        }

        masks_grid: Dict[str, Optional[np.ndarray]] = {
            "arm_core": None,
            "arm_guard": None,
            "gripper_core": None,
            "gripper_guard": None,
        }
        if grid_shape is not None:
            masks_grid = {
                "arm_core": _mask_to_grid_mask(masks_px["arm_core"], grid_shape),
                "arm_guard": _mask_to_grid_mask(masks_px["arm_guard"], grid_shape),
                "gripper_core": _mask_to_grid_mask(masks_px["gripper_core"], grid_shape),
                "gripper_guard": _mask_to_grid_mask(masks_px["gripper_guard"], grid_shape),
            }

        valid = bool(
            masks_px["arm_guard"].any()
            or masks_px["gripper_guard"].any()
            or masks_px["arm_core"].any()
            or masks_px["gripper_core"].any()
        )

        return SafetyRegionBundle(
            valid=valid,
            masks_px=masks_px,
            penalty_map_px=penalty,
            boxes_px=boxes_px,
            masks_grid=masks_grid,
        )
