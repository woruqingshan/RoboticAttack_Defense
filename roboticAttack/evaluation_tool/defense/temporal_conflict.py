"""Temporal conflict resolver for static patch mask vs dynamic arm/gripper regions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import numpy as np


@dataclass
class TemporalConflictConfig:
    core_hard_on: float = 0.10
    core_hard_off: float = 0.04
    guard_soft_on: float = 0.25
    guard_soft_off: float = 0.12
    hard_on_frames: int = 2
    soft_on_frames: int = 2
    soft_alpha: float = 0.7


@dataclass
class TemporalConflictDecision:
    mode: str  # SAFE | SOFT | HARD
    alpha_scale: float
    reason: str
    stats: Dict[str, float] = field(default_factory=dict)


class TemporalConflictResolver:
    """Resolve mask-vs-geometry conflicts with hysteresis and frame persistence."""

    def __init__(self, config: Optional[TemporalConflictConfig] = None):
        self.config = config if config is not None else TemporalConflictConfig()
        self.reset()

    def reset(self) -> None:
        self.state = "SAFE"
        self._hard_cnt = 0
        self._soft_cnt = 0

    def _compute_overlap(self, roi_mask: np.ndarray, safety_bundle: Optional[Any]) -> Dict[str, float]:
        area = float(max(int(roi_mask.sum()), 1))
        if safety_bundle is None:
            return {
                "core_overlap_ratio": 0.0,
                "guard_overlap_ratio": 0.0,
                "mask_area": area,
            }
        masks = getattr(safety_bundle, "masks_px", {})
        arm_core = masks.get("arm_core", np.zeros_like(roi_mask, dtype=bool))
        gripper_core = masks.get("gripper_core", np.zeros_like(roi_mask, dtype=bool))
        arm_guard = masks.get("arm_guard", np.zeros_like(roi_mask, dtype=bool))
        gripper_guard = masks.get("gripper_guard", np.zeros_like(roi_mask, dtype=bool))

        core = np.logical_or(arm_core.astype(bool), gripper_core.astype(bool))
        guard = np.logical_or(arm_guard.astype(bool), gripper_guard.astype(bool))
        core_overlap = float(np.logical_and(roi_mask, core).sum()) / area
        guard_overlap = float(np.logical_and(roi_mask, guard).sum()) / area
        return {
            "core_overlap_ratio": core_overlap,
            "guard_overlap_ratio": guard_overlap,
            "mask_area": area,
        }

    def update(self, roi_mask: np.ndarray, safety_bundle: Optional[Any]) -> TemporalConflictDecision:
        if roi_mask is None or roi_mask.ndim != 2 or int(roi_mask.sum()) <= 0:
            self.reset()
            return TemporalConflictDecision(
                mode="SAFE",
                alpha_scale=1.0,
                reason="empty_mask",
                stats={"core_overlap_ratio": 0.0, "guard_overlap_ratio": 0.0, "mask_area": 0.0},
            )

        st = self._compute_overlap(roi_mask.astype(bool), safety_bundle)
        core = float(st["core_overlap_ratio"])
        guard = float(st["guard_overlap_ratio"])

        if core >= float(self.config.core_hard_on):
            self._hard_cnt += 1
        else:
            self._hard_cnt = 0

        if guard >= float(self.config.guard_soft_on):
            self._soft_cnt += 1
        else:
            self._soft_cnt = 0

        if self.state != "HARD" and self._hard_cnt >= int(max(1, self.config.hard_on_frames)):
            self.state = "HARD"
        elif self.state == "HARD" and core <= float(self.config.core_hard_off):
            # HARD 退出后至少进入 SOFT，避免震荡。
            self.state = "SOFT"

        if self.state == "SAFE" and self._soft_cnt >= int(max(1, self.config.soft_on_frames)):
            self.state = "SOFT"
        elif self.state == "SOFT" and guard <= float(self.config.guard_soft_off):
            self.state = "SAFE"

        if self.state == "HARD":
            return TemporalConflictDecision(
                mode="HARD",
                alpha_scale=0.0,
                reason=f"hard_conflict core={core:.4f}",
                stats=st,
            )
        if self.state == "SOFT":
            return TemporalConflictDecision(
                mode="SOFT",
                alpha_scale=float(np.clip(self.config.soft_alpha, 0.0, 1.0)),
                reason=f"soft_conflict guard={guard:.4f}",
                stats=st,
            )
        return TemporalConflictDecision(
            mode="SAFE",
            alpha_scale=1.0,
            reason="safe",
            stats=st,
        )
