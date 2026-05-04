# -*- coding: utf-8 -*-
"""
PatchSelector (Step 2): Patch candidate selection using geometric prior, no PRAC.

Filters top_k candidates by overlap with the projected gripper / arm protection masks.
The older coarse GridBox input is kept as a compatibility fallback, but mask overlap is
now the primary filtering signal. Applies tau_patch_strength so weak anomalies yield
NO_PATCH. Outputs: NO_PATCH (no purify), PATCH_FOUND (lock best ROI), or
NEAR_TASK_PATCH (lock but force mask refinement in Step 4).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from .temporal import GridBox, grid_iou


def _grid_mask_overlap_ratio(roi: GridBox, grid_mask: Optional[object]) -> float:
    """Compute the fraction of ROI grid cells covered by a boolean grid mask."""
    if grid_mask is None:
        return 0.0
    try:
        mask = grid_mask.astype(bool)
    except Exception:
        return 0.0
    if mask.ndim != 2:
        return 0.0
    y0, y1 = max(0, int(roi.gy0)), min(int(roi.gy1), int(mask.shape[0]))
    x0, x1 = max(0, int(roi.gx0)), min(int(roi.gx1), int(mask.shape[1]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi_area = float((x1 - x0) * (y1 - y0))
    return float(mask[y0:y1, x0:x1].sum() / max(roi_area, 1.0))

@dataclass
class PatchSelectResult:
    verdict: str           # "NO_PATCH", "PATCH_FOUND", or "NEAR_TASK_PATCH"
    roi: Optional[GridBox]
    score: float           # The anomaly score / mass of the selected ROI
    reason: str

@dataclass
class PatchSelectorConfig:
    tau_g: float = 0.3                # Overlap threshold with G_grid; candidates with IoU > tau_g are excluded
    tau_arm: float = 0.3              # Overlap threshold with arm_region_grid when provided; same semantics as tau_g
    tau_patch_strength: float = 0.05  # Minimum anomaly mass required to be considered a valid patch
    near_task_tau: float = 0.08       # If all overlap > tau_g, but mass >= near_task_tau, mark as NEAR_TASK_PATCH
    allow_near_task_patch: bool = False

class PatchSelector:
    def __init__(self, config: PatchSelectorConfig):
        self.config = config

    def select(
        self,
        top_k_candidates: List[Tuple[GridBox, float]],
        G_grid: Optional[GridBox] = None,
        gripper_core_grid_mask: Optional[object] = None,
        gripper_guard_grid_mask: Optional[object] = None,
        arm_region_grid: Optional[GridBox] = None,
        arm_core_grid_mask: Optional[object] = None,
        arm_guard_grid_mask: Optional[object] = None,
    ) -> PatchSelectResult:
        """
        Selects the true patch ROI from candidates using geometric constraints.

        A candidate is considered "away from task/arm" only if its overlap with the
        gripper protection region is <= tau_g and its overlap with the arm protection
        region is <= tau_arm.

        Args:
            top_k_candidates: List of (GridBox, anomaly_score) from localizer.
            G_grid: Compatibility fallback coarse gripper/task protection zone.
            gripper_core_grid_mask: Boolean grid mask for the hard gripper exclusion area.
            gripper_guard_grid_mask: Boolean grid mask for the wider gripper guard area.
            arm_region_grid: Optional grid-level arm region; when provided, candidates
                overlapping with it above tau_arm are also treated as near-task.

        Returns:
            PatchSelectResult indicating the decision.
        """
        if not top_k_candidates:
            return PatchSelectResult(verdict="NO_PATCH", roi=None, score=0.0, reason="No candidates from localizer")

        filtered_candidates = []
        near_task_candidates = []

        # 1. Geometric Filtering (G_grid and optionally arm_region_grid)
        for roi, score in top_k_candidates:
            if roi is None:
                continue

            iou_g = grid_iou(roi, G_grid) if G_grid is not None else 0.0
            overlap_gripper_core = _grid_mask_overlap_ratio(roi, gripper_core_grid_mask)
            overlap_gripper_guard = _grid_mask_overlap_ratio(roi, gripper_guard_grid_mask)
            iou_arm = grid_iou(roi, arm_region_grid) if arm_region_grid is not None else 0.0
            overlap_arm_core = _grid_mask_overlap_ratio(roi, arm_core_grid_mask)
            overlap_arm_guard = _grid_mask_overlap_ratio(roi, arm_guard_grid_mask)

            over_g = (
                (G_grid is not None and iou_g > self.config.tau_g)
                or overlap_gripper_core > self.config.tau_g
                or overlap_gripper_guard > self.config.tau_g
            )
            over_arm = (
                (arm_region_grid is not None and iou_arm > self.config.tau_arm)
                or overlap_arm_core > self.config.tau_arm
                or overlap_arm_guard > self.config.tau_arm
            )
            if not over_g and not over_arm:
                filtered_candidates.append(
                    (
                        roi,
                        float(score),
                        iou_g,
                        overlap_gripper_core,
                        overlap_gripper_guard,
                        iou_arm,
                        overlap_arm_core,
                        overlap_arm_guard,
                    )
                )
            else:
                near_task_candidates.append(
                    (
                        roi,
                        float(score),
                        iou_g,
                        overlap_gripper_core,
                        overlap_gripper_guard,
                        iou_arm,
                        overlap_arm_core,
                        overlap_arm_guard,
                    )
                )

        # 2. Check filtered candidates (those away from the gripper and arm)
        if filtered_candidates:
            # Sort by anomaly score (descending)
            filtered_candidates.sort(key=lambda x: x[1], reverse=True)
            (
                best_roi,
                best_score,
                best_iou_g,
                best_gripper_core,
                best_gripper_guard,
                best_iou_arm,
                best_core,
                best_guard,
            ) = filtered_candidates[0]

            # Anomaly strength threshold check
            if best_score < self.config.tau_patch_strength:
                return PatchSelectResult(
                    verdict="NO_PATCH",
                    roi=None,
                    score=best_score,
                    reason=f"Best isolated candidate score {best_score:.3f} < tau_patch_strength {self.config.tau_patch_strength:.3f}"
                )
            reason_ious = f"iou_G={best_iou_g:.3f}"
            if gripper_core_grid_mask is not None:
                reason_ious += f" gripper_core={best_gripper_core:.3f}"
            if gripper_guard_grid_mask is not None:
                reason_ious += f" gripper_guard={best_gripper_guard:.3f}"
            if arm_region_grid is not None:
                reason_ious += f" iou_arm={best_iou_arm:.3f}"
            if arm_core_grid_mask is not None:
                reason_ious += f" arm_core={best_core:.3f}"
            if arm_guard_grid_mask is not None:
                reason_ious += f" arm_guard={best_guard:.3f}"
            return PatchSelectResult(
                verdict="PATCH_FOUND",
                roi=best_roi,
                score=best_score,
                reason=f"Isolated patch found: score={best_score:.3f}, {reason_ious}"
            )

        # 3. Near-task exception
        # If all candidates overlap significantly with G_grid or arm_region, but we have strong anomaly signals
        if near_task_candidates:
            near_task_candidates.sort(key=lambda x: x[1], reverse=True)
            (
                best_roi,
                best_score,
                best_iou_g,
                best_gripper_core,
                best_gripper_guard,
                best_iou_arm,
                best_core,
                best_guard,
            ) = near_task_candidates[0]

            if not bool(getattr(self.config, "allow_near_task_patch", False)):
                return PatchSelectResult(
                    verdict="NO_PATCH",
                    roi=None,
                    score=float(best_score),
                    reason=(
                        f"Near-task candidate rejected by safe-region policy: "
                        f"score={best_score:.3f}, "
                        f"iou_G={best_iou_g:.3f}, "
                        f"gripper_core={best_gripper_core:.3f}, "
                        f"gripper_guard={best_gripper_guard:.3f}, "
                        f"iou_arm={best_iou_arm:.3f}, "
                        f"arm_core={best_core:.3f}, "
                        f"arm_guard={best_guard:.3f}"
                    ),
                )

            if best_score >= self.config.near_task_tau:
                reason_ious = f"iou_G={best_iou_g:.3f}"
                if gripper_core_grid_mask is not None:
                    reason_ious += f" gripper_core={best_gripper_core:.3f}"
                if gripper_guard_grid_mask is not None:
                    reason_ious += f" gripper_guard={best_gripper_guard:.3f}"
                if arm_region_grid is not None:
                    reason_ious += f" iou_arm={best_iou_arm:.3f}"
                if arm_core_grid_mask is not None:
                    reason_ious += f" arm_core={best_core:.3f}"
                if arm_guard_grid_mask is not None:
                    reason_ious += f" arm_guard={best_guard:.3f}"
                return PatchSelectResult(
                    verdict="NEAR_TASK_PATCH",
                    roi=best_roi,
                    score=best_score,
                    reason=f"Near-task patch detected: score={best_score:.3f} >= {self.config.near_task_tau:.3f}, {reason_ious}"
                )

        # 4. Fallback: all candidates were near the gripper/arm but scores were too low
        return PatchSelectResult(
            verdict="NO_PATCH",
            roi=None,
            score=0.0,
            reason="All candidates were near task/arm area and had low anomaly scores"
        )
