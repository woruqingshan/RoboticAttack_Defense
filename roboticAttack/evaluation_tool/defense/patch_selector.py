# -*- coding: utf-8 -*-
"""
PatchSelector (Step 2): Patch candidate selection using geometric prior, no PRAC.

Filters top_k candidates by overlap with G_grid (exclude if overlap > tau_g) and
optionally with arm_region_grid (exclude if overlap > tau_arm). Applies
tau_patch_strength so weak anomalies yield NO_PATCH. Outputs: NO_PATCH (no purify),
PATCH_FOUND (lock best ROI), or NEAR_TASK_PATCH (lock but force mask refinement in Step 4).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from .temporal import GridBox, grid_iou

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

class PatchSelector:
    def __init__(self, config: PatchSelectorConfig):
        self.config = config

    def select(
        self,
        top_k_candidates: List[Tuple[GridBox, float]],
        G_grid: GridBox,
        arm_region_grid: Optional[GridBox] = None,
    ) -> PatchSelectResult:
        """
        Selects the true patch ROI from candidates using geometric constraints.

        A candidate is considered "away from task/arm" only if IoU with G_grid <= tau_g
        and (when arm_region_grid is provided) IoU with arm_region_grid <= tau_arm.

        Args:
            top_k_candidates: List of (GridBox, anomaly_score) from localizer.
            G_grid: The grid-level protection zone of the gripper/task area.
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

            iou_g = grid_iou(roi, G_grid)
            iou_arm = grid_iou(roi, arm_region_grid) if arm_region_grid is not None else 0.0

            over_g = iou_g > self.config.tau_g
            over_arm = arm_region_grid is not None and iou_arm > self.config.tau_arm
            if not over_g and not over_arm:
                filtered_candidates.append((roi, float(score), iou_g, iou_arm))
            else:
                near_task_candidates.append((roi, float(score), iou_g, iou_arm))

        # 2. Check filtered candidates (those away from the gripper and arm)
        if filtered_candidates:
            # Sort by anomaly score (descending)
            filtered_candidates.sort(key=lambda x: x[1], reverse=True)
            best_roi, best_score, best_iou_g, best_iou_arm = filtered_candidates[0]

            # Anomaly strength threshold check
            if best_score < self.config.tau_patch_strength:
                return PatchSelectResult(
                    verdict="NO_PATCH",
                    roi=None,
                    score=best_score,
                    reason=f"Best isolated candidate score {best_score:.3f} < tau_patch_strength {self.config.tau_patch_strength:.3f}"
                )
            reason_ious = f"iou_G={best_iou_g:.3f}"
            if arm_region_grid is not None:
                reason_ious += f" iou_arm={best_iou_arm:.3f}"
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
            best_roi, best_score, best_iou_g, best_iou_arm = near_task_candidates[0]

            if best_score >= self.config.near_task_tau:
                reason_ious = f"iou_G={best_iou_g:.3f}"
                if arm_region_grid is not None:
                    reason_ious += f" iou_arm={best_iou_arm:.3f}"
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
