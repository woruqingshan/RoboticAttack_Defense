# -*- coding: utf-8 -*-
"""
PatchSelector (Step 2): Patch candidate selection using geometric prior, no PRAC.

Filters top_k candidates by overlap with G_grid (exclude if overlap > tau_g). Applies
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
    tau_patch_strength: float = 0.05  # Minimum anomaly mass required to be considered a valid patch
    near_task_tau: float = 0.08       # If all overlap > tau_g, but mass >= near_task_tau, mark as NEAR_TASK_PATCH

class PatchSelector:
    def __init__(self, config: PatchSelectorConfig):
        self.config = config

    def select(
        self, 
        top_k_candidates: List[Tuple[GridBox, float]], 
        G_grid: GridBox
    ) -> PatchSelectResult:
        """
        Selects the true patch ROI from candidates using geometric constraints.
        
        Args:
            top_k_candidates: List of (GridBox, anomaly_score) from localizer.
            G_grid: The grid-level protection zone of the gripper/task area.
            
        Returns:
            PatchSelectResult indicating the decision.
        """
        if not top_k_candidates:
            return PatchSelectResult(verdict="NO_PATCH", roi=None, score=0.0, reason="No candidates from localizer")

        filtered_candidates = []
        near_task_candidates = []

        # 1. Geometric Filtering
        for roi, score in top_k_candidates:
            if roi is None:
                continue
                
            iou = grid_iou(roi, G_grid)
            
            if iou <= self.config.tau_g:
                filtered_candidates.append((roi, float(score), iou))
            else:
                near_task_candidates.append((roi, float(score), iou))

        # 2. Check filtered candidates (those away from the gripper)
        if filtered_candidates:
            # Sort by anomaly score (descending)
            filtered_candidates.sort(key=lambda x: x[1], reverse=True)
            best_roi, best_score, best_iou = filtered_candidates[0]
            
            # Anomaly strength threshold check
            if best_score < self.config.tau_patch_strength:
                return PatchSelectResult(
                    verdict="NO_PATCH", 
                    roi=None, 
                    score=best_score, 
                    reason=f"Best isolated candidate score {best_score:.3f} < tau_patch_strength {self.config.tau_patch_strength:.3f}"
                )
            else:
                return PatchSelectResult(
                    verdict="PATCH_FOUND", 
                    roi=best_roi, 
                    score=best_score, 
                    reason=f"Isolated patch found: score={best_score:.3f}, iou_G={best_iou:.3f}"
                )

        # 3. Near-task exception
        # If all candidates overlap significantly with G_grid, but we have strong anomaly signals
        if near_task_candidates:
            near_task_candidates.sort(key=lambda x: x[1], reverse=True)
            best_roi, best_score, best_iou = near_task_candidates[0]
            
            if best_score >= self.config.near_task_tau:
                return PatchSelectResult(
                    verdict="NEAR_TASK_PATCH", 
                    roi=best_roi, 
                    score=best_score, 
                    reason=f"Near-task patch detected: score={best_score:.3f} >= {self.config.near_task_tau:.3f}, iou_G={best_iou:.3f}"
                )

        # 4. Fallback: all candidates were near the gripper but scores were too low
        return PatchSelectResult(
            verdict="NO_PATCH", 
            roi=None, 
            score=0.0, 
            reason="All candidates were near task area and had low anomaly scores"
        )
