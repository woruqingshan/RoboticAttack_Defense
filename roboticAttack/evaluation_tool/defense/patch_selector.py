# -*- coding: utf-8 -*-
"""
PatchSelector (Step 2): Patch candidate selection using geometric prior, no PRAC.

Filters top_k candidates by overlap with the projected gripper / arm protection masks.
The older coarse GridBox input is kept as a compatibility fallback, but mask overlap is
now the primary filtering signal. Applies tau_patch_strength so weak anomalies yield
NO_PATCH. Outputs: NO_PATCH (no purify), PATCH_FOUND (lock best ROI), or
NEAR_TASK_PATCH (lock but force mask refinement in Step 4).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math

import numpy as np

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


def _roi_grid_dict(
    roi: Optional[GridBox],
    *,
    grid_h: Optional[int] = None,
    grid_w: Optional[int] = None,
) -> Optional[Dict[str, int]]:
    if roi is None:
        return None
    out = {
        "gx0": int(roi.gx0),
        "gy0": int(roi.gy0),
        "gx1": int(roi.gx1),
        "gy1": int(roi.gy1),
    }
    if grid_w is not None:
        out["gw"] = int(grid_w)
    if grid_h is not None:
        out["gh"] = int(grid_h)
    return out


def _compute_area_ratio(roi: GridBox, grid_h: int = 16, grid_w: int = 16) -> float:
    if grid_h <= 0 or grid_w <= 0:
        return 0.0
    w = max(0, int(roi.gx1) - int(roi.gx0))
    h = max(0, int(roi.gy1) - int(roi.gy0))
    area = float(w * h)
    denom = float(grid_h * grid_w)
    if denom <= 0:
        return 0.0
    return float(area / denom)


def _compute_aspect(roi: GridBox) -> float:
    w = float(max(0, int(roi.gx1) - int(roi.gx0)))
    h = float(max(0, int(roi.gy1) - int(roi.gy0)))
    if w <= 0.0 or h <= 0.0:
        return 0.0
    return float(w / max(h, 1e-12))


def _compute_size_prior(area_ratio: float, expected: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    z = abs(float(area_ratio) - float(expected)) / float(sigma)
    return float(math.exp(-z))


def _compute_square_prior(aspect: float, sigma: float) -> float:
    if sigma <= 0.0 or aspect <= 0.0:
        return 0.0
    z = abs(float(math.log(float(aspect)))) / float(sigma)
    return float(math.exp(-z))


def _compute_corner_prior(
    roi: GridBox,
    grid_h: int = 16,
    grid_w: int = 16,
    corner_type: str = "top_right",
    sigma: float = 0.35,
    enabled: bool = False,
) -> float:
    if not enabled or sigma <= 0.0:
        return 0.0
    if str(corner_type) == "none":
        return 0.0
    if grid_h <= 0 or grid_w <= 0:
        return 0.0
    cx = (float(roi.gx0) + float(roi.gx1)) * 0.5
    cy = (float(roi.gy0) + float(roi.gy1)) * 0.5
    if corner_type == "top_left":
        tx, ty = 0.0, 0.0
    elif corner_type == "bottom_left":
        tx, ty = 0.0, float(grid_h - 1)
    elif corner_type == "bottom_right":
        tx, ty = float(grid_w - 1), float(grid_h - 1)
    else:
        tx, ty = float(grid_w - 1), 0.0
    dx = cx - tx
    dy = cy - ty
    norm = float(max(grid_w - 1, grid_h - 1, 1))
    dist = float(math.sqrt(dx * dx + dy * dy) / norm)
    z = dist / float(sigma)
    return float(math.exp(-(z * z)))


def _compute_candidate_evidence_score(
    roi: GridBox,
    score_grid: np.ndarray,
    config: "PatchSelectorConfig",
) -> Dict[str, float]:
    """Compute selector-owned evidence score for one candidate ROI."""
    eps = 1e-8
    grid = np.asarray(score_grid, dtype=np.float32)
    if grid.ndim != 2:
        return {
            "mass": 0.0,
            "peak": 0.0,
            "area_ratio": 0.0,
            "density": 0.0,
            "aspect": 0.0,
            "shape_penalty": 0.0,
            "final_score": 0.0,
        }

    gh, gw = int(grid.shape[0]), int(grid.shape[1])
    y0, y1 = max(0, int(roi.gy0)), min(int(roi.gy1), gh)
    x0, x1 = max(0, int(roi.gx0)), min(int(roi.gx1), gw)
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    grid_area = float(max(gh * gw, 1))
    area_ratio = float((width * height) / grid_area)
    aspect = float(width / max(float(height), eps)) if width > 0 and height > 0 else 0.0
    shape_penalty = float(abs(math.log(aspect + eps))) if aspect > 0.0 else 0.0

    total = float(grid.sum()) + eps
    global_peak = float(grid.max()) + eps if grid.size > 0 else eps
    if width <= 0 or height <= 0:
        mass = 0.0
        peak = 0.0
    else:
        roi_grid = grid[y0:y1, x0:x1]
        mass = float(roi_grid.sum() / total)
        peak = float(roi_grid.max() / global_peak) if roi_grid.size > 0 else 0.0
    density = float(mass / (area_ratio + eps))
    final_score = (
        float(config.final_w_mass) * mass
        + float(config.final_w_density) * (density / (density + 1.0))
        + float(config.final_w_peak) * peak
        - float(config.final_w_area) * math.sqrt(max(area_ratio, 0.0))
        - float(config.final_w_shape) * shape_penalty
    )
    return {
        "mass": float(mass),
        "peak": float(peak),
        "area_ratio": float(area_ratio),
        "density": float(density),
        "aspect": float(aspect),
        "shape_penalty": float(shape_penalty),
        "final_score": float(final_score),
    }

@dataclass
class PatchSelectResult:
    verdict: str           # "NO_PATCH", "PATCH_FOUND", or "NEAR_TASK_PATCH"
    roi: Optional[GridBox]
    score: float           # The anomaly score / mass of the selected ROI
    reason: str
    debug: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatchSelectorConfig:
    tau_g: float = 0.3                # Overlap threshold with G_grid; candidates with IoU > tau_g are excluded
    tau_arm: float = 0.3              # Overlap threshold with arm_region_grid when provided; same semantics as tau_g
    tau_patch_strength: float = 0.05  # Minimum anomaly mass required to be considered a valid patch
    near_task_tau: float = 0.08       # If all overlap > tau_g, but mass >= near_task_tau, mark as NEAR_TASK_PATCH
    allow_near_task_patch: bool = False
    selector_debug_enabled: bool = False
    selector_debug_topk: int = 3
    expected_patch_area_ratio: float = 0.04
    patch_area_sigma: float = 0.03
    patch_aspect_sigma: float = 0.4
    corner_prior_enabled: bool = False
    corner_prior_type: str = "top_right"
    corner_prior_sigma: float = 0.35
    final_w_mass: float = 1.0
    final_w_density: float = 0.5
    final_w_peak: float = 0.3
    final_w_area: float = 0.2
    final_w_shape: float = 0.2
    use_final_score: bool = True

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
        score_grid: Optional[np.ndarray] = None,
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
            score_grid: Optional evidence grid used by the selector to re-score
                candidate ROIs. If absent, localizer scores are used as before.

        Returns:
            PatchSelectResult indicating the decision.
        """
        score_grid_arr = None
        if score_grid is not None and bool(getattr(self.config, "use_final_score", True)):
            try:
                score_grid_arr = np.asarray(score_grid, dtype=np.float32)
                if score_grid_arr.ndim != 2:
                    score_grid_arr = None
            except Exception:
                score_grid_arr = None
        use_final_score = score_grid_arr is not None

        selector_debug = {
            "enabled": bool(getattr(self.config, "selector_debug_enabled", False)),
            "use_final_score": bool(use_final_score),
            "topk": [],
            "selected": None,
            "selected_bucket": None,
            "selected_rank_input": None,
            "selected_raw_score": None,
            "selected_diagnostic_score": None,
            "selected_final_score": None,
            "selected_evidence_mass": None,
            "selected_density": None,
            "selected_peak": None,
            "selected_area_ratio": None,
            "selected_shape_penalty": None,
            "verdict": None,
            "reason": None,
        }

        if not top_k_candidates:
            selector_debug["verdict"] = "NO_PATCH"
            selector_debug["reason"] = "No candidates from localizer"
            return PatchSelectResult(
                verdict="NO_PATCH",
                roi=None,
                score=0.0,
                reason="No candidates from localizer",
                debug=selector_debug,
            )

        filtered_candidates = []
        near_task_candidates = []
        debug_topk_limit = int(max(0, getattr(self.config, "selector_debug_topk", 3)))
        debug_enabled = bool(getattr(self.config, "selector_debug_enabled", False))
        debug_by_roi = {}
        grid_h = 16
        grid_w = 16
        if score_grid_arr is not None:
            grid_h = int(score_grid_arr.shape[0])
            grid_w = int(score_grid_arr.shape[1])
        else:
            for grid_mask in (
                gripper_core_grid_mask,
                gripper_guard_grid_mask,
                arm_core_grid_mask,
                arm_guard_grid_mask,
            ):
                if grid_mask is None or not hasattr(grid_mask, "shape"):
                    continue
                shape = getattr(grid_mask, "shape", None)
                if isinstance(shape, tuple) and len(shape) == 2:
                    grid_h = int(shape[0])
                    grid_w = int(shape[1])
                    break

        def _roi_key(roi: GridBox) -> Tuple[int, int, int, int]:
            return (int(roi.gx0), int(roi.gy0), int(roi.gx1), int(roi.gy1))

        def _sort_score(candidate: Dict[str, Any]) -> float:
            final_score = candidate.get("final_score")
            if final_score is not None:
                return float(final_score)
            return float(candidate.get("raw_score", 0.0))

        def _fill_selected_debug(candidate: Dict[str, Any], bucket: str) -> None:
            roi = candidate["roi"]
            selector_debug["selected"] = _roi_grid_dict(roi, grid_h=grid_h, grid_w=grid_w)
            selector_debug["selected_bucket"] = bucket
            selector_debug["selected_raw_score"] = float(candidate["raw_score"])
            selector_debug["selected_final_score"] = candidate.get("final_score")
            selector_debug["selected_evidence_mass"] = candidate.get("evidence_mass")
            selector_debug["selected_density"] = candidate.get("density")
            selector_debug["selected_peak"] = candidate.get("peak")
            selector_debug["selected_area_ratio"] = candidate.get("area_ratio")
            selector_debug["selected_shape_penalty"] = candidate.get("shape_penalty")
            selected_entry = debug_by_roi.get(_roi_key(roi))
            if selected_entry is not None:
                selector_debug["selected_rank_input"] = selected_entry.get("rank_input")
                selector_debug["selected_diagnostic_score"] = selected_entry.get("diagnostic_score")

        def _reason_ious(candidate: Dict[str, Any]) -> str:
            reason_ious = f"iou_G={candidate['iou_g']:.3f}"
            if gripper_core_grid_mask is not None:
                reason_ious += f" gripper_core={candidate['gripper_core_overlap']:.3f}"
            if gripper_guard_grid_mask is not None:
                reason_ious += f" gripper_guard={candidate['gripper_guard_overlap']:.3f}"
            if arm_region_grid is not None:
                reason_ious += f" iou_arm={candidate['iou_arm']:.3f}"
            if arm_core_grid_mask is not None:
                reason_ious += f" arm_core={candidate['arm_core_overlap']:.3f}"
            if arm_guard_grid_mask is not None:
                reason_ious += f" arm_guard={candidate['arm_guard_overlap']:.3f}"
            return reason_ious

        # 1. Geometric Filtering (G_grid and optionally arm_region_grid)
        for rank_input, (roi, raw_score) in enumerate(top_k_candidates, start=1):
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

            area_ratio = _compute_area_ratio(roi, grid_h=grid_h, grid_w=grid_w)
            aspect = _compute_aspect(roi)
            evidence = {
                "mass": None,
                "peak": None,
                "area_ratio": float(area_ratio),
                "density": None,
                "aspect": float(aspect),
                "shape_penalty": None,
                "final_score": None,
            }
            if use_final_score:
                evidence = _compute_candidate_evidence_score(roi, score_grid_arr, self.config)

            candidate_strength = (
                float(evidence["mass"])
                if use_final_score and evidence.get("mass") is not None
                else float(raw_score)
            )
            candidate = {
                "roi": roi,
                "raw_score": float(raw_score),
                "final_score": evidence.get("final_score") if use_final_score else None,
                "evidence_mass": evidence.get("mass") if use_final_score else None,
                "density": evidence.get("density") if use_final_score else None,
                "peak": evidence.get("peak") if use_final_score else None,
                "area_ratio": float(evidence.get("area_ratio", area_ratio)),
                "aspect": float(evidence.get("aspect", aspect)),
                "shape_penalty": evidence.get("shape_penalty") if use_final_score else None,
                "candidate_strength": float(candidate_strength),
                "iou_g": float(iou_g),
                "gripper_core_overlap": float(overlap_gripper_core),
                "gripper_guard_overlap": float(overlap_gripper_guard),
                "iou_arm": float(iou_arm),
                "arm_core_overlap": float(overlap_arm_core),
                "arm_guard_overlap": float(overlap_arm_guard),
            }

            if debug_enabled:
                size_prior = _compute_size_prior(
                    area_ratio,
                    getattr(self.config, "expected_patch_area_ratio", 0.04),
                    getattr(self.config, "patch_area_sigma", 0.03),
                )
                square_prior = _compute_square_prior(
                    aspect,
                    getattr(self.config, "patch_aspect_sigma", 0.4),
                )
                corner_prior = _compute_corner_prior(
                    roi,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    corner_type=str(getattr(self.config, "corner_prior_type", "top_right")),
                    sigma=float(getattr(self.config, "corner_prior_sigma", 0.35)),
                    enabled=bool(getattr(self.config, "corner_prior_enabled", False)),
                )
                diagnostic_score = (
                    float(raw_score)
                    + 0.35 * float(size_prior)
                    + 0.25 * float(square_prior)
                    + 0.30 * float(corner_prior)
                    - 1.00 * float(overlap_arm_core)
                    - 0.80 * float(overlap_arm_guard)
                    - 0.80 * float(overlap_gripper_core)
                    - 0.50 * float(overlap_gripper_guard)
                )
                bucket = "near_task" if (over_g or over_arm) else "filtered"
                debug_entry = {
                    "rank_input": int(rank_input),
                    "roi_grid": _roi_grid_dict(roi, grid_h=grid_h, grid_w=grid_w),
                    "raw_score": float(raw_score),
                    "final_score": candidate.get("final_score"),
                    "evidence_mass": candidate.get("evidence_mass"),
                    "density": candidate.get("density"),
                    "peak": candidate.get("peak"),
                    "diagnostic_score": float(diagnostic_score),
                    "area_ratio": float(candidate["area_ratio"]),
                    "aspect": float(candidate["aspect"]),
                    "shape_penalty": candidate.get("shape_penalty"),
                    "candidate_strength": float(candidate_strength),
                    "size_prior": float(size_prior),
                    "square_prior": float(square_prior),
                    "corner_prior": float(corner_prior),
                    "iou_g": float(iou_g),
                    "gripper_core_overlap": float(overlap_gripper_core),
                    "gripper_guard_overlap": float(overlap_gripper_guard),
                    "iou_arm": float(iou_arm),
                    "arm_core_overlap": float(overlap_arm_core),
                    "arm_guard_overlap": float(overlap_arm_guard),
                    "over_g": bool(over_g),
                    "over_arm": bool(over_arm),
                    "bucket": bucket,
                }
                if len(selector_debug["topk"]) < debug_topk_limit:
                    selector_debug["topk"].append(debug_entry)
                debug_by_roi[_roi_key(roi)] = debug_entry

            if not over_g and not over_arm:
                filtered_candidates.append(candidate)
            else:
                near_task_candidates.append(candidate)

        # 2. Check filtered candidates (those away from the gripper and arm)
        if filtered_candidates:
            filtered_candidates.sort(key=_sort_score, reverse=True)
            best = filtered_candidates[0]
            best_roi = best["roi"]
            best_score = _sort_score(best)
            best_strength = float(best["candidate_strength"])

            # Anomaly strength threshold check
            if best_strength < self.config.tau_patch_strength:
                selector_debug["verdict"] = "NO_PATCH"
                selector_debug["reason"] = (
                    f"Best isolated candidate strength {best_strength:.3f} < tau_patch_strength {self.config.tau_patch_strength:.3f}"
                )
                _fill_selected_debug(best, "filtered_rejected")
                return PatchSelectResult(
                    verdict="NO_PATCH",
                    roi=None,
                    score=best_score,
                    reason=f"Best isolated candidate strength {best_strength:.3f} < tau_patch_strength {self.config.tau_patch_strength:.3f}",
                    debug=selector_debug,
                )
            reason_ious = _reason_ious(best)
            _fill_selected_debug(best, "filtered")
            selector_debug["verdict"] = "PATCH_FOUND"
            selector_debug["reason"] = (
                f"Isolated patch found: score={best_score:.3f}, strength={best_strength:.3f}, {reason_ious}"
            )
            return PatchSelectResult(
                verdict="PATCH_FOUND",
                roi=best_roi,
                score=best_score,
                reason=f"Isolated patch found: score={best_score:.3f}, strength={best_strength:.3f}, {reason_ious}",
                debug=selector_debug,
            )

        # 3. Near-task exception
        # If all candidates overlap significantly with G_grid or arm_region, but we have strong anomaly signals
        if near_task_candidates:
            near_task_candidates.sort(key=_sort_score, reverse=True)
            best = near_task_candidates[0]
            best_roi = best["roi"]
            best_score = _sort_score(best)
            best_strength = float(best["candidate_strength"])

            if not bool(getattr(self.config, "allow_near_task_patch", False)):
                _fill_selected_debug(best, "near_task_rejected")
                selector_debug["verdict"] = "NO_PATCH"
                selector_debug["reason"] = (
                    f"Near-task candidate rejected by safe-region policy: "
                    f"score={best_score:.3f}, strength={best_strength:.3f}, {_reason_ious(best)}"
                )
                return PatchSelectResult(
                    verdict="NO_PATCH",
                    roi=None,
                    score=best_score,
                    reason=(
                        f"Near-task candidate rejected by safe-region policy: "
                        f"score={best_score:.3f}, strength={best_strength:.3f}, {_reason_ious(best)}"
                    ),
                    debug=selector_debug,
                )

            if best_strength >= self.config.near_task_tau:
                reason_ious = _reason_ious(best)
                _fill_selected_debug(best, "near_task")
                selector_debug["verdict"] = "NEAR_TASK_PATCH"
                selector_debug["reason"] = (
                    f"Near-task patch detected: strength={best_strength:.3f} >= {self.config.near_task_tau:.3f}, "
                    f"score={best_score:.3f}, {reason_ious}"
                )
                return PatchSelectResult(
                    verdict="NEAR_TASK_PATCH",
                    roi=best_roi,
                    score=best_score,
                    reason=(
                        f"Near-task patch detected: strength={best_strength:.3f} >= {self.config.near_task_tau:.3f}, "
                        f"score={best_score:.3f}, {reason_ious}"
                    ),
                    debug=selector_debug,
                )

        # 4. Fallback: all candidates were near the gripper/arm but scores were too low
        selector_debug["verdict"] = "NO_PATCH"
        selector_debug["reason"] = "All candidates were near task/arm area and had low anomaly scores"
        return PatchSelectResult(
            verdict="NO_PATCH",
            roi=None,
            score=0.0,
            reason="All candidates were near task/arm area and had low anomaly scores",
            debug=selector_debug,
        )
