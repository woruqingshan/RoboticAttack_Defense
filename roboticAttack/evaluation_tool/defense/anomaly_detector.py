"""
Anomaly detection + localization + controller for attention-based patch defense.

This module provides:

Legacy API (known patch location):
- PatchAttentionAnomalyDetector.detect(heatmap, patch_box) -> DetectionResult

Auto-mode pipeline (multimodal geometric prior + PatchSelector, no PRAC in main path):
- Step 0: GripperPrior (optional) yields G_px, G_grid from eef_pos.
- Step 1: PatchAttentionLocalizer on stable grid -> main_roi + top_k outlier candidates.
- Step 2: PatchSelector filters by G_grid overlap and tau_patch_strength; outputs
  NO_PATCH | PATCH_FOUND | NEAR_TASK_PATCH and selected ROI.
- Step 3: Gate by verdict (NO_PATCH -> no purify; PATCH_FOUND/NEAR_TASK_PATCH -> lock ROI).
- Step 4: Mask refinement via verifier.refine_mask_with_constraints(roi, G_px, tau_protect, tau_cover).
- Step 5: ACQUIRE/TRACK; locked ROI is purified every step; optional counterfactual verifier on TRACK.

Key types:
- PatchAttentionLocalizer, TemporalGate, OnlinePatchDefenseController (optional GripperPrior, PatchSelector).
- DefenseDecision (should_purify, roi_box, phase, patch_verdict, gripper_box, ...).
- UnifiedDefenseInterface: one-step interface (known vs auto mode).

Design goals:
- Backward compatible: known-mode and auto without gripper prior still work.
- Minimal deps: NumPy; optional GripperPrior/PatchSelector for multimodal path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

# --- NEW: decoupled modules ---
from .localizer import TemporalPatchAttentionLocalizer, TemporalLocalizeResult
from .temporal import (
    GridBox,
    RunningStats2D,
    TemporalGate,
    GateDecision,
    grid_iou,
)
from .verifier import VerifierProtocol, roi_mass as heatmap_roi_mass, refine_mask_with_constraints
from .gripper_prior import GripperPrior
from .arm_skeleton_prior import ArmSkeletonPrior, GeometryRuntimeContext
from .patch_selector import PatchSelector
from .safety_region import SafetyRegionBuilder
from .pixel_mask_refiner import PixelMaskRefiner
from .temporal_conflict import TemporalConflictResolver

# PRAC checker (optional import to avoid circular dependency)
try:
    from .prac_checker import PRACChecker, PRACConfig
except ImportError:
    PRACChecker = None
    PRACConfig = None

if TYPE_CHECKING:  # pragma: no cover
    # Only for type hints; avoid runtime circular imports.
    from .online_defense import OnlineAttentionHook


# =============================================================================
# Legacy data types (kept as-is for backward compatibility)
# =============================================================================


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
    """Legacy detector: requires a known patch box (oracle location)."""

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


# =========================
# Auto localization + smoothing control (NEW)
# =========================

def _pad_grid_box(box: GridBox, pad_cells: int, *, gw: int, gh: int) -> GridBox:
    """Pad a GridBox by `pad_cells` and clamp to the grid shape.

    Note:
        `temporal.GridBox` intentionally does not include `.pad()` to keep it minimal.
        This helper is used to preserve the old behavior of expanding the ROI slightly.
    """
    p = int(max(0, pad_cells))
    b = GridBox(gx0=int(box.gx0) - p, gy0=int(box.gy0) - p, gx1=int(box.gx1) + p, gy1=int(box.gy1) + p)
    return b.clamp(gw=int(gw), gh=int(gh))


class PatchAttentionLocalizer:
    """
    Auto-mode localizer (v3): temporal mainland-vs-island localizer.

    This keeps the old class name (`PatchAttentionLocalizer`) so existing scripts
    (e.g., `run_libero_eval_args_geo_batch.py`) don't break, but it **no longer**
    uses the legacy single-frame "small & dense" heuristic.

    Internally it delegates to `localizer.TemporalPatchAttentionLocalizer` and returns
    `TemporalLocalizeResult` (main_roi/outlier_roi/outlier_score/debug).
    """

    def __init__(
        self,
        top_p: float = 0.07,
        min_area_frac: float = 0.003,
        max_area_frac: float = 0.12,
        connectivity: int = 4,   # kept for compatibility (unused in v3)
        pad_cells: int = 1,
        eps: float = 1e-12,
    ) -> None:
        self.top_p = float(top_p)
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.connectivity = int(connectivity)
        self.pad_cells = int(pad_cells)
        self.eps = float(eps)

        # Temporal mainland-vs-island localizer (decoupled, pure numpy)
        self._impl = TemporalPatchAttentionLocalizer(
            top_p=self.top_p,
            min_area=float(self.min_area_frac),
            max_area=float(self.max_area_frac),
            eps=float(max(self.eps, 1e-8)),
        )

    def reset(self) -> None:
        self._impl.reset()

    def localize(self, stable_grid: np.ndarray) -> TemporalLocalizeResult:
        """Localize mainland ROI + outlier island ROI on a (stable) grid."""
        if stable_grid.ndim != 2:
            raise ValueError(f"stable_grid must be 2D, got shape={stable_grid.shape}")
        gh, gw = int(stable_grid.shape[0]), int(stable_grid.shape[1])

        r = self._impl.localize(stable_grid)

        # Preserve old behavior: slightly pad the candidate outlier ROI (mask more context).
        if r.outlier_roi is not None:
            outlier = _pad_grid_box(r.outlier_roi, self.pad_cells, gw=gw, gh=gh)
        else:
            outlier = None

        # Also pad Top-K candidates (if available) for consistent downstream masking.
        padded_top_k = []
        raw_top_k = getattr(r, "top_k_candidates", [])
        if isinstance(raw_top_k, list):
            for item in raw_top_k:
                try:
                    roi_k, score_k = item
                except Exception:
                    continue
                if roi_k is None:
                    continue
                padded_top_k.append((_pad_grid_box(roi_k, self.pad_cells, gw=gw, gh=gh), float(score_k)))

        # NOTE: we intentionally do not pad the mainland ROI; it is used for debugging
        # and for future verifier/task-preservation checks.
        # Create a new TemporalLocalizeResult with padded outlier_roi, preserving all other fields from r.
        return TemporalLocalizeResult(
            main_roi=r.main_roi,
            outlier_roi=outlier,
            outlier_score=float(r.outlier_score),
            threshold=float(r.threshold),
            chosen_label=r.chosen_label,
            main_label=r.main_label,
            reason=str(r.reason),
            debug=dict(r.debug) if isinstance(r.debug, dict) else {},
            top_k_candidates=padded_top_k,
        )


@dataclass(frozen=True)
class DefenseDecision:
    """Decision for one step (auto-mode)."""
    # Required fields (no default values) - must come first
    should_purify: bool
    roi_box: Optional[PatchBox]
    grid_box: Optional[GridBox]
    mass_ema: float
    raw_mass: float
    state: str
    reason: str
    
    # Optional fields (with default values) - must come after required fields
    main_grid_box: Optional[GridBox] = None
    outlier_score: float = 0.0
    # Optional: caller can use it as purifier strength (0..1). Not required.
    strength: Optional[float] = None
    verified: Optional[bool] = None
    verify_stats: Optional[dict] = None
    # New: quality signals in heatmap space (aligned with verifier.roi_mass).
    mass_heatmap: Optional[float] = None
    quality_ok: Optional[bool] = None
    quality_reason: str = ""
    gate_checked: Optional[bool] = None
    verdict_code: str = "NA"
    # Control-plane fields (for unified logging/statistics).
    phase: str = "NA"  # "ACQUIRE" | "TRACK" | "KNOWN" | "NA"
    reacquire_needed: Optional[bool] = None
    verify_performed: Optional[bool] = None
    # PRAC fields (new)
    prac_performed: Optional[bool] = None
    prac_verdict: Optional[str] = None  # "PASS" | "REACQUIRE" | "NEAR_OBJECT" | "SKIP"
    prac_odr: Optional[float] = None  # Outlier Dependency Ratio
    prac_mer: Optional[float] = None  # Mainland Erosion Risk
    prac_stats: Optional[dict] = None  # Full PRAC statistics
    
    # New geometric prior fields
    gripper_box: Optional[PatchBox] = None
    patch_verdict: Optional[str] = None  # "NO_PATCH" | "PATCH_FOUND" | "NEAR_TASK_PATCH"
    selector_debug: Optional[dict] = None
    # Arm region (pixel box (x0,y0,x1,y1)) for visualization; from GripperPrior when arm_extend_px > 0
    arm_region_box: Optional[Tuple[int, int, int, int]] = None
    arm_core_box: Optional[Tuple[int, int, int, int]] = None
    arm_guard_box: Optional[Tuple[int, int, int, int]] = None
    joint_points_2d: Optional[List[Tuple[int, int]]] = None
    joint_names_used: Optional[List[str]] = None
    joint_points_render_2d: Optional[List[Tuple[int, int]]] = None
    gripper_points_2d: Optional[List[Tuple[int, int]]] = None
    gripper_points_render_2d: Optional[List[Tuple[int, int]]] = None
    gripper_point_keys_used: Optional[List[str]] = None
    gripper_link_segments_2d: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None
    gripper_link_name_pairs: Optional[List[Tuple[str, str]]] = None
    gripper_link_quads_2d: Optional[List[List[Tuple[int, int]]]] = None
    arm_link_segments_2d: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None
    arm_link_name_pairs: Optional[List[Tuple[str, str]]] = None
    arm_link_quads_2d: Optional[List[List[Tuple[int, int]]]] = None
    roi_mask: Optional[np.ndarray] = None
    conflict_mode: Optional[str] = None
    conflict_reason: str = ""
    conflict_stats: Optional[dict] = None


class OnlinePatchDefenseController:
    """
    Auto-mode controller (v2):
      raw grid -> temporal stable grid -> localize candidate -> ROI track ->
      stable-mass score -> low-frequency pulsed gate -> (optional) counterfactual verify

    NOTE:
      - Uses RunningStats2D to compute stable_score (long-term high + low variance)
        which naturally filters out moving objects like gripper/target.
      - Stationarity checks are replaced by stable_score (more robust).
      - Optional counterfactual verification reduces false positives.
    """

    def __init__(
        self,
        hook: Any,
        localizer: PatchAttentionLocalizer,
        gate: TemporalGate,
        *,
        stats_alpha: float = 0.2,
        # Strength mapping (kept for backward compatibility)
        strength_min: float = 0.35,
        strength_max: float = 0.85,
        # verifier (optional, plugin)
        verifier: Optional[VerifierProtocol] = None,
        verify_every_k: int = 3,          # only verify at low frequency (TRACK stage)
        require_verify: bool = False,
        # v4: hard quality gate to prevent triggering on empty/low-mass ROI
        min_trigger_patch_mass: float = 0.02,
        # v5: quality gate aligned with verifier space (heatmap). If heatmap_fn is provided,
        # this threshold is preferred over min_trigger_patch_mass (grid-space).
        min_trigger_mass_heatmap: float = 0.02,
        quality_mass_source: str = "heatmap",  # "heatmap" | "grid"
        # PRAC checker (optional, plugin)
        prac_checker: Optional[Any] = None,  # PRACChecker
        prac_enabled: bool = True,  # Enable PRAC if prac_checker is provided
        
        # Multimodal prior and selector
        gripper_prior: Optional[GripperPrior] = None,
        arm_skeleton_prior: Optional[ArmSkeletonPrior] = None,
        safety_region_builder: Optional[SafetyRegionBuilder] = None,
        pixel_mask_refiner: Optional[PixelMaskRefiner] = None,
        temporal_conflict_resolver: Optional[TemporalConflictResolver] = None,
        patch_selector: Optional[PatchSelector] = None,
        tau_protect: float = 0.1,  # Maximum allowed overlap ratio with G_px
        tau_cover: float = 0.5,    # Minimum required coverage ratio of original ROI
    ) -> None:
        self.hook = hook
        self.localizer = localizer
        self.gate = gate

        self.stats_alpha = float(stats_alpha)
        self._stats: Optional[RunningStats2D] = None
        self._stats_shape: Optional[Tuple[int, int]] = None

        self.strength_min = float(strength_min)
        self.strength_max = float(strength_max)

        self.verifier = verifier
        self.verify_every_k = int(max(1, verify_every_k))
        self._t = 0

        # NOTE: In the simplified pipeline we split "ACQUIRE" (localize) and "TRACK" (mask + verify).
        # Only when reacquire_needed=True do we run localizer again.
        self.require_verify = bool(require_verify)
        # Hard gate: do not trigger if the localized ROI has too little attention mass.
        self.min_trigger_patch_mass = float(min_trigger_patch_mass)
        self.min_trigger_mass_heatmap = float(min_trigger_mass_heatmap)
        self.quality_mass_source = str(quality_mass_source)

        # PRAC checker
        self.prac_checker = prac_checker
        self.prac_enabled = bool(prac_enabled) and (prac_checker is not None)

        # Multimodal prior
        self.gripper_prior = gripper_prior
        self.arm_skeleton_prior = arm_skeleton_prior
        self.safety_region_builder = safety_region_builder
        self.pixel_mask_refiner = pixel_mask_refiner
        self.temporal_conflict_resolver = temporal_conflict_resolver
        self.patch_selector = patch_selector
        self.tau_protect = float(tau_protect)
        self.tau_cover = float(tau_cover)

        # --- Simplified controller state (ACQUIRE/TRACK) ---
        self._tracking: bool = False
        self._reacquire_needed: bool = True  # start by acquiring an ROI
        self._current_outlier_grid: Optional[GridBox] = None
        self._current_main_grid: Optional[GridBox] = None
        self._current_outlier_score: float = 0.0
        # PRAC state: protected ROIs for negative prior (avoid re-localizing same region)
        self._protected_rois: List[GridBox] = []

        # PRAC lock mode (episode scope):
        # Run localize + PRAC once (first step), lock ROI, then force purify every step.
        self._locked: bool = False
        self._locked_outlier_grid: Optional[GridBox] = None
        self._locked_main_grid: Optional[GridBox] = None
        self._lock_prac_debug: Optional[dict] = None

    def reset(self) -> None:
        self.gate.reset()
        self._stats = None
        self._stats_shape = None
        self._t = 0
        self._tracking = False
        self._reacquire_needed = True
        self._current_outlier_grid = None
        self._current_main_grid = None
        self._current_outlier_score = 0.0
        self._protected_rois = []
        self._locked = False
        self._locked_outlier_grid = None
        self._locked_main_grid = None
        self._lock_prac_debug = None
        # Reset localizer temporal state for a new episode.
        if hasattr(self.localizer, "reset"):
            self.localizer.reset()

    def _mass_to_strength(self, mass_ema: float) -> float:
        """
        Map mass_ema to purifier strength for smoother visuals.
        Caller may ignore this and use fixed strength.
        """
        # Normalize roughly around theta_on..(theta_on*2)
        lo = float(self.gate.theta_on)
        hi = float(max(lo * 2.0, lo + 1e-6))
        t = float(np.clip((mass_ema - lo) / (hi - lo + 1e-12), 0.0, 1.0))
        return float(self.strength_min + (self.strength_max - self.strength_min) * t)

    def step(
        self,
        grid: np.ndarray,
        *,
        image: Optional[np.ndarray] = None,
        purify_fn: Optional[Callable[[np.ndarray, PatchBox], np.ndarray]] = None,
        forward_fn: Optional[Callable[[np.ndarray], Any]] = None,
        heatmap_fn: Optional[Callable[[], np.ndarray]] = None,
        hm_current: Optional[np.ndarray] = None,
        eef_pos: Optional[np.ndarray] = None,
        geometry_ctx: Optional[GeometryRuntimeContext] = None,
    ) -> DefenseDecision:
        """
        Args:
            grid: saliency grid (e.g., 16x16), requires the hook cache already populated.
            image/purify_fn/forward_fn/heatmap_fn: only needed if you enable counterfactual verification.
            eef_pos: End-effector pose array for multimodal geometric prior.
            geometry_ctx: Simulator and camera context for the arm skeleton prior.
        """
        self._t += 1
        if grid.ndim != 2:
            raise ValueError(f"grid must be 2D, got shape={grid.shape}")

        # --- (A) temporal stability ---
        grid_shape = (grid.shape[0], grid.shape[1])
        if self._stats is None or self._stats_shape != grid_shape:
            # Initialize or reinitialize if shape changed
            self._stats = RunningStats2D(shape=grid_shape, alpha=self.stats_alpha)
            self._stats_shape = grid_shape
        st = self._stats.update(grid)
        stable_grid = st.stable  # long-term high + low variance

        # --- Multimodal Geometric Prior (Step 0) ---
        G_px = None
        G_grid = None
        gripper_core_grid_mask = None
        gripper_guard_grid_mask = None
        arm_core_mask_px = None
        arm_guard_mask_px = None
        arm_core_grid_mask = None
        arm_guard_grid_mask = None
        arm_region_grid: Optional[GridBox] = None
        gripper_box = None
        arm_region_box: Optional[Tuple[int, int, int, int]] = None
        arm_core_box: Optional[Tuple[int, int, int, int]] = None
        arm_guard_box: Optional[Tuple[int, int, int, int]] = None
        joint_points_2d: Optional[List[Tuple[int, int]]] = None
        joint_names_used: Optional[List[str]] = None
        joint_points_render_2d: Optional[List[Tuple[int, int]]] = None
        gripper_points_2d: Optional[List[Tuple[int, int]]] = None
        gripper_points_render_2d: Optional[List[Tuple[int, int]]] = None
        gripper_point_keys_used: Optional[List[str]] = None
        gripper_link_segments_2d: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None
        gripper_link_name_pairs: Optional[List[Tuple[str, str]]] = None
        gripper_link_quads_2d: Optional[List[List[Tuple[int, int]]]] = None
        arm_link_segments_2d: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None
        arm_link_name_pairs: Optional[List[Tuple[str, str]]] = None
        arm_link_quads_2d: Optional[List[List[Tuple[int, int]]]] = None
        img_shape = (256, 256)
        if image is not None:
            img_shape = image.shape[:2]
        elif hm_current is not None:
            img_shape = hm_current.shape[:2]
        
        gripper_res = None
        skeleton_res = None
        safety_bundle = None

        if self.gripper_prior is not None and geometry_ctx is not None:
            gripper_res = self.gripper_prior.compute(geometry_ctx, grid.shape)
            if bool(gripper_res.valid) and gripper_res.gripper_guard_box_px is not None and gripper_res.gripper_guard_grid is not None:
                g_px_tuple = gripper_res.gripper_guard_box_px
                G_px = PatchBox(x0=g_px_tuple[0], y0=g_px_tuple[1], x1=g_px_tuple[2], y1=g_px_tuple[3])
                G_grid = gripper_res.gripper_guard_grid
                gripper_core_grid_mask = gripper_res.gripper_core_grid_mask
                gripper_guard_grid_mask = gripper_res.gripper_guard_grid_mask
                gripper_box = G_px
                gripper_points_2d = list(gripper_res.gripper_points_policy_px)
                gripper_points_render_2d = list(gripper_res.gripper_points_render_px)
                gripper_point_keys_used = list(gripper_res.point_keys_used)
                gripper_link_segments_2d = [
                    (tuple(seg.start_point_policy_px), tuple(seg.end_point_policy_px))
                    for seg in gripper_res.gripper_segments_2d
                ]
                gripper_link_name_pairs = [
                    (str(seg.start_point_name), str(seg.end_point_name))
                    for seg in gripper_res.gripper_segments_2d
                ]
                gripper_link_quads_2d = [list(seg.core_quad_xy) for seg in gripper_res.gripper_segments_2d]

        if self.arm_skeleton_prior is not None and geometry_ctx is not None:
            skeleton_res = self.arm_skeleton_prior.compute(geometry_ctx, grid.shape)
            if bool(skeleton_res.valid):
                arm_core_mask_px = skeleton_res.arm_core_mask_px
                arm_guard_mask_px = skeleton_res.arm_guard_mask_px
                arm_core_grid_mask = skeleton_res.arm_core_grid_mask
                arm_guard_grid_mask = skeleton_res.arm_guard_grid_mask
                arm_core_box = skeleton_res.arm_core_box_px
                arm_guard_box = skeleton_res.arm_guard_box_px
                joint_points_2d = list(skeleton_res.joint_points_policy_px)
                joint_names_used = list(skeleton_res.joint_names_used)
                joint_points_render_2d = list(skeleton_res.joint_points_render_px)
                arm_link_segments_2d = [
                    (tuple(seg.start_point_policy_px), tuple(seg.end_point_policy_px))
                    for seg in skeleton_res.link_segments_2d
                ]
                arm_link_name_pairs = [
                    (str(seg.start_joint_name), str(seg.end_joint_name))
                    for seg in skeleton_res.link_segments_2d
                ]
                arm_link_quads_2d = [list(seg.core_quad_xy) for seg in skeleton_res.link_segments_2d]
                # Keep the old arm_region_box field as a guard-region visualization fallback.
                if arm_guard_box is not None:
                    arm_region_box = arm_guard_box

        if self.safety_region_builder is not None:
            safety_bundle = self.safety_region_builder.build(
                policy_hw=img_shape,
                gripper_res=gripper_res,
                arm_res=skeleton_res,
                grid_shape=grid.shape,
            )
            if bool(safety_bundle.valid):
                # Feed bundle back into legacy variables so existing selector/refine paths keep working.
                if gripper_core_grid_mask is None:
                    gripper_core_grid_mask = safety_bundle.masks_grid.get("gripper_core", None)
                if gripper_guard_grid_mask is None:
                    gripper_guard_grid_mask = safety_bundle.masks_grid.get("gripper_guard", None)
                if arm_core_grid_mask is None:
                    arm_core_grid_mask = safety_bundle.masks_grid.get("arm_core", None)
                if arm_guard_grid_mask is None:
                    arm_guard_grid_mask = safety_bundle.masks_grid.get("arm_guard", None)

        # ---------------------------------------------------------------------
        # LOCKED MODE (Multimodal or PRAC):
        # - First step: localize + select best patch ROI, lock it.
        # - Subsequent steps: purify the locked ROI (with mask refinement if applicable).
        # ---------------------------------------------------------------------
        if bool(self._locked) and (self._locked_outlier_grid is not None):
            roi_grid_locked = self._locked_outlier_grid
            main_grid_locked = self._locked_main_grid
            x0, y0, x1, y1 = roi_grid_locked.gx0, roi_grid_locked.gy0, roi_grid_locked.gx1, roi_grid_locked.gy1
            roi_box = self.hook.grid_bbox_to_patch_box(int(x0), int(y0), int(x1), int(y1))
            roi_mask = None
            conflict_mode = None
            conflict_reason = ""
            conflict_stats = None
            should_purify_locked = True
            strength_locked = 1.0

            # --- Mask verification/refinement (Step 4) ---
            if G_px is not None:
                rx0, ry0, rx1, ry1 = refine_mask_with_constraints(
                    initial_roi_px=roi_box, 
                    G_px=G_px, 
                    tau_protect=self.tau_protect, 
                    tau_cover=self.tau_cover
                )
                roi_box = PatchBox(x0=rx0, y0=ry0, x1=rx1, y1=ry1)

            if self.pixel_mask_refiner is not None:
                hm_for_refine = hm_current
                if hm_for_refine is None and heatmap_fn is not None:
                    try:
                        hm_for_refine = heatmap_fn()
                    except Exception:
                        hm_for_refine = None
                if hm_for_refine is not None:
                    try:
                        pmr = self.pixel_mask_refiner.refine(
                            initial_roi_px=roi_box,
                            heatmap=hm_for_refine,
                            safety_bundle=safety_bundle,
                        )
                        if bool(pmr.valid) and pmr.tight_box_xyxy is not None:
                            tx0, ty0, tx1, ty1 = pmr.tight_box_xyxy
                            roi_box = PatchBox(x0=tx0, y0=ty0, x1=tx1, y1=ty1)
                            roi_mask = pmr.mask_px
                    except Exception:
                        pass

            if self.temporal_conflict_resolver is not None and roi_mask is not None:
                try:
                    tc = self.temporal_conflict_resolver.update(roi_mask=roi_mask, safety_bundle=safety_bundle)
                    conflict_mode = tc.mode
                    conflict_reason = tc.reason
                    conflict_stats = dict(tc.stats)
                    if str(tc.mode) == "HARD":
                        should_purify_locked = False
                    elif str(tc.mode) == "SOFT":
                        strength_locked = float(np.clip(tc.alpha_scale, 0.0, 1.0))
                except Exception:
                    pass

            # Best-effort heatmap mass (for logging only; does not affect decision).
            mass_heatmap = None
            if hm_current is not None:
                try:
                    mass_heatmap = float(heatmap_roi_mass(hm_current, roi_box))
                except Exception:
                    mass_heatmap = None
            elif heatmap_fn is not None:
                try:
                    mass_heatmap = float(heatmap_roi_mass(heatmap_fn(), roi_box))
                except Exception:
                    mass_heatmap = None

            return DefenseDecision(
                should_purify=bool(should_purify_locked),
                roi_box=roi_box,
                grid_box=roi_grid_locked,
                main_grid_box=main_grid_locked,
                outlier_score=1.0,
                mass_ema=1.0,
                raw_mass=1.0,
                state="LOCKED",
                reason="locked_force_purify",
                strength=float(strength_locked),
                verified=None,
                verify_stats=None,
                mass_heatmap=mass_heatmap,
                quality_ok=True,
                quality_reason="locked_force",
                gate_checked=False,
                verdict_code="LOCKED",
                phase="LOCKED",
                reacquire_needed=False,
                verify_performed=False,
                prac_performed=False,
                prac_verdict=None,
                prac_odr=None,
                prac_mer=None,
                prac_stats=None,
                gripper_box=gripper_box,
                patch_verdict=getattr(self, "_locked_patch_verdict", "PATCH_FOUND"),
                arm_region_box=arm_region_box,
                arm_core_box=arm_core_box,
                arm_guard_box=arm_guard_box,
                joint_points_2d=joint_points_2d,
                joint_names_used=joint_names_used,
                joint_points_render_2d=joint_points_render_2d,
                gripper_points_2d=gripper_points_2d,
                gripper_points_render_2d=gripper_points_render_2d,
                gripper_point_keys_used=gripper_point_keys_used,
                gripper_link_segments_2d=gripper_link_segments_2d,
                gripper_link_name_pairs=gripper_link_name_pairs,
                gripper_link_quads_2d=gripper_link_quads_2d,
                arm_link_segments_2d=arm_link_segments_2d,
                arm_link_name_pairs=arm_link_name_pairs,
                arm_link_quads_2d=arm_link_quads_2d,
                roi_mask=roi_mask,
                conflict_mode=conflict_mode,
                conflict_reason=str(conflict_reason),
                conflict_stats=conflict_stats,
            )

        if not bool(self._locked):
            # One-time ACQUIRE + Lock.
            tlr = self.localizer.localize(stable_grid)
            top_k_candidates = getattr(tlr, "top_k_candidates", [])
            if (not top_k_candidates) and (tlr.outlier_roi is not None):
                top_k_candidates = [(tlr.outlier_roi, float(tlr.outlier_score))]

            best_roi: Optional[GridBox] = None
            best_reason: str = "lock_no_candidate"
            patch_verdict_str = "NO_PATCH"
            selector_debug = None

            # Step 2: PatchSelector (filter by gripper / arm masks, with G_grid kept as fallback)
            if self.patch_selector is not None and (
                G_grid is not None
                or gripper_core_grid_mask is not None
                or gripper_guard_grid_mask is not None
            ):
                ps_res = self.patch_selector.select(
                    top_k_candidates,
                    G_grid=G_grid,
                    gripper_core_grid_mask=gripper_core_grid_mask,
                    gripper_guard_grid_mask=gripper_guard_grid_mask,
                    arm_region_grid=arm_region_grid,
                    arm_core_grid_mask=arm_core_grid_mask,
                    arm_guard_grid_mask=arm_guard_grid_mask,
                    score_grid=stable_grid,
                )
                patch_verdict_str = ps_res.verdict
                best_roi = ps_res.roi
                best_reason = ps_res.reason
                selector_debug = getattr(ps_res, "debug", None)

                if patch_verdict_str == "NO_PATCH":
                    # Step 3: NO_PATCH -> return without purifying
                    return DefenseDecision(
                        should_purify=False,
                        roi_box=None,
                        grid_box=None,
                        main_grid_box=tlr.main_roi,
                        outlier_score=0.0,
                        mass_ema=0.0,
                        raw_mass=0.0,
                        state="NO_PATCH",
                        reason=best_reason,
                        strength=None,
                        verified=None,
                        verify_stats=None,
                        mass_heatmap=None,
                        quality_ok=None,
                        quality_reason="no_patch_candidate",
                        gate_checked=False,
                        verdict_code="NA",
                        phase="ACQUIRE",
                        reacquire_needed=True,
                        verify_performed=False,
                        prac_performed=False,
                        prac_verdict=None,
                        prac_odr=None,
                        prac_mer=None,
                        prac_stats=None,
                        gripper_box=gripper_box,
                        patch_verdict=patch_verdict_str,
                        selector_debug=selector_debug,
                        arm_region_box=arm_region_box,
                        arm_core_box=arm_core_box,
                        arm_guard_box=arm_guard_box,
                        joint_points_2d=joint_points_2d,
                        joint_names_used=joint_names_used,
                        joint_points_render_2d=joint_points_render_2d,
                        gripper_points_2d=gripper_points_2d,
                        gripper_points_render_2d=gripper_points_render_2d,
                        gripper_point_keys_used=gripper_point_keys_used,
                        gripper_link_segments_2d=gripper_link_segments_2d,
                        gripper_link_name_pairs=gripper_link_name_pairs,
                        gripper_link_quads_2d=gripper_link_quads_2d,
                        arm_link_segments_2d=arm_link_segments_2d,
                        arm_link_name_pairs=arm_link_name_pairs,
                        arm_link_quads_2d=arm_link_quads_2d,
                    )
            else:
                # Fallback: Top-1
                if isinstance(top_k_candidates, list) and top_k_candidates:
                    best_roi, _ = top_k_candidates[0]
                    best_reason = "lock_fallback_top1"
                    patch_verdict_str = "PATCH_FOUND"

            if best_roi is not None:
                # Lock ROI for the rest of the episode.
                self._locked = True
                self._locked_outlier_grid = best_roi
                self._locked_main_grid = tlr.main_roi
                self._locked_patch_verdict = patch_verdict_str

                # Return the forced purify decision immediately.
                x0, y0, x1, y1 = best_roi.gx0, best_roi.gy0, best_roi.gx1, best_roi.gy1
                roi_box = self.hook.grid_bbox_to_patch_box(int(x0), int(y0), int(x1), int(y1))
                roi_mask = None
                conflict_mode = None
                conflict_reason = ""
                conflict_stats = None
                should_purify_locked = True
                strength_locked = 1.0

                # Step 4: Mask Verification/Refinement
                if G_px is not None:
                    rx0, ry0, rx1, ry1 = refine_mask_with_constraints(
                        initial_roi_px=roi_box, 
                        G_px=G_px, 
                        tau_protect=self.tau_protect, 
                        tau_cover=self.tau_cover
                    )
                    roi_box = PatchBox(x0=rx0, y0=ry0, x1=rx1, y1=ry1)

                if self.pixel_mask_refiner is not None:
                    hm_for_refine = hm_current
                    if hm_for_refine is None and heatmap_fn is not None:
                        try:
                            hm_for_refine = heatmap_fn()
                        except Exception:
                            hm_for_refine = None
                    if hm_for_refine is not None:
                        try:
                            pmr = self.pixel_mask_refiner.refine(
                                initial_roi_px=roi_box,
                                heatmap=hm_for_refine,
                                safety_bundle=safety_bundle,
                            )
                            if bool(pmr.valid) and pmr.tight_box_xyxy is not None:
                                tx0, ty0, tx1, ty1 = pmr.tight_box_xyxy
                                roi_box = PatchBox(x0=tx0, y0=ty0, x1=tx1, y1=ty1)
                                roi_mask = pmr.mask_px
                        except Exception:
                            pass

                if self.temporal_conflict_resolver is not None and roi_mask is not None:
                    try:
                        tc = self.temporal_conflict_resolver.update(roi_mask=roi_mask, safety_bundle=safety_bundle)
                        conflict_mode = tc.mode
                        conflict_reason = tc.reason
                        conflict_stats = dict(tc.stats)
                        if str(tc.mode) == "HARD":
                            should_purify_locked = False
                        elif str(tc.mode) == "SOFT":
                            strength_locked = float(np.clip(tc.alpha_scale, 0.0, 1.0))
                    except Exception:
                        pass

                mass_heatmap = None
                if hm_current is not None:
                    try:
                        mass_heatmap = float(heatmap_roi_mass(hm_current, roi_box))
                    except Exception:
                        pass
                elif heatmap_fn is not None:
                    try:
                        mass_heatmap = float(heatmap_roi_mass(heatmap_fn(), roi_box))
                    except Exception:
                        pass

                return DefenseDecision(
                    should_purify=bool(should_purify_locked),
                    roi_box=roi_box,
                    grid_box=best_roi,
                    main_grid_box=tlr.main_roi,
                    outlier_score=1.0,
                    mass_ema=1.0,
                    raw_mass=1.0,
                    state="LOCKED",
                    reason=best_reason,
                    strength=float(strength_locked),
                    verified=None,
                    verify_stats=None,
                    mass_heatmap=mass_heatmap,
                    quality_ok=True,
                    quality_reason="locked_force",
                    gate_checked=False,
                    verdict_code="LOCKED",
                    phase="LOCKED",
                    reacquire_needed=False,
                    verify_performed=False,
                    prac_performed=False,
                    prac_verdict=None,
                    prac_odr=None,
                    prac_mer=None,
                    prac_stats=None,
                    gripper_box=gripper_box,
                    patch_verdict=patch_verdict_str,
                    selector_debug=selector_debug,
                    arm_region_box=arm_region_box,
                    arm_core_box=arm_core_box,
                    arm_guard_box=arm_guard_box,
                    joint_points_2d=joint_points_2d,
                    joint_names_used=joint_names_used,
                    joint_points_render_2d=joint_points_render_2d,
                    gripper_points_2d=gripper_points_2d,
                    gripper_points_render_2d=gripper_points_render_2d,
                    gripper_point_keys_used=gripper_point_keys_used,
                    gripper_link_segments_2d=gripper_link_segments_2d,
                    gripper_link_name_pairs=gripper_link_name_pairs,
                    gripper_link_quads_2d=gripper_link_quads_2d,
                    arm_link_segments_2d=arm_link_segments_2d,
                    arm_link_name_pairs=arm_link_name_pairs,
                    arm_link_quads_2d=arm_link_quads_2d,
                    roi_mask=roi_mask,
                    conflict_mode=conflict_mode,
                    conflict_reason=str(conflict_reason),
                    conflict_stats=conflict_stats,
                )

        # --- (B) ACQUIRE/TRACK controller (Legacy continuous tracking) ---
        # ACQUIRE: run localizer only when needed; TRACK: keep using the last ROI.
        main_grid: Optional[GridBox] = self._current_main_grid
        roi_grid: Optional[GridBox] = self._current_outlier_grid
        raw_outlier_score: float = float(self._current_outlier_score)
        loc_reason = "track"
        loc_debug: dict = {}
        
        # PRAC state
        prac_result = None
        prac_performed = False
        consensus_grid = None  # Shared consensus grid for Top-K parallel evaluation

        if (not bool(self._tracking)) or bool(self._reacquire_needed):
            # ACQUIRE phase: Top-K parallel PRAC evaluation
            # Step 1: Localize to get top-K candidates
            tlr = self.localizer.localize(stable_grid)
            top_k_candidates = getattr(tlr, "top_k_candidates", [])
            main_grid = tlr.main_roi
            
            # Fallback to top-1 if top_k_candidates is empty
            if not top_k_candidates and tlr.outlier_roi is not None:
                top_k_candidates = [(tlr.outlier_roi, float(tlr.outlier_score))]
            
            best_roi: Optional[GridBox] = None
            best_main: Optional[GridBox] = main_grid
            best_score: float = 0.0
            best_reason = "no_candidate"
            best_debug: dict = {}
            
            # Step 2: Top-K parallel PRAC evaluation
            if self.prac_enabled and top_k_candidates and image is not None and forward_fn is not None:
                # Build forward context (clears hook + forward)
                def forward_ctx(img: np.ndarray) -> None:
                    """Forward context: clear hook and run forward."""
                    if hasattr(self.hook, "clear"):
                        self.hook.clear()
                    forward_fn(img)  # Forward pass (action discarded)
                
                # Grid readout function
                def grid_readout() -> np.ndarray:
                    """Read attention grid after forward."""
                    return self.hook.get_saliency_grid()
                
                # Build consensus attention ONCE (N forwards for all candidates)
                # Use the first candidate to build consensus (or use original image)
                consensus_grid = self.prac_checker._build_consensus_attention(
                    image, forward_ctx, grid_readout
                )
                prac_performed = True
                
                # Step 3: Evaluate all top-K candidates in parallel (no additional forwards)
                candidate_results = []
                for candidate_roi, candidate_score in top_k_candidates:
                    # Check if candidate overlaps with protected ROIs (negative prior)
                    overlaps_protected = False
                    for prot_roi in self._protected_rois:
                        iou = grid_iou(candidate_roi, prot_roi)
                        if iou > 0.3:  # Threshold for overlap
                            overlaps_protected = True
                            break
                    
                    if overlaps_protected:
                        continue  # Skip protected candidates
                    
                    # Evaluate candidate using pre-computed consensus (no forward)
                    prac_result_candidate = self.prac_checker.evaluate_candidate_with_consensus(
                        base_grid=grid,
                        consensus_grid=consensus_grid,
                        outlier_roi=candidate_roi,
                        main_roi=main_grid,
                    )
                    
                    candidate_results.append({
                        "roi": candidate_roi,
                        "score": candidate_score,
                        "prac_result": prac_result_candidate,
                        "odr": prac_result_candidate.stats.odr,
                        "mer": prac_result_candidate.stats.mer,
                        "verdict": prac_result_candidate.verdict,
                    })
                
                # Step 4: Select best candidate based on PRAC evaluation
                if candidate_results:
                    # Filter: prefer PASS candidates, then by quality score
                    pass_candidates = [c for c in candidate_results if c["verdict"] == "PASS"]
                    near_object_candidates = [c for c in candidate_results if c["verdict"] == "NEAR_OBJECT"]
                    
                    if pass_candidates:
                        # Select best PASS candidate: maximize (ODR - lambda * MER)
                        # Higher ODR = more consistent, lower MER = less overlap risk
                        lambda_mer = 0.5  # Weight for MER penalty
                        best_candidate = max(
                            pass_candidates,
                            key=lambda c: c["odr"] - lambda_mer * c["mer"]
                        )
                        best_roi = best_candidate["roi"]
                        best_score = best_candidate["score"]
                        best_reason = f"topk_pass_odr={best_candidate['odr']:.3f}_mer={best_candidate['mer']:.3f}"
                        prac_result = best_candidate["prac_result"]
                    elif near_object_candidates:
                        # All candidates are near object: select one with lowest MER
                        best_candidate = min(near_object_candidates, key=lambda c: c["mer"])
                        best_roi = best_candidate["roi"]
                        best_score = best_candidate["score"]
                        best_reason = f"topk_near_object_mer={best_candidate['mer']:.3f}"
                        prac_result = best_candidate["prac_result"]
                    else:
                        # All candidates failed PRAC: use fallback (top-1 by localizer score)
                        # Add failed candidates to protected list for next frame
                        for c in candidate_results:
                            if c["prac_result"].protected_roi is not None:
                                self._protected_rois.append(c["prac_result"].protected_roi)
                        
                        # Fallback: use top-1 candidate
                        if top_k_candidates:
                            best_roi, best_score = top_k_candidates[0]
                            best_reason = "topk_fallback_all_failed"
                else:
                    # No valid candidates (all protected)
                    best_reason = "topk_no_valid_candidates"
            else:
                # PRAC disabled or missing inputs: use top-1 from localizer
                if top_k_candidates:
                    best_roi, best_score = top_k_candidates[0]
                    best_reason = "topk_prac_disabled"
                elif tlr.outlier_roi is not None:
                    best_roi = tlr.outlier_roi
                    best_score = float(tlr.outlier_score)
                    best_reason = str(tlr.reason) if hasattr(tlr, "reason") else "ok"
            
            # Update controller state with best candidate
            roi_grid = best_roi
            raw_outlier_score = best_score
            loc_reason = best_reason
            loc_debug = dict(tlr.debug) if isinstance(tlr.debug, dict) else {}
            
            self._current_main_grid = main_grid
            self._current_outlier_grid = roi_grid
            self._current_outlier_score = float(raw_outlier_score)
            self._reacquire_needed = False

        if roi_grid is None:
            # No ROI candidate: not tracking, request reacquire next frame.
            self._tracking = False
            self._reacquire_needed = True
            gd0 = self.gate.step(0.0)
            mass_ema = float(gd0.score_ema)
            state = str(gd0.state)
            greason = str(gd0.reason)
            # Prepare PRAC fields
            prac_verdict = None
            prac_odr = None
            prac_mer = None
            prac_stats_dict = None
            if prac_result is not None:
                prac_verdict = prac_result.verdict
                prac_odr = prac_result.stats.odr
                prac_mer = prac_result.stats.mer
                prac_stats_dict = prac_result.stats.debug
            
            return DefenseDecision(
                should_purify=False,
                roi_box=None,
                grid_box=None,
                main_grid_box=main_grid,
                outlier_score=0.0,
                mass_ema=float(mass_ema),
                raw_mass=0.0,
                state=str(state),
                reason=f"no_outlier_roi | loc={loc_reason} | {greason}",
                strength=None,
                verified=None,
                verify_stats=None,
                mass_heatmap=None,
                quality_ok=None,
                quality_reason="no_roi",
                gate_checked=bool(gd0.checked),
                verdict_code="NA",
                phase="ACQUIRE",
                reacquire_needed=bool(self._reacquire_needed),
                verify_performed=False,
                prac_performed=prac_performed,
                prac_verdict=prac_verdict,
                prac_odr=prac_odr,
                prac_mer=prac_mer,
                prac_stats=prac_stats_dict,
                gripper_box=gripper_box,
                arm_region_box=arm_region_box,
                arm_core_box=arm_core_box,
                arm_guard_box=arm_guard_box,
                joint_points_2d=joint_points_2d,
                joint_names_used=joint_names_used,
                joint_points_render_2d=joint_points_render_2d,
                gripper_points_2d=gripper_points_2d,
                gripper_points_render_2d=gripper_points_render_2d,
                gripper_point_keys_used=gripper_point_keys_used,
                gripper_link_segments_2d=gripper_link_segments_2d,
                gripper_link_name_pairs=gripper_link_name_pairs,
                gripper_link_quads_2d=gripper_link_quads_2d,
                arm_link_segments_2d=arm_link_segments_2d,
                arm_link_name_pairs=arm_link_name_pairs,
                arm_link_quads_2d=arm_link_quads_2d,
            )

        # --- (C0) map ROI grid -> pixel PatchBox early (needed for heatmap-space quality gate) ---
        x0, y0, x1, y1 = roi_grid.gx0, roi_grid.gy0, roi_grid.gx1, roi_grid.gy1
        roi_box = self.hook.grid_bbox_to_patch_box(int(x0), int(y0), int(x1), int(y1))

        # Gate expects a bounded score (historically in [0,1]).
        score = float(np.clip(raw_outlier_score, 0.0, 1.0))

        # --- (C1) hard quality gate (aligned with verifier space when possible) ---
        # We keep grid-space mass for debugging (loc_debug["best"]["m"]), but prefer heatmap-space ROI mass
        # because it is consistent with verifier.roi_mass().
        mass_grid: Optional[float] = None
        if isinstance(loc_debug, dict):
            try:
                mass_grid = float(loc_debug.get("best", {}).get("m", 0.0))
            except Exception:
                mass_grid = None

        mass_heatmap: Optional[float] = None
        hm_now: Optional[np.ndarray] = None
        if hm_current is not None:
            hm_now = hm_current
        elif heatmap_fn is not None:
            try:
                hm_now = heatmap_fn()
            except Exception:
                hm_now = None
        if hm_now is not None:
            try:
                mass_heatmap = float(heatmap_roi_mass(hm_now, roi_box))
            except Exception:
                mass_heatmap = None

        # Decide which mass to use for hard quality gate.
        use_heatmap_mass = (self.quality_mass_source.lower() == "heatmap") and (mass_heatmap is not None)
        thr = float(self.min_trigger_mass_heatmap if use_heatmap_mass else self.min_trigger_patch_mass)
        m_used = float(mass_heatmap) if use_heatmap_mass else float(mass_grid if mass_grid is not None else 0.0)
        allow_trigger = bool(m_used >= thr)
        quality_reason = ("heatmap" if use_heatmap_mass else "grid") + f"_mass={m_used:.4f}>=thr={thr:.4f}"
        if not allow_trigger:
            # Prevent new trigger. Note: if we are already in HOLD, we must force-exit below.
            score = 0.0
            quality_reason = ("heatmap" if use_heatmap_mass else "grid") + f"_mass={m_used:.4f}<thr={thr:.4f}"

        # If verifier is configured and marked as required, but required inputs are missing,
        # raise exception to force caller to provide all verification callbacks.
        if self.verifier is not None and bool(self.require_verify):
            missing = []
            if image is None:
                missing.append("image")
            if purify_fn is None:
                missing.append("purify_fn")
            if forward_fn is None:
                missing.append("forward_fn")
            if heatmap_fn is None:
                missing.append("heatmap_fn")
            
            if missing:
                raise ValueError(
                    f"require_verify=True but missing required parameters: {', '.join(missing)}. "
                    f"Please provide all verification callbacks in controller.step()."
                )

        gd = self.gate.step(score)
        should_gate = bool(gd.should_purify)
        mass_ema = float(gd.score_ema)
        state = str(gd.state)
        greason = str(gd.reason)

        # ACQUIRE decision: only start tracking if gate says ON and quality is OK.
        if not bool(self._tracking):
            if (not bool(should_gate)) or (not bool(allow_trigger)):
                # Not entering TRACK yet; request reacquire again next frame.
                self._reacquire_needed = True
            # Prepare PRAC fields
            prac_verdict = None
            prac_odr = None
            prac_mer = None
            prac_stats_dict = None
            if prac_result is not None:
                prac_verdict = prac_result.verdict
                prac_odr = prac_result.stats.odr
                prac_mer = prac_result.stats.mer
                prac_stats_dict = prac_result.stats.debug
            
            return DefenseDecision(
                should_purify=False,
                roi_box=None,
                grid_box=None,
                main_grid_box=main_grid,
                outlier_score=float(score),
                mass_ema=float(mass_ema),
                raw_mass=float(mass_grid if mass_grid is not None else 0.0),
                state=str(state),
                reason=f"acquire_not_ready | loc={loc_reason} | {greason}",
                strength=None,
                verified=None,
                verify_stats=None,
                mass_heatmap=mass_heatmap,
                quality_ok=allow_trigger,
                quality_reason=str(quality_reason),
                gate_checked=bool(gd.checked),
                verdict_code="NA",
                phase="ACQUIRE",
                reacquire_needed=bool(self._reacquire_needed),
                verify_performed=False,
                prac_performed=prac_performed,
                prac_verdict=prac_verdict,
                prac_odr=prac_odr,
                prac_mer=prac_mer,
                prac_stats=prac_stats_dict,
                gripper_box=gripper_box,
                patch_verdict=None,
                arm_region_box=arm_region_box,
                arm_core_box=arm_core_box,
                arm_guard_box=arm_guard_box,
                joint_points_2d=joint_points_2d,
                joint_names_used=joint_names_used,
                joint_points_render_2d=joint_points_render_2d,
                gripper_points_2d=gripper_points_2d,
                gripper_points_render_2d=gripper_points_render_2d,
                gripper_point_keys_used=gripper_point_keys_used,
                gripper_link_segments_2d=gripper_link_segments_2d,
                gripper_link_name_pairs=gripper_link_name_pairs,
                gripper_link_quads_2d=gripper_link_quads_2d,
                arm_link_segments_2d=arm_link_segments_2d,
                arm_link_name_pairs=arm_link_name_pairs,
                arm_link_quads_2d=arm_link_quads_2d,
            )
            # Enter TRACK
            self._tracking = True

        # --- (F) verifier plugin (TRACK stage) ---
        # When verifier fails, we do NOT stop masking; we only request a re-acquire next frame.
        do_verify = (
            (self.verifier is not None)
            and (image is not None)
            and (purify_fn is not None)
            and (forward_fn is not None)
            and (heatmap_fn is not None)
            and (int(self.verify_every_k) > 0)
            and ((int(self._t) % int(self.verify_every_k)) == 0)
        )

        verify_stats: Optional[dict] = None
        verified: Optional[bool] = None
        verify_performed: bool = False

        if do_verify:
            verify_performed = True
            hm_before = heatmap_fn()

            # IMPORTANT: ensure the next forward produces a fresh attention/heatmap.
            def _forward_with_clear(img: np.ndarray) -> Any:
                if hasattr(self.hook, "clear"):
                    self.hook.clear()
                return forward_fn(img)  # type: ignore[misc]

            # Map main_grid to pixel coordinates for verifier (if available)
            main_roi_box = None
            if main_grid is not None:
                mx0, my0, mx1, my1 = main_grid.gx0, main_grid.gy0, main_grid.gx1, main_grid.gy1
                main_roi_box = self.hook.grid_bbox_to_patch_box(int(mx0), int(my0), int(mx1), int(my1))

            vr = self.verifier.verify(
                image=image,
                roi_box=roi_box,
                purify_fn=purify_fn,
                forward_fn=_forward_with_clear,
                heatmap_fn=heatmap_fn,
                hm_before=hm_before,
                main_roi_box=main_roi_box,  # Pass mainland ROI for task preservation check
            )
            verified = bool(vr.verified)
            verify_stats = dict(vr.stats)
            # Failure semantics: request reacquire next frame.
            if not bool(vr.verified):
                self._reacquire_needed = True
            pass_verdict_code = str(vr.verdict_code)
        else:
            pass_verdict_code = "NA"

        # --- (G) final decision ---
        strength = self._mass_to_strength(float(mass_ema))

        # Prepare PRAC fields
        prac_verdict = None
        prac_odr = None
        prac_mer = None
        prac_stats_dict = None
        if prac_result is not None:
            prac_verdict = prac_result.verdict
            prac_odr = prac_result.stats.odr
            prac_mer = prac_result.stats.mer
            prac_stats_dict = prac_result.stats.debug
        
        return DefenseDecision(
            # TRACK stage: continuous masking (ignore gate OFF while tracking).
            should_purify=True,
            roi_box=roi_box,
            grid_box=GridBox(gx0=int(x0), gy0=int(y0), gx1=int(x1), gy1=int(y1)),
            main_grid_box=main_grid,
            outlier_score=float(score),
            mass_ema=float(mass_ema),
            raw_mass=float(mass_grid if mass_grid is not None else 0.0),
            state=str(state),
            reason=f"TRACK | score={score:.4f} raw_outlier_score={raw_outlier_score:.4f} | loc={loc_reason} | {greason}"
                   + (" | reacquire_next" if bool(self._reacquire_needed) else ""),
            strength=strength,
            verified=verified,
            verify_stats=verify_stats,
            mass_heatmap=mass_heatmap,
            quality_ok=allow_trigger,
            quality_reason=str(quality_reason),
            gate_checked=bool(gd.checked),
            verdict_code=str(pass_verdict_code),
            phase="TRACK",
            reacquire_needed=bool(self._reacquire_needed),
            verify_performed=bool(verify_performed),
            prac_performed=prac_performed,
            prac_verdict=prac_verdict,
            prac_odr=prac_odr,
            prac_mer=prac_mer,
            prac_stats=prac_stats_dict,
            gripper_box=gripper_box,
            patch_verdict=None,
            arm_region_box=arm_region_box,
            arm_core_box=arm_core_box,
            arm_guard_box=arm_guard_box,
            joint_points_2d=joint_points_2d,
            joint_names_used=joint_names_used,
            joint_points_render_2d=joint_points_render_2d,
            gripper_points_2d=gripper_points_2d,
            gripper_points_render_2d=gripper_points_render_2d,
            gripper_point_keys_used=gripper_point_keys_used,
            gripper_link_segments_2d=gripper_link_segments_2d,
            gripper_link_name_pairs=gripper_link_name_pairs,
            gripper_link_quads_2d=gripper_link_quads_2d,
            arm_link_segments_2d=arm_link_segments_2d,
            arm_link_name_pairs=arm_link_name_pairs,
            arm_link_quads_2d=arm_link_quads_2d,
        )


# =========================
# Unified interface (optional but aligns with __init__.py)
# =========================

@dataclass(frozen=True)
class UnifiedDefenseResult:
    """Unified result for both known-mode and auto-mode."""
    should_purify: bool
    roi_box: Optional[PatchBox]

    # Unified control-plane fields (consistent across known/auto).
    phase: str = "NA"  # "KNOWN" | "ACQUIRE" | "TRACK" | "NA"
    reason: str = ""
    reacquire_needed: Optional[bool] = None

    # Gating/strength/debug fields (mainly for auto mode).
    mass_ema: float = 0.0
    strength: Optional[float] = None
    gate_checked: Optional[bool] = None

    # Quality signals (auto mode).
    mass_heatmap: Optional[float] = None
    quality_ok: Optional[bool] = None
    quality_reason: str = ""

    # Verifier fields (auto mode; plugin).
    verify_performed: Optional[bool] = None
    verified: Optional[bool] = None
    verify_stats: Optional[dict] = None
    verdict_code: str = "NA"

    # Optional visualization (both modes; only populated when enabled).
    heatmap: Optional[np.ndarray] = None

    # New geometric prior fields
    gripper_box: Optional[PatchBox] = None
    patch_verdict: Optional[str] = None
    selector_debug: Optional[dict] = None
    # Aggregate arm guard box kept for compatibility / fallback visualization.
    arm_region_box: Optional[Tuple[int, int, int, int]] = None
    arm_core_box: Optional[Tuple[int, int, int, int]] = None
    arm_guard_box: Optional[Tuple[int, int, int, int]] = None
    joint_points_2d: Optional[List[Tuple[int, int]]] = None
    joint_names_used: Optional[List[str]] = None
    joint_points_render_2d: Optional[List[Tuple[int, int]]] = None
    gripper_points_2d: Optional[List[Tuple[int, int]]] = None
    gripper_points_render_2d: Optional[List[Tuple[int, int]]] = None
    gripper_point_keys_used: Optional[List[str]] = None
    gripper_link_segments_2d: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None
    gripper_link_name_pairs: Optional[List[Tuple[str, str]]] = None
    gripper_link_quads_2d: Optional[List[List[Tuple[int, int]]]] = None
    arm_link_segments_2d: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None
    arm_link_name_pairs: Optional[List[Tuple[str, str]]] = None
    arm_link_quads_2d: Optional[List[List[Tuple[int, int]]]] = None
    roi_mask: Optional[np.ndarray] = None
    conflict_mode: Optional[str] = None
    conflict_reason: str = ""
    conflict_stats: Optional[dict] = None


class UnifiedDefenseInterface:
    """
    Unified defense interface:
      - mode="known": always masks the patch at the given location (no detection needed)
      - mode="auto" : uses OnlinePatchDefenseController (localize + smooth + track)
    """

    def __init__(
        self,
        hook: Any,
        mode: str = "known",
        # known-mode
        patch_x: Optional[int] = None,
        patch_y: Optional[int] = None,
        patch_w: int = 50,
        patch_h: int = 50,
        detector: Optional[PatchAttentionAnomalyDetector] = None,  # Deprecated in known mode; kept for backward compatibility
        # auto-mode
        controller: Optional[OnlinePatchDefenseController] = None,
        # viz
        use_heatmap_for_viz: bool = False,
    ) -> None:
        self.hook = hook
        self.mode = str(mode)
        self.use_heatmap_for_viz = bool(use_heatmap_for_viz)

        # known-mode params
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_w = int(patch_w)
        self.patch_h = int(patch_h)
        self.detector = detector  # Deprecated; no longer used in known mode

        # auto-mode params
        self.controller = controller

        if self.mode == "known":
            if self.patch_x is None or self.patch_y is None:
                raise ValueError("known mode requires patch_x/patch_y")
            # detector is no longer required in known mode (deprecated)
        elif self.mode == "auto":
            if self.controller is None:
                raise ValueError("auto mode requires controller")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def clear(self) -> None:
        self.hook.clear()

    def reset(self) -> None:
        """Reset stateful controller for a new episode."""
        if self.mode == "auto" and self.controller is not None:
            self.controller.reset()

    def step(
        self,
        *,
        image: Optional[np.ndarray] = None,
        purify_fn: Optional[Callable[[np.ndarray, PatchBox], np.ndarray]] = None,
        forward_fn: Optional[Callable[[np.ndarray], Any]] = None,
        heatmap_fn: Optional[Callable[[], np.ndarray]] = None,
        hm_current: Optional[np.ndarray] = None,
        eef_pos: Optional[np.ndarray] = None,
        geometry_ctx: Optional[GeometryRuntimeContext] = None,
    ) -> UnifiedDefenseResult:
        """
        Args:
            image/purify_fn/forward_fn/heatmap_fn: optional parameters for counterfactual verification.
            If not provided, verification is skipped (backward compatible).
        """
        if self.mode == "known":
            # Known mode: always mask the patch at the given location (no detection needed)
            patch_box = PatchBox(
                x0=int(self.patch_x),
                y0=int(self.patch_y),
                x1=int(self.patch_x) + int(self.patch_w),
                y1=int(self.patch_y) + int(self.patch_h),
            )
            # Optional: get heatmap only for visualization
            heatmap = self.hook.get_heatmap() if self.use_heatmap_for_viz else None
            return UnifiedDefenseResult(
                should_purify=True,  # Always mask in known mode
                roi_box=patch_box,  # Always return patch_box
                phase="KNOWN",
                reason="known_mode_always_mask",
                reacquire_needed=False,
                mass_ema=0.0,
                strength=None,
                gate_checked=None,
                mass_heatmap=None,
                quality_ok=None,
                quality_reason="",
                verify_performed=None,
                verified=None,
                verify_stats=None,
                verdict_code="NA",
                heatmap=heatmap,
                gripper_box=None,
                patch_verdict=None,
            )

        # auto mode
        grid = self.hook.get_saliency_grid()

        # default heatmap_fn if user didn't pass it (works after forward_fn too)
        if heatmap_fn is None:
            heatmap_fn = self.hook.get_heatmap

        dd = self.controller.step(
            grid,
            image=image,
            purify_fn=purify_fn,
            forward_fn=forward_fn,
            heatmap_fn=heatmap_fn,
            hm_current=hm_current,
            eef_pos=eef_pos,
            geometry_ctx=geometry_ctx,
        )
        heatmap = self.hook.get_heatmap() if self.use_heatmap_for_viz else None

        return UnifiedDefenseResult(
            should_purify=bool(dd.should_purify),
            roi_box=dd.roi_box,
            phase=str(getattr(dd, "phase", "NA")),
            reason=str(dd.reason),
            reacquire_needed=getattr(dd, "reacquire_needed", None),
            mass_ema=float(dd.mass_ema),
            strength=dd.strength,
            gate_checked=dd.gate_checked,
            verified=dd.verified,
            verify_stats=dd.verify_stats,
            mass_heatmap=dd.mass_heatmap,
            quality_ok=dd.quality_ok,
            quality_reason=str(dd.quality_reason),
            verify_performed=getattr(dd, "verify_performed", None),
            verdict_code=str(dd.verdict_code),
            heatmap=heatmap,
            gripper_box=getattr(dd, "gripper_box", None),
            patch_verdict=getattr(dd, "patch_verdict", None),
            selector_debug=getattr(dd, "selector_debug", None),
            arm_region_box=getattr(dd, "arm_region_box", None),
            arm_core_box=getattr(dd, "arm_core_box", None),
            arm_guard_box=getattr(dd, "arm_guard_box", None),
            joint_points_2d=getattr(dd, "joint_points_2d", None),
            joint_names_used=getattr(dd, "joint_names_used", None),
            joint_points_render_2d=getattr(dd, "joint_points_render_2d", None),
            gripper_points_2d=getattr(dd, "gripper_points_2d", None),
            gripper_points_render_2d=getattr(dd, "gripper_points_render_2d", None),
            gripper_point_keys_used=getattr(dd, "gripper_point_keys_used", None),
            gripper_link_segments_2d=getattr(dd, "gripper_link_segments_2d", None),
            gripper_link_name_pairs=getattr(dd, "gripper_link_name_pairs", None),
            gripper_link_quads_2d=getattr(dd, "gripper_link_quads_2d", None),
            arm_link_segments_2d=getattr(dd, "arm_link_segments_2d", None),
            arm_link_name_pairs=getattr(dd, "arm_link_name_pairs", None),
            arm_link_quads_2d=getattr(dd, "arm_link_quads_2d", None),
            roi_mask=getattr(dd, "roi_mask", None),
            conflict_mode=getattr(dd, "conflict_mode", None),
            conflict_reason=str(getattr(dd, "conflict_reason", "")),
            conflict_stats=getattr(dd, "conflict_stats", None),
        )
