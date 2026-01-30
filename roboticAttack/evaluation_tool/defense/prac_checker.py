# prac_checker.py
# -*- coding: utf-8 -*-
"""
PRAC (Patch-wise Randomized Attention Consistency) Checker.

This module implements perception-level check to distinguish patch-induced anomalies
from task-semantic attention using randomized patch-wise perturbations.

Design:
- Given a candidate outlier ROI, apply N random patch-wise perturbations
- Aggregate attention from N views to compute consensus attention
- Compare base attention vs consensus attention to compute:
  - ODR (Outlier Dependency Ratio): consistency of outlier attention
  - MER (Mainland Erosion Risk): overlap risk with task-relevant regions
- Output verdict: PASS / REACQUIRE / NEAR_OBJECT

Integration:
- Called in ACQUIRE phase, after localizer.localize(), before gate.step()
- Only runs when image and forward_fn are available
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Literal, Dict, Any, Tuple, List
import numpy as np

from .temporal import GridBox

# Verdict type
Verdict = Literal["PASS", "REACQUIRE", "NEAR_OBJECT", "SKIP"]


@dataclass
class PRACConfig:
    """Configuration for PRAC checker."""
    # Number of random views for consensus
    n_views: int = 6
    
    # Patch-wise masking granularity (pixels per patch)
    patch_size: int = 16
    
    # Fraction of patches to perturb (0..1)
    mask_ratio: float = 0.25
    
    # Perturbation mode: "blur" | "gray" | "noise"
    transform_mode: str = "blur"
    
    # Consistency threshold: if ODR < tau_odr, verdict=REACQUIRE
    tau_odr: float = 1.8
    
    # Near-object threshold: if MER > tau_mer, verdict=NEAR_OBJECT
    tau_mer: float = 0.20
    
    # Maximum re-localization attempts within one ACQUIRE frame
    max_attempts: int = 2
    
    # Random seed for reproducibility (None = random)
    seed: Optional[int] = None
    
    # Blur parameters (if transform_mode="blur")
    blur_ksize: int = 7
    blur_sigma: float = 2.0
    
    # Gray value (if transform_mode="gray")
    gray_value: int = 127
    
    # Noise std (if transform_mode="noise")
    noise_std: float = 0.1
    
    # =============================================================================
    # RPA-based optimization parameters (Phase 1)
    # =============================================================================
    
    # Multi-scale patch configuration (RPA paper: alternate between 1, 3, 5, 7)
    use_multi_scale_patches: bool = True
    patch_sizes: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    
    # Mixed transform configuration (RPA paper: randomly select blur/gray/noise per view)
    use_mixed_transform: bool = True
    transform_modes: List[str] = field(default_factory=lambda: ["blur", "gray", "noise"])
    
    # Dynamic mask ratio configuration (RPA paper: p_m=0.1-0.3 for defense scenarios)
    use_dynamic_mask_ratio: bool = True
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.3


@dataclass
class PRACStats:
    """Statistics from PRAC check."""
    # Outlier Dependency Ratio (consistency measure)
    odr: float
    
    # Mainland Erosion Risk (overlap risk)
    mer: float
    
    # Raw outlier mass in base attention
    raw_outlier_mass: float
    
    # Consensus outlier mass (from aggregated attention)
    cons_outlier_mass: float
    
    # Consensus mainland mass
    cons_main_mass: float
    
    # Additional debug statistics
    debug: Dict[str, float]


@dataclass
class PRACResult:
    """PRAC check result."""
    # Verdict: PASS / REACQUIRE / NEAR_OBJECT / SKIP
    verdict: Verdict
    
    # Statistics
    stats: PRACStats
    
    # Protected ROI for negative prior (if REACQUIRE)
    # This ROI should be avoided in next localization attempt
    protected_roi: Optional[GridBox] = None
    
    # Consensus attention grid (for mask extraction in future)
    consensus_grid: Optional[np.ndarray] = None
    
    # Human-readable reason
    reason: str = ""


@dataclass
class PRACCandidateScore:
    """Candidate score for PRAC-based patch selection.

    Note:
        This score is designed for *relative* comparison among Top-K candidates.
        Higher `score` means the candidate ROI is more likely to be the adversarial patch.
    """

    score: float
    drop_rate: float
    l1_shift: float
    main_rise: float
    base_outlier_mass: float
    cons_outlier_mass: float
    base_main_mass: float
    cons_main_mass: float
    debug: Dict[str, float]


class PRACChecker:
    """
    PRAC Checker: validates outlier ROI using randomized patch-wise perturbations.
    
    Strategy:
    1. Apply N random patch-wise perturbations to the image
    2. For each perturbation, run forward pass and extract attention grid
    3. Aggregate attention grids to compute consensus attention
    4. Compare base attention vs consensus attention:
       - ODR = raw_outlier_mass / cons_outlier_mass (higher = more consistent)
       - MER = cons_main_mass / (cons_outlier_mass + eps) (higher = closer to task object)
    5. Output verdict based on thresholds
    """
    
    def __init__(self, cfg: Optional[PRACConfig] = None):
        """
        Initialize PRAC checker.
        
        Args:
            cfg: PRAC configuration. If None, uses default config.
        """
        self.cfg = cfg if cfg is not None else PRACConfig()
        self._rng = np.random.RandomState(self.cfg.seed)
        
        # Optional: OpenCV for blur (if available)
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            self._cv2 = None
            if self.cfg.transform_mode == "blur":
                # Fallback to gray if blur not available
                self.cfg.transform_mode = "gray"
    
    def check(
        self,
        image: np.ndarray,
        base_grid: np.ndarray,  # Base attention grid (16x16) from current forward
        outlier_roi: GridBox,
        main_roi: Optional[GridBox],
        forward_ctx: Callable[[np.ndarray], None],  # forward_ctx(img) -> None (clears hook + forward)
        grid_readout: Callable[[], np.ndarray],  # Returns 16x16 grid after forward
    ) -> PRACResult:
        """
        Perform PRAC check on outlier ROI.
        
        Args:
            image: Original image (H, W, 3) uint8
            base_grid: Base attention grid (16x16) from current forward pass
            outlier_roi: Outlier ROI in grid coordinates
            main_roi: Mainland ROI in grid coordinates (optional)
            forward_ctx: Forward context function that:
                - Clears hook cache
                - Runs forward pass on given image
                - Does NOT return action (to avoid side effects)
            grid_readout: Function to read attention grid after forward_ctx
        
        Returns:
            PRACResult with verdict and statistics
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape={image.shape}")
        if base_grid.ndim != 2:
            raise ValueError(f"Expected 2D grid, got shape={base_grid.shape}")
        
        gh, gw = base_grid.shape
        
        # Clamp ROI to grid bounds
        outlier_roi = outlier_roi.clamp(gw=gw, gh=gh)
        if main_roi is not None:
            main_roi = main_roi.clamp(gw=gw, gh=gh)
        
        # Compute base attention statistics
        base_outlier_mass = self._roi_mass_in_grid(base_grid, outlier_roi)
        base_main_mass = self._roi_mass_in_grid(base_grid, main_roi) if main_roi is not None else 0.0
        
        # Build consensus attention from N random views
        consensus_grid = self._build_consensus_attention(
            image, forward_ctx, grid_readout
        )
        
        # Compute consensus statistics
        cons_outlier_mass = self._roi_mass_in_grid(consensus_grid, outlier_roi)
        cons_main_mass = self._roi_mass_in_grid(consensus_grid, main_roi) if main_roi is not None else 0.0
        
        # Compute ODR (Outlier Dependency Ratio)
        # Higher ODR = more consistent outlier attention across views
        eps = 1e-8
        if cons_outlier_mass < eps:
            odr = 0.0  # Outlier disappeared in consensus (inconsistent)
        else:
            odr = base_outlier_mass / (cons_outlier_mass + eps)
        
        # Compute MER (Mainland Erosion Risk)
        # Higher MER = outlier attention overlaps with mainland (near task object)
        if cons_outlier_mass < eps:
            mer = 1.0  # Outlier disappeared, likely near object
        else:
            mer = cons_main_mass / (cons_outlier_mass + cons_main_mass + eps)
        
        # Decision logic
        if odr >= self.cfg.tau_odr and mer <= self.cfg.tau_mer:
            verdict = "PASS"
            reason = f"ODR={odr:.3f}>=tau_odr={self.cfg.tau_odr}, MER={mer:.3f}<=tau_mer={self.cfg.tau_mer}"
            protected_roi = None
        elif mer > self.cfg.tau_mer:
            verdict = "NEAR_OBJECT"
            reason = f"MER={mer:.3f}>tau_mer={self.cfg.tau_mer} (too close to task object)"
            protected_roi = None
        else:
            verdict = "REACQUIRE"
            reason = f"ODR={odr:.3f}<tau_odr={self.cfg.tau_odr} (inconsistent attention)"
            # Mark this ROI as protected (negative prior for next localization)
            protected_roi = outlier_roi
        
        # Build statistics
        stats = PRACStats(
            odr=float(odr),
            mer=float(mer),
            raw_outlier_mass=float(base_outlier_mass),
            cons_outlier_mass=float(cons_outlier_mass),
            cons_main_mass=float(cons_main_mass),
            debug={
                "base_outlier_mass": float(base_outlier_mass),
                "base_main_mass": float(base_main_mass),
                "cons_outlier_mass": float(cons_outlier_mass),
                "cons_main_mass": float(cons_main_mass),
                "odr": float(odr),
                "mer": float(mer),
            }
        )
        
        return PRACResult(
            verdict=verdict,
            stats=stats,
            protected_roi=protected_roi,
            consensus_grid=consensus_grid,
            reason=reason,
        )
    
    def evaluate_candidate_with_consensus(
        self,
        base_grid: np.ndarray,
        consensus_grid: np.ndarray,
        outlier_roi: GridBox,
        main_roi: Optional[GridBox],
    ) -> PRACResult:
        """
        Evaluate a candidate ROI using pre-computed consensus attention.
        This method does NOT perform forward passes, only computes statistics.
        
        Used for Top-K parallel evaluation: build consensus once, evaluate all candidates.
        
        Args:
            base_grid: Base attention grid (16x16) from current forward pass
            consensus_grid: Pre-computed consensus attention grid (16x16)
            outlier_roi: Candidate outlier ROI in grid coordinates
            main_roi: Mainland ROI in grid coordinates (optional)
        
        Returns:
            PRACResult with verdict and statistics
        """
        if base_grid.ndim != 2 or consensus_grid.ndim != 2:
            raise ValueError(f"Expected 2D grids, got base={base_grid.shape}, consensus={consensus_grid.shape}")
        
        gh, gw = base_grid.shape
        if consensus_grid.shape != (gh, gw):
            raise ValueError(f"Grid shape mismatch: base={base_grid.shape}, consensus={consensus_grid.shape}")
        
        # Clamp ROI to grid bounds
        outlier_roi = outlier_roi.clamp(gw=gw, gh=gh)
        if main_roi is not None:
            main_roi = main_roi.clamp(gw=gw, gh=gh)
        
        # Compute base attention statistics
        base_outlier_mass = self._roi_mass_in_grid(base_grid, outlier_roi)
        base_main_mass = self._roi_mass_in_grid(base_grid, main_roi) if main_roi is not None else 0.0
        
        # Compute consensus statistics (using pre-computed consensus_grid)
        cons_outlier_mass = self._roi_mass_in_grid(consensus_grid, outlier_roi)
        cons_main_mass = self._roi_mass_in_grid(consensus_grid, main_roi) if main_roi is not None else 0.0
        
        # Compute ODR (Outlier Dependency Ratio)
        eps = 1e-8
        if cons_outlier_mass < eps:
            odr = 0.0  # Outlier disappeared in consensus (inconsistent)
        else:
            odr = base_outlier_mass / (cons_outlier_mass + eps)
        
        # Compute MER (Mainland Erosion Risk)
        if cons_outlier_mass < eps:
            mer = 1.0  # Outlier disappeared, likely near object
        else:
            mer = cons_main_mass / (cons_outlier_mass + cons_main_mass + eps)
        
        # Decision logic
        if odr >= self.cfg.tau_odr and mer <= self.cfg.tau_mer:
            verdict = "PASS"
            reason = f"ODR={odr:.3f}>=tau_odr={self.cfg.tau_odr}, MER={mer:.3f}<=tau_mer={self.cfg.tau_mer}"
            protected_roi = None
        elif mer > self.cfg.tau_mer:
            verdict = "NEAR_OBJECT"
            reason = f"MER={mer:.3f}>tau_mer={self.cfg.tau_mer} (too close to task object)"
            protected_roi = None
        else:
            verdict = "REACQUIRE"
            reason = f"ODR={odr:.3f}<tau_odr={self.cfg.tau_odr} (inconsistent attention)"
            protected_roi = outlier_roi
        
        # Build statistics
        stats = PRACStats(
            odr=float(odr),
            mer=float(mer),
            raw_outlier_mass=float(base_outlier_mass),
            cons_outlier_mass=float(cons_outlier_mass),
            cons_main_mass=float(cons_main_mass),
            debug={
                "base_outlier_mass": float(base_outlier_mass),
                "base_main_mass": float(base_main_mass),
                "cons_outlier_mass": float(cons_outlier_mass),
                "cons_main_mass": float(cons_main_mass),
                "odr": float(odr),
                "mer": float(mer),
            }
        )
        
        return PRACResult(
            verdict=verdict,
            stats=stats,
            protected_roi=protected_roi,
            consensus_grid=consensus_grid,  # Reuse same consensus
            reason=reason,
        )
    
    def _build_consensus_attention(
        self,
        image: np.ndarray,
        forward_ctx: Callable[[np.ndarray], None],
        grid_readout: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """
        Build consensus attention by aggregating N random views.
        
        Args:
            image: Original image
            forward_ctx: Forward context (clears hook + forward)
            grid_readout: Read attention grid after forward
        
        Returns:
            Aggregated attention grid (16x16)
        """
        grids = []
        
        for i in range(self.cfg.n_views):
            # Apply random patch-wise perturbation
            perturbed_img = self._apply_patch_perturbation(image.copy())
            
            # Forward pass (forward_ctx handles hook.clear())
            forward_ctx(perturbed_img)
            
            # Read attention grid
            grid = grid_readout()
            grids.append(grid)
        
        # Aggregate grids (mean aggregation)
        stacked = np.stack(grids, axis=0)  # [N, H, W]
        consensus = np.mean(stacked, axis=0).astype(np.float32)
        
        return consensus

    def score_candidate_with_roi_mask(
        self,
        *,
        image: np.ndarray,
        base_grid: np.ndarray,
        outlier_roi: GridBox,
        main_roi: Optional[GridBox],
        roi_box_xyxy: Tuple[int, int, int, int],
        forward_ctx: Callable[[np.ndarray], None],
        grid_readout: Callable[[], np.ndarray],
    ) -> PRACCandidateScore:
        """Score one candidate by masking *inside the candidate ROI box* and measuring global attention shift.

        This implements the intended PRAC mechanism for patch selection:
        - Apply randomized block-wise perturbations *restricted to the candidate ROI* in pixel space.
        - Run multiple forwards to obtain attention grids for the perturbed images.
        - Aggregate perturbed grids (mean) to form a candidate-specific consensus grid.
        - Score the candidate by how much attention mass on the candidate ROI drops, and how much
          the global attention distribution shifts.

        Args:
            image: Original image (H, W, 3) uint8.
            base_grid: Attention grid from the original (unperturbed) forward.
            outlier_roi: Candidate ROI in grid coordinates.
            main_roi: Optional mainland ROI in grid coordinates (for debugging).
            roi_box_xyxy: Candidate ROI bbox in pixel coordinates (x0, y0, x1, y1).
            forward_ctx: Forward context that clears hook caches and runs forward(img).
            grid_readout: Read attention grid after forward_ctx.

        Returns:
            PRACCandidateScore for relative ranking among candidates.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape={image.shape}")
        if base_grid.ndim != 2:
            raise ValueError(f"Expected 2D base_grid, got shape={base_grid.shape}")

        # Clamp ROI grid boxes to grid bounds.
        gh, gw = int(base_grid.shape[0]), int(base_grid.shape[1])
        outlier_roi_g = outlier_roi.clamp(gw=gw, gh=gh)
        main_roi_g = main_roi.clamp(gw=gw, gh=gh) if main_roi is not None else None

        base_outlier_mass = float(self._roi_mass_in_grid(base_grid, outlier_roi_g))
        base_main_mass = float(self._roi_mass_in_grid(base_grid, main_roi_g)) if main_roi_g is not None else 0.0

        consensus_grid = self._build_consensus_attention_in_box(
            image=image,
            roi_box_xyxy=roi_box_xyxy,
            forward_ctx=forward_ctx,
            grid_readout=grid_readout,
        )

        cons_outlier_mass = float(self._roi_mass_in_grid(consensus_grid, outlier_roi_g))
        cons_main_mass = float(self._roi_mass_in_grid(consensus_grid, main_roi_g)) if main_roi_g is not None else 0.0

        # (1) Candidate ROI attention drop rate (primary signal for ranking).
        eps = 1e-8
        if base_outlier_mass <= eps:
            drop_rate = 0.0
        else:
            drop_rate = float((base_outlier_mass - cons_outlier_mass) / (base_outlier_mass + eps))
            drop_rate = float(np.clip(drop_rate, 0.0, 1.0))

        # (2) Mainland attention rise (auxiliary debug signal).
        main_rise = float(cons_main_mass - base_main_mass)

        # (3) Global distribution shift (auxiliary signal).
        p = self._normalize_grid(base_grid)
        q = self._normalize_grid(consensus_grid)
        l1_shift = float(np.abs(p - q).sum())

        # Final score: use drop_rate as the primary ranking key (threshold-free).
        score = float(drop_rate)

        debug = {
            "base_outlier_mass": float(base_outlier_mass),
            "cons_outlier_mass": float(cons_outlier_mass),
            "drop_rate": float(drop_rate),
            "base_main_mass": float(base_main_mass),
            "cons_main_mass": float(cons_main_mass),
            "main_rise": float(main_rise),
            "l1_shift": float(l1_shift),
        }

        return PRACCandidateScore(
            score=float(score),
            drop_rate=float(drop_rate),
            l1_shift=float(l1_shift),
            main_rise=float(main_rise),
            base_outlier_mass=float(base_outlier_mass),
            cons_outlier_mass=float(cons_outlier_mass),
            base_main_mass=float(base_main_mass),
            cons_main_mass=float(cons_main_mass),
            debug=debug,
        )

    def _build_consensus_attention_in_box(
        self,
        *,
        image: np.ndarray,
        roi_box_xyxy: Tuple[int, int, int, int],
        forward_ctx: Callable[[np.ndarray], None],
        grid_readout: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """Build consensus attention from N views by perturbing only inside a given pixel ROI box.
        
        RPA-based optimizations: passes view_idx to enable multi-scale, mixed transform, and dynamic mask_ratio.
        """
        grids: List[np.ndarray] = []
        for i in range(int(self.cfg.n_views)):
            # Pass view_idx to enable RPA optimizations (multi-scale, mixed transform, dynamic mask_ratio)
            perturbed = self._apply_patch_perturbation_in_box(
                image.copy(), 
                roi_box_xyxy=roi_box_xyxy,
                view_idx=i  # Pass view index for alternating patch sizes
            )
            forward_ctx(perturbed)
            grids.append(grid_readout())
        stacked = np.stack(grids, axis=0)
        return np.mean(stacked, axis=0).astype(np.float32)

    def _apply_patch_perturbation_in_box(
        self,
        image: np.ndarray,
        *,
        roi_box_xyxy: Tuple[int, int, int, int],
        view_idx: int = 0,
    ) -> np.ndarray:
        """Apply randomized patch-wise perturbations *restricted to the given ROI box* (pixel space).
        
        RPA-based optimizations (Phase 1):
        - Multi-scale patches: alternate between different patch sizes per view
        - Mixed transforms: randomly select transform mode per view
        - Dynamic mask ratio: randomly select mask ratio per view within min/max range
        
        Args:
            image: Input image (H, W, 3) uint8
            roi_box_xyxy: ROI bounding box in pixel coordinates (x0, y0, x1, y1)
            view_idx: View index for alternating patch sizes (default: 0)
        """
        h, w, _ = image.shape
        x0, y0, x1, y1 = roi_box_xyxy
        # Clamp ROI to image bounds.
        x0 = int(max(0, min(int(x0), w)))
        x1 = int(max(0, min(int(x1), w)))
        y0 = int(max(0, min(int(y0), h)))
        y1 = int(max(0, min(int(y1), h)))
        if x1 <= x0 or y1 <= y0:
            return image

        # =============================================================================
        # RPA Optimization 1: Multi-scale patch selection
        # =============================================================================
        if self.cfg.use_multi_scale_patches and hasattr(self.cfg, 'patch_sizes') and self.cfg.patch_sizes:
            # Alternate between different patch sizes (RPA strategy: 1, 3, 5, 7)
            patch_size_idx = view_idx % len(self.cfg.patch_sizes)
            patch_size = int(self.cfg.patch_sizes[patch_size_idx])
        else:
            patch_size = int(self.cfg.patch_size)
        patch_size = max(1, patch_size)

        # Compute patch index ranges that intersect the ROI box.
        ph0 = int(y0 // patch_size)
        ph1 = int((y1 - 1) // patch_size)
        pw0 = int(x0 // patch_size)
        pw1 = int((x1 - 1) // patch_size)

        n_patches_h = int((h + patch_size - 1) // patch_size)
        n_patches_w = int((w + patch_size - 1) // patch_size)

        ph0 = max(0, min(ph0, n_patches_h - 1))
        ph1 = max(0, min(ph1, n_patches_h - 1))
        pw0 = max(0, min(pw0, n_patches_w - 1))
        pw1 = max(0, min(pw1, n_patches_w - 1))

        roi_patch_indices: List[Tuple[int, int]] = []
        for ph in range(ph0, ph1 + 1):
            for pw in range(pw0, pw1 + 1):
                roi_patch_indices.append((ph, pw))

        if not roi_patch_indices:
            return image

        # =============================================================================
        # RPA Optimization 3: Dynamic mask ratio
        # =============================================================================
        total = len(roi_patch_indices)
        if self.cfg.use_dynamic_mask_ratio and hasattr(self.cfg, 'mask_ratio_min') and hasattr(self.cfg, 'mask_ratio_max'):
            # Randomly select mask_ratio per view (RPA strategy: p_m=0.1-0.3 for defense)
            mask_ratio = float(self._rng.uniform(self.cfg.mask_ratio_min, self.cfg.mask_ratio_max))
        else:
            mask_ratio = float(self.cfg.mask_ratio)
        n_perturb = int(round(float(total) * mask_ratio))
        n_perturb = max(1, min(total, n_perturb))  # ensure at least one patch is perturbed
        chosen = self._rng.choice(total, size=n_perturb, replace=False)

        # =============================================================================
        # RPA Optimization 2: Mixed transform mode selection
        # =============================================================================
        if self.cfg.use_mixed_transform and hasattr(self.cfg, 'transform_modes') and self.cfg.transform_modes:
            # Randomly select transform mode per view (RPA strategy)
            transform_mode = self._rng.choice(self.cfg.transform_modes)
        else:
            transform_mode = self.cfg.transform_mode

        for idx in chosen:
            ph, pw = roi_patch_indices[int(idx)]
            py0 = ph * patch_size
            py1 = min(py0 + patch_size, h)
            px0 = pw * patch_size
            px1 = min(px0 + patch_size, w)
            patch_roi = image[py0:py1, px0:px1]

            if transform_mode == "blur":
                if self._cv2 is not None:
                    k = int(self.cfg.blur_ksize)
                    if k % 2 == 0:
                        k += 1
                    patch_roi[:] = self._cv2.GaussianBlur(
                        patch_roi, (k, k),
                        sigmaX=float(self.cfg.blur_sigma),
                        sigmaY=float(self.cfg.blur_sigma),
                    )
                else:
                    patch_roi[:] = int(self.cfg.gray_value)
            elif transform_mode == "gray":
                patch_roi[:] = int(self.cfg.gray_value)
            elif transform_mode == "noise":
                noise = self._rng.normal(0, float(self.cfg.noise_std) * 255.0, patch_roi.shape)
                patch_roi[:] = np.clip(patch_roi.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return image

    @staticmethod
    def _normalize_grid(grid: np.ndarray) -> np.ndarray:
        """Normalize a grid to a probability distribution (sum=1) in float32."""
        g = grid.astype(np.float32)
        g = g - float(g.min())
        s = float(g.sum())
        if s <= 1e-8:
            return np.full_like(g, 1.0 / float(g.size), dtype=np.float32)
        return (g / s).astype(np.float32)
    
    def _apply_patch_perturbation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random patch-wise perturbation to image.
        
        Strategy:
        - Divide image into patches of size patch_size x patch_size
        - Randomly select mask_ratio fraction of patches
        - Apply perturbation (blur/gray/noise) to selected patches
        
        Args:
            image: Original image (H, W, 3) uint8
        
        Returns:
            Perturbed image (H, W, 3) uint8
        """
        h, w, _ = image.shape
        patch_size = self.cfg.patch_size
        
        # Compute number of patches
        n_patches_h = (h + patch_size - 1) // patch_size
        n_patches_w = (w + patch_size - 1) // patch_size
        
        # Randomly select patches to perturb
        total_patches = n_patches_h * n_patches_w
        n_perturb = int(total_patches * self.cfg.mask_ratio)
        selected_indices = self._rng.choice(total_patches, size=n_perturb, replace=False)
        
        # Apply perturbation to selected patches
        for idx in selected_indices:
            ph = idx // n_patches_w
            pw = idx % n_patches_w
            
            y0 = ph * patch_size
            y1 = min(y0 + patch_size, h)
            x0 = pw * patch_size
            x1 = min(x0 + patch_size, w)
            
            patch_roi = image[y0:y1, x0:x1]
            
            if self.cfg.transform_mode == "blur":
                if self._cv2 is not None:
                    k = self.cfg.blur_ksize
                    if k % 2 == 0:
                        k += 1
                    patch_roi[:] = self._cv2.GaussianBlur(
                        patch_roi, (k, k),
                        sigmaX=self.cfg.blur_sigma,
                        sigmaY=self.cfg.blur_sigma
                    )
                else:
                    # Fallback to gray
                    patch_roi[:] = self.cfg.gray_value
            elif self.cfg.transform_mode == "gray":
                patch_roi[:] = self.cfg.gray_value
            elif self.cfg.transform_mode == "noise":
                noise = self._rng.normal(0, self.cfg.noise_std * 255, patch_roi.shape)
                patch_roi[:] = np.clip(
                    patch_roi.astype(np.float32) + noise, 0, 255
                ).astype(np.uint8)
        
        return image
    
    def _roi_mass_in_grid(self, grid: np.ndarray, roi: Optional[GridBox]) -> float:
        """
        Compute ROI mass in grid space.
        
        Args:
            grid: Attention grid (H, W)
            roi: ROI in grid coordinates (None = return 0.0)
        
        Returns:
            ROI mass (sum in ROI / sum in grid)
        """
        if roi is None:
            return 0.0
        
        gh, gw = grid.shape
        roi_clamped = roi.clamp(gw=gw, gh=gh)
        
        roi_patch = grid[
            roi_clamped.gy0:roi_clamped.gy1,
            roi_clamped.gx0:roi_clamped.gx1
        ]
        
        total = float(grid.sum()) + 1e-8
        roi_sum = float(roi_patch.sum())
        
        return float(roi_sum / total)

