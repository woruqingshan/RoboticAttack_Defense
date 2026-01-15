"""
Anomaly detection + localization + temporal gating for attention-based patch defense.

This module keeps the legacy detector API:
    PatchAttentionAnomalyDetector.detect(heatmap, patch_box) -> DetectionResult

And extends the system with:
- PatchAttentionLocalizer: localize suspicious ROI from a low-res saliency grid
- TemporalGate: smooth / pulsed defense control to avoid on-off flicker
- OnlinePatchDefenseController: localize + gate -> pixel ROI decision
- UnifiedDefenseInterface: one-step interface (known/oracle vs auto/localize modes)

Design goals:
- Backward compatible: existing scripts that assume known patch location still work.
- No new dependencies: connected components are implemented via pure NumPy.
- Practical control: supports pulsed masking (hold/cooldown) to reduce collateral damage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Deque, List, Optional, Tuple, TYPE_CHECKING
from collections import deque
import math

import numpy as np

# --- NEW: decoupled modules ---
from .localizer import AttentionLocalizer
from .temporal import (
    RunningStats2D,
    AttentionStabilityScorer,
    ROITracker,
    TemporalGate as _PulseGate,
)
from .verifier import CounterfactualVerifier

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

@dataclass(frozen=True)
class GridBox:
    """Axis-aligned bbox in low-res saliency grid coordinates [0..W/H]."""
    gx0: int
    gy0: int
    gx1: int
    gy1: int

    def clamp(self, gw: int, gh: int) -> "GridBox":
        gx0 = int(max(0, min(self.gx0, gw)))
        gx1 = int(max(0, min(self.gx1, gw)))
        gy0 = int(max(0, min(self.gy0, gh)))
        gy1 = int(max(0, min(self.gy1, gh)))
        if gx1 < gx0:
            gx0, gx1 = gx1, gx0
        if gy1 < gy0:
            gy0, gy1 = gy1, gy0
        return GridBox(gx0=gx0, gy0=gy0, gx1=gx1, gy1=gy1)

    def area(self) -> int:
        return int(max(0, self.gx1 - self.gx0) * max(0, self.gy1 - self.gy0))

    def pad(self, p: int) -> "GridBox":
        p = int(max(0, p))
        return GridBox(self.gx0 - p, self.gy0 - p, self.gx1 + p, self.gy1 + p)

    def centroid(self) -> Tuple[float, float]:
        cx = (self.gx0 + self.gx1) * 0.5
        cy = (self.gy0 + self.gy1) * 0.5
        return float(cx), float(cy)

    def iou(self, other: "GridBox") -> float:
        ax0, ay0, ax1, ay1 = self.gx0, self.gy0, self.gx1, self.gy1
        bx0, by0, bx1, by1 = other.gx0, other.gy0, other.gx1, other.gy1
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        union = self.area() + other.area() - inter
        return float(inter / union) if union > 0 else 0.0


@dataclass(frozen=True)
class LocalizationResult:
    """Per-frame localization result on the saliency grid."""
    found: bool
    grid_box: Optional[GridBox]
    mass: float
    area_frac: float
    concentration: float
    centroid: Optional[Tuple[float, float]]
    reason: str = ""


def _connected_components(mask: np.ndarray, connectivity: int = 4) -> List[List[Tuple[int, int]]]:
    """
    Very small-grid connected components (BFS), no external deps.
    mask: bool HxW
    returns: list of components, each a list of (y,x)
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []

    if connectivity == 8:
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        neigh = [(-1,0),(1,0),(0,-1),(0,1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q: Deque[Tuple[int, int]] = deque()
            q.append((y, x))
            visited[y, x] = True
            comp: List[Tuple[int, int]] = [(y, x)]
            while q:
                cy, cx = q.popleft()
                for dy, dx in neigh:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and (not visited[ny, nx]):
                        visited[ny, nx] = True
                        q.append((ny, nx))
                        comp.append((ny, nx))
            comps.append(comp)
    return comps


class PatchAttentionLocalizer:
    """
    Wrapper over localizer.AttentionLocalizer.
    Keeps the old name + return type (LocalizationResult) for compatibility.

    NOTE:
    - `AttentionLocalizer` uses pure-numpy CC and returns roi box as (x0,y0,x1,y1).
    - We convert it into your GridBox/LocalizationResult.
    - Parameter mapping: min_area_frac/max_area_frac are passed directly to underlying localizer
      (defaults may differ: 0.003/0.12 vs 0.01/0.25, but user can override).
    - connectivity parameter is kept for compatibility but underlying impl always uses 4-neighborhood.
    """

    def __init__(
        self,
        top_p: float = 0.07,
        min_area_frac: float = 0.003,
        max_area_frac: float = 0.12,
        connectivity: int = 4,   # kept for compatibility; underlying impl uses 4-neigh
        pad_cells: int = 1,
        eps: float = 1e-12,
    ) -> None:
        self.top_p = float(top_p)
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.connectivity = int(connectivity)
        self.pad_cells = int(pad_cells)
        self.eps = float(eps)

        # decoupled localizer
        self._impl = AttentionLocalizer(
            top_p=self.top_p,
            min_area=self.min_area_frac,
            max_area=self.max_area_frac,
        )

    def localize(self, grid: np.ndarray, top_k: int = 1) -> List[LocalizationResult]:
        if grid.ndim != 2:
            raise ValueError(f"grid must be 2D, got shape={grid.shape}")
        gh, gw = grid.shape

        results = self._impl.localize(grid, top_k=top_k)
        if not results:
            return [LocalizationResult(False, None, 0.0, 0.0, 0.0, None, reason="no_roi")]

        localized_results = []
        for res in results:
            # GridBox is now a dataclass, access attributes instead of unpacking
            x0, y0, x1, y1 = res.roi.gx0, res.roi.gy0, res.roi.gx1, res.roi.gy1
            gb = GridBox(gx0=int(x0), gy0=int(y0), gx1=int(x1), gy1=int(y1))
            gb = gb.pad(self.pad_cells).clamp(gw=gw, gh=gh)

            # compute concentration (mass / area_frac)
            conc = float(res.roi_mass / (res.area_ratio + self.eps))
            cx, cy = gb.centroid()

            localized_results.append(LocalizationResult(
                found=True,
                grid_box=gb,
                mass=float(res.roi_mass),
                area_frac=float(res.area_ratio),
                concentration=float(conc),
                centroid=(float(cx), float(cy)),
                reason=f"thr={res.threshold:.6f} mass={res.roi_mass:.3f} area={res.area_ratio:.3f}",
            ))

        return localized_results


@dataclass(frozen=True)
class TemporalGateState:
    """Simple state labels."""
    state: str  # SAFE / ON / HOLD / COOLDOWN


class TemporalGate:
    """
    Compatibility wrapper:
    - keeps old .step() signature -> (should_purify, mass_ema, state, reason)
    - internally uses temporal.TemporalGate for low-frequency + hold/cooldown
    - NOTE: mass_ema is always updated (even when gate skips check), for consistency with old behavior
    """

    def __init__(
        self,
        theta_on: float = 0.07,
        theta_off: float = 0.05,
        ema_alpha: float = 0.3,
        hold_frames: int = 5,
        cooldown_frames: int = 3,
        check_every_k: int = 3,   # NEW (important): only check every k frames when SAFE
    ) -> None:
        self.theta_on = float(theta_on)
        self.theta_off = float(theta_off)
        self.ema_alpha = float(ema_alpha)

        self.mass_ema = 0.0
        self._pulse = _PulseGate(
            theta_on=float(theta_on),
            theta_off=float(theta_off),
            hold_frames=int(hold_frames),
            cooldown_frames=int(cooldown_frames),
            check_every_k=int(check_every_k),
        )

    def reset(self) -> None:
        self.mass_ema = 0.0
        self._pulse.reset()

    def step(self, mass: float) -> Tuple[bool, float, str, str]:
        m = float(max(0.0, mass))
        # Always update mass_ema (even if gate skips check) for consistency
        self.mass_ema = (1.0 - self.ema_alpha) * self.mass_ema + self.ema_alpha * m

        gd = self._pulse.step(self.mass_ema)
        # Provide more detailed reason for debugging
        if gd.checked:
            if gd.should_purify:
                reason = f"trigger mass_ema={self.mass_ema:.3f} >= theta_on={self.theta_on:.3f} state={gd.state}"
            else:
                reason = f"safe mass_ema={self.mass_ema:.3f} state={gd.state}"
        else:
            reason = f"skip_check mass_ema={self.mass_ema:.3f} state={gd.state}"
        # states: SAFE/HOLD/COOLDOWN (from temporal gate)
        return bool(gd.should_purify), float(self.mass_ema), str(gd.state), str(reason)


@dataclass(frozen=True)
class DefenseDecision:
    """Decision for one step (auto-mode)."""
    should_purify: bool
    roi_box: Optional[PatchBox]
    grid_box: Optional[GridBox]
    mass_ema: float
    raw_mass: float
    state: str
    reason: str
    # Optional: caller can use it as purifier strength (0..1). Not required.
    strength: Optional[float] = None


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
        tracker_iou_keep: float = 0.30,
        tracker_ema: float = 0.50,
        # Strength mapping (kept for backward compatibility)
        strength_min: float = 0.35,
        strength_max: float = 0.85,
        # Motion exclusion (NEW)
        motion_threshold_cells: float = 1.5,  # max centroid movement to be considered static
        motion_penalty_weight: float = 0.3,   # how much to penalize moving ROIs
        top_k_candidates: int = 3,            # number of top candidates to evaluate
        # verifier (optional)
        verifier: Optional[CounterfactualVerifier] = None,
        verify_every_k: int = 3,          # only verify at low frequency
        verify_block_frames: int = 6,     # if verification fails, block purify for a few frames
    ) -> None:
        self.hook = hook
        self.localizer = localizer
        self.gate = gate

        self.stats_alpha = float(stats_alpha)
        self._stats: Optional[RunningStats2D] = None
        self._stats_shape: Optional[Tuple[int, int]] = None
        self._scorer = AttentionStabilityScorer()
        self._tracker = ROITracker(iou_keep=float(tracker_iou_keep), ema=float(tracker_ema))

        self.strength_min = float(strength_min)
        self.strength_max = float(strength_max)

        # Motion exclusion
        self.motion_threshold_cells = float(motion_threshold_cells)
        self.motion_penalty_weight = float(motion_penalty_weight)
        self.top_k_candidates = int(top_k_candidates)
        self._prev_image: Optional[np.ndarray] = None

        self.verifier = verifier
        self.verify_every_k = int(verify_every_k)
        self.verify_block_frames = int(verify_block_frames)
        self._verify_block_left = 0
        self._t = 0

    def reset(self) -> None:
        self.gate.reset()
        self._stats = None
        self._stats_shape = None
        self._tracker.reset()
        self._prev_image = None
        self._verify_block_left = 0
        self._t = 0

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

    def _compute_motion_energy(self, current_image: np.ndarray, roi_box: GridBox) -> float:
        """
        Compute motion energy for a candidate ROI using absdiff with previous frame.
        Returns normalized motion energy (0-1), higher means more motion.
        """
        if self._prev_image is None or current_image is None:
            return 0.0

        # Convert grid box to pixel coordinates for image cropping
        # GridBox is a dataclass, access attributes instead of unpacking
        x0, y0, x1, y1 = roi_box.gx0, roi_box.gy0, roi_box.gx1, roi_box.gy1
        # Map grid coords to pixel coords (assuming image is 224x224, grid is 16x16)
        img_h, img_w = current_image.shape[:2]
        grid_h, grid_w = self._stats_shape if self._stats_shape else (16, 16)

        px0 = int(x0 * img_w / grid_w)
        py0 = int(y0 * img_h / grid_h)
        px1 = int(x1 * img_w / grid_w)
        py1 = int(y1 * img_h / grid_h)

        px0, px1 = max(0, px0), min(img_w, px1)
        py0, py1 = max(0, py0), min(img_h, py1)

        if px1 <= px0 or py1 <= py0:
            return 0.0

        # Convert to grayscale for motion detection
        curr_roi = current_image[py0:py1, px0:px1]
        prev_roi = self._prev_image[py0:py1, px0:px1]

        if curr_roi.shape != prev_roi.shape:
            return 0.0

        # Convert to grayscale if needed
        if curr_roi.ndim == 3:
            curr_gray = np.mean(curr_roi.astype(np.float32), axis=2)
            prev_gray = np.mean(prev_roi.astype(np.float32), axis=2)
        else:
            curr_gray = curr_roi.astype(np.float32)
            prev_gray = prev_roi.astype(np.float32)

        # Compute absolute difference
        absdiff = np.abs(curr_gray - prev_gray)
        motion_energy = float(np.mean(absdiff) / 255.0)  # normalize to 0-1

        return motion_energy

    def step(
        self,
        grid: np.ndarray,
        *,
        image: Optional[np.ndarray] = None,
        purify_fn: Optional[Callable[[np.ndarray, PatchBox], np.ndarray]] = None,
        forward_fn: Optional[Callable[[np.ndarray], Any]] = None,
        heatmap_fn: Optional[Callable[[], np.ndarray]] = None,
    ) -> DefenseDecision:
        """
        Args:
            grid: saliency grid (e.g., 16x16), requires the hook cache already populated.
            image/purify_fn/forward_fn/heatmap_fn: only needed if you enable counterfactual verification.
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

        # --- (B) candidate localization on stable grid (Top-K) ---
        candidates = self.localizer.localize(stable_grid, top_k=self.top_k_candidates)

        if not candidates:
            should, mass_ema, state, greason = self.gate.step(0.0)
            return DefenseDecision(
                should_purify=False,
                roi_box=None,
                grid_box=None,
                mass_ema=float(mass_ema),
                raw_mass=0.0,
                state=str(state),
                reason=f"no_candidates | {greason}",
                strength=None,
            )

        # --- (B.1) Evaluate candidates with motion penalty ---
        best_candidate = None
        best_score = -1.0

        for loc in candidates:
            if not loc.found or loc.grid_box is None:
                continue

            # Base score from stable mass
            base_score = float(self._scorer.score(stable_grid, loc.grid_box))

            # Motion penalty
            motion_energy = self._compute_motion_energy(image, loc.grid_box)
            motion_penalty = self.motion_penalty_weight * motion_energy

            # Combined score: favor high mass + low motion
            combined_score = base_score * (1.0 - motion_penalty)

            if combined_score > best_score:
                best_score = combined_score
                best_candidate = loc

        if best_candidate is None:
            should, mass_ema, state, greason = self.gate.step(0.0)
            return DefenseDecision(
                should_purify=False,
                roi_box=None,
                grid_box=None,
                mass_ema=float(mass_ema),
                raw_mass=0.0,
                state=str(state),
                reason=f"no_valid_candidate | {greason}",
                strength=None,
            )

        loc = best_candidate

        # --- (C) ROI track to reduce jitter ---
        proposal = loc.grid_box  # GridBox object, not tuple
        upd = self._tracker.update(proposal)
        roi_grid = upd.roi
        if roi_grid is None:
            should, mass_ema, state, greason = self.gate.step(0.0)
            return DefenseDecision(
                should_purify=False,
                roi_box=None,
                grid_box=None,
                mass_ema=float(mass_ema),
                raw_mass=float(loc.mass),
                state=str(state),
                reason=f"tracker_empty | {greason}",
                strength=None,
            )

        # --- (D) score on stable grid (NOT on raw heatmap) ---
        stable_mass = float(self._scorer.score(stable_grid, roi_grid))
        # Calculate area penalty: penalize large ROIs (often gripper/object), favor compact patches
        x0, y0, x1, y1 = roi_grid.gx0, roi_grid.gy0, roi_grid.gx1, roi_grid.gy1
        roi_area = float(max(0, x1 - x0) * max(0, y1 - y0))
        grid_area = float(stable_grid.size)
        area_frac = float(roi_area / (grid_area + 1e-12))
        # Penalize large ROIs (often gripper/object), keep score in [0, stable_mass]
        # area_ref ~ (4x4)/256 for 16x16 grid, typical patch size
        area_ref = 0.06
        penalty = float(min(1.0, area_ref / (area_frac + 1e-12)))
        score = float(stable_mass * penalty)

        # verification fail blocking (avoid repeated false masking)
        if self._verify_block_left > 0:
            self._verify_block_left -= 1
            should, mass_ema, state, greason = self.gate.step(0.0)
            return DefenseDecision(
                should_purify=False,
                roi_box=None,
                grid_box=None,
                mass_ema=float(mass_ema),
                raw_mass=float(loc.mass),
                state=str(state),
                reason=f"verify_block({self._verify_block_left}) | score={score:.4f} stable_mass={stable_mass:.4f} "
                       f"area_frac={area_frac:.3f} penalty={penalty:.2f} | {loc.reason}",
                strength=None,
            )

        should, mass_ema, state, greason = self.gate.step(score)

        if not should:
            return DefenseDecision(
                should_purify=False,
                roi_box=None,
                grid_box=None,
                mass_ema=float(mass_ema),
                raw_mass=float(loc.mass),
                state=str(state),
                reason=f"gate_off | score={score:.4f} stable_mass={stable_mass:.4f} "
                       f"area_frac={area_frac:.3f} penalty={penalty:.2f} | {loc.reason} | {greason}",
                strength=None,
            )

        # --- (E) map ROI grid -> pixel PatchBox ---
        x0, y0, x1, y1 = roi_grid.gx0, roi_grid.gy0, roi_grid.gx1, roi_grid.gy1
        roi_box = self.hook.grid_bbox_to_patch_box(int(x0), int(y0), int(x1), int(y1))

        # --- (F) optional counterfactual verify (low frequency) ---
        do_verify = (
            (self.verifier is not None)
            and (image is not None)
            and (purify_fn is not None)
            and (forward_fn is not None)
            and (heatmap_fn is not None)
            and (self.verify_every_k > 0)
            and ((self._t % self.verify_every_k) == 0)
        )

        if do_verify:
            hm_before = heatmap_fn()
            
            # IMPORTANT: ensure the next forward produces a fresh attention/heatmap
            # Wrap forward_fn to clear hook cache before forward, ensuring verify reads new heatmap
            def _forward_with_clear(img: np.ndarray) -> Any:
                if hasattr(self.hook, "clear"):
                    self.hook.clear()
                return forward_fn(img)  # type: ignore[misc]

            vr = self.verifier.verify(
                image=image,
                roi_box=roi_box,
                purify_fn=purify_fn,
                forward_fn=_forward_with_clear,
                heatmap_fn=heatmap_fn,
                hm_before=hm_before,
            )
            if not vr.verified:
                self._verify_block_left = int(self.verify_block_frames)
                return DefenseDecision(
                    should_purify=False,
                    roi_box=None,
                    grid_box=None,
                    mass_ema=float(mass_ema),
                    raw_mass=float(loc.mass),
                    state=str(state),
                    reason=f"verify_fail -> block | stats={vr.stats}",
                    strength=None,
                )

        # --- (G) final decision ---
        strength = self._mass_to_strength(float(mass_ema))

        # Update prev_image for next frame motion detection
        if image is not None:
            self._prev_image = image.copy()

        return DefenseDecision(
            should_purify=True,
            roi_box=roi_box,
            grid_box=GridBox(gx0=int(x0), gy0=int(y0), gx1=int(x1), gy1=int(y1)),
            mass_ema=float(mass_ema),
            raw_mass=float(loc.mass),
            state=str(state),
            reason=f"TRIGGER | score={score:.4f} stable_mass={stable_mass:.4f} "
                   f"area_frac={area_frac:.3f} penalty={penalty:.2f} | motion_penalty_applied | {loc.reason} | {greason}",
            strength=strength,
        )


# =========================
# Unified interface (optional but aligns with __init__.py)
# =========================

@dataclass(frozen=True)
class UnifiedDefenseResult:
    """Unified result for both known-mode and auto-mode."""
    should_purify: bool
    roi_box: Optional[PatchBox]

    # Backward-compat fields
    is_anomaly: bool = False
    patch_mass: float = 0.0
    entropy: Optional[float] = None

    # Auto-mode fields
    mass_ema: float = 0.0
    state: str = "SAFE"
    reason: str = ""
    strength: Optional[float] = None

    # Optional visualization
    heatmap: Optional[np.ndarray] = None


class UnifiedDefenseInterface:
    """
    Unified defense interface:
      - mode="known": uses PatchAttentionAnomalyDetector with a known patch_box
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
        detector: Optional[PatchAttentionAnomalyDetector] = None,
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
        self.detector = detector

        # auto-mode params
        self.controller = controller

        if self.mode == "known":
            if self.patch_x is None or self.patch_y is None:
                raise ValueError("known mode requires patch_x/patch_y")
            if self.detector is None:
                raise ValueError("known mode requires detector")
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
    ) -> UnifiedDefenseResult:
        """
        Args:
            image/purify_fn/forward_fn/heatmap_fn: optional parameters for counterfactual verification.
            If not provided, verification is skipped (backward compatible).
        """
        if self.mode == "known":
            heatmap = self.hook.get_heatmap()
            patch_box = PatchBox(
                x0=int(self.patch_x),
                y0=int(self.patch_y),
                x1=int(self.patch_x) + int(self.patch_w),
                y1=int(self.patch_y) + int(self.patch_h),
            )
            d = self.detector.detect(heatmap=heatmap, patch_box=patch_box)
            return UnifiedDefenseResult(
                should_purify=bool(d.is_anomaly),
                roi_box=patch_box if d.is_anomaly else None,
                is_anomaly=bool(d.is_anomaly),
                patch_mass=float(d.patch_mass),
                entropy=d.entropy,
                mass_ema=float(d.patch_mass),
                state="TRIGGERED" if d.is_anomaly else "SAFE",
                reason="known_mode",
                strength=None,
                heatmap=heatmap if self.use_heatmap_for_viz else None,
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
        )
        heatmap = self.hook.get_heatmap() if self.use_heatmap_for_viz else None

        return UnifiedDefenseResult(
            should_purify=bool(dd.should_purify),
            roi_box=dd.roi_box,
            is_anomaly=bool(dd.should_purify),
            patch_mass=float(dd.raw_mass),  # Use raw_mass (instantaneous) instead of mass_ema (smoothed)
            entropy=None,
            mass_ema=float(dd.mass_ema),
            state=str(dd.state),
            reason=str(dd.reason),
            strength=dd.strength,
            heatmap=heatmap,
        )
