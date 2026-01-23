"""
Anomaly detection + localization + temporal gating for attention-based patch defense.

This module keeps the legacy detector API:
    PatchAttentionAnomalyDetector.detect(heatmap, patch_box) -> DetectionResult

And extends the system with:
- PatchAttentionLocalizer: localize suspicious ROI from a low-res saliency grid
- TemporalGate: minimal ON/OFF hysteresis gate (continuous masking)
- OnlinePatchDefenseController: localize + gate -> pixel ROI decision
- UnifiedDefenseInterface: one-step interface (known/oracle vs auto/localize modes)

Design goals:
- Backward compatible: existing scripts that assume known patch location still work.
- No new dependencies: connected components are implemented via pure NumPy.
- Practical control: supports continuous masking with verifier-triggered reacquire.
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
)
from .verifier import VerifierProtocol, roi_mass as heatmap_roi_mass

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

        # --- Simplified controller state (ACQUIRE/TRACK) ---
        self._tracking: bool = False
        self._reacquire_needed: bool = True  # start by acquiring an ROI
        self._current_outlier_grid: Optional[GridBox] = None
        self._current_main_grid: Optional[GridBox] = None
        self._current_outlier_score: float = 0.0

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

        # --- (B) ACQUIRE/TRACK controller ---
        # ACQUIRE: run localizer only when needed; TRACK: keep using the last ROI.
        main_grid: Optional[GridBox] = self._current_main_grid
        roi_grid: Optional[GridBox] = self._current_outlier_grid
        raw_outlier_score: float = float(self._current_outlier_score)
        loc_reason = "track"
        loc_debug: dict = {}

        if (not bool(self._tracking)) or bool(self._reacquire_needed):
            tlr = self.localizer.localize(stable_grid)
            main_grid = tlr.main_roi
            roi_grid = tlr.outlier_roi
            raw_outlier_score = float(tlr.outlier_score)
            loc_reason = str(tlr.reason) if hasattr(tlr, "reason") else "ok"
            loc_debug = dict(tlr.debug) if isinstance(tlr.debug, dict) else {}

            # Update controller state (even if roi_grid is None; we want to drop stale ROIs).
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
        )
