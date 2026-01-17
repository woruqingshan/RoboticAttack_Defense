# verifier.py
# -*- coding: utf-8 -*-
"""
Counterfactual verification for candidate ROI (V2: three-tier evidence).

This module is decoupled and depends only on numpy.
It uses user-provided callbacks:
- purify_fn(image, roi_box) -> purified_image
- forward_fn(purified_image) -> np.ndarray action vector (7-dim)
- heatmap_fn() -> np.ndarray heatmap (H x W), e.g., 224 x 224

Main API:
- CounterfactualVerifier.verify(...)

Three-tier verification:
1. Tier 1 (Island Suppression): Check if outlier ROI mass drops after purification
2. Tier 2 (Task Preservation): Check if mainland ROI mass does NOT drop significantly
3. Tier 3 (Action-level Evidence): Check if policy action changes after purification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np

# Accept either (x0,y0,x1,y1) or an object with .x0,.y0,.x1,.y1
BoxLike = Union[Tuple[int, int, int, int], Any]


def _as_xyxy(box: BoxLike) -> Tuple[int, int, int, int]:
    if isinstance(box, tuple) or isinstance(box, list):
        x0, y0, x1, y1 = box
        return int(x0), int(y0), int(x1), int(y1)
    # object with attributes
    return int(box.x0), int(box.y0), int(box.x1), int(box.y1)


def normalized_entropy(hm: np.ndarray, eps: float = 1e-8) -> float:
    """
    Normalized entropy in [0,1]. High => diffuse attention; Low => concentrated.
    """
    x = hm.astype(np.float32)
    x = x - float(x.min())
    s = float(x.sum())
    if s <= eps:
        return 1.0
    p = x.reshape(-1) / (s + eps)
    ent = -float((p * np.log(p + eps)).sum())
    ent /= float(np.log(p.size + eps))
    return float(ent)


def roi_mass(hm: np.ndarray, roi_box: BoxLike, eps: float = 1e-8) -> float:
    """
    ROI mass = sum(hm in ROI) / sum(hm)
    """
    x = hm.astype(np.float32)
    x = x - float(x.min())
    total = float(x.sum()) + eps

    x0, y0, x1, y1 = _as_xyxy(roi_box)
    H, W = x.shape[:2]
    x0 = max(0, min(x0, W))
    x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H))
    y1 = max(0, min(y1, H))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    return float(x[y0:y1, x0:x1].sum() / total)


@dataclass
class VerifyResult:
    """Counterfactual verification result with detailed diagnostics."""
    verified: bool
    stats: Dict[str, float]
    
    # Failure reason code (for logging and blocklist strategy)
    verdict_code: str  # "PASS" / "FAIL_TASK_HARM" / "FAIL_NO_SUPPRESSION" / 
                       # "FAIL_NO_ACTION_CHANGE" / "FAIL_INPUT_MISSING"
    
    # Calibration suggestions (for credibility/blocklist)
    credibility_delta: float  # Suggested credibility change (-1 to +1)
    block_suggest_frames: int  # Suggested block frames (0 means no block)
    
    # Optional human-readable reason description
    reason: str = ""  # Human-readable failure reason


@dataclass
class CounterfactualVerifier:
    """
    Enhanced counterfactual verifier with three-tier evidence:
    1. Island suppression (outlier ROI mass drop + entropy gain)
    2. Task preservation (mainland ROI mass should not drop significantly)
    3. Action-level evidence (policy output should change after purification)
    """
    # Tier 1: Island suppression (existing)
    min_mass_drop_rel: float = 0.15   # require roi_mass_after <= (1 - rel)*before
    min_entropy_gain_abs: float = 0.02
    # If False, entropy_gain becomes a soft signal (logged + affects credibility slightly),
    # and Tier-1 suppression is decided primarily by roi_mass_rel_drop.
    entropy_hard: bool = False
    
    # Tier 2: Task preservation (NEW)
    max_main_mass_drop_rel: float = 0.10  # mainland mass drop should not exceed this
    require_mainland_check: bool = True    # whether to require mainland ROI
    
    # Tier 3: Action-level evidence (NEW)
    min_action_diff_l2: float = 0.01      # absolute L2 threshold for action change
    min_action_diff_rel: float = 0.05     # relative threshold (5% change)
    min_gripper_diff: float = 0.1         # gripper-specific threshold
    use_action_verification: bool = True   # whether to check action change
    
    # Calibration weights (for credibility_delta calculation)
    credibility_pass_bonus: float = 0.1    # bonus when verification passes
    credibility_fail_task_penalty: float = -0.5  # penalty for task harm
    credibility_fail_suppression_penalty: float = -0.3  # penalty for no suppression
    credibility_fail_action_penalty: float = -0.2  # penalty for no action change
    
    # Block suggestion (for block_suggest_frames)
    block_frames_task_harm: int = 10      # block frames when task is harmed
    block_frames_no_suppression: int = 0  # don't block if just no suppression
    block_frames_no_action: int = 0       # don't block if just no action change
    
    eps: float = 1e-8

    def verify(
        self,
        image: np.ndarray,
        roi_box: BoxLike,
        purify_fn: Callable[[np.ndarray, BoxLike], np.ndarray],
        forward_fn: Callable[[np.ndarray], np.ndarray],  # returns action vector (7-dim)
        heatmap_fn: Callable[[], np.ndarray],
        *,
        hm_before: Optional[np.ndarray] = None,
        main_roi_box: Optional[BoxLike] = None,  # mainland ROI (task region, optional)
    ) -> VerifyResult:
        """
        Three-tier counterfactual verification:
        
        Tier 1 (Island Suppression):
            - Check if outlier ROI mass drops after purification
            - Check if entropy increases (attention diffuses)
        
        Tier 2 (Task Preservation):
            - Check if mainland ROI mass does NOT drop significantly
            - Only checked if main_roi_box is provided
        
        Tier 3 (Action-level Evidence):
            - Check if policy action changes after purification
            - Measures L2 distance and relative change
        
        Args:
            image: Original image (H, W, 3) numpy array
            roi_box: Outlier ROI box (PatchBox or (x0,y0,x1,y1))
            purify_fn: Function to purify ROI (masks the region)
            forward_fn: Function that runs policy forward pass, returns action vector (7-dim)
            heatmap_fn: Function that returns current heatmap from hooks
            hm_before: Optional precomputed heatmap for BEFORE state
            main_roi_box: Optional mainland ROI box (for task preservation check)
        
        Returns:
            VerifyResult with verified flag, stats, verdict_code, and calibration suggestions
        """
        # === BEFORE: Run policy on original image ===
        if hm_before is None:
            hm0 = heatmap_fn()
        else:
            hm0 = hm_before

        # Extract action from forward pass (forward_fn returns action vector directly)
        action_raw = forward_fn(image)
        if not isinstance(action_raw, np.ndarray):
            raise TypeError(f"forward_fn must return np.ndarray action vector, got {type(action_raw)}")
        if action_raw.ndim != 1 or action_raw.shape[0] != 7:
            raise ValueError(f"Expected 7-dim action vector, got shape={action_raw.shape}")

        # Compute BEFORE metrics
        m0_outlier = roi_mass(hm0, roi_box, eps=self.eps)
        e0 = normalized_entropy(hm0, eps=self.eps)

        m0_mainland: Optional[float] = None
        if main_roi_box is not None:
            m0_mainland = roi_mass(hm0, main_roi_box, eps=self.eps)

        # === AFTER: Purify and run policy again ===
        img_purified = purify_fn(image, roi_box)
        action_pur = forward_fn(img_purified)  # This is the +1 forward cost
        if not isinstance(action_pur, np.ndarray):
            raise TypeError(f"forward_fn must return np.ndarray action vector, got {type(action_pur)}")
        if action_pur.ndim != 1 or action_pur.shape[0] != 7:
            raise ValueError(f"Expected 7-dim action vector, got shape={action_pur.shape}")
        hm1 = heatmap_fn()

        # Compute AFTER metrics
        m1_outlier = roi_mass(hm1, roi_box, eps=self.eps)
        e1 = normalized_entropy(hm1, eps=self.eps)

        m1_mainland: Optional[float] = None
        if main_roi_box is not None:
            m1_mainland = roi_mass(hm1, main_roi_box, eps=self.eps)

        # === Tier 1: Island Suppression ===
        m0_safe = max(m0_outlier, float(self.eps))
        outlier_mass_drop = (m0_safe - m1_outlier) / m0_safe
        entropy_gain = e1 - e0
        mass_ok = outlier_mass_drop >= float(self.min_mass_drop_rel)
        entropy_ok = entropy_gain >= float(self.min_entropy_gain_abs)
        island_suppressed = bool(mass_ok and (entropy_ok if bool(self.entropy_hard) else True))

        # === Tier 2: Task Preservation ===
        task_preserved = True
        main_mass_drop = 0.0
        if main_roi_box is not None and m0_mainland is not None:
            m0_main_safe = max(m0_mainland, float(self.eps))
            if m1_mainland is not None:
                main_mass_drop = (m0_main_safe - m1_mainland) / m0_main_safe
            task_preserved = main_mass_drop <= float(self.max_main_mass_drop_rel)
        elif self.require_mainland_check and main_roi_box is None:
            # If mainland check is required but not provided, fail
            task_preserved = False

        # === Tier 3: Action-level Evidence ===
        action_changed = True
        action_diff_l2 = 0.0
        action_diff_rel = 0.0
        gripper_diff = 0.0

        if self.use_action_verification:
            action_diff_l2 = float(np.linalg.norm(action_raw - action_pur))
            action_norm = float(np.linalg.norm(action_raw))
            action_diff_rel = action_diff_l2 / (action_norm + self.eps)
            gripper_diff = float(abs(action_raw[6] - action_pur[6]))  # Last dim is gripper

            action_changed = (
                action_diff_l2 >= float(self.min_action_diff_l2)
                or action_diff_rel >= float(self.min_action_diff_rel)
                or gripper_diff >= float(self.min_gripper_diff)
            )

        # === Final Verdict ===
        if not island_suppressed:
            verdict_code = "FAIL_NO_SUPPRESSION"
            credibility_delta = float(self.credibility_fail_suppression_penalty)
            block_suggest_frames = int(self.block_frames_no_suppression)
            if not mass_ok:
                reason = f"Island not suppressed: mass_drop={outlier_mass_drop:.3f} < {self.min_mass_drop_rel}"
            else:
                # Only reachable when entropy_hard=True
                reason = f"Island not suppressed: entropy_gain={entropy_gain:.3f} < {self.min_entropy_gain_abs}"
        elif not task_preserved:
            verdict_code = "FAIL_TASK_HARM"
            credibility_delta = float(self.credibility_fail_task_penalty)
            block_suggest_frames = int(self.block_frames_task_harm)
            reason = f"Task region harmed: main_mass_drop={main_mass_drop:.3f} > {self.max_main_mass_drop_rel}"
        elif not action_changed:
            verdict_code = "FAIL_NO_ACTION_CHANGE"
            credibility_delta = float(self.credibility_fail_action_penalty)
            block_suggest_frames = int(self.block_frames_no_action)
            reason = (
                f"Action unchanged: diff_l2={action_diff_l2:.4f} < {self.min_action_diff_l2}, "
                f"diff_rel={action_diff_rel:.3f} < {self.min_action_diff_rel}, "
                f"gripper_diff={gripper_diff:.3f} < {self.min_gripper_diff}"
            )
        else:
            verdict_code = "PASS"
            credibility_delta = float(self.credibility_pass_bonus)
            block_suggest_frames = 0
            # Soft entropy signal: if entropy_hard=False, we keep PASS but slightly downweight confidence.
            if (not bool(self.entropy_hard)) and (not entropy_ok) and (float(self.min_entropy_gain_abs) > 0.0):
                credibility_delta = float(credibility_delta - 0.05)
                reason = (
                    "All hard checks passed (entropy soft-fail): "
                    f"entropy_gain={entropy_gain:.3f} < {self.min_entropy_gain_abs}"
                )
            else:
                reason = "All checks passed"

        # === Build Stats Dict ===
        stats = {
            # Tier 1: Island suppression
            "roi_mass_before": float(m0_outlier),
            "roi_mass_after": float(m1_outlier),
            "roi_mass_rel_drop": float(outlier_mass_drop),
            "entropy_before": float(e0),
            "entropy_after": float(e1),
            "entropy_gain": float(entropy_gain),
            "tier1_mass_ok": 1.0 if mass_ok else 0.0,
            "tier1_entropy_ok": 1.0 if entropy_ok else 0.0,
            
            # Tier 2: Task preservation
            "main_mass_before": float(m0_mainland) if m0_mainland is not None else 0.0,
            "main_mass_after": float(m1_mainland) if m1_mainland is not None else 0.0,
            "main_mass_drop": float(main_mass_drop),
            
            # Tier 3: Action evidence
            "action_diff_l2": float(action_diff_l2),
            "action_diff_rel": float(action_diff_rel),
            "gripper_diff": float(gripper_diff),
        }

        return VerifyResult(
            verified=(verdict_code == "PASS"),
            stats=stats,
            verdict_code=verdict_code,
            credibility_delta=credibility_delta,
            block_suggest_frames=block_suggest_frames,
            reason=reason,
        )

