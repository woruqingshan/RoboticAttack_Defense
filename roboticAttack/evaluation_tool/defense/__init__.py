"""Online defense: attention-based detection, multimodal prior (GripperPrior, PatchSelector), localization, and purification."""

from .online_defense import OnlineAttentionHook

from .anomaly_detector import (
    PatchBox,
    DetectionResult,
    PatchAttentionAnomalyDetector,
    # Auto-mode components
    PatchAttentionLocalizer,
    DefenseDecision,
    OnlinePatchDefenseController,
    # Unified interface
    UnifiedDefenseResult,
    UnifiedDefenseInterface,
)

from .purifier import ImagePurifier

from .temporal import GridBox, TemporalGate, GateDecision

from .verifier import (
    CounterfactualVerifier,
    NoOpVerifier,
    VerifyResult,
    VerifierProtocol,
    normalized_entropy,
    roi_mass,
)

from .logging_utils import (
    defense_result_to_log_dict,
    format_defense_log_line,
    format_defense_geometry_lines,
    format_defense_selector_lines,
)

from .gripper_prior import GripperPrior, GripperPriorConfig, GripperPriorResult, GripperSegment2D
from .arm_skeleton_prior import (
    GeometryRuntimeContext,
    ArmSkeletonPrior,
    ArmSkeletonPriorConfig,
    LinkSegment2D,
    ArmSkeletonResult,
)
from .patch_selector import PatchSelector, PatchSelectorConfig, PatchSelectResult
from .safety_region import SafetyRegionConfig, SafetyRegionBundle, SafetyRegionBuilder
from .pixel_mask_refiner import PixelMaskRefinerConfig, PixelMaskRefineResult, PixelMaskRefiner
from .temporal_conflict import TemporalConflictConfig, TemporalConflictDecision, TemporalConflictResolver
from .mask_postprocess import erode_mask, dilate_mask, mask_to_tight_box
from .mask_metrics import mask_to_box, overlap_ratio, summarize_mask
from .recovery_metrics import (
    action_metrics,
    attention_metrics,
    binary_mask_metrics,
    mask_overlap_ratio,
    safe_float_dict,
    safe_json_value,
)
from .patch_mask_utils import (
    build_axis_aligned_patch_mask,
    box_to_mask,
    infer_patch_hw,
    mask_to_box_xyxy,
    patch_box_xyxy,
    roi_to_mask,
)
from .metrics_logger import JsonlMetricsLogger

__all__ = [
    # Sensor
    "OnlineAttentionHook",
    # Legacy detector (known/oracle)
    "PatchAttentionAnomalyDetector",
    "DetectionResult",
    "PatchBox",
    # Auto-mode localization + temporal control
    "GridBox",
    "PatchAttentionLocalizer",
    "TemporalGate",
    "DefenseDecision",
    "OnlinePatchDefenseController",
    # Unified interface
    "UnifiedDefenseResult",
    "UnifiedDefenseInterface",
    # Purifier
    "ImagePurifier",
    "GateDecision",
    # New decoupled modules: verifier
    "CounterfactualVerifier",
    "NoOpVerifier",
    "VerifyResult",
    "VerifierProtocol",
    "normalized_entropy",
    "roi_mass",
    # Logging helpers
    "defense_result_to_log_dict",
    "format_defense_log_line",
    "format_defense_geometry_lines",
    "format_defense_selector_lines",
    # Multimodal geometry prior components
    "GripperPrior",
    "GripperPriorConfig",
    "GripperPriorResult",
    "GripperSegment2D",
    "GeometryRuntimeContext",
    "ArmSkeletonPrior",
    "ArmSkeletonPriorConfig",
    "LinkSegment2D",
    "ArmSkeletonResult",
    "PatchSelector",
    "PatchSelectorConfig",
    "PatchSelectResult",
    "SafetyRegionConfig",
    "SafetyRegionBundle",
    "SafetyRegionBuilder",
    "PixelMaskRefinerConfig",
    "PixelMaskRefineResult",
    "PixelMaskRefiner",
    "TemporalConflictConfig",
    "TemporalConflictDecision",
    "TemporalConflictResolver",
    "erode_mask",
    "dilate_mask",
    "mask_to_tight_box",
    "mask_to_box",
    "overlap_ratio",
    "summarize_mask",
    "action_metrics",
    "attention_metrics",
    "binary_mask_metrics",
    "mask_overlap_ratio",
    "safe_float_dict",
    "safe_json_value",
    "build_axis_aligned_patch_mask",
    "box_to_mask",
    "infer_patch_hw",
    "mask_to_box_xyxy",
    "patch_box_xyxy",
    "roi_to_mask",
    "JsonlMetricsLogger",
]
