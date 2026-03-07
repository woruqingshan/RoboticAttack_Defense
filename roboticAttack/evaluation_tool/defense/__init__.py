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

from .logging_utils import defense_result_to_log_dict, format_defense_log_line

from .gripper_prior import GripperPrior, GripperPriorConfig
from .patch_selector import PatchSelector, PatchSelectorConfig, PatchSelectResult

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
    # Multimodal geometry prior components
    "GripperPrior",
    "GripperPriorConfig",
    "PatchSelector",
    "PatchSelectorConfig",
    "PatchSelectResult",
]
