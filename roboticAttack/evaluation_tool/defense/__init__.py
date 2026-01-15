"""Online defense utilities (attention-based detection + localization + purification)."""

from .online_defense import OnlineAttentionHook

from .anomaly_detector import (
    PatchBox,
    DetectionResult,
    PatchAttentionAnomalyDetector,
    # Auto-mode components
    GridBox,
    LocalizationResult,
    PatchAttentionLocalizer,
    TemporalGate,
    TemporalGateState,
    DefenseDecision,
    OnlinePatchDefenseController,
    # Unified interface
    UnifiedDefenseResult,
    UnifiedDefenseInterface,
)

from .purifier import ImagePurifier

# New decoupled modules
from .temporal import (
    RunningStats2D,
    Stats2DResult,
    AttentionStabilityScorer,
    ROITracker,
    ROIUpdate,
    TemporalGate as NewTemporalGate,
    GateDecision,
)

from .localizer import (
    AttentionLocalizer,
    LocalizeResult,
)

from .verifier import (
    CounterfactualVerifier,
    VerifyResult,
    normalized_entropy,
    roi_mass,
)

__all__ = [
    # Sensor
    "OnlineAttentionHook",
    # Legacy detector (known/oracle)
    "PatchAttentionAnomalyDetector",
    "DetectionResult",
    "PatchBox",
    # Auto-mode localization + temporal control
    "GridBox",
    "LocalizationResult",
    "PatchAttentionLocalizer",
    "TemporalGate",
    "TemporalGateState",
    "DefenseDecision",
    "OnlinePatchDefenseController",
    # Unified interface
    "UnifiedDefenseResult",
    "UnifiedDefenseInterface",
    # Purifier
    "ImagePurifier",
    # New decoupled modules: temporal
    "RunningStats2D",
    "Stats2DResult",
    "AttentionStabilityScorer",
    "ROITracker",
    "ROIUpdate",
    "NewTemporalGate",
    "GateDecision",
    # New decoupled modules: localizer
    "AttentionLocalizer",
    "LocalizeResult",
    # New decoupled modules: verifier
    "CounterfactualVerifier",
    "VerifyResult",
    "normalized_entropy",
    "roi_mass",
]
