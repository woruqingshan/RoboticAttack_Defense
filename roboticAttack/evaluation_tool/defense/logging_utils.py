"""Structured logging helpers for the defense pipeline.

Design goals:
- Keep the main evaluation script clean (no ad-hoc string parsing).
- Provide consistent fields across modes (KNOWN / ACQUIRE / TRACK).
- Avoid logging legacy metrics (e.g., patch_mass/entropy) unless the caller explicitly adds them.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional


def _box_to_dict(box: Any) -> Optional[Dict[str, int]]:
    """Convert PatchBox-like objects into a stable dict format."""
    if box is None:
        return None
    # Expect attributes x0,y0,x1,y1 (PatchBox) or tuple/list (x0,y0,x1,y1).
    if isinstance(box, (tuple, list)) and len(box) == 4:
        x0, y0, x1, y1 = box
        return {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)}
    for k in ["x0", "y0", "x1", "y1"]:
        if not hasattr(box, k):
            return None
    return {"x0": int(box.x0), "y0": int(box.y0), "x1": int(box.x1), "y1": int(box.y1)}


def defense_result_to_log_dict(result: Any) -> Dict[str, Any]:
    """Convert a UnifiedDefenseResult-like object into a JSON-serializable dict."""
    if is_dataclass(result):
        base = asdict(result)
    else:
        # Best-effort conversion for non-dataclass results.
        base = {k: getattr(result, k) for k in dir(result) if not k.startswith("_")}

    # Normalize ROI box.
    base["roi_box"] = _box_to_dict(getattr(result, "roi_box", None))

    # Ensure commonly used keys exist (avoid KeyError in scripts).
    for k in [
        "phase",
        "should_purify",
        "reacquire_needed",
        "verdict_code",
        "verify_performed",
        "verified",
        "mass_heatmap",
        "quality_ok",
        "quality_reason",
        "gate_checked",
        "mass_ema",
        "strength",
        "reason",
    ]:
        base.setdefault(k, None)

    # Remove any legacy keys if present.
    base.pop("patch_mass", None)
    base.pop("entropy", None)
    base.pop("is_anomaly", None)
    base.pop("state", None)

    return base


def format_defense_log_line(step: int, result: Any) -> str:
    """Format a single-line log for human reading (stable fields, no legacy metrics)."""
    d = defense_result_to_log_dict(result)
    roi = d.get("roi_box")
    roi_str = "None" if roi is None else f"{roi['x0']},{roi['y0']},{roi['x1']},{roi['y1']}"

    parts = [
        f"step={int(step)}",
        f"phase={d.get('phase')}",
        f"mask={bool(d.get('should_purify'))}",
        f"roi={roi_str}",
        f"reacquire={d.get('reacquire_needed')}",
        f"quality_ok={d.get('quality_ok')}",
        f"mass_heatmap={d.get('mass_heatmap')}",
        f"gate_checked={d.get('gate_checked')}",
        f"verdict={d.get('verdict_code')}",
        f"verified={d.get('verified')}",
    ]
    reason = d.get("reason")
    if isinstance(reason, str) and reason:
        parts.append(f"reason={reason}")
    return " | ".join(parts)


