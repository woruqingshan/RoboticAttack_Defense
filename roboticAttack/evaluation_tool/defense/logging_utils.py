"""Structured logging helpers for the defense pipeline.

Converts DefenseDecision/UnifiedDefenseResult to a JSON-serializable dict and a
one-line log string. Supports phase, should_purify, roi_box, patch_verdict,
gripper_box, and legacy PRAC/verifier fields. Keeps the main evaluation script
free of ad-hoc parsing; consistent across KNOWN / ACQUIRE / TRACK / LOCKED.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from .geometry_alignment import points_rc_to_xyxy_box
from .mask_metrics import summarize_mask


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
    base["gripper_box"] = _box_to_dict(getattr(result, "gripper_box", None))
    base["arm_region_box"] = _box_to_dict(getattr(result, "arm_region_box", None))
    base["arm_core_box"] = _box_to_dict(getattr(result, "arm_core_box", None))
    base["arm_guard_box"] = _box_to_dict(getattr(result, "arm_guard_box", None))
    base["roi_mask_stats"] = summarize_mask(getattr(result, "roi_mask", None))
    # Keep logs JSON-safe: drop raw ndarray payload.
    base.pop("roi_mask", None)

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
        # PRAC fields
        "prac_performed",
        "prac_verdict",
        "prac_odr",
        "prac_mer",
        "prac_stats",
        # New geometric fields
        "patch_verdict",
        "gripper_box",
        "arm_region_box",
        "arm_core_box",
        "arm_guard_box",
        "joint_points_2d",
        "gripper_points_2d",
        "gripper_points_render_2d",
        "gripper_point_keys_used",
        "gripper_link_name_pairs",
        "gripper_link_segments_2d",
        "gripper_link_quads_2d",
        "joint_names_used",
        "joint_points_render_2d",
        "arm_link_name_pairs",
        "arm_link_segments_2d",
        "arm_link_quads_2d",
        "roi_mask_stats",
        "conflict_mode",
        "conflict_reason",
        "conflict_stats",
        "selector_debug",
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
    
    if d.get("patch_verdict"):
        parts.append(f"patch={d.get('patch_verdict')}")
    if d.get("roi_mask_stats") is not None:
        parts.append(f"mask_area={d.get('roi_mask_stats', {}).get('area', 0)}")
    if d.get("conflict_mode"):
        parts.append(f"conflict={d.get('conflict_mode')}")
    if d.get("arm_core_box") is not None:
        parts.append(f"arm_core={d.get('arm_core_box')}")
    if d.get("arm_guard_box") is not None:
        parts.append(f"arm_guard={d.get('arm_guard_box')}")
    if d.get("gripper_points_2d") is not None:
        parts.append(f"gripper_pts={len(d.get('gripper_points_2d') or [])}")
    if d.get("gripper_link_segments_2d") is not None:
        parts.append(f"gripper_links={len(d.get('gripper_link_segments_2d') or [])}")
    if d.get("joint_points_2d") is not None:
        parts.append(f"joints={len(d.get('joint_points_2d') or [])}")
    if d.get("arm_link_segments_2d") is not None:
        parts.append(f"links={len(d.get('arm_link_segments_2d') or [])}")
    
    # Add PRAC fields if available
    if d.get("prac_performed"):
        prac_verdict = d.get("prac_verdict", "N/A")
        prac_odr = d.get("prac_odr")
        prac_mer = d.get("prac_mer")
        prac_str = f"PRAC={prac_verdict}"
        if prac_odr is not None:
            prac_str += f",ODR={prac_odr:.3f}"
        if prac_mer is not None:
            prac_str += f",MER={prac_mer:.3f}"
        parts.append(prac_str)
    
    reason = d.get("reason")
    if isinstance(reason, str) and reason:
        parts.append(f"reason={reason}")
    conflict_reason = d.get("conflict_reason")
    if isinstance(conflict_reason, str) and conflict_reason:
        parts.append(f"conflict_reason={conflict_reason}")
    return " | ".join(parts)


def _rc_range(points: Any) -> Optional[str]:
    """Summarize a list of (row, col) points with explicit axis names."""
    if not isinstance(points, list) or not points:
        return None
    try:
        rows = [int(p[0]) for p in points]
        cols = [int(p[1]) for p in points]
    except Exception:
        return None
    return f"col=[{min(cols)},{max(cols)}] row=[{min(rows)},{max(rows)}]"


def _rc_box(points: Any) -> Optional[str]:
    """Summarize a list of (row, col) points as an xyxy box."""
    if not isinstance(points, list) or not points:
        return None
    try:
        max_row = max(int(p[0]) for p in points) + 1
        max_col = max(int(p[1]) for p in points) + 1
        box = points_rc_to_xyxy_box(points, hw=(max_row, max_col))
    except Exception:
        return None
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return f"xyxy=({x0},{y0},{x1},{y1})"


def format_defense_geometry_lines(step: int, result: Any) -> List[str]:
    """Format verbose geometry debug lines for terminal output."""
    d = defense_result_to_log_dict(result)
    lines: List[str] = []

    gripper_policy = d.get("gripper_points_2d")
    gripper_render = d.get("gripper_points_render_2d")
    gripper_names = d.get("gripper_point_keys_used")
    gripper_links = d.get("gripper_link_name_pairs")
    gripper_segments = d.get("gripper_link_segments_2d")
    if gripper_policy:
        summary = _rc_range(gripper_policy)
        box_summary = _rc_box(gripper_policy)
        lines.append(
            f"[DEFENSE][GEOM][GRIPPER] step={int(step)} names={gripper_names} "
            f"render_rc={gripper_render} policy_rc={gripper_policy}"
            + (f" | {summary}" if summary else "")
            + (f" | {box_summary}" if box_summary else "")
        )
    if gripper_links and gripper_segments and len(gripper_links) == len(gripper_segments):
        link_entries = [
            (gripper_links[i][0], gripper_links[i][1], gripper_segments[i][0], gripper_segments[i][1])
            for i in range(len(gripper_links))
        ]
        lines.append(f"[DEFENSE][GEOM][GRIPPER_LINKS] step={int(step)} segments={link_entries}")
    elif gripper_links:
        lines.append(f"[DEFENSE][GEOM][GRIPPER_LINKS] step={int(step)} links={gripper_links}")

    joint_policy = d.get("joint_points_2d")
    joint_render = d.get("joint_points_render_2d")
    joint_names = d.get("joint_names_used")
    arm_links = d.get("arm_link_name_pairs")
    arm_segments = d.get("arm_link_segments_2d")
    if joint_policy:
        summary = _rc_range(joint_policy)
        box_summary = _rc_box(joint_policy)
        lines.append(
            f"[DEFENSE][GEOM][ARM] step={int(step)} joints={joint_names} "
            f"render_rc={joint_render} policy_rc={joint_policy}"
            + (f" | {summary}" if summary else "")
            + (f" | {box_summary}" if box_summary else "")
        )
    if arm_links and arm_segments and len(arm_links) == len(arm_segments):
        link_entries = [
            (arm_links[i][0], arm_links[i][1], arm_segments[i][0], arm_segments[i][1])
            for i in range(len(arm_links))
        ]
        lines.append(f"[DEFENSE][GEOM][ARM_LINKS] step={int(step)} segments={link_entries}")
    elif arm_links:
        lines.append(f"[DEFENSE][GEOM][ARM_LINKS] step={int(step)} links={arm_links}")

    return lines


def format_defense_selector_lines(step: int, result: Any) -> List[str]:
    """Format selector debug lines for terminal output."""
    d = defense_result_to_log_dict(result)
    selector_debug = d.get("selector_debug")
    if not isinstance(selector_debug, dict):
        return []
    if not bool(selector_debug.get("enabled", False)):
        return []

    def _fmt(val: Any) -> str:
        if val is None:
            return "None"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    lines: List[str] = []
    summary = (
        f"[DEFENSE][SELECTOR_SUMMARY] step={int(step)} "
        f"verdict={_fmt(selector_debug.get('verdict'))} "
        f"selected={_fmt(selector_debug.get('selected'))} "
        f"selected_bucket={_fmt(selector_debug.get('selected_bucket'))} "
        f"selected_raw_score={_fmt(selector_debug.get('selected_raw_score'))} "
        f"selected_diagnostic_score={_fmt(selector_debug.get('selected_diagnostic_score'))} "
        f"reason={_fmt(selector_debug.get('reason'))}"
    )
    lines.append(summary)

    for entry in selector_debug.get("topk", []) or []:
        line = (
            f"[DEFENSE][SELECTOR_TOPK] step={int(step)} "
            f"rank_input={_fmt(entry.get('rank_input'))} "
            f"bucket={_fmt(entry.get('bucket'))} "
            f"roi_grid={_fmt(entry.get('roi_grid'))} "
            f"raw_score={_fmt(entry.get('raw_score'))} "
            f"diagnostic_score={_fmt(entry.get('diagnostic_score'))} "
            f"area_ratio={_fmt(entry.get('area_ratio'))} "
            f"aspect={_fmt(entry.get('aspect'))} "
            f"size_prior={_fmt(entry.get('size_prior'))} "
            f"square_prior={_fmt(entry.get('square_prior'))} "
            f"corner_prior={_fmt(entry.get('corner_prior'))} "
            f"iou_g={_fmt(entry.get('iou_g'))} "
            f"gripper_core_overlap={_fmt(entry.get('gripper_core_overlap'))} "
            f"gripper_guard_overlap={_fmt(entry.get('gripper_guard_overlap'))} "
            f"iou_arm={_fmt(entry.get('iou_arm'))} "
            f"arm_core_overlap={_fmt(entry.get('arm_core_overlap'))} "
            f"arm_guard_overlap={_fmt(entry.get('arm_guard_overlap'))} "
            f"over_g={_fmt(entry.get('over_g'))} "
            f"over_arm={_fmt(entry.get('over_arm'))}"
        )
        lines.append(line)

    return lines


