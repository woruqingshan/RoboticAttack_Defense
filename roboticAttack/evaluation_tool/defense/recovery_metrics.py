"""Recovery and mechanism metrics for VLA patch defense experiments.

The functions in this module are deliberately side-effect free. They only turn
actions, masks, and attention heatmaps into JSON-friendly scalar evidence for
paper analysis; they do not participate in the defense decision path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

EPS = 1e-8


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    """Convert numpy / torch-like inputs to a numpy array when possible."""
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)
    except Exception:
        return None


def safe_json_value(x: Any) -> Any:
    """Convert common numpy values into JSON-safe Python values.

    Non-finite floats are represented as None so JSONL stays standards-compliant.
    """
    if x is None:
        return None
    if isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, np.generic):
        return safe_json_value(x.item())
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return safe_json_value(x.item())
        return [safe_json_value(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [safe_json_value(v) for v in x]
    if isinstance(x, dict):
        return {str(k): safe_json_value(v) for k, v in x.items()}
    try:
        return float(x)
    except Exception:
        return str(x)


def safe_float_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe copy of a flat or nested metric dictionary."""
    return {str(k): safe_json_value(v) for k, v in dict(d).items()}


def _direction_error_deg(v_a: np.ndarray, v_b: np.ndarray) -> Optional[float]:
    """Angle between two vectors in degrees; None when either vector is zero."""
    na = float(np.linalg.norm(v_a))
    nb = float(np.linalg.norm(v_b))
    if na <= EPS or nb <= EPS:
        return None
    cos = float(np.clip(np.dot(v_a, v_b) / (na * nb + EPS), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def action_metrics(a_clean: Any, a_adv: Any, a_def: Any) -> Dict[str, Any]:
    """Compute paired action recovery metrics under the same environment state.

    Expected action layout is OpenVLA/LIBERO-style 7D:
    translation-like dims [0:3], rotation-like dims [3:6], gripper dim [-1].
    The function also works for shorter vectors and omits unavailable slices.
    """
    clean = _to_numpy(a_clean)
    adv = _to_numpy(a_adv)
    defended = _to_numpy(a_def)
    if clean is None or adv is None or defended is None:
        return {}

    clean = clean.astype(np.float32).reshape(-1)
    adv = adv.astype(np.float32).reshape(-1)
    defended = defended.astype(np.float32).reshape(-1)
    n = min(clean.size, adv.size, defended.size)
    if n <= 0:
        return {}
    clean = clean[:n]
    adv = adv[:n]
    defended = defended[:n]

    diff_adv = adv - clean
    diff_def = defended - clean
    d_adv_l1 = float(np.mean(np.abs(diff_adv)))
    d_def_l1 = float(np.mean(np.abs(diff_def)))
    d_adv_l2 = float(np.linalg.norm(diff_adv))
    d_def_l2 = float(np.linalg.norm(diff_def))

    out: Dict[str, Any] = {
        "action_l1_adv_to_clean": d_adv_l1,
        "action_l1_def_to_clean": d_def_l1,
        "action_l2_adv_to_clean": d_adv_l2,
        "action_l2_def_to_clean": d_def_l2,
        "nar_l2": float(1.0 - d_def_l2 / (d_adv_l2 + EPS)),
        "action_dim": int(n),
    }

    if n >= 3:
        d_adv_xyz = float(np.linalg.norm(adv[:3] - clean[:3]))
        d_def_xyz = float(np.linalg.norm(defended[:3] - clean[:3]))
        out.update(
            {
                "action_l2_xyz_adv_to_clean": d_adv_xyz,
                "action_l2_xyz_def_to_clean": d_def_xyz,
                "nar_xyz": float(1.0 - d_def_xyz / (d_adv_xyz + EPS)),
                "direction_error_adv_deg": _direction_error_deg(adv[:3], clean[:3]),
                "direction_error_def_deg": _direction_error_deg(defended[:3], clean[:3]),
            }
        )

    if n >= 6:
        d_adv_rot = float(np.linalg.norm(adv[3:6] - clean[3:6]))
        d_def_rot = float(np.linalg.norm(defended[3:6] - clean[3:6]))
        out.update(
            {
                "action_l2_rot_adv_to_clean": d_adv_rot,
                "action_l2_rot_def_to_clean": d_def_rot,
                "nar_rot": float(1.0 - d_def_rot / (d_adv_rot + EPS)),
            }
        )

    if n >= 7:
        adv_grip_diff = float(abs(float(adv[-1]) - float(clean[-1])))
        def_grip_diff = float(abs(float(defended[-1]) - float(clean[-1])))
        out.update(
            {
                "gripper_abs_adv_to_clean": adv_grip_diff,
                "gripper_abs_def_to_clean": def_grip_diff,
                "gripper_match_adv": bool(np.sign(adv[-1]) == np.sign(clean[-1])),
                "gripper_match_def": bool(np.sign(defended[-1]) == np.sign(clean[-1])),
            }
        )

    return safe_float_dict(out)


def binary_mask_metrics(mask: Any, patch_mask: Any, prefix: str = "mask") -> Dict[str, Any]:
    """Compute binary mask-vs-patch coverage metrics."""
    m_arr = _to_numpy(mask)
    p_arr = _to_numpy(patch_mask)
    if m_arr is None or p_arr is None or m_arr.ndim != 2 or p_arr.ndim != 2:
        return {}
    if m_arr.shape != p_arr.shape:
        return {}

    m = m_arr.astype(bool)
    p = p_arr.astype(bool)
    inter = float(np.logical_and(m, p).sum())
    union = float(np.logical_or(m, p).sum())
    m_area = float(m.sum())
    p_area = float(p.sum())
    image_area = float(max(m.size, 1))

    out = {
        f"{prefix}_patch_iou": inter / max(union, 1.0),
        f"{prefix}_patch_recall": inter / max(p_area, 1.0),
        f"{prefix}_patch_precision": inter / max(m_area, 1.0),
        f"{prefix}_area_ratio": m_area / image_area,
        "patch_area_ratio": p_area / image_area,
    }
    return safe_float_dict(out)


def mask_overlap_ratio(mask: Any, region_mask: Any, name: str) -> Dict[str, Any]:
    """Compute fraction of mask area overlapping a protected region."""
    m_arr = _to_numpy(mask)
    r_arr = _to_numpy(region_mask)
    if m_arr is None or r_arr is None or m_arr.ndim != 2 or r_arr.ndim != 2:
        return {}
    if m_arr.shape != r_arr.shape:
        return {}
    m = m_arr.astype(bool)
    r = r_arr.astype(bool)
    denom = float(max(int(m.sum()), 1))
    return {f"{name}_overlap": float(np.logical_and(m, r).sum() / denom)}


def attention_metrics(
    heatmap: Any,
    patch_mask: Any,
    top_quantile: float = 0.90,
    prefix: str = "",
) -> Dict[str, Any]:
    """Compute patch attention mass and attention localization metrics."""
    h_arr = _to_numpy(heatmap)
    p_arr = _to_numpy(patch_mask)
    if h_arr is None or p_arr is None or h_arr.ndim != 2 or p_arr.ndim != 2:
        return {}
    if h_arr.shape != p_arr.shape:
        return {}

    h = h_arr.astype(np.float32)
    h = h - float(np.nanmin(h))
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(h.sum())
    if total <= EPS:
        return {}

    p = p_arr.astype(bool)
    pam = float(h[p].sum() / max(total, EPS))

    q = float(np.clip(top_quantile, 0.0, 1.0))
    thresh = float(np.quantile(h, q))
    top_mask = h >= thresh
    inter = float(np.logical_and(top_mask, p).sum())
    union = float(np.logical_or(top_mask, p).sum())

    ys, xs = np.indices(h.shape)
    cx = float((h * xs).sum() / total)
    cy = float((h * ys).sum() / total)

    py, px = np.where(p)
    if px.size > 0:
        pcx = float(px.mean())
        pcy = float(py.mean())
        dist_patch = float(np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2))
    else:
        dist_patch = None

    stem = f"{prefix}_" if prefix else ""
    out = {
        f"{stem}pam": pam,
        f"{stem}topk_attn_iou_patch": inter / max(union, 1.0),
        f"{stem}attn_center_x": cx,
        f"{stem}attn_center_y": cy,
        f"{stem}attn_center_dist_patch": dist_patch,
    }
    return safe_float_dict(out)

