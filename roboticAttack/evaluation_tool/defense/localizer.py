# localizer.py
# -*- coding: utf-8 -*-
"""
ROI localizer for stable attention grids.

This module is decoupled and depends only on numpy.
It provides connected-component localization on a thresholded stable grid.

Main API:
- AttentionLocalizer.localize(stable_grid) -> Optional[LocalizeResult]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .temporal import GridBox, ROITracker, grid_iou


@dataclass
class LocalizeResult:
    roi: GridBox
    roi_mass: float
    area_ratio: float
    threshold: float


def _percentile_threshold(x: np.ndarray, top_p: float) -> float:
    """
    Compute a value threshold such that approximately top_p fraction of elements are kept.
    top_p in (0,1). Larger top_p keeps more pixels.
    """
    flat = x.reshape(-1)
    if flat.size == 0:
        return float("inf")
    # Keep top_p => drop (1-top_p)
    k = int(round((1.0 - float(top_p)) * float(flat.size)))
    k = max(0, min(k, flat.size - 1))
    return float(np.partition(flat, k)[k])


def _connected_components_4(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Minimal 4-neighborhood connected components labeling (BFS).
    Returns (num_labels, labels) with labels in {0..num}.
    """
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    label = 0

    for y in range(H):
        for x in range(W):
            if not bool(mask[y, x]) or labels[y, x] != 0:
                continue
            label += 1
            stack = [(y, x)]
            labels[y, x] = label
            while stack:
                cy, cx = stack.pop()
                # 4-neighbors
                if cy > 0 and mask[cy - 1, cx] and labels[cy - 1, cx] == 0:
                    labels[cy - 1, cx] = label
                    stack.append((cy - 1, cx))
                if cy + 1 < H and mask[cy + 1, cx] and labels[cy + 1, cx] == 0:
                    labels[cy + 1, cx] = label
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and labels[cy, cx - 1] == 0:
                    labels[cy, cx - 1] = label
                    stack.append((cy, cx - 1))
                if cx + 1 < W and mask[cy, cx + 1] and labels[cy, cx + 1] == 0:
                    labels[cy, cx + 1] = label
                    stack.append((cy, cx + 1))

    return label, labels


def _bbox_from_label(labels: np.ndarray, lab: int) -> Optional[GridBox]:
    ys, xs = np.where(labels == lab)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    return GridBox(gx0=x0, gy0=y0, gx1=x1, gy1=y1)


@dataclass
class AttentionLocalizer:
    """
    Localize suspicious ROI from a stable attention grid.

    Strategy:
    1) Normalize stable_grid (shift by min)
    2) threshold by top_p percentile
    3) CC labeling
    4) choose Top-K components with highest roi_mass under area constraints
    """
    top_p: float = 0.07
    min_area: float = 0.01   # relative to grid area
    max_area: float = 0.25
    eps: float = 1e-6

    def localize(self, stable_grid: np.ndarray, top_k: int = 1) -> List[LocalizeResult]:
        g = stable_grid.astype(np.float32)
        g = g - float(g.min())
        total_sum = float(g.sum())

        if total_sum <= self.eps:
            return []

        thr = _percentile_threshold(g, float(self.top_p))
        mask = g >= thr

        num, labels = _connected_components_4(mask)
        if num <= 0:
            return []

        H, W = g.shape
        total_area = float(H * W)

        candidates = []

        for lab in range(1, num + 1):
            box = _bbox_from_label(labels, lab)
            if box is None:
                continue
            # GridBox is a dataclass, access attributes instead of unpacking
            x0, y0, x1, y1 = box.gx0, box.gy0, box.gx1, box.gy1
            area = float((x1 - x0) * (y1 - y0))
            area_ratio = area / (total_area + self.eps)
            if area_ratio < float(self.min_area) or area_ratio > float(self.max_area):
                continue

            roi_sum = float(g[y0:y1, x0:x1].sum())
            roi_mass = roi_sum / (total_sum + self.eps)

            candidates.append(LocalizeResult(
                roi=box,
                roi_mass=float(roi_mass),
                area_ratio=float(area_ratio),
                threshold=float(thr),
            ))

        # Sort by density (mass/area) descending, prefer compact & peaky regions
        # Patches are typically small and dense (high density), while gripper/object
        # regions are often large with high mass but lower density
        # Use density as primary sort key, roi_mass as secondary tie-breaker
        candidates.sort(key=lambda x: (x.roi_mass / (x.area_ratio + self.eps), x.roi_mass), reverse=True)
        return candidates[:top_k]


# ------------------------------
# New: component extraction + temporal outlier localizer (v3)
# ------------------------------

from dataclasses import field
from typing import Dict, Any


@dataclass
class ComponentStats:
    """Connected-component stats on a thresholded stable grid."""
    label: int
    roi: GridBox
    roi_mass: float
    area_ratio: float
    density: float
    peak: float
    centroid: Tuple[float, float]


@dataclass
class FrameComponents:
    """Per-frame components extracted from a stable grid."""
    components: List[ComponentStats]
    main_component: Optional[ComponentStats]
    threshold: float
    total_mass: float


@dataclass
class TemporalLocalizeResult:
    """Temporal outlier localization output.

    This output is designed for debugging and downstream decision-making.
    """
    main_roi: Optional[GridBox]
    outlier_roi: Optional[GridBox]
    outlier_score: float
    threshold: float
    chosen_label: Optional[int]
    main_label: Optional[int]
    reason: str
    debug: Dict[str, Any] = field(default_factory=dict)


def _extract_components(
    stable_grid: np.ndarray,
    *,
    top_p: float,
    min_area: float,
    max_area: float,
    eps: float,
) -> FrameComponents:
    """Extract connected components from a stable grid.

    Args:
        stable_grid: 2D stable score grid.
        top_p: keep approximately top_p fraction of pixels.
        min_area: min area ratio to keep.
        max_area: max area ratio to keep.
        eps: small number for numerical stability.

    Returns:
        FrameComponents with per-component statistics + a single main_component candidate.
    """
    g = stable_grid.astype(np.float32)
    g = g - float(g.min())
    total_sum = float(g.sum())
    if total_sum <= float(eps):
        return FrameComponents(components=[], main_component=None, threshold=float("inf"), total_mass=0.0)

    thr = _percentile_threshold(g, float(top_p))
    mask = g >= thr
    num, labels = _connected_components_4(mask)
    if num <= 0:
        return FrameComponents(components=[], main_component=None, threshold=float(thr), total_mass=float(total_sum))

    H, W = g.shape
    total_area = float(H * W)

    comps: List[ComponentStats] = []
    for lab in range(1, num + 1):
        box = _bbox_from_label(labels, lab)
        if box is None:
            continue

        x0, y0, x1, y1 = box.gx0, box.gy0, box.gx1, box.gy1
        area = float(max(0, x1 - x0) * max(0, y1 - y0))
        if area <= 0:
            continue
        area_ratio = area / total_area
        if area_ratio < float(min_area) or area_ratio > float(max_area):
            continue

        roi_patch = g[y0:y1, x0:x1]
        roi_mass = float(roi_patch.sum()) / (float(total_sum) + float(eps))
        peak = float(roi_patch.max()) if roi_patch.size > 0 else 0.0
        density = float(roi_mass / (area_ratio + float(eps)))

        ys, xs = np.where(labels == lab)
        if xs.size == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())

        comps.append(ComponentStats(
            label=int(lab),
            roi=box,
            roi_mass=float(roi_mass),
            area_ratio=float(area_ratio),
            density=float(density),
            peak=float(peak),
            centroid=(cx, cy),
        ))

    # Main component heuristic: largest roi_mass (fallback to density if tie)
    main_comp: Optional[ComponentStats] = None
    if comps:
        main_comp = max(comps, key=lambda c: (c.roi_mass, c.area_ratio, c.density))

    return FrameComponents(components=comps, main_component=main_comp, threshold=float(thr), total_mass=float(total_sum))


def _center_prior_box(H: int, W: int, radius_frac: float) -> GridBox:
    """Build a center prior box in grid coordinates."""
    r = max(1, int(round(min(H, W) * float(radius_frac))))
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    x0 = int(max(0, round(cx - r)))
    x1 = int(min(W, round(cx + r + 1)))
    y0 = int(max(0, round(cy - r)))
    y1 = int(min(H, round(cy + r + 1)))
    return GridBox(gx0=x0, gy0=y0, gx1=x1, gy1=y1)


def _select_mainland(components: List[ComponentStats], center_box: GridBox) -> Optional[ComponentStats]:
    """Select a 'mainland' component that likely corresponds to task-relevant attention.

    Priority:
        1) Components overlapping the center prior box: maximize overlap IoU with center_box
        2) Fallback: maximize roi_mass
    """
    if not components:
        return None

    best: Optional[ComponentStats] = None
    best_key = (-1.0, -1.0)
    for c in components:
        ov = grid_iou(c.roi, center_box)
        key = (float(ov), float(c.roi_mass))
        if key > best_key:
            best_key = key
            best = c

    if best is not None and best_key[0] > 0.0:
        return best

    return max(components, key=lambda c: (c.roi_mass, c.area_ratio, c.density))


@dataclass
class TemporalPatchAttentionLocalizer:
    """Temporal 'high-attention island' localizer.

    This localizer reduces false positives where the detector mistakenly selects
    the gripper/object region (typically centered and spatially continuous)
    instead of an adversarial patch region (often off-center and isolated).

    Pipeline:
      - Extract components per frame from stable grid.
      - Identify task-relevant 'mainland' with center prior.
      - Score other components as outlier islands (weighted formula).
      - Use ROITracker + keepalive + winner-keep to reduce flicker.

    Output:
      - mainland ROI (for debugging)
      - outlier ROI (candidate patch region on the grid)
      - outlier_score + rich debug fields
    """
    # Component extraction parameters
    top_p: float = 0.07
    min_area: float = 0.01
    max_area: float = 0.30
    eps: float = 1e-6

    # Mainland selection prior (center region)
    mainland_center_radius_frac: float = 0.25

    # Association / stability
    assoc_iou_thr: float = 0.10
    keepalive_frames: int = 5
    switch_margin: float = 0.05  # require score improvement to switch tracks

    # Scoring weights (see spec)
    w_mass: float = 1.2
    w_peak: float = 0.3
    w_density: float = 0.2
    w_dist: float = 0.6
    w_area_penalty: float = 0.5
    w_center_penalty: float = 0.6
    w_mainland_iou_penalty: float = 1.0
    center_sigma: float = 0.35  # normalized by diag

    def __post_init__(self) -> None:
        self._tracker = ROITracker(iou_keep=0.30, ema=0.5)
        self._last_roi: Optional[GridBox] = None
        self._last_score: float = 0.0
        self._miss_left: int = 0
        self._frame_idx: int = 0
        self._last_label: Optional[int] = None
        self._last_main_label: Optional[int] = None

    def reset(self) -> None:
        self._tracker.reset()
        self._last_roi = None
        self._last_score = 0.0
        self._miss_left = 0
        self._frame_idx = 0
        self._last_label = None
        self._last_main_label = None

    def localize(self, stable_grid: np.ndarray) -> TemporalLocalizeResult:
        """Return mainland ROI + best outlier island ROI on this frame."""
        self._frame_idx += 1

        H, W = stable_grid.shape
        center_box = _center_prior_box(H, W, float(self.mainland_center_radius_frac))

        fc = _extract_components(
            stable_grid,
            top_p=float(self.top_p),
            min_area=float(self.min_area),
            max_area=float(self.max_area),
            eps=float(self.eps),
        )

        if not fc.components:
            if self._last_roi is not None and self._miss_left > 0:
                self._miss_left -= 1
                return TemporalLocalizeResult(
                    main_roi=None,
                    outlier_roi=self._last_roi,
                    outlier_score=float(self._last_score),
                    threshold=float(fc.threshold),
                    chosen_label=self._last_label,
                    main_label=self._last_main_label,
                    reason="keepalive_no_components",
                    debug={"miss_left": int(self._miss_left)},
                )
            return TemporalLocalizeResult(
                main_roi=None,
                outlier_roi=None,
                outlier_score=0.0,
                threshold=float(fc.threshold),
                chosen_label=None,
                main_label=None,
                reason="no_components",
                debug={},
            )

        mainland = _select_mainland(fc.components, center_box)
        mainland_roi = mainland.roi if mainland is not None else None
        main_label = mainland.label if mainland is not None else None

        g = stable_grid.astype(np.float32)
        g = g - float(g.min())
        peak_global = float(g.max()) + float(self.eps)
        diag = float(np.sqrt(H * H + W * W)) + float(self.eps)
        cx0, cy0 = (W - 1) * 0.5, (H - 1) * 0.5

        candidates: List[Tuple[float, ComponentStats, Dict[str, float]]] = []

        for c in fc.components:
            if mainland is not None and c.label == mainland.label:
                continue

            m = float(c.roi_mass)
            p = float(c.peak / peak_global)
            dens = float(c.density)
            area = float(c.area_ratio)

            dx = float(c.centroid[0] - cx0)
            dy = float(c.centroid[1] - cy0)
            dist = float(np.sqrt(dx * dx + dy * dy) / diag)

            sigma = float(self.center_sigma)
            center_prox = float(np.exp(-(dist * dist) / (2.0 * sigma * sigma + float(self.eps))))

            mainland_iou = float(grid_iou(c.roi, mainland_roi)) if mainland_roi is not None else 0.0

            score = (
                float(self.w_mass) * m
                + float(self.w_peak) * p
                + float(self.w_density) * (dens / (dens + 1.0))
                + float(self.w_dist) * dist
                - float(self.w_area_penalty) * np.sqrt(max(0.0, area))
                - float(self.w_center_penalty) * center_prox
                - float(self.w_mainland_iou_penalty) * mainland_iou
            )

            iou_prev = float(grid_iou(c.roi, self._last_roi)) if self._last_roi is not None else 0.0
            if iou_prev >= float(self.assoc_iou_thr):
                score += 0.15 * iou_prev

            feats = {
                "m": m,
                "p": p,
                "dens": dens,
                "area": area,
                "dist": dist,
                "center_prox": center_prox,
                "mainland_iou": mainland_iou,
                "iou_prev": iou_prev,
            }
            candidates.append((float(score), c, feats))

        if not candidates:
            return TemporalLocalizeResult(
                main_roi=mainland_roi,
                outlier_roi=None,
                outlier_score=0.0,
                threshold=float(fc.threshold),
                chosen_label=None,
                main_label=main_label,
                reason="no_islands_excluding_mainland",
                debug={"main_label": main_label},
            )

        candidates.sort(key=lambda t: t[0], reverse=True)
        best_score, best_comp, best_feats = candidates[0]

        if self._last_roi is not None and self._last_label is not None and best_comp.label != self._last_label:
            if float(best_score) < float(self._last_score) + float(self.switch_margin):
                self._miss_left = max(0, int(self.keepalive_frames) - 1)
                return TemporalLocalizeResult(
                    main_roi=mainland_roi,
                    outlier_roi=self._last_roi,
                    outlier_score=float(self._last_score),
                    threshold=float(fc.threshold),
                    chosen_label=self._last_label,
                    main_label=main_label,
                    reason="winner_keep",
                    debug={"kept_score": float(self._last_score), "new_score": float(best_score)},
                )

        upd = self._tracker.update(best_comp.roi)
        out_roi = upd.roi

        self._last_roi = out_roi
        self._last_score = float(best_score)
        self._last_label = int(best_comp.label)
        self._last_main_label = main_label
        self._miss_left = int(self.keepalive_frames)

        debug = {
            "best": {"label": int(best_comp.label), "score": float(best_score), **best_feats},
            "main": {"label": int(main_label) if main_label is not None else None},
            "top3": [
                {"label": int(c.label), "score": float(s), **f}
                for (s, c, f) in candidates[:3]
            ],
            "tracker": {
                "changed": bool(upd.changed),
                "iou_prev": float(upd.iou_with_prev) if upd.iou_with_prev is not None else None,
            },
        }

        return TemporalLocalizeResult(
            main_roi=mainland_roi,
            outlier_roi=out_roi,
            outlier_score=float(best_score),
            threshold=float(fc.threshold),
            chosen_label=int(best_comp.label),
            main_label=main_label,
            reason="ok",
            debug=debug,
        )

