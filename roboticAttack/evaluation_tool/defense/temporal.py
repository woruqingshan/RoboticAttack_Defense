# temporal.py
# -*- coding: utf-8 -*-
"""
Temporal utilities for online patch defense.

This module is intentionally decoupled from the rest of the project.
It only depends on numpy.

Key components:
- RunningStats2D: EMA mean/variance + stable score for 2D grids
- AttentionStabilityScorer: computes ROI mass score on stable grids
- ROITracker: smooth ROI updates to reduce jitter
- TemporalGate: low-frequency hysteresis + hold/cooldown pulse gate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


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

    def centroid(self) -> Tuple[float, float]:
        cx = (self.gx0 + self.gx1) * 0.5
        cy = (self.gy0 + self.gy1) * 0.5
        return float(cx), float(cy)


@dataclass
class Stats2DResult:
    mu: np.ndarray
    var: np.ndarray
    stable: np.ndarray


@dataclass
class RunningStats2D:
    """
    EMA running mean and variance for 2D grids (e.g., 16x16).

    stable = mu / (sqrt(var) + eps)

    Notes:
    - If normalize_input=True, each input grid is normalized to sum=1
      to make scores more comparable across frames.
    """
    shape: Tuple[int, int]
    alpha: float = 0.2
    eps: float = 1e-6
    normalize_input: bool = True

    def __post_init__(self) -> None:
        H, W = self.shape
        self.mu = np.zeros((H, W), dtype=np.float32)
        self.m2 = np.zeros((H, W), dtype=np.float32)
        self.initialized = False

    def reset(self) -> None:
        self.mu.fill(0.0)
        self.m2.fill(0.0)
        self.initialized = False

    def _prep(self, grid: np.ndarray) -> np.ndarray:
        g = grid.astype(np.float32)
        g = g - float(g.min())
        s = float(g.sum())
        if self.normalize_input and s > self.eps:
            g = g / s
        return g

    def update(self, grid: np.ndarray) -> Stats2DResult:
        g = self._prep(grid)

        if not self.initialized:
            self.mu = g.copy()
            self.m2 = (g * g).copy()
            self.initialized = True
        else:
            a = float(self.alpha)
            self.mu = (1.0 - a) * self.mu + a * g
            self.m2 = (1.0 - a) * self.m2 + a * (g * g)

        var = np.maximum(self.m2 - self.mu * self.mu, 0.0)
        stable = self.mu / (np.sqrt(var) + float(self.eps))
        return Stats2DResult(mu=self.mu.copy(), var=var.astype(np.float32), stable=stable.astype(np.float32))


@dataclass
class AttentionStabilityScorer:
    """
    Compute a scalar score for a candidate ROI on a (stable) grid.

    score = ROI_mass on the stable grid
    """
    eps: float = 1e-6

    def score(self, stable_grid: np.ndarray, roi: GridBox) -> float:
        x = stable_grid.astype(np.float32)
        x = x - float(x.min())
        total = float(x.sum()) + float(self.eps)

        x0, y0, x1, y1 = roi.gx0, roi.gy0, roi.gx1, roi.gy1
        x0 = max(0, min(x0, x.shape[1]))
        x1 = max(0, min(x1, x.shape[1]))
        y0 = max(0, min(y0, x.shape[0]))
        y1 = max(0, min(y1, x.shape[0]))
        if x1 <= x0 or y1 <= y0:
            return 0.0

        roi_sum = float(x[y0:y1, x0:x1].sum())
        return float(roi_sum / total)


def _iou(a: GridBox, b: GridBox) -> float:
    ax0, ay0, ax1, ay1 = a.gx0, a.gy0, a.gx1, a.gy1
    bx0, by0, bx1, by1 = b.gx0, b.gy0, b.gx1, b.gy1
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = float(area_a + area_b - inter) + 1e-6
    return float(inter / denom)


@dataclass
class ROIUpdate:
    roi: Optional[GridBox]
    changed: bool
    iou_with_prev: Optional[float]


@dataclass
class ROITracker:
    """
    Smooth ROI updates to reduce jitter.
    This tracker is decoupled: it only processes grid boxes.

    - If IoU is small, accept the new ROI as a jump (changed=True).
    - Otherwise EMA-smooth the coordinates.
    """
    iou_keep: float = 0.30
    ema: float = 0.5

    def __post_init__(self) -> None:
        self._roi: Optional[GridBox] = None

    def reset(self) -> None:
        self._roi = None

    @property
    def roi(self) -> Optional[GridBox]:
        return self._roi

    def update(self, proposal: Optional[GridBox]) -> ROIUpdate:
        if proposal is None:
            return ROIUpdate(roi=self._roi, changed=False, iou_with_prev=None)

        if self._roi is None:
            self._roi = proposal
            return ROIUpdate(roi=self._roi, changed=True, iou_with_prev=None)

        i = _iou(self._roi, proposal)
        if i < self.iou_keep:
            self._roi = proposal
            return ROIUpdate(roi=self._roi, changed=True, iou_with_prev=i)

        # EMA smooth
        px0, py0, px1, py1 = proposal.gx0, proposal.gy0, proposal.gx1, proposal.gy1
        rx0, ry0, rx1, ry1 = self._roi.gx0, self._roi.gy0, self._roi.gx1, self._roi.gy1
        e = float(self.ema)
        x0 = int(round(e * rx0 + (1.0 - e) * px0))
        y0 = int(round(e * ry0 + (1.0 - e) * py0))
        x1 = int(round(e * rx1 + (1.0 - e) * px1))
        y1 = int(round(e * ry1 + (1.0 - e) * py1))
        new_roi = GridBox(gx0=x0, gy0=y0, gx1=x1, gy1=y1)

        changed = (new_roi != self._roi)
        self._roi = new_roi
        return ROIUpdate(roi=self._roi, changed=changed, iou_with_prev=i)


@dataclass
class GateDecision:
    should_purify: bool
    state: str
    checked: bool


@dataclass
class TemporalGate:
    """
    Low-frequency hysteresis + hold/cooldown gate to avoid flicker.

    States:
    - SAFE: not purifying; only checks every K frames
    - HOLD: purify for hold_frames frames (pulse)
    - COOLDOWN: do not purify for cooldown_frames frames
    """
    theta_on: float = 0.07
    theta_off: float = 0.05
    hold_frames: int = 3
    cooldown_frames: int = 2
    check_every_k: int = 3

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = "SAFE"
        self._t = 0
        self._hold_left = 0
        self._cool_left = 0

    def step(self, score: float) -> GateDecision:
        self._t += 1

        # HOLD: always purify
        if self.state == "HOLD":
            self._hold_left -= 1
            if self._hold_left <= 0:
                self.state = "COOLDOWN"
                self._cool_left = int(self.cooldown_frames)
            return GateDecision(should_purify=True, state=self.state, checked=False)

        # COOLDOWN: never purify
        if self.state == "COOLDOWN":
            self._cool_left -= 1
            if self._cool_left <= 0:
                self.state = "SAFE"
            return GateDecision(should_purify=False, state=self.state, checked=False)

        # SAFE: low-frequency check
        if int(self.check_every_k) > 1 and (self._t % int(self.check_every_k)) != 0:
            return GateDecision(should_purify=False, state=self.state, checked=False)

        # checked now
        if score >= float(self.theta_on):
            self.state = "HOLD"
            self._hold_left = int(self.hold_frames)
            return GateDecision(should_purify=True, state=self.state, checked=True)

        # Optional: hysteresis off (not strictly needed in SAFE)
        if score <= float(self.theta_off):
            return GateDecision(should_purify=False, state=self.state, checked=True)

        return GateDecision(should_purify=False, state=self.state, checked=True)

