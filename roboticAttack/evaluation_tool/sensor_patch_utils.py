"""
Utility helpers for sensor-layer patch detection.

This module wraps YOLO detectors, exposes a tiny IoU-based tracker for
multi-frame confirmation, and leaves hooks for future feature-space
patch masking / cleaning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np
from ultralytics import YOLO


def iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class SimplePatchTracker:
    """
    Minimal IoU-based tracker that requires a patch to persist for a few
    frames before being considered valid.
    """

    def __init__(self, iou_thresh: float = 0.5, min_len: int = 3):
        self.iou_thresh = iou_thresh
        self.min_len = min_len
        self.tracks: List[dict] = []

    def update(self, frame_idx: int, boxes: List[List[float]]) -> None:
        used = set()
        for track in self.tracks:
            best_iou = 0.0
            best_j = None
            for j, box in enumerate(boxes):
                if j in used:
                    continue
                score = iou(track["boxes"][-1], box)
                if score > best_iou:
                    best_iou = score
                    best_j = j
            if best_j is not None and best_iou >= self.iou_thresh:
                track["boxes"].append(boxes[best_j])
                track["last_frame"] = frame_idx
                used.add(best_j)

        for j, box in enumerate(boxes):
            if j in used:
                continue
            self.tracks.append({"boxes": [box], "last_frame": frame_idx})

    def get_active_boxes(self, frame_idx: int) -> List[List[float]]:
        active: List[List[float]] = []
        for track in self.tracks:
            if len(track["boxes"]) >= self.min_len and track["last_frame"] == frame_idx:
                active.append(track["boxes"][-1])
        return active


def _clip_boxes(box: Sequence[float], width: int, height: int) -> List[float]:
    x1 = float(np.clip(box[0], 0, width - 1))
    y1 = float(np.clip(box[1], 0, height - 1))
    x2 = float(np.clip(box[2], 0, width - 1))
    y2 = float(np.clip(box[3], 0, height - 1))
    return [x1, y1, x2, y2]


@dataclass
class PatchDetector:
    """
    Thin wrapper around an Ultralytics YOLO model plus a tracker helper.
    """

    model_path: str
    conf_th: float = 0.5
    iou_th: float = 0.5
    min_track_len: int = 3

    def __post_init__(self):
        self.model = YOLO(self.model_path)

    def build_tracker(self) -> SimplePatchTracker:
        return SimplePatchTracker(iou_thresh=self.iou_th, min_len=self.min_track_len)

    def detect_boxes(self, frame_bgr: np.ndarray) -> List[List[float]]:
        """
        Run YOLO detection on the frame and return list of boxes in pixel coords.
        """
        height, width = frame_bgr.shape[:2]
        results = self.model.predict(source=frame_bgr, verbose=False)
        boxes: List[List[float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                score = float(box.conf[0].cpu().numpy())
                if score < self.conf_th:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                boxes.append(_clip_boxes(xyxy, width, height))
        return boxes

    def detect_and_track(
        self, frame_bgr: np.ndarray, tracker: SimplePatchTracker, frame_idx: int
    ) -> List[List[float]]:
        boxes = self.detect_boxes(frame_bgr)
        tracker.update(frame_idx, boxes)
        return tracker.get_active_boxes(frame_idx)

    def mask_or_clean(self, frame_bgr: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
        """
        Placeholder for feature/pixel-space mitigation.
        Currently applies a blur over detected boxes for quick visualization.
        """
        if not boxes:
            return frame_bgr
        masked = frame_bgr.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            patch = masked[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            patch = cv2.blur(patch, (25, 25))
            masked[y1:y2, x1:x2] = patch
        return masked

