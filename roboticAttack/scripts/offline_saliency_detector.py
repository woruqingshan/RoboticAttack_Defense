#!/usr/bin/env python3
"""Offline anomaly scoring based on saliency statistics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Set

import numpy as np

# Helper functions are copied from the alignment script to avoid import coupling.


def load_yolo_boxes(label_file: Path, image_size: int, target_class: Optional[int]):
    boxes = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if target_class is not None and cls != target_class:
                continue
            cx, cy, w, h = map(float, parts[1:])
            x_min = (cx - w / 2) * image_size
            y_min = (cy - h / 2) * image_size
            x_max = (cx + w / 2) * image_size
            y_max = (cy + h / 2) * image_size
            boxes.append((x_min, y_min, x_max, y_max))
    return boxes


def compute_iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union


def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def topk_mask(heatmap: np.ndarray, k: float) -> np.ndarray:
    flat = heatmap.flatten()
    k = min(max(k, 0.0), 1.0)
    threshold_index = int((1 - k) * len(flat))
    threshold_value = np.partition(flat, threshold_index)[threshold_index]
    return (heatmap >= threshold_value).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate anomaly scores from saliency heatmaps.")
    parser.add_argument("--saliency-dir", type=str, required=True, help="Directory with .npy heatmaps.")
    parser.add_argument("--labels-dir", type=str, default=None, help="Directory with YOLO labels for patches.")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to store per-frame anomaly scores.")
    parser.add_argument("--topk", type=float, default=0.05, help="Top-k threshold for pseudo bbox extraction.")
    parser.add_argument("--image-size", type=int, default=224, help="Resolution of labels and heatmaps.")
    parser.add_argument("--class-id", type=int, default=None, help="Optional patch class id.")
    parser.add_argument("--weight-iou", type=float, default=0.7, help="Weight for IoU-based component.")
    parser.add_argument("--weight-entropy", type=float, default=0.3, help="Weight for entropy component.")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Score threshold for attack flag.")
    parser.add_argument("--attack-list", type=str, default=None, help="Optional file containing attack frame ids.")
    return parser.parse_args()


def normalized_entropy(heatmap: np.ndarray) -> float:
    flat = heatmap.flatten().astype(np.float64)
    flat = flat - flat.min()
    if flat.max() > 0:
        flat = flat / flat.max()
    probs = flat / (flat.sum() + 1e-8)
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    max_entropy = math.log(len(flat))
    return float(entropy / (max_entropy + 1e-8))


def load_attack_set(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    entries = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if token:
                entries.add(token)
    return entries


def main() -> None:
    args = parse_args()
    saliency_dir = Path(args.saliency_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    attack_set = load_attack_set(args.attack_list)

    rows = []
    for heatmap_file in sorted(saliency_dir.glob("*.npy")):
        frame_id = heatmap_file.stem
        heatmap = np.load(heatmap_file)
        entropy_score = normalized_entropy(heatmap)

        if labels_dir:
            label_file = labels_dir / f"{frame_id}.txt"
            if label_file.exists():
                mask = topk_mask(heatmap, args.topk)
                pred_bbox = mask_to_bbox(mask)
                gt_boxes = load_yolo_boxes(label_file, args.image_size, args.class_id)
                max_iou = max((compute_iou(pred_bbox, box) for box in gt_boxes), default=0.0) if pred_bbox else 0.0
            else:
                max_iou = 0.0
        else:
            max_iou = 0.0

        score = args.weight_iou * (1.0 - max_iou) + args.weight_entropy * entropy_score
        is_attack = frame_id in attack_set
        predicted = score >= args.decision_threshold
        rows.append(
            {
                "frame_id": frame_id,
                "score": score,
                "max_iou": max_iou,
                "entropy": entropy_score,
                "is_attack": int(is_attack),
                "predicted_attack": int(predicted),
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_id", "score", "max_iou", "entropy", "is_attack", "predicted_attack"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if rows:
        attack_predictions = sum(row["predicted_attack"] for row in rows)
        print(f"Wrote {len(rows)} scores. Predicted attacks: {attack_predictions}")
    else:
        print("No heatmaps found to score.")


if __name__ == "__main__":
    main()

