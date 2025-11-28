#!/usr/bin/env python3
"""Compute IoU alignment statistics between saliency maps and known patch boxes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saliency alignment with patch labels.")
    parser.add_argument("--saliency-dir", type=str, required=True, help="Directory containing .npy heatmaps.")
    parser.add_argument("--labels-dir", type=str, required=True, help="Directory containing YOLO txt labels.")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to save frame-level metrics.")
    parser.add_argument("--topk", type=float, default=0.05, help="Top-k percentage for saliency mask (0-1).")
    parser.add_argument("--class-id", type=int, default=None, help="Optional class id to filter labels.")
    parser.add_argument("--image-size", type=int, default=224, help="Image resolution of labels and heatmaps.")
    return parser.parse_args()


def load_yolo_boxes(label_file: Path, image_size: int, target_class: Optional[int]) -> List[Tuple[float, float, float, float]]:
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


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


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


def topk_mask(heatmap: np.ndarray, k: float) -> np.ndarray:
    flat = heatmap.flatten()
    k = min(max(k, 0.0), 1.0)
    threshold_index = int((1 - k) * len(flat))
    threshold_value = np.partition(flat, threshold_index)[threshold_index]
    return (heatmap >= threshold_value).astype(np.uint8)


def main() -> None:
    args = parse_args()
    saliency_dir = Path(args.saliency_dir)
    labels_dir = Path(args.labels_dir)
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for heatmap_file in sorted(saliency_dir.glob("*.npy")):
        frame_id = heatmap_file.stem
        label_file = labels_dir / f"{frame_id}.txt"
        if not label_file.exists():
            continue
        heatmap = np.load(heatmap_file)
        mask = topk_mask(heatmap, args.topk)
        pred_bbox = mask_to_bbox(mask)
        if pred_bbox is None:
            continue

        gt_boxes = load_yolo_boxes(label_file, args.image_size, args.class_id)
        if not gt_boxes:
            continue
        ious = [compute_iou(pred_bbox, gt_box) for gt_box in gt_boxes]
        record = {
            "frame_id": frame_id,
            "max_iou": float(np.max(ious)),
            "mean_iou": float(np.mean(ious)),
            "num_gt": len(gt_boxes),
        }
        records.append(record)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "max_iou", "mean_iou", "num_gt"])
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    if records:
        max_iou = np.mean([row["max_iou"] for row in records])
        mean_iou = np.mean([row["mean_iou"] for row in records])
        print(f"Processed {len(records)} frames. Avg max IoU: {max_iou:.3f}, avg mean IoU: {mean_iou:.3f}")
    else:
        print("No matching saliency-label pairs were found.")


if __name__ == "__main__":
    main()

