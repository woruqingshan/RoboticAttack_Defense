#!/usr/bin/env python
"""
Extract frames from rollout videos and build a YOLO dataset with automatic labels.

Workflow:
1. Traverse directories named `object_xy_<X>_<Y>` under --base-dir.
2. Parse patch coordinates (X, Y) from the directory name (top-left corner).
3. Collect all .mp4 files (recursively) inside each directory.
4. Split videos into train/val subsets (2/3 train, 1/3 val by default).
5. For each video, extract frames at the desired stride and write YOLO labels
   assuming a fixed patch size (default 50x50).

Example:
    python evaluation_tool/sensor_dataset/extract_frames_for_yolo.py \
        --base-dir rollouts/rollouts/libero_object \
        --dataset-root datasets/patch_yolo \
        --stride 10
"""

'''
cd /root/autodl-tmp/code/roboticAttack

python evaluation_tool/sensor_dataset/extract_frames_for_yolo.py \
  --base-dir rollouts/rollouts/libero_object \
  --dataset-root datasets/patch_yolo \
  --stride 10 \
  --patch-size 50 \
  --img-size 224 \
  --train-ratio 0.6667 \
  --shuffle \
  --seed 42

'''


import argparse
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2


def parse_position_from_name(name: str) -> Tuple[int, int]:
    """
    Extract (x, y) from folder names like object_xy_20_144.
    """
    if not name.startswith("object_xy_"):
        raise ValueError(f"Invalid folder name for patch position: {name}")
    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse coordinates from {name}")
    return int(parts[-2]), int(parts[-1])


def ensure_dirs(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def clamp_bbox(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(width - 1, x1))
    y1 = max(0.0, min(height - 1, y1))
    x2 = max(0.0, min(width - 1, x2))
    y2 = max(0.0, min(height - 1, y2))
    return x1, y1, x2, y2


def bbox_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def extract_frames_with_label(
    video_path: Path,
    img_dir: Path,
    lbl_dir: Path,
    stride: int,
    patch_x: int,
    patch_y: int,
    patch_size: int,
    target_size: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    need_resize = (orig_width != target_size) or (orig_height != target_size)

    if need_resize:
        scale_x = target_size / orig_width
        scale_y = target_size / orig_height
        scaled_x = patch_x * scale_x
        scaled_y = patch_y * scale_y
        scaled_size_x = patch_size * scale_x
        scaled_size_y = patch_size * scale_y
        bbox = clamp_bbox(
            scaled_x,
            scaled_y,
            scaled_x + scaled_size_x,
            scaled_y + scaled_size_y,
            target_size,
            target_size,
        )
        yolo_label = bbox_to_yolo(*bbox, target_size, target_size)
    else:
        bbox = clamp_bbox(
            patch_x,
            patch_y,
            patch_x + patch_size,
            patch_y + patch_size,
            orig_width,
            orig_height,
        )
        yolo_label = bbox_to_yolo(*bbox, orig_width, orig_height)

    frame_idx = 0
    saved = 0
    base = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            img_name = f"{base}_frame{frame_idx:06d}.jpg"
            img_path = img_dir / img_name
            lbl_path = lbl_dir / f"{img_name[:-4]}.txt"
            if need_resize:
                frame_to_save = cv2.resize(frame, (target_size, target_size))
            else:
                frame_to_save = frame
            cv2.imwrite(str(img_path), frame_to_save)
            lbl_path.write_text(yolo_label)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames and auto-label patches for YOLO training.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="rollouts/rollouts/libero_object",
        help="Base directory containing object_xy_<X>_<Y> folders.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/patch_yolo",
        help="Output dataset root (images/ and labels/ subdirs will be created).",
    )
    parser.add_argument("--stride", type=int, default=10, help="Frame sampling stride.")
    parser.add_argument("--patch-size", type=int, default=50, help="Patch size in pixels (width=height).")
    parser.add_argument("--img-size", type=int, default=224, help="Target image size for YOLO training.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=2.0 / 3.0,
        help="Fraction of videos to use for training (rest for val).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle video order before splitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    dataset_root = Path(args.dataset_root)
    total_videos = 0
    total_frames = {"train": 0, "val": 0}

    if args.shuffle:
        random.seed(args.seed)

    for pos_dir in sorted(base_dir.glob("object_xy_*")):
        if not pos_dir.is_dir():
            continue
        try:
            patch_x, patch_y = parse_position_from_name(pos_dir.name)
        except ValueError as exc:
            print(f"[WARN] {exc}; skip {pos_dir}")
            continue

        # Collect all videos under this directory (recursively)
        video_files = sorted(pos_dir.rglob("*.mp4"))
        if not video_files:
            print(f"[WARN] No videos found in {pos_dir}")
            continue

        if args.shuffle:
            random.shuffle(video_files)

        num_videos = len(video_files)
        train_count = max(1, math.ceil(num_videos * args.train_ratio))
        train_videos = video_files[:train_count]
        val_videos = video_files[train_count:]

        print(f"\n[INFO] Processing {pos_dir.name} (videos={num_videos}, train={len(train_videos)}, val={len(val_videos)})")
        print(f"[INFO] Patch position: ({patch_x}, {patch_y}), size={args.patch_size}")

        # Process train videos
        img_train, lbl_train = ensure_dirs(dataset_root, "train")
        for vid in train_videos:
            count = extract_frames_with_label(
                vid,
                img_train,
                lbl_train,
                args.stride,
                patch_x,
                patch_y,
                args.patch_size,
                args.img_size,
            )
            total_frames["train"] += count
            total_videos += 1
            print(f"[TRAIN] {vid.relative_to(base_dir)} -> frames {count}")

        # Process val videos
        if val_videos:
            img_val, lbl_val = ensure_dirs(dataset_root, "val")
            for vid in val_videos:
                count = extract_frames_with_label(
                    vid,
                    img_val,
                    lbl_val,
                    args.stride,
                    patch_x,
                    patch_y,
                    args.patch_size,
                    args.img_size,
                )
                total_frames["val"] += count
                total_videos += 1
                print(f"[VAL]   {vid.relative_to(base_dir)} -> frames {count}")

    print("\n=== SUMMARY ===")
    print(f"Total videos processed: {total_videos}")
    print(f"Train frames: {total_frames['train']}")
    print(f"Val frames:   {total_frames['val']}")
    print(f"Dataset root: {dataset_root}")


if __name__ == "__main__":
    main()

