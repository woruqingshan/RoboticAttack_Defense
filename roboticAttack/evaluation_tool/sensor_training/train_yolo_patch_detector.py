#!/usr/bin/env python
"""
Train a YOLO model to detect adversarial patches.

Example:
    python evaluation_tool/sensor_training/train_yolo_patch_detector.py \
        --data evaluation_tool/sensor_training/patch.yaml \
        --epochs 50 --batch 16 --imgsz 640
"""

'''
python evaluation_tool/sensor_training/train_yolo_patch_detector.py \
    --data evaluation_tool/sensor_training/patch.yaml \
    --model yolov8n.pt \
    --epochs 50 --batch 16 --imgsz 640 \
    --name patch_yolo_1120 --project runs/patch_yolo

'''

import argparse
import os
from pathlib import Path

# Disable automatic model download and update checks
os.environ['ULTRALYTICS_OFFLINE'] = '1'
# Note: Don't set YOLO_VERBOSE=False here, as it will suppress training progress

from ultralytics import YOLO
import ultralytics

# Disable cloud sync and API calls
ultralytics.settings.update({'sync': False})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/data/zifeng/siyuan/Download/yolov8n.pt",
        help="Base YOLO checkpoint (local path).",
    )
    parser.add_argument(
        "--data",
        default="evaluation_tool/sensor_training/patch.yaml",
        help="Dataset YAML path.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="patch_yolo_v1", help="Run name suffix.")
    parser.add_argument(
        "--project",
        default="runs/patch_yolo",
        help="Directory where Ultralytics saves experiment folders.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Print training configuration
    print("=" * 60)
    print("[INFO] YOLO Patch Detector Training")
    print("=" * 60)
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Dataset config: {args.data}")
    print(f"[INFO] Epochs: {args.epochs}")
    print(f"[INFO] Batch size: {args.batch}")
    print(f"[INFO] Image size: {args.imgsz}")
    print(f"[INFO] Run name: {args.name}")
    print(f"[INFO] Project: {args.project}")
    print("=" * 60)
    
    # Check dataset
    data_path = Path(args.data)
    if data_path.exists():
        print(f"[INFO] Dataset config file found: {data_path}")
    else:
        print(f"[WARN] Dataset config file not found: {data_path}")
    
    # Load model
    print(f"\n[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"[INFO] Model loaded successfully")
    
    # Start training
    print("\n" + "=" * 60)
    print("[INFO] Starting training...")
    print("=" * 60)
    print("[INFO] Training progress will be displayed below:")
    print("[INFO] You will see:")
    print("      - Epoch progress (Epoch X/Y)")
    print("      - Batch progress bar for each epoch")
    print("      - Training metrics (loss, mAP, etc.)")
    print("-" * 60)
    print()
    
    # Enable verbose output for training progress
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        project=args.project,
        save=True,
        verbose=True,  # Enable verbose output - shows progress bars and metrics
        amp=False,  # Disable AMP to avoid downloading yolo11n.pt
    )
    
    print("\n" + "=" * 60)
    print("[INFO] Training finished!")
    print("=" * 60)
    print(f"[INFO] Results summary:")
    print(results)
    
    # Check saved weights
    best_ckpt = Path(args.project) / args.name / "weights" / "best.pt"
    last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
    
    if best_ckpt.exists():
        print(f"\n[INFO] ✓ Best weights saved at: {best_ckpt}")
    else:
        print(f"\n[WARN] Best weights not found at: {best_ckpt}")
    
    if last_ckpt.exists():
        print(f"[INFO] ✓ Last epoch weights saved at: {last_ckpt}")
    
    print("\n[INFO] Training logs and plots saved in:")
    print(f"      {Path(args.project) / args.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

