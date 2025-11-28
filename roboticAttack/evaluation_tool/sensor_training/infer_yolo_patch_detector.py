#!/usr/bin/env python
"""
Run a trained YOLO patch detector on videos or image folders.

Example:
    python evaluation_tool/sensor_training/infer_yolo_patch_detector.py \
        --weights runs/patch_yolo/patch_yolo_v1/weights/best.pt \
        --inputs rollouts/libero_spatial_test/author_uada_spatial/2025_11_16 \
        --output-json outputs/patch_detect/spatial.json
"""
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from evaluation_tool.sensor_patch_utils import SimplePatchTracker, _clip_boxes  # type: ignore


def iter_frames(path: Path) -> Iterable[Tuple[int, np.ndarray]]:
    if path.is_file() and path.suffix.lower() == ".mp4":
        cap = cv2.VideoCapture(str(path))
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1
        cap.release()
    elif path.is_dir():
        files = sorted(
            [p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        for idx, img_path in enumerate(files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            yield idx, frame
    else:
        raise ValueError(f"Unsupported input: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Trained YOLO weights.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Video file(s) or directories with frames.",
    )
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--output-json", required=True, help="Where to save detection results.")
    parser.add_argument("--track", action="store_true", help="Enable SimplePatchTracker.")
    parser.add_argument("--min-track-len", type=int, default=3)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    tracker = SimplePatchTracker(iou_thresh=args.iou, min_len=args.min_track_len)
    all_results = {}
    for input_path in args.inputs:
        path = Path(input_path)
        seq_key = str(path)
        seq_results = {}
        tracker.tracks.clear()
        for frame_idx, frame in iter_frames(path):
            height, width = frame.shape[:2]
            detections = model.predict(source=frame, conf=args.conf, verbose=False)
            boxes: List[List[float]] = []
            for r in detections:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    if float(box.conf[0].cpu().numpy()) < args.conf:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes.append(_clip_boxes(xyxy, width, height))
            stable = boxes
            if args.track:
                tracker.update(frame_idx, boxes)
                stable = tracker.get_active_boxes(frame_idx)
            seq_results[frame_idx] = {
                "cand_boxes": boxes,
                "stable_boxes": stable,
            }
        all_results[seq_key] = seq_results
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[INFO] Saved detections to {output_path}")


if __name__ == "__main__":
    main()

