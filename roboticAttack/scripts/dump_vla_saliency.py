#!/usr/bin/env python3
"""Dump OpenVLA saliency heatmaps for rollout frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from evaluation_tool.saliency import VLAAttentionExtractor, save_overlay_heatmap

'''

python scripts/dump_vla_saliency.py \
  --episode-path rollouts/rollouts/libero_object/object_xy_20_164/2025_11_19/2025_11_19-16_40_25--episode=1--success=False--task=pick_up_the_alphabet_soup_and_place_it_in_the_bask.mp4 \
  --out-dir results/saliency/libero_object/object_xy_20_164/ep1 \
  --instruction "pick up the alphabet soup and place it in the basket" \
  --dataset libero_object \
  --frame-stride 10 \
  --device cuda:1 \
  --model-root /data/zifeng/siyuan/data/models

python scripts/dump_vla_saliency.py \
  --episode-path rollouts/rollouts/libero_object_clean/object_clean/2025_11_16/2025_11_16-17_13_41--episode=11--success=True--task=pick_up_the_cream_cheese_and_place_it_in_the_baske.mp4 \
  --out-dir results/saliency/libero_object_clean/object_clean/ep2 \
  --instruction "pick up the cream cheese and place it in the basket" \
  --dataset libero_object \
  --frame-stride 10 \
  --device cuda:1 \
  --model-root /data/zifeng/siyuan/data/models

'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump OpenVLA saliency heatmaps for a rollout.")
    parser.add_argument("--episode-path", type=str, required=True, help="Path to rollout frames or a video file.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for heatmaps and overlays.")
    parser.add_argument("--instruction", type=str, default=None, help="Natural language instruction for the rollout.")
    parser.add_argument("--instruction-file", type=str, default=None, help="Optional text file containing the instruction.")
    parser.add_argument("--dataset", type=str, default="libero_spatial", help="Dataset spec to resolve OpenVLA weights.")
    parser.add_argument("--frame-stride", type=int, default=10, help="Sample stride for frames.")
    parser.add_argument("--frame-limit", type=int, default=None, help="Maximum number of frames to process.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for OpenVLA.")
    parser.add_argument("--model-root", type=str, default=None, help="Optional override for cached model directory.")
    parser.add_argument("--attn-module", type=str, default=None, help="Specific module name to hook for attention.")
    parser.add_argument("--image-size", type=int, default=224, help="Heatmap resize target.")
    return parser.parse_args()


def load_instruction(args: argparse.Namespace) -> str:
    if args.instruction:
        return args.instruction
    if args.instruction_file:
        with open(args.instruction_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    raise ValueError("Either --instruction or --instruction-file must be provided.")


def iter_frames(path: Path, stride: int, limit: Optional[int]) -> Iterator[Tuple[str, np.ndarray]]:
    # Check if path exists
    if not path.exists():
        raise FileNotFoundError(
            f"Path does not exist: {path}\n"
            f"Please check the --episode-path argument. "
            f"It should point to either:\n"
            f"  1. A directory containing image files (.png, .jpg, .jpeg)\n"
            f"  2. A video file (.mp4, .avi, etc.)"
        )
    
    # Handle directory case
    if path.is_dir():
        image_files = sorted([p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not image_files:
            raise ValueError(
                f"Directory {path} does not contain any image files (.png, .jpg, .jpeg).\n"
                f"Found files: {list(path.iterdir())[:10] if list(path.iterdir()) else 'empty directory'}"
            )
        for idx, img_path in enumerate(image_files):
            if idx % stride != 0:
                continue
            if limit is not None and idx // stride >= limit:
                break
            image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            yield img_path.stem, image
    
    # Handle file case (assumed to be video)
    else:
        # Check if file has a video extension (optional check, cv2 can handle some files without extension)
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
        if path.suffix.lower() not in video_extensions and path.suffix:
            # File exists but doesn't have a video extension - warn but try anyway
            print(f"Warning: File {path} does not have a common video extension. Attempting to open anyway...")
        
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(
                f"Could not open video file: {path}\n"
                f"The file exists but OpenCV cannot open it. Possible reasons:\n"
                f"  1. File is corrupted\n"
                f"  2. File format is not supported by OpenCV\n"
                f"  3. File is actually a directory (check if path ends with a slash)\n"
                f"  4. Missing codec or video codec not installed\n"
                f"If this is a directory, make sure the path is correct and ends with the directory name."
            )
        frame_id = 0
        kept = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield f"frame_{frame_id:06d}", rgb
                kept += 1
                if limit is not None and kept >= limit:
                    break
            frame_id += 1
        cap.release()


def main() -> None:
    args = parse_args()
    instruction = load_instruction(args)
    episode_path = Path(args.episode_path)
    
    # Resolve relative paths relative to current working directory
    if not episode_path.is_absolute():
        episode_path = Path.cwd() / episode_path
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = VLAAttentionExtractor(
        dataset=args.dataset,
        instruction_template=instruction,
        device=args.device,
        model_root=args.model_root,
        attn_module_name=args.attn_module,
        image_size=args.image_size,
    )

    metadata = {
        "episode_path": str(episode_path),
        "instruction": instruction,
        "dataset": args.dataset,
        "frame_stride": args.frame_stride,
        "frame_limit": args.frame_limit,
        "attn_module": args.attn_module,
    }

    stats = {"frames": 0}
    for frame_name, rgb in iter_frames(episode_path, args.frame_stride, args.frame_limit):
        result = extractor.get_saliency(rgb)
        heatmap = result["heatmap"]
        heatmap_path = out_dir / f"{frame_name}.npy"
        np.save(heatmap_path, heatmap)

        overlay_path = out_dir / f"{frame_name}_overlay.png"
        save_overlay_heatmap(rgb, heatmap, overlay_path)
        stats["frames"] += 1

    metadata["processed_frames"] = stats["frames"]
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {stats['frames']} heatmaps to {out_dir}")


if __name__ == "__main__":
    main()

