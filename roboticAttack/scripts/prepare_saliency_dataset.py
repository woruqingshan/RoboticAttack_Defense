#!/usr/bin/env python3
"""
Batch helper that prepares paired clean / attack saliency dumps for OpenVLA.

Phase-0 goals covered by this script:
  * Enumerate rollouts under rollouts/rollouts/libero_object(_clean)
  * Map each LIBERO-Object task string to a stable task_id (persisted registry)
  * Create deterministic output directories inside results/saliency/...
  * Support resume/auto-skip by inspecting existing ep directories
  * Invoke dump_vla_saliency.py for every required episode and augment metadata

instructions:

   python scripts/prepare_saliency_dataset.py \
     --dataset libero_object \
     --frame-stride 10 \
     --device cuda:1 \
     --model-root /data/zifeng/siyuan/data/models


"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


###############################################################################
# Data classes and helpers
###############################################################################


@dataclass
class EpisodeInfo:
    """Container describing a single rollout episode."""

    path: Path
    task_text: str
    episode_id: int
    success: bool
    patch_tag: Optional[str]  # object_xy_* for attack; None for clean
    is_attack: bool
    date_folder: str

    @property
    def instruction(self) -> str:
        """Derive a natural language instruction from the task token."""
        return self.task_text.replace("_", " ").strip()


@dataclass
class RunPlan:
    """Executable plan for one saliency dump."""

    episode: EpisodeInfo
    task_id: int
    output_dir: Path
    repeat_slot: Optional[int]  # clean repeat index (episode % 10), else None


class TaskRegistry:
    """Persisted mapping between LIBERO task text and numeric task IDs."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self._data: Dict[str, Dict[str, object]] = {"tasks": {}}
        if registry_path.exists():
            with open(registry_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        self._tasks: Dict[str, Dict[str, object]] = self._data.setdefault("tasks", {})

    def get_task_id(self, task_text: str) -> int:
        if task_text not in self._tasks:
            next_id = 1
            if self._tasks:
                next_id = max(int(info["id"]) for info in self._tasks.values()) + 1
            self._tasks[task_text] = {"id": next_id, "instruction": task_text.replace("_", " ").strip()}
            self.save()
        return int(self._tasks[task_text]["id"])

    def save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)


###############################################################################
# Rollout discovery
###############################################################################


def parse_episode_metadata(mp4_path: Path) -> Dict[str, str]:
    """Parse --key=value tokens from rollout filenames."""
    segments = mp4_path.stem.split("--")
    metadata: Dict[str, str] = {}
    for token in segments[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            metadata[key] = value
    return metadata


def discover_attack_rollouts(root: Path) -> Iterable[EpisodeInfo]:
    """Yield EpisodeInfo for every attack rollout."""
    if not root.exists():
        return []
    for patch_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("object_xy_")):
        videos = sorted(patch_dir.glob("*/*.mp4"))
        for video in videos:
            meta = parse_episode_metadata(video)
            task_text = meta.get("task")
            if not task_text:
                continue
            episode_id = int(meta.get("episode", "-1"))
            success = meta.get("success", "").lower() == "true"
            yield EpisodeInfo(
                path=video,
                task_text=task_text,
                episode_id=episode_id,
                success=success,
                patch_tag=patch_dir.name,
                is_attack=True,
                date_folder=video.parent.name,
            )


def discover_clean_rollouts(root: Path) -> Iterable[EpisodeInfo]:
    """Yield EpisodeInfo for every clean rollout."""
    if not root.exists():
        return []
    videos = sorted(root.glob("*/*.mp4"))
    for video in videos:
        meta = parse_episode_metadata(video)
        task_text = meta.get("task")
        if not task_text:
            continue
        episode_id = int(meta.get("episode", "-1"))
        success = meta.get("success", "").lower() == "true"
        yield EpisodeInfo(
            path=video,
            task_text=task_text,
            episode_id=episode_id,
            success=success,
            patch_tag=None,
            is_attack=False,
            date_folder=video.parent.name,
        )


###############################################################################
# Planning utilities
###############################################################################


def build_attack_output_dir(results_root: Path, patch_tag: str, task_id: int) -> Path:
    """Return output dir for attack saliency (one run per patch/task)."""
    subdir = f"ep{task_id}"
    return results_root.joinpath("libero_object", patch_tag, subdir)


def build_clean_output_dir(results_root: Path, task_id: int, repeat_slot: int) -> Path:
    """
    Return output dir for clean saliency with nested ep structure.

    Layout:
        .../libero_object_clean/object_clean/ep<task_id>/ep<task_id>_<repeat_slot>/
    """
    root = results_root.joinpath("libero_object_clean", "object_clean", f"ep{task_id}")
    return root.joinpath(f"ep{task_id}_{repeat_slot}")


def should_skip_run(output_dir: Path, overwrite: bool) -> bool:
    """Decide whether to skip based on existing metadata."""
    if overwrite:
        return False
    metadata = output_dir / "metadata.json"
    return metadata.exists()


def update_metadata(output_dir: Path, extra: Dict[str, object]) -> None:
    """Merge additional metadata fields into dump_vla_saliency output."""
    metadata_file = output_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    metadata.update(extra)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def append_index(record_path: Path, record: Dict[str, object]) -> None:
    """Append a JSON line describing a completed run."""
    path = Path(record_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


###############################################################################
# Execution
###############################################################################


def run_saliency_dump(plan: RunPlan, args: argparse.Namespace) -> None:
    """Invoke dump_vla_saliency.py for a single episode."""
    cmd = [
        sys.executable,
        str(args.dump_script),
        "--episode-path",
        str(plan.episode.path),
        "--out-dir",
        str(plan.output_dir),
        "--instruction",
        plan.episode.instruction,
        "--dataset",
        args.dataset,
        "--frame-stride",
        str(args.frame_stride),
    ]
    if args.frame_limit is not None:
        cmd.extend(["--frame-limit", str(args.frame_limit)])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.model_root:
        cmd.extend(["--model-root", args.model_root])
    if args.attn_module:
        cmd.extend(["--attn-module", args.attn_module])
    if args.image_size:
        cmd.extend(["--image-size", str(args.image_size)])

    if args.dry_run:
        print("[DRY-RUN]", " ".join(cmd))
        return

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] Task {plan.task_id} ({'attack' if plan.episode.is_attack else 'clean'}) -> {plan.output_dir}")
    subprocess.run(cmd, check=True)

    meta_extra = {
        "task_id": plan.task_id,
        "task_text": plan.episode.task_text,
        "instruction": plan.episode.instruction,
        "episode_id": plan.episode.episode_id,
        "is_attack": plan.episode.is_attack,
        "patch_tag": plan.episode.patch_tag,
        "repeat_slot": plan.repeat_slot,
        "source_video": str(plan.episode.path),
    }
    update_metadata(plan.output_dir, meta_extra)
    append_index(
        args.index_file,
        {
            **meta_extra,
            "output_dir": str(plan.output_dir),
            "frame_stride": args.frame_stride,
            "frame_limit": args.frame_limit,
            "device": args.device,
            "date_folder": plan.episode.date_folder,
        },
    )


###############################################################################
# Main orchestration
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paired clean/attack saliency datasets.")
    parser.add_argument("--dataset", type=str, default="libero_object", help="Dataset spec for OpenVLA.")
    parser.add_argument("--rollouts-root", type=str, default="rollouts/rollouts/libero_object")
    parser.add_argument(
        "--clean-rollouts-root",
        type=str,
        default="rollouts/rollouts/libero_object_clean/object_clean",
    )
    parser.add_argument("--results-root", type=str, default="results/saliency")
    parser.add_argument(
        "--dump-script",
        type=str,
        default="scripts/dump_vla_saliency.py",
        help="Path to dump_vla_saliency.py",
    )
    parser.add_argument("--task-registry", type=str, default="results/saliency/task_registry.json")
    parser.add_argument("--index-file", type=str, default="results/saliency/saliency_runs_index.jsonl")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model-root", type=str, default=None)
    parser.add_argument("--attn-module", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-clean", type=int, default=50, help="Clean episodes per task.")
    parser.add_argument("--max-attack", type=int, default=1, help="Attack episodes per patch/task.")
    parser.add_argument("--patch-filter", type=str, nargs="*", default=None, help="Subset of object_xy_* dirs.")
    parser.add_argument("--task-filter", type=str, nargs="*", default=None, help="Only process tasks containing substrings.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if metadata exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return parser.parse_args()


def task_matches_filters(task_text: str, filters: Optional[List[str]]) -> bool:
    if not filters:
        return True
    lowered = task_text.lower()
    return any(f.lower() in lowered for f in filters)


def build_plans(args: argparse.Namespace, registry: TaskRegistry) -> List[RunPlan]:
    rollouts_root = Path(args.rollouts_root)
    clean_root = Path(args.clean_rollouts_root)
    results_root = Path(args.results_root)

    plans: List[RunPlan] = []
    clean_counts: Dict[int, int] = {}
    attack_counts: Dict[Tuple[int, str], int] = {}

    # Attack episodes
    for episode in discover_attack_rollouts(rollouts_root):
        if args.patch_filter and episode.patch_tag not in args.patch_filter:
            continue
        if not task_matches_filters(episode.task_text, args.task_filter):
            continue
        task_id = registry.get_task_id(episode.task_text)
        key = (task_id, episode.patch_tag or "")
        if attack_counts.get(key, 0) >= args.max_attack:
            continue
        out_dir = build_attack_output_dir(results_root, episode.patch_tag or "unknown_patch", task_id)
        if should_skip_run(out_dir, args.overwrite):
            continue
        plans.append(RunPlan(episode=episode, task_id=task_id, output_dir=out_dir, repeat_slot=None))
        attack_counts[key] = attack_counts.get(key, 0) + 1

    # Clean episodes
    for episode in discover_clean_rollouts(clean_root):
        if not task_matches_filters(episode.task_text, args.task_filter):
            continue
        task_id = registry.get_task_id(episode.task_text)
        if clean_counts.get(task_id, 0) >= args.max_clean:
            continue
        repeat_slot = episode.episode_id % 10
        repeat_slot = repeat_slot if repeat_slot != 0 else 10
        out_dir = build_clean_output_dir(results_root, task_id, repeat_slot)
        if should_skip_run(out_dir, args.overwrite):
            continue
        plans.append(RunPlan(episode=episode, task_id=task_id, output_dir=out_dir, repeat_slot=repeat_slot))
        clean_counts[task_id] = clean_counts.get(task_id, 0) + 1

    plans.sort(key=lambda p: (not p.episode.is_attack, p.task_id, p.episode.episode_id))
    return plans


def main() -> None:
    args = parse_args()
    registry = TaskRegistry(Path(args.task_registry))
    plans = build_plans(args, registry)
    if not plans:
        print("No pending episodes to process. All caught up!")
        return
    print(f"[INFO] Prepared {len(plans)} episodes (clean + attack).")
    for idx, plan in enumerate(plans, 1):
        print(f"[{idx}/{len(plans)}] {plan.episode.path.name} -> {plan.output_dir}")
        run_saliency_dump(plan, args)


if __name__ == "__main__":
    main()

