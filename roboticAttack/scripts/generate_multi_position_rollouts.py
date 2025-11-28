#!/usr/bin/env python
"""
Generate rollout videos for multiple patch positions (serial execution).

For each patch position, runs all tasks in libero suite (object or spatial), 1 trial per task.
Saves videos to: rollouts/rollouts/{task_suite}/{prefix}_xy_{x}_{y}/{date}/

Usage:
    # For libero_object:
    python scripts/generate_multi_position_rollouts.py \
        --patch-path /home/zifeng/siyuan/code/roboticAttack/adversarial_patches/simulation/untargeted/UADA-dof1-b55bb4ee-f3df-4410-b7ea-fcfe68ae4132/patch.pt \
        --cudaid 0 \
        --task-suite libero_object
    
    # For libero_spatial:
    python scripts/generate_multi_position_rollouts.py \
        --patch-path /home/zifeng/siyuan/code/roboticAttack/adversarial_patches/simulation/untargeted/UADA-dof1-b55bb4ee-f3df-4410-b7ea-fcfe68ae4132/patch.pt \
        --cudaid 0 \
        --task-suite libero_spatial
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Patch positions (22 positions)
PATCH_POSITIONS: List[Tuple[int, int]] = [
    (0, 0),
    (10, 10),
    (0, 10),
    (10, 0),
    (20, 10),
    (10, 20),
    (154, 0),
    (154, 10),
    (154, 20),
    (164, 0),
    (164, 10),
    (164, 20),
    (174, 0),
    (174, 10),
    (174, 20),
    (10, 144),
    (10, 154),
    (10, 164),
    (20, 144),
    (20, 154),
    (20, 164),
    (174, 174),
]

# Default configuration
DEFAULT_PATCH_PATH = "/home/zifeng/siyuan/code/roboticAttack/adversarial_patches/simulation/untargeted/UADA-dof1-b55bb4ee-f3df-4410-b7ea-fcfe68ae4132/patch.pt"
DEFAULT_TASK_SUITE = "libero_object"
DEFAULT_NUM_TRIALS = 1  # 1 trial per task

# Paths for models and datasets
DEFAULT_MODEL_ROOT = "/data/zifeng/siyuan/data/models"
DEFAULT_DATASET_PATH = "/data/zifeng/siyuan/data/datasets"

# Model mapping for different task suites
MODEL_MAP = {
    "libero_object": "openvla/openvla-7b-finetuned-libero-object",
    "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
}


def get_folder_prefix(task_suite: str) -> str:
    """
    Get folder prefix based on task suite.
    
    Args:
        task_suite: Task suite name (libero_object or libero_spatial)
    
    Returns:
        Prefix string (e.g., "object" or "spatial")
    """
    if task_suite == "libero_object":
        return "object"
    elif task_suite == "libero_spatial":
        return "spatial"
    else:
        # Default to extracting from task_suite name
        return task_suite.replace("libero_", "")


def get_default_model(task_suite: str) -> str:
    """
    Get default model checkpoint based on task suite.
    
    Args:
        task_suite: Task suite name
    
    Returns:
        Model checkpoint path
    """
    return MODEL_MAP.get(task_suite, MODEL_MAP["libero_object"])


def run_single_position(
    x: int,
    y: int,
    patch_path: str,
    cuda_id: int,
    num_trials: int = 1,
    model: str = None,
    task_suite: str = DEFAULT_TASK_SUITE,
) -> bool:
    """
    Run evaluation for a single patch position.
    
    Args:
        x: Patch x coordinate
        y: Patch y coordinate
        patch_path: Path to patch file
        cuda_id: CUDA device ID
        num_trials: Number of trials per task
        model: Model checkpoint path (if None, uses default for task_suite)
        task_suite: Task suite name (libero_object or libero_spatial)
    
    Returns:
        True if successful, False otherwise
    """
    # Use default model if not provided
    if model is None:
        model = get_default_model(task_suite)
    
    # Get folder prefix based on task suite
    prefix = get_folder_prefix(task_suite)
    
    # Generate date string
    date_str = datetime.now().strftime("%Y_%m_%d")
    
    # Build exp_name for video saving
    # Note: save_rollout_video uses f"./rollouts/{exp_name}/{DATE}"
    # So exp_name should be: rollouts/{task_suite}/{prefix}_xy_{x}_{y}
    exp_name = f"rollouts/{task_suite}/{prefix}_xy_{x}_{y}"
    
    # Build folder name for logs and run_id
    folder_name = f"{prefix}_xy_{x}_{y}"
    
    # Build command
    cmd = [
        "python",
        "experiments/robot/libero/run_libero_eval_args_geo_batch.py",
        "--model_family", "openvla",
        "--pretrained_checkpoint", model,
        "--task_suite_name", task_suite,
        "--num_trials_per_task", str(num_trials),
        "--center_crop", "True",
        "--patchroot", patch_path,
        "--x", str(x),
        "--y", str(y),
        "--angle", "0",
        "--shx", "0",
        "--shy", "0",
        "--cudaid", str(cuda_id),
        "--use_wandb", "False",
        "--local_log_dir", f"./experiments/logs/{prefix}_attack_xy_{x}_{y}",
        "--run_id_note", folder_name,
        "--exp_name", exp_name,
        "--use_patch", "True",
    ]
    
    print(f"\n{'='*80}")
    print(f"Running position ({x}, {y})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Set environment variables for models and datasets paths
    env = os.environ.copy()
    env["ROBOTIC_ATTACK_MODEL_ROOT"] = DEFAULT_MODEL_ROOT
    env["LIBERO_DATASET_PATH"] = DEFAULT_DATASET_PATH
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/home/zifeng/siyuan/code/roboticAttack",
            env=env,
            check=True,
            capture_output=False,  # Show output in real-time
        )
        print(f"\n[SUCCESS] Position ({x}, {y}) completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Position ({x}, {y}) failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Position ({x}, {y}) failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate rollout videos for multiple patch positions"
    )
    parser.add_argument(
        "--patch-path",
        type=str,
        default=DEFAULT_PATCH_PATH,
        help="Path to patch .pt file",
    )
    parser.add_argument(
        "--cudaid",
        type=int,
        default=0,
        help="CUDA device ID",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="Number of trials per task (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model checkpoint path (default: auto-selected based on task-suite)",
    )
    parser.add_argument(
        "--task-suite",
        type=str,
        default=DEFAULT_TASK_SUITE,
        choices=["libero_object", "libero_spatial"],
        help="Task suite name: libero_object or libero_spatial (default: libero_object)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from position index (for resuming)",
    )
    parser.add_argument(
        "--positions",
        type=str,
        nargs="+",
        help="Override positions list (format: 'x,y x,y ...')",
    )
    args = parser.parse_args()
    
    # Use custom positions if provided
    if args.positions:
        positions = []
        for pos_str in args.positions:
            x, y = map(int, pos_str.split(","))
            positions.append((x, y))
    else:
        positions = PATCH_POSITIONS
    
    # Validate patch path
    patch_path = Path(args.patch_path)
    if not patch_path.exists():
        print(f"[ERROR] Patch file not found: {patch_path}")
        sys.exit(1)
    
    # Get model (use default if not provided)
    model = args.model if args.model else get_default_model(args.task_suite)
    
    # Get folder prefix
    prefix = get_folder_prefix(args.task_suite)
    
    print("=" * 80)
    print("Multi-Position Rollout Generation")
    print("=" * 80)
    print(f"Task suite: {args.task_suite}")
    print(f"Total positions: {len(positions)}")
    print(f"Trials per task: {args.num_trials}")
    print(f"Model: {model}")
    print(f"Model root: {DEFAULT_MODEL_ROOT}")
    print(f"Dataset path: {DEFAULT_DATASET_PATH}")
    print(f"Patch path: {args.patch_path}")
    print(f"CUDA device: {args.cudaid}")
    print(f"Start from index: {args.start_from}")
    print(f"Output folder prefix: {prefix}_xy_{{x}}_{{y}}")
    print("=" * 80)
    
    # Run for each position
    successful = 0
    failed = 0
    failed_positions = []
    
    for idx, (x, y) in enumerate(positions[args.start_from:], start=args.start_from):
        print(f"\n[{idx+1}/{len(positions)}] Processing position ({x}, {y})")
        
        success = run_single_position(
            x=x,
            y=y,
            patch_path=str(patch_path),
            cuda_id=args.cudaid,
            num_trials=args.num_trials,
            model=model,
            task_suite=args.task_suite,
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_positions.append((x, y))
        
        # Progress summary
        print(f"\n[Progress] Completed: {idx+1}/{len(positions)}")
        print(f"  Successful: {successful}, Failed: {failed}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total positions: {len(positions)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_positions:
        print(f"Failed positions: {failed_positions}")
    print("=" * 80)
    
    if failed > 0:
        print("\n[WARNING] Some positions failed. You can resume with:")
        if failed_positions:
            first_failed_idx = positions.index(failed_positions[0])
            print(f"  --start-from {first_failed_idx}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All positions completed successfully!")


if __name__ == "__main__":
    main()

