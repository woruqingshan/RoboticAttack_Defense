# -*- coding: utf-8 -*-
"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch or any CUDA-using modules
# Parse --cudaid from command line arguments early to set environment variable
# This allows multiple terminals to use different GPUs independently
if '--cudaid' in sys.argv:
    cudaid_idx = sys.argv.index('--cudaid')
    if cudaid_idx + 1 < len(sys.argv):
        cudaid_value = sys.argv[cudaid_idx + 1]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid_value)
        print(f"[*] Early setting: CUDA_VISIBLE_DEVICES={cudaid_value} (before torch import)")

# Set MuJoCo rendering backend for headless environments
# This must be set BEFORE importing mujoco/robosuite
# System-level graphics libraries are required for rendering
# EGL is preferred for GPU environments (hardware-accelerated rendering)
# Fallback options: osmesa (software rendering) or glfw (requires virtual display)
if 'MUJOCO_GL' not in os.environ:
    # Use EGL for headless rendering with GPU (hardware-accelerated)
    # Requires system libraries: libegl1-mesa-dev, libgles2-mesa-dev
    # If EGL fails, fallback to osmesa: export MUJOCO_GL=osmesa
    os.environ['MUJOCO_GL'] = 'egl'

# Add LIBERO to Python path if not already present
# This handles the case where editable install didn't work properly
LIBERO_DIR = "/home/zifeng/siyuan/code/LIBERO"
if LIBERO_DIR not in sys.path:
    sys.path.insert(0, LIBERO_DIR)

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
import sys
from typing import Optional

# Add white_patch directory to path for RandomPatchTransform
WHITE_PATCH_DIR = os.path.join(os.path.dirname(__file__), "../../../VLAAttacker/white_patch")
if WHITE_PATCH_DIR not in sys.path:
    sys.path.insert(0, WHITE_PATCH_DIR)
from appply_random_transform import RandomPatchTransform
import torch
import os
import random


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
ROBOTIC_ATTACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROBOTIC_ATTACK_ROOT not in sys.path:
    sys.path.insert(0, ROBOTIC_ATTACK_ROOT)
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from evaluation_tool.defense import (
    ImagePurifier,
    OnlineAttentionHook,
    PatchBox,
    # Unified interface and auto-mode components
    UnifiedDefenseInterface,
    PatchAttentionLocalizer,
    TemporalGate,
    OnlinePatchDefenseController,
    CounterfactualVerifier,
    NoOpVerifier,
    format_defense_log_line,
    format_defense_geometry_lines,
    format_defense_selector_lines,
    defense_result_to_log_dict,
    # Multimodal geometry prior components
    GripperPrior,
    GripperPriorConfig,
    ArmSkeletonPrior,
    ArmSkeletonPriorConfig,
    GeometryRuntimeContext,
    PatchSelector,
    PatchSelectorConfig,
    SafetyRegionConfig,
    SafetyRegionBuilder,
    PixelMaskRefinerConfig,
    PixelMaskRefiner,
    TemporalConflictConfig,
    TemporalConflictResolver,
    JsonlMetricsLogger,
    action_metrics,
    attention_metrics,
    binary_mask_metrics,
    build_axis_aligned_patch_mask,
    box_to_mask,
    infer_patch_hw,
    mask_overlap_ratio,
    patch_box_xyxy,
    roi_to_mask,
    safe_float_dict,
)

# PRAC checker (optional import)
try:
    from evaluation_tool.defense.prac_checker import PRACChecker, PRACConfig
    PRAC_AVAILABLE = True
except ImportError:
    PRACChecker = None
    PRACConfig = None
    PRAC_AVAILABLE = False

def _defense_debug_print(cfg, msg: str, log_file=None) -> None:
    """Print defense debug logs when enabled."""
    if not getattr(cfg, "defense_debug", False):
        return
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")

def _defense_debug_every_step(cfg) -> bool:
    """Return True when step-level defense debug logs are enabled."""
    return bool(getattr(cfg, "defense_debug", False) and getattr(cfg, "defense_debug_every_step", False))


def _defense_debug_geometry(cfg) -> bool:
    """Return True when verbose geometry debug logs are enabled."""
    return bool(getattr(cfg, "defense_debug", False) and getattr(cfg, "defense_debug_geometry", False))

def _normalize_heatmap_uint8(heatmap) -> "np.ndarray":
    """Normalize a float heatmap into uint8 [0,255] for visualization."""
    hm = heatmap.astype(np.float32)
    hm = hm - float(hm.min())
    denom = float(hm.max()) + 1e-6
    hm = hm / denom
    return (hm * 255.0).clip(0, 255).astype("uint8")

def _make_overlay_rgb(image_rgb, heatmap, alpha: float):
    """
    Create an RGB overlay image using a heatmap (JET colormap when OpenCV is available).

    Notes:
    - image_rgb: uint8 HxWx3
    - heatmap: float HxW
    """
    try:
        import cv2  # Local import to avoid hard dependency at import time.
        hm_u8 = _normalize_heatmap_uint8(heatmap)
        colored_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        a = float(alpha)
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
        overlay = (a * colored_rgb.astype("float32") + (1.0 - a) * image_rgb.astype("float32")).clip(0, 255).astype("uint8")
        return overlay
    except Exception:
        # Fallback: grayscale heatmap stacked to RGB.
        hm_u8 = _normalize_heatmap_uint8(heatmap)
        return np.stack([hm_u8, hm_u8, hm_u8], axis=-1)

def _maybe_pack_replay_frame(
    cfg,
    image_rgb,
    heatmap: Optional["np.ndarray"],
    gripper_box=None,
    arm_region_box=None,
    gripper_points_2d=None,
    gripper_link_segments_2d=None,
    gripper_link_quads_2d=None,
    joint_points_2d=None,
    arm_link_segments_2d=None,
    arm_link_quads_2d=None,
):
    """Optionally concatenate the policy input and heatmap overlay side-by-side."""
    # If a gripper box is provided, draw it on the image_rgb (and overlay if created)
    img_to_pack = image_rgb.copy()
    try:
        import cv2
        joint_radius = int(getattr(cfg, "defense_viz_joint_radius_px", 4))
        line_thickness = int(getattr(cfg, "defense_viz_link_line_thickness", 2))
        quad_thickness = int(getattr(cfg, "defense_viz_link_quad_thickness", 2))

        if gripper_box is not None:
            # Draw a green bounding box for the gripper prior
            cv2.rectangle(
                img_to_pack,
                (int(gripper_box.x0), int(gripper_box.y0)),
                (int(gripper_box.x1), int(gripper_box.y1)),
                (0, 255, 0), 2
            )
        if gripper_points_2d is not None:
            for row, col in gripper_points_2d:
                cv2.circle(img_to_pack, (int(col), int(row)), joint_radius, (0, 255, 0), -1)
        if gripper_link_quads_2d is not None:
            for quad_xy in gripper_link_quads_2d:
                if quad_xy is None or len(quad_xy) < 4:
                    continue
                quad_np = np.array([[int(x), int(y)] for x, y in quad_xy], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_to_pack, [quad_np], isClosed=True, color=(0, 200, 0), thickness=quad_thickness)
        if gripper_link_segments_2d is not None:
            for p0_rc, p1_rc in gripper_link_segments_2d:
                cv2.line(
                    img_to_pack,
                    (int(p0_rc[1]), int(p0_rc[0])),
                    (int(p1_rc[1]), int(p1_rc[0])),
                    (0, 180, 0),
                    line_thickness,
                )
        if arm_link_quads_2d is not None:
            for quad_xy in arm_link_quads_2d:
                if quad_xy is None or len(quad_xy) < 4:
                    continue
                quad_np = np.array([[int(x), int(y)] for x, y in quad_xy], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_to_pack, [quad_np], isClosed=True, color=(255, 0, 0), thickness=quad_thickness)
        if arm_link_segments_2d is not None:
            for p0_rc, p1_rc in arm_link_segments_2d:
                cv2.line(
                    img_to_pack,
                    (int(p0_rc[1]), int(p0_rc[0])),
                    (int(p1_rc[1]), int(p1_rc[0])),
                    (255, 255, 0),
                    line_thickness,
                )
        if joint_points_2d is not None:
            for row, col in joint_points_2d:
                cv2.circle(img_to_pack, (int(col), int(row)), joint_radius, (255, 255, 0), -1)
        if arm_region_box is not None and not arm_link_quads_2d:
            # Draw a blue bounding box for the arm region (for verification)
            x0, y0, x1, y1 = arm_region_box
            cv2.rectangle(img_to_pack, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
    except Exception:
        pass

    if not getattr(cfg, "defense_viz", False):
        return img_to_pack
    if heatmap is None:
        return img_to_pack
        
    overlay = _make_overlay_rgb(img_to_pack, heatmap, alpha=getattr(cfg, "defense_viz_alpha", 0.45))
    try:
        return np.concatenate([img_to_pack, overlay], axis=1)
    except Exception:
        return img_to_pack


def _metrics_enabled(cfg) -> bool:
    return bool(getattr(cfg, "metrics_enabled", False))


def _metrics_should_sample(cfg, step: int) -> bool:
    if not _metrics_enabled(cfg):
        return False
    if bool(getattr(cfg, "metrics_every_step", True)):
        return True
    n = int(max(1, getattr(cfg, "metrics_sample_every_n", 1)))
    return (int(step) % n) == 0


def _copy_heatmap_from_hook(defense_hook) -> Optional[np.ndarray]:
    if defense_hook is None:
        return None
    try:
        hm = defense_hook.get_heatmap()
        return np.asarray(hm, dtype=np.float32).copy()
    except Exception:
        return None


def _copy_action_for_metrics(action) -> Optional[np.ndarray]:
    if action is None:
        return None
    try:
        return np.asarray(action, dtype=np.float32).copy()
    except Exception:
        return None


def _box_dict_to_list(box: Any) -> Optional[List[int]]:
    if box is None:
        return None
    if isinstance(box, dict):
        try:
            return [int(box["x0"]), int(box["y0"]), int(box["x1"]), int(box["y1"])]
        except Exception:
            return None
    if isinstance(box, (tuple, list)) and len(box) == 4:
        return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    try:
        return [int(box.x0), int(box.y0), int(box.x1), int(box.y1)]
    except Exception:
        return None


def _prefix_metric_keys(metrics: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Rename unprefixed attention metric keys into compact adv/def/clean fields."""
    mapping = {
        "pam": f"pam_{suffix}",
        "topk_attn_iou_patch": f"topk_attn_iou_patch_{suffix}",
        "attn_center_x": f"attn_center_x_{suffix}",
        "attn_center_y": f"attn_center_y_{suffix}",
        "attn_center_dist_patch": f"attn_center_dist_patch_{suffix}",
    }
    return {mapping.get(k, f"{k}_{suffix}"): v for k, v in metrics.items()}


def _mean_numeric(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float, np.integer, np.floating)) and np.isfinite(float(val)):
            vals.append(float(val))
    if not vals:
        return None
    return float(np.mean(vals))


def _build_patch_mask_for_metrics(cfg, patch, image_hw):
    patch_h = int(getattr(cfg, "metrics_patch_h", 50))
    patch_w = int(getattr(cfg, "metrics_patch_w", 50))
    patch_size_source = "cli"
    if patch is not None:
        try:
            patch_h, patch_w = infer_patch_hw(patch)
            patch_size_source = "patch_tensor"
        except Exception:
            patch_size_source = "cli_fallback"

    h, w = int(image_hw[0]), int(image_hw[1])
    patch_box = patch_box_xyxy(
        int(cfg.x),
        int(cfg.y),
        patch_w=patch_w,
        patch_h=patch_h,
        image_w=w,
        image_h=h,
    )
    patch_mask = build_axis_aligned_patch_mask(
        image_hw=(h, w),
        x=int(cfg.x),
        y=int(cfg.y),
        patch_w=patch_w,
        patch_h=patch_h,
    )
    transformed = (
        abs(float(getattr(cfg, "angle", 0.0))) > 1e-8
        or abs(float(getattr(cfg, "shx", 0.0))) > 1e-8
        or abs(float(getattr(cfg, "shy", 0.0))) > 1e-8
    )
    patch_mask_mode = "bbox_fallback" if transformed else "axis_aligned"
    return patch_mask, patch_box, patch_w, patch_h, patch_size_source, patch_mask_mode


def _defense_mask_for_metrics(image_hw, defense_result, dlog: Dict[str, Any]):
    roi_mask = getattr(defense_result, "roi_mask", None) if defense_result is not None else None
    if isinstance(roi_mask, np.ndarray) and roi_mask.ndim == 2 and roi_mask.shape == tuple(image_hw):
        return roi_mask.astype(bool), "pixel_mask"
    roi_box = dlog.get("roi_box") if isinstance(dlog, dict) else None
    if roi_box is not None:
        return roi_to_mask(image_hw, roi_box), "roi_fallback"
    return None, None


def _compute_frame_metrics_row(
    *,
    cfg,
    task_id: int,
    task_description: str,
    episode_idx: int,
    global_episode_id: int,
    step: int,
    obs: Dict[str, Any],
    patch_mask: np.ndarray,
    patch_box: tuple,
    patch_w: int,
    patch_h: int,
    patch_size_source: str,
    patch_mask_mode: str,
    defense_result,
    dlog: Dict[str, Any],
    action_clean,
    action_adv,
    action_def,
    hm_clean,
    hm_adv,
    hm_def,
) -> Dict[str, Any]:
    image_hw = tuple(patch_mask.shape)
    roi_box_list = _box_dict_to_list(dlog.get("roi_box")) if isinstance(dlog, dict) else None
    defense_mask, mask_source = _defense_mask_for_metrics(image_hw, defense_result, dlog)

    row: Dict[str, Any] = {
        "record_type": "frame_metrics",
        "metrics_version": 1,
        "task_suite_name": str(getattr(cfg, "task_suite_name", "")),
        "run_id_note": getattr(cfg, "run_id_note", None),
        "task_id": int(task_id),
        "task_description": str(task_description),
        "episode": int(episode_idx + 1),
        "episode_id": int(global_episode_id),
        "step": int(step),
        "use_patch": bool(getattr(cfg, "use_patch", False)),
        "defense_enabled": bool(getattr(cfg, "defense_enabled", False)),
        "patch_x": int(cfg.x),
        "patch_y": int(cfg.y),
        "patch_w": int(patch_w),
        "patch_h": int(patch_h),
        "patch_size_source": str(patch_size_source),
        "patch_mask_mode": str(patch_mask_mode),
        "patch_box": [int(v) for v in patch_box],
        "patch_x0": int(patch_box[0]),
        "patch_y0": int(patch_box[1]),
        "patch_x1": int(patch_box[2]),
        "patch_y1": int(patch_box[3]),
        "phase": dlog.get("phase") if isinstance(dlog, dict) else None,
        "defense_phase": dlog.get("phase") if isinstance(dlog, dict) else None,
        "defense_reason": dlog.get("reason") if isinstance(dlog, dict) else None,
        "should_purify": bool(dlog.get("should_purify")) if isinstance(dlog, dict) else False,
        "masked": bool(dlog.get("should_purify") and dlog.get("roi_box") is not None) if isinstance(dlog, dict) else False,
        "roi_box": roi_box_list,
        "mask_source": mask_source,
        "mass_heatmap": dlog.get("mass_heatmap") if isinstance(dlog, dict) else None,
        "conflict_mode": dlog.get("conflict_mode") if isinstance(dlog, dict) else None,
        "patch_verdict": dlog.get("patch_verdict") if isinstance(dlog, dict) else None,
        "quality_ok": dlog.get("quality_ok") if isinstance(dlog, dict) else None,
    }

    try:
        eef = np.asarray(obs.get("robot0_eef_pos", []), dtype=np.float32).reshape(-1)
        if eef.size >= 3:
            row.update({"eef_x": float(eef[0]), "eef_y": float(eef[1]), "eef_z": float(eef[2])})
    except Exception:
        pass

    if bool(getattr(cfg, "metrics_mask_recovery", True)) and defense_mask is not None:
        row.update(binary_mask_metrics(defense_mask, patch_mask, prefix="mask"))
        if roi_box_list is not None:
            roi_mask = roi_to_mask(image_hw, roi_box_list)
            row.update(binary_mask_metrics(roi_mask, patch_mask, prefix="selected_roi"))
        # Approximate robot/gripper overlap from logged geometry boxes when raw safety masks are unavailable.
        arm_core_box = dlog.get("arm_core_box") if isinstance(dlog, dict) else None
        arm_guard_box = dlog.get("arm_guard_box") if isinstance(dlog, dict) else None
        gripper_box = dlog.get("gripper_box") if isinstance(dlog, dict) else None
        if arm_core_box is not None:
            row.update(mask_overlap_ratio(defense_mask, box_to_mask(image_hw, arm_core_box), "mask_robot_core"))
        if arm_guard_box is not None:
            row.update(mask_overlap_ratio(defense_mask, box_to_mask(image_hw, arm_guard_box), "mask_robot_guard"))
        if gripper_box is not None:
            row.update(mask_overlap_ratio(defense_mask, box_to_mask(image_hw, gripper_box), "mask_gripper"))

    if bool(getattr(cfg, "metrics_attention_recovery", False)):
        top_q = float(getattr(cfg, "metrics_top_quantile", 0.90))
        row.update(_prefix_metric_keys(attention_metrics(hm_adv, patch_mask, top_quantile=top_q), "adv"))
        row.update(_prefix_metric_keys(attention_metrics(hm_def, patch_mask, top_quantile=top_q), "def"))
        row.update(_prefix_metric_keys(attention_metrics(hm_clean, patch_mask, top_quantile=top_q), "clean"))
        if row.get("pam_adv") is not None and row.get("pam_def") is not None:
            row["pam_reduction"] = float(row["pam_adv"] - row["pam_def"])

    if bool(getattr(cfg, "metrics_action_recovery", False)):
        row.update(action_metrics(action_clean, action_adv, action_def))

    return safe_float_dict(row)


def _episode_summary_from_rows(
    *,
    cfg,
    task_id: int,
    task_description: str,
    episode_idx: int,
    global_episode_id: int,
    success: bool,
    masked_frames: int,
    acquire_count: int,
    reacquire_count: int,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    first_masked = next((r for r in rows if bool(r.get("masked"))), None)
    summary: Dict[str, Any] = {
        "record_type": "episode_summary",
        "metrics_version": 1,
        "task_suite_name": str(getattr(cfg, "task_suite_name", "")),
        "run_id_note": getattr(cfg, "run_id_note", None),
        "task_id": int(task_id),
        "task_description": str(task_description),
        "episode": int(episode_idx + 1),
        "episode_id": int(global_episode_id),
        "success": bool(success),
        "masked_frames": int(masked_frames),
        "acquire_count": int(acquire_count),
        "reacquire_count": int(reacquire_count),
        "first_lock_step": first_masked.get("step") if first_masked else None,
        "first_roi_box": first_masked.get("roi_box") if first_masked else None,
    }
    for key in [
        "mask_patch_iou",
        "mask_patch_recall",
        "mask_patch_precision",
        "mask_area_ratio",
        "selected_roi_patch_iou",
        "selected_roi_patch_recall",
        "selected_roi_patch_precision",
        "pam_adv",
        "pam_def",
        "pam_clean",
        "pam_reduction",
        "topk_attn_iou_patch_adv",
        "topk_attn_iou_patch_def",
        "attn_center_dist_patch_adv",
        "attn_center_dist_patch_def",
        "action_l2_adv_to_clean",
        "action_l2_def_to_clean",
        "action_l2_xyz_adv_to_clean",
        "action_l2_xyz_def_to_clean",
        "nar_l2",
        "nar_xyz",
    ]:
        summary[f"mean_{key}"] = _mean_numeric(rows, key)
    return safe_float_dict(summary)


# @dataclass
# class GenerateConfig:
#     # fmt: off
#
#     #################################################################################################################
#     # Model-specific parameters
#     #################################################################################################################
#     model_family: str = "openvla"                    # Model family
#     pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
#     load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
#     load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
#
#     center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
#
#     #################################################################################################################
#     # LIBERO environment-specific parameters
#     #################################################################################################################
#     task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
#     num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
#     num_trials_per_task: int = 50                    # Number of rollouts per task
#
#     #################################################################################################################
#     # Utils
#     #################################################################################################################
#     run_id_note: Optional[str] = "spatial1"                # Extra note to add in run ID for logging
#     local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
#
#     use_wandb: bool = False                          # Whether to also log results in Weights & Biases
#     wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
#     wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
#
#     seed: int = 7                                    # Random Seed (for reproducibility)
#
#     # fmt: on

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# @draccus.wrap()
def eval_libero(cfg) -> None:
    # Note: CUDA_VISIBLE_DEVICES is now set in if __name__ == "__main__" before this function is called
    # This ensures each process can independently use different GPUs
    
    # Update DEVICE in openvla_utils module
    from experiments.robot import openvla_utils
    openvla_utils.DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Check Flash Attention availability (optional verification)
    try:
        import flash_attn
        print(f"[*] Flash Attention 2 is available (version: {flash_attn.__version__})")
        print(f"[*] Flash Attention will be automatically used when loading the model")
    except ImportError:
        print("[*] Flash Attention 2 is not available - will use default attention implementation")
    
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    randomPatchTransform = None
    patch = None
    if cfg.use_patch:
        # Ensure patch file exists when applying adversarial patch
        assert cfg.patchroot and os.path.isfile(cfg.patchroot), "patchroot must be a valid file when use_patch=True"
        randomPatchTransform = RandomPatchTransform('cpu', False)
        patch = torch.load(cfg.patchroot)
    else:
        print("[*] Running CLEAN (no patch) evaluation")
    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model (DEVICE is set via global variable in openvla_utils)
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    defense_interface = None  # Unified defense interface
    defense_purifier = None  # Keep for direct access if needed
    defense_hook = None
    defense_mode = getattr(cfg, "defense_mode", "known")  # Default: backward compatible
    metrics_need_attention = bool(
        getattr(cfg, "metrics_enabled", False) and getattr(cfg, "metrics_attention_recovery", False)
    )
    if getattr(cfg, "defense_enabled", False) or metrics_need_attention:
        # Create hook for defense and/or attention evidence logging.
        defense_hook = OnlineAttentionHook(
            model=model,
            attn_module_name=getattr(cfg, "defense_attn_module", None),
            aggregate_mode=getattr(cfg, "defense_aggregate_mode", "mean"),
            image_size=get_image_resize_size(cfg),
        )

    if getattr(cfg, "defense_enabled", False):
        # Create purifier (required for both modes)
        defense_purifier = ImagePurifier(
            strategy=getattr(cfg, "defense_purifier_strategy", "mask_mean"),
            pad=getattr(cfg, "defense_purifier_pad", 0),
            gray_value=getattr(cfg, "defense_gray_value", 127),
            alpha=getattr(cfg, "defense_purifier_alpha", 0.8),
        )

        # Create unified interface based on mode
        if defense_mode == "known":
            # Known location mode: always masks the patch at the given location (no detection needed)
            defense_interface = UnifiedDefenseInterface(
                hook=defense_hook,
                mode="known",
                detector=None,  # No longer needed in known mode (deprecated)
                patch_x=int(cfg.x),
                patch_y=int(cfg.y),
                patch_w=int(getattr(cfg, "defense_patch_w", 50)),
                patch_h=int(getattr(cfg, "defense_patch_h", 50)),
                use_heatmap_for_viz=getattr(cfg, "defense_viz", False),
            )
        
        elif defense_mode == "auto":
            # Auto localization mode (new feature)
            localizer = PatchAttentionLocalizer(
                top_p=getattr(cfg, "defense_localizer_top_p", 0.07),
                min_area_frac=getattr(cfg, "defense_localizer_min_area", 0.003),
                max_area_frac=getattr(cfg, "defense_localizer_max_area", 0.12),
            )
            gate = TemporalGate(
                theta_on=getattr(cfg, "defense_gate_theta_on", 0.07),
                theta_off=getattr(cfg, "defense_gate_theta_off", 0.05),
                ema_alpha=getattr(cfg, "defense_gate_ema_alpha", 0.3),
                check_every_k=getattr(cfg, "defense_gate_check_every_k", 3),
            )
            
            # Create verifier if enabled
            if getattr(cfg, "defense_verifier_enabled", True):  # Default: enabled
                verifier = CounterfactualVerifier(
                    min_mass_drop_rel=getattr(cfg, "defense_verifier_min_mass_drop", 0.15),
                    min_entropy_gain_abs=getattr(cfg, "defense_verifier_min_entropy_gain", 0.02),
                    entropy_hard=getattr(cfg, "defense_verifier_entropy_hard", False),
                    max_main_mass_drop_rel=getattr(cfg, "defense_verifier_max_main_drop", 0.10),
                    min_action_diff_l2=getattr(cfg, "defense_verifier_min_action_diff_l2", 0.01),
                    min_action_diff_rel=getattr(cfg, "defense_verifier_min_action_diff_rel", 0.05),
                    min_gripper_diff=getattr(cfg, "defense_verifier_min_gripper_diff", 0.1),
                    use_action_verification=getattr(cfg, "defense_verifier_use_action", True),
                )
            else:
                verifier = NoOpVerifier()
            
            # Create PRAC checker if enabled (kept for fallback)
            prac_checker = None
            if PRAC_AVAILABLE and getattr(cfg, "defense_prac_enabled", True):  # Default: enabled if available
                prac_cfg = PRACConfig(
                    n_views=getattr(cfg, "defense_prac_n_views", 6),
                    patch_size=getattr(cfg, "defense_prac_patch_size", 16),
                    mask_ratio=getattr(cfg, "defense_prac_mask_ratio", 0.25),
                    transform_mode=getattr(cfg, "defense_prac_transform_mode", "blur"),
                    tau_odr=getattr(cfg, "defense_prac_tau_odr", 1.8),
                    tau_mer=getattr(cfg, "defense_prac_tau_mer", 0.20),
                    max_attempts=getattr(cfg, "defense_prac_max_attempts", 2),
                    seed=getattr(cfg, "defense_prac_seed", None),
                )
                prac_checker = PRACChecker(cfg=prac_cfg)
            
            # Create Multimodal prior components if enabled
            gripper_prior = None
            arm_skeleton_prior = None
            patch_selector = None
            safety_region_builder = None
            pixel_mask_refiner = None
            temporal_conflict_resolver = None
            if getattr(cfg, "defense_gripper_prior_enabled", True):
                gp_cfg = GripperPriorConfig(
                    enabled=True,
                    site_names=parse_csv_list(getattr(cfg, "defense_gripper_site_names", "grip_site,ft_frame")),
                    body_names=parse_csv_list(
                        getattr(
                            cfg,
                            "defense_gripper_body_names",
                            "right_hand,right_gripper,eef,leftfinger,rightfinger,finger_joint1_tip,finger_joint2_tip",
                        )
                    ),
                    segment_point_pairs=parse_csv_pairs(
                        getattr(
                            cfg,
                            "defense_gripper_segment_pairs",
                            "right_hand:ft_frame,ft_frame:grip_site,grip_site:finger_joint1_tip,grip_site:finger_joint2_tip",
                        )
                    ),
                    name_prefixes=[""] + parse_csv_list(getattr(cfg, "defense_gripper_name_prefixes", "robot0_,Panda0_,Panda_")),
                    camera_name=getattr(cfg, "defense_gripper_camera_name", "agentview"),
                    point_radius_px=int(getattr(cfg, "defense_gripper_point_radius_px", 5)),
                    core_thickness_px=int(getattr(cfg, "defense_gripper_core_thickness_px", 10)),
                    guard_scale=float(getattr(cfg, "defense_gripper_guard_scale", 2.0)),
                    segment_core_thicknesses=parse_csv_ints(getattr(cfg, "defense_gripper_segment_core_thicknesses", "")),
                    segment_guard_scales=parse_csv_floats(getattr(cfg, "defense_gripper_segment_guard_scales", "")),
                    min_valid_points=int(getattr(cfg, "defense_gripper_min_valid_points", 1)),
                )
                gripper_prior = GripperPrior(gp_cfg)

                if getattr(cfg, "defense_arm_skeleton_enabled", False):
                    arm_cfg = ArmSkeletonPriorConfig(
                        enabled=True,
                        geometry_source=getattr(cfg, "defense_arm_skeleton_source", "body"),
                        body_names=parse_csv_list(
                            getattr(cfg, "defense_arm_body_names", "base,link1,link2,link3,link4,link5,link6,link7,right_hand")
                        ),
                        site_names=parse_csv_list(getattr(cfg, "defense_arm_site_names", "")),
                        name_prefixes=[""] + parse_csv_list(getattr(cfg, "defense_arm_name_prefixes", "robot0_,Panda0_,Panda_")),
                        camera_name=getattr(cfg, "defense_arm_camera_name", "agentview"),
                        joint_radius_px=int(getattr(cfg, "defense_arm_joint_radius_px", 4)),
                        core_thickness_px=int(getattr(cfg, "defense_arm_core_thickness_px", 14)),
                        guard_scale=float(getattr(cfg, "defense_arm_guard_scale", 1.8)),
                        link_core_thicknesses=parse_csv_ints(getattr(cfg, "defense_arm_link_core_thicknesses", "")),
                        link_guard_scales=parse_csv_floats(getattr(cfg, "defense_arm_link_guard_scales", "")),
                        min_valid_points=int(getattr(cfg, "defense_arm_min_valid_points", 3)),
                    )
                    arm_skeleton_prior = ArmSkeletonPrior(arm_cfg)
                
                ps_cfg = PatchSelectorConfig(
                    tau_g=getattr(cfg, "defense_tau_g", 0.3),
                    tau_arm=getattr(cfg, "defense_tau_arm", 0.3),
                    tau_patch_strength=getattr(cfg, "defense_tau_patch_strength", 0.05),
                    near_task_tau=getattr(cfg, "defense_near_task_tau", 0.08),
                    allow_near_task_patch=getattr(cfg, "defense_allow_near_task_patch", False),
                    selector_debug_enabled=getattr(cfg, "defense_selector_debug_enabled", False),
                    selector_debug_topk=int(getattr(cfg, "defense_selector_debug_topk", 3)),
                    expected_patch_area_ratio=float(getattr(cfg, "defense_expected_patch_area_ratio", 0.04)),
                    patch_area_sigma=float(getattr(cfg, "defense_patch_area_sigma", 0.03)),
                    patch_aspect_sigma=float(getattr(cfg, "defense_patch_aspect_sigma", 0.4)),
                    corner_prior_enabled=bool(getattr(cfg, "defense_corner_prior_enabled", False)),
                    corner_prior_type=str(getattr(cfg, "defense_corner_prior_type", "top_right")),
                    corner_prior_sigma=float(getattr(cfg, "defense_corner_prior_sigma", 0.35)),
                )
                patch_selector = PatchSelector(ps_cfg)

            if getattr(cfg, "defense_safety_region_enabled", True):
                sr_cfg = SafetyRegionConfig(
                    w_arm_core=float(getattr(cfg, "defense_safety_w_arm_core", 1.0)),
                    w_arm_guard=float(getattr(cfg, "defense_safety_w_arm_guard", 0.6)),
                    w_gripper_core=float(getattr(cfg, "defense_safety_w_gripper_core", 1.0)),
                    w_gripper_guard=float(getattr(cfg, "defense_safety_w_gripper_guard", 0.7)),
                    smooth_kernel=int(getattr(cfg, "defense_safety_smooth_kernel", 0)),
                )
                safety_region_builder = SafetyRegionBuilder(sr_cfg)

            if getattr(cfg, "defense_pixel_mask_refine_enabled", True):
                pm_cfg = PixelMaskRefinerConfig(
                    lambda_safety=float(getattr(cfg, "defense_pixel_lambda_safety", 0.75)),
                    score_quantile=float(getattr(cfg, "defense_pixel_score_quantile", 0.65)),
                    min_area_ratio=float(getattr(cfg, "defense_pixel_min_area_ratio", 0.08)),
                    min_cover_ratio=float(getattr(cfg, "defense_pixel_min_cover_ratio", 0.40)),
                    keep_largest_component=bool(getattr(cfg, "defense_pixel_keep_largest_component", False)),
                    hard_forbid_core=bool(getattr(cfg, "defense_pixel_hard_forbid_core", False)),
                )
                pixel_mask_refiner = PixelMaskRefiner(pm_cfg)

            if getattr(cfg, "defense_temporal_conflict_enabled", True):
                tc_cfg = TemporalConflictConfig(
                    core_hard_on=float(getattr(cfg, "defense_conflict_core_hard_on", 0.10)),
                    core_hard_off=float(getattr(cfg, "defense_conflict_core_hard_off", 0.04)),
                    guard_soft_on=float(getattr(cfg, "defense_conflict_guard_soft_on", 0.25)),
                    guard_soft_off=float(getattr(cfg, "defense_conflict_guard_soft_off", 0.12)),
                    hard_on_frames=int(getattr(cfg, "defense_conflict_hard_on_frames", 2)),
                    soft_on_frames=int(getattr(cfg, "defense_conflict_soft_on_frames", 2)),
                    soft_alpha=float(getattr(cfg, "defense_conflict_soft_alpha", 0.7)),
                )
                temporal_conflict_resolver = TemporalConflictResolver(tc_cfg)

            controller = OnlinePatchDefenseController(
                hook=defense_hook,
                localizer=localizer,
                gate=gate,
                verifier=verifier,
                verify_every_k=getattr(cfg, "defense_verifier_every_k", 3),  # Verify every K frames during TRACK
                require_verify=getattr(cfg, "defense_require_verify", False),  # Default: False (backward compatible)
                min_trigger_mass_heatmap=getattr(cfg, "defense_min_trigger_mass_heatmap", 0.02),
                quality_mass_source=getattr(cfg, "defense_quality_mass_source", "heatmap"),
                strength_min=getattr(cfg, "defense_strength_min", 0.35),
                strength_max=getattr(cfg, "defense_strength_max", 0.85),
                prac_checker=prac_checker,
                prac_enabled=getattr(cfg, "defense_prac_enabled", True),  # Default: enabled if prac_checker is provided
                gripper_prior=gripper_prior,
                arm_skeleton_prior=arm_skeleton_prior,
                safety_region_builder=safety_region_builder,
                pixel_mask_refiner=pixel_mask_refiner,
                temporal_conflict_resolver=temporal_conflict_resolver,
                patch_selector=patch_selector,
                tau_protect=getattr(cfg, "defense_tau_protect", 0.1),
                tau_cover=getattr(cfg, "defense_tau_cover", 0.5),
                use_residual_candidate_grid=getattr(cfg, "defense_use_residual_candidate_grid", False),
                geometry_residual_gamma=getattr(cfg, "defense_geometry_residual_gamma", 1.0),
                geometry_guard_weight_arm=getattr(cfg, "defense_geometry_guard_weight_arm", 0.6),
                geometry_guard_weight_gripper=getattr(cfg, "defense_geometry_guard_weight_gripper", 0.7),
            )
            defense_interface = UnifiedDefenseInterface(
                hook=defense_hook,
                mode="auto",
                controller=controller,
                use_heatmap_for_viz=getattr(cfg, "defense_viz", False),
            )

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    print(f"Log Path:{str(os.path.join(cfg.local_log_dir, cfg.task_suite_name, '.txt'))}")
    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Log Path:{str(os.path.join(cfg.local_log_dir, cfg.task_suite_name, '.txt'))}")
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if getattr(cfg, "single_task_id", None) is not None and task_id != int(cfg.single_task_id):
            continue

        # New semantics under ACQUIRE/TRACK:
        # - masked_frames: number of frames where masking was applied
        # - acquire_count: number of times we entered TRACK in this task
        # - reacquire_count: number of times verifier requested a reacquire
        task_masked_frames = 0
        task_acquire_count = 0
        task_reacquire_count = 0

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        _defense_debug_print(
            cfg,
            f"[DEFENSE][TASK_START] task_id={task_id} task='{task_description.strip()}' "
            f"enabled={getattr(cfg, 'defense_enabled', False)} "
            f"patch_xy=({int(cfg.x)},{int(cfg.y)}) angle={float(cfg.angle)} shx={float(cfg.shx)} shy={float(cfg.shy)} "
            f"purifier={getattr(cfg, 'defense_purifier_strategy', 'NA')} "
            f"th_patch_mass={getattr(cfg, 'defense_patch_mass_threshold', 'NA')}",
            log_file=log_file,
        )

        metrics_logger = None
        if _metrics_enabled(cfg) and bool(getattr(cfg, "metrics_save_jsonl", True)):
            note = str(getattr(cfg, "run_id_note", "run") or "run").replace(os.sep, "_")
            metrics_path = os.path.join(cfg.local_log_dir, f"metrics_task{task_id}_{note}.jsonl")
            metrics_logger = JsonlMetricsLogger(
                metrics_path,
                enabled=True,
                strict=bool(getattr(cfg, "metrics_strict", False)),
            )
            log_file.write(f"[METRICS] Writing recovery metrics to {metrics_path}\n")
            log_file.flush()

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()
            
            # Reset defense stateful components for new episode
            if defense_interface is not None:
                try:
                    defense_interface.reset()
                except Exception as reset_error:
                    error_msg = f"FATAL DEFENSE RESET ERROR: {reset_error}"
                    print(error_msg)
                    print("Exiting immediately to prevent empty episode analysis...")
                    log_file.write(error_msg + "\n")
                    log_file.close()
                    sys.exit(1)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx]) #

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 193  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 254  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 270  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 505  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 373  # longest training demo has 373 steps
            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            episode_masked_frames = 0
            episode_acquire_count = 0
            episode_reacquire_count = 0
            prev_phase = None
            done = False
            episode_metric_rows: List[Dict[str, Any]] = []
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image. Metrics keep clean / adversarial / defended
                    # copies as evidence only; the executed action path below is unchanged.
                    policy_image_rotate_180 = bool(getattr(cfg, "defense_geometry_rotate_180", True))
                    img_clean_for_metric = get_libero_image(obs, resize_size, rotate_180=policy_image_rotate_180)
                    img = img_clean_for_metric.copy() # Preprocess image for model
                    if cfg.use_patch:
                        img = randomPatchTransform.simulation_random_patch(
                            img, patch, geometry=True, colorjitter=False,
                            angle=cfg.angle, shx=cfg.shx, shy=cfg.shy, position=(cfg.x, cfg.y)
                        )
                    img_adv_for_metric = img.copy()
                    img_def_for_metric = img_adv_for_metric
                    img_for_policy = img
                    patch_mask_for_metric, patch_box_for_metric, patch_w_for_metric, patch_h_for_metric, patch_size_source, patch_mask_mode = (
                        _build_patch_mask_for_metrics(cfg, patch, img_clean_for_metric.shape[:2])
                    )

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img_for_policy,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action (this forward pass is also used to populate attention hooks)
                    if defense_hook is not None:
                        try:
                            defense_hook.clear()
                        except Exception as clear_error:
                            error_msg = f"FATAL DEFENSE CLEAR ERROR: {clear_error}"
                            print(error_msg)
                            print("Exiting immediately to prevent empty episode analysis...")
                            log_file.write(error_msg + "\n")
                            log_file.close()
                            sys.exit(1)
                    action = get_action(cfg, model, observation, task_description, processor=processor)
                    action_adv_raw_for_metric = _copy_action_for_metrics(action)
                    action_def_raw_for_metric = action_adv_raw_for_metric
                    action_clean_raw_for_metric = None
                    hm_adv_for_metric = _copy_heatmap_from_hook(defense_hook)
                    hm_def_for_metric = hm_adv_for_metric
                    hm_clean_for_metric = None
                    defense_result = None
                    dlog: Dict[str, Any] = {
                        "phase": None,
                        "reason": None,
                        "should_purify": False,
                        "roi_box": None,
                    }

                    # Unified defense step (works for both known and auto modes)
                    heatmap_for_viz = None
                    gripper_box_for_viz = None
                    arm_region_box_for_viz = None
                    gripper_points_for_viz = None
                    gripper_link_segments_for_viz = None
                    gripper_link_quads_for_viz = None
                    joint_points_for_viz = None
                    arm_link_segments_for_viz = None
                    arm_link_quads_for_viz = None
                    if defense_interface is not None:
                        try:
                            # Extract EEF pos for Multimodal Geometric Prior
                            # We concatenate pos, quat (as axis-angle), and gripper qpos
                            eef_pos = np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            )
                            geometry_ctx = None
                            if "agentview_image" in obs:
                                geometry_ctx = GeometryRuntimeContext(
                                    sim=env.sim,
                                    camera_name=getattr(cfg, "defense_arm_camera_name", "agentview"),
                                    render_hw=tuple(obs["agentview_image"].shape[:2]),
                                    policy_hw=tuple(img.shape[:2]),
                                    policy_image_rotate_180=policy_image_rotate_180,
                                )

                            # For auto mode, pass additional parameters
                            if defense_mode == "auto":
                                defense_result = defense_interface.step(
                                    image=img,
                                    purify_fn=lambda img, box: defense_purifier.purify(img, box),
                                    forward_fn=lambda img: get_action(cfg, model, {
                                        "full_image": img,
                                        "state": eef_pos,
                                    }, task_description, processor=processor),
                                    heatmap_fn=lambda: defense_hook.get_heatmap(),
                                    hm_current=defense_hook.get_heatmap(),
                                    eef_pos=eef_pos,
                                    geometry_ctx=geometry_ctx,
                                )
                            else:
                                defense_result = defense_interface.step(eef_pos=eef_pos, geometry_ctx=geometry_ctx)
                        except Exception as defense_error:
                            # Exit immediately on defense errors to avoid empty episode analysis
                            error_msg = f"FATAL DEFENSE ERROR: {defense_error}"
                            print(error_msg)
                            print("Exiting immediately to prevent empty episode analysis...")
                            log_file.write(error_msg + "\n")
                            log_file.close()
                            sys.exit(1)
                        
                        # Get heatmap, gripper box, and arm region for visualization
                        if getattr(cfg, "defense_viz", False):
                            heatmap_for_viz = defense_result.heatmap
                            gripper_box_for_viz = getattr(defense_result, "gripper_box", None)
                            gripper_points_for_viz = getattr(defense_result, "gripper_points_2d", None)
                            gripper_link_segments_for_viz = getattr(defense_result, "gripper_link_segments_2d", None)
                            gripper_link_quads_for_viz = getattr(defense_result, "gripper_link_quads_2d", None)
                            joint_points_for_viz = getattr(defense_result, "joint_points_2d", None)
                            arm_link_segments_for_viz = getattr(defense_result, "arm_link_segments_2d", None)
                            arm_link_quads_for_viz = getattr(defense_result, "arm_link_quads_2d", None)
                        arm_region_box_for_viz = getattr(defense_result, "arm_region_box", None)
                        
                        # Statistics (new semantics)
                        dlog = defense_result_to_log_dict(defense_result)
                        if bool(dlog.get("should_purify")) and dlog.get("roi_box") is not None:
                            task_masked_frames += 1
                            episode_masked_frames += 1
                        # Count entering TRACK (ACQUIRE -> TRACK edge)
                        if prev_phase != "TRACK" and dlog.get("phase") == "TRACK":
                            task_acquire_count += 1
                            episode_acquire_count += 1
                        # Count verifier-triggered reacquire
                        if bool(dlog.get("reacquire_needed")):
                            task_reacquire_count += 1
                            episode_reacquire_count += 1
                        prev_phase = dlog.get("phase")
                        
                        # Debug output (structured; no legacy patch_mass/entropy)
                        if _defense_debug_every_step(cfg):
                            _defense_debug_print(
                                cfg,
                                f"[DEFENSE] {format_defense_log_line(step=int(t), result=defense_result)}",
                                log_file=log_file,
                            )
                        if _defense_debug_geometry(cfg):
                            for geom_line in format_defense_geometry_lines(step=int(t), result=defense_result):
                                _defense_debug_print(cfg, geom_line, log_file=log_file)
                        if getattr(cfg, "defense_debug_selector", False):
                            for selector_line in format_defense_selector_lines(step=int(t), result=defense_result):
                                _defense_debug_print(cfg, selector_line, log_file=log_file)
                        
                        # Check if purification is needed (unified field)
                        roi_mask = getattr(defense_result, "roi_mask", None)
                        if defense_result.should_purify and (roi_mask is not None or defense_result.roi_box is not None):
                            
                            # Purify image (unified interface)
                            # Use strength from defense_result if available (auto mode provides dynamic strength)
                            if isinstance(roi_mask, np.ndarray):
                                img_for_policy = defense_purifier.purify_with_mask(
                                    img_for_policy,
                                    roi_mask,
                                    strength=getattr(defense_result, "strength", None),
                                )
                            else:
                                img_for_policy = defense_purifier.purify(
                                    img_for_policy,
                                    defense_result.roi_box,
                                    strength=getattr(defense_result, "strength", None),  # Use dynamic strength if available
                                )
                            
                            # Recompute action on purified image
                            observation["full_image"] = img_for_policy
                            img_def_for_metric = img_for_policy.copy()
                            if getattr(cfg, "defense_recompute_action", True):
                                if defense_hook is not None:
                                    defense_hook.clear()
                                action = get_action(cfg, model, observation, task_description, processor=processor)
                                action_def_raw_for_metric = _copy_action_for_metrics(action)
                                hm_def_for_metric = _copy_heatmap_from_hook(defense_hook)
                            
                            _defense_debug_print(
                                cfg,
                                f"[DEFENSE][MASK] {format_defense_log_line(step=int(t), result=defense_result)}",
                                log_file=log_file,
                            )

                    if _metrics_enabled(cfg) and bool(getattr(cfg, "metrics_action_recovery", False)):
                        clean_observation = {
                            "full_image": img_clean_for_metric,
                            "state": observation["state"],
                        }
                        if defense_hook is not None:
                            try:
                                defense_hook.clear()
                            except Exception:
                                pass
                        clean_action_tmp = get_action(
                            cfg,
                            model,
                            clean_observation,
                            task_description,
                            processor=processor,
                        )
                        action_clean_raw_for_metric = _copy_action_for_metrics(clean_action_tmp)
                        hm_clean_for_metric = _copy_heatmap_from_hook(defense_hook)

                    if _metrics_should_sample(cfg, int(t)):
                        frame_row = _compute_frame_metrics_row(
                            cfg=cfg,
                            task_id=int(task_id),
                            task_description=task_description,
                            episode_idx=int(episode_idx),
                            global_episode_id=int(total_episodes + 1),
                            step=int(t),
                            obs=obs,
                            patch_mask=patch_mask_for_metric,
                            patch_box=patch_box_for_metric,
                            patch_w=patch_w_for_metric,
                            patch_h=patch_h_for_metric,
                            patch_size_source=patch_size_source,
                            patch_mask_mode=patch_mask_mode,
                            defense_result=defense_result,
                            dlog=dlog,
                            action_clean=action_clean_raw_for_metric,
                            action_adv=action_adv_raw_for_metric,
                            action_def=action_def_raw_for_metric,
                            hm_clean=hm_clean_for_metric,
                            hm_adv=hm_adv_for_metric,
                            hm_def=hm_def_for_metric,
                        )
                        episode_metric_rows.append(frame_row)
                        if metrics_logger is not None:
                            metrics_logger.write(frame_row)

                    # Save replay frame:
                    # - default: policy input image
                    # - optional: side-by-side with real-time heatmap overlay; green=gripper, blue=arm region
                    replay_images.append(_maybe_pack_replay_frame(
                        cfg, img_for_policy, heatmap_for_viz,
                        gripper_box=gripper_box_for_viz,
                        arm_region_box=arm_region_box_for_viz,
                        gripper_points_2d=gripper_points_for_viz,
                        gripper_link_segments_2d=gripper_link_segments_for_viz,
                        gripper_link_quads_2d=gripper_link_quads_for_viz,
                        joint_points_2d=joint_points_for_viz,
                        arm_link_segments_2d=arm_link_segments_for_viz,
                        arm_link_quads_2d=arm_link_quads_for_viz,
                    ))

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            # When rollout_per_task_dir is True: save under rollout_root_dir / exp_name / task{N}_suffix / DATE (e.g. clean_baseline/task0_no_attack_no_defense/2026_03_07/)
            if getattr(cfg, "rollout_per_task_dir", False):
                task_suffix = getattr(cfg, "exp_name_task_suffix", "_no_attack_no_defense")
                exp_name_save = os.path.join(cfg.exp_name, f"task{task_id}{task_suffix}")
            else:
                exp_name_save = cfg.exp_name
            print(f"Saving replay video...")
            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                exp_name=exp_name_save,
                rollout_root_dir=getattr(cfg, "rollout_root_dir", "./rollouts"),
            )
            _defense_debug_print(
                cfg,
                f"[DEFENSE][EPISODE_END] episode={task_episodes+1} success={done} "
                f"masked_frames={episode_masked_frames} acquire_count={episode_acquire_count} reacquire_count={episode_reacquire_count}",
                log_file=log_file,
            )
            if _metrics_enabled(cfg):
                episode_summary_row = _episode_summary_from_rows(
                    cfg=cfg,
                    task_id=int(task_id),
                    task_description=task_description,
                    episode_idx=int(episode_idx),
                    global_episode_id=int(total_episodes),
                    success=bool(done),
                    masked_frames=int(episode_masked_frames),
                    acquire_count=int(episode_acquire_count),
                    reacquire_count=int(episode_reacquire_count),
                    rows=episode_metric_rows,
                )
                if metrics_logger is not None:
                    metrics_logger.write(episode_summary_row)

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        _defense_debug_print(
            cfg,
            f"[DEFENSE][TASK_END] task_id={task_id} episodes={task_episodes} "
            f"success_rate={float(task_successes) / float(task_episodes):.3f} "
            f"masked_frames={task_masked_frames} acquire_count={task_acquire_count} reacquire_count={task_reacquire_count}",
            log_file=log_file,
        )
        if metrics_logger is not None:
            metrics_logger.close()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()


    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)
    # ?????????????????????
    with open(os.path.join(cfg.local_log_dir,cfg.task_suite_name+".txt"), "a") as file:
        file.write(f"success_rate/total:{float(total_successes) / float(total_episodes)}, num_episodes/total:{total_episodes} position_info:{cfg.angle}_{cfg.shx}_{cfg.shy}_{cfg.x}_{cfg.y} \n")  # ???????????

import argparse
from pathlib import Path
from typing import Optional, Union

def str2bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_csv_list(value):
    """Parse a comma-separated CLI string into a clean list."""
    if value is None:
        return []
    parts = [item.strip() for item in str(value).split(",")]
    return [item for item in parts if item]


def parse_csv_ints(value):
    """Parse a comma-separated CLI string into a list of ints."""
    return [int(item) for item in parse_csv_list(value)]


def parse_csv_floats(value):
    """Parse a comma-separated CLI string into a list of floats."""
    return [float(item) for item in parse_csv_list(value)]


def parse_csv_pairs(value):
    """Parse a comma-separated list like a:b,c:d into a list of string pairs."""
    pairs = []
    for item in parse_csv_list(value):
        left, sep, right = item.partition(":")
        if not sep:
            continue
        left = left.strip()
        right = right.strip()
        if left and right:
            pairs.append((left, right))
    return pairs

def parse_args():
    parser = argparse.ArgumentParser(description="Generate configuration for model training/evaluation")
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    parser.add_argument("--model_family", type=str, default="openvla", help="Model family")
    parser.add_argument("--exp_name", type=str, default=f"libero_object", help="Model family")
    parser.add_argument("--pretrained_checkpoint", type=str, default="openvla/openvla-7b-finetuned-libero-object", help="Pretrained checkpoint path")
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    parser.add_argument("--load_in_4bit", type=bool, default=False)
    parser.add_argument("--center_crop", type=bool, default=True, help="Center crop? (if trained w/ random crop image aug)")

    ################################################################################################# ################
    # LIBERO environment-specific parameters
    #################################################################################################################
    parser.add_argument("--task_suite_name", type=str, default="libero_object", help="Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90")
    parser.add_argument("--num_steps_wait", type=int, default=10, help="Number of steps to wait for objects to stabilize in sim")
    parser.add_argument("--num_trials_per_task", type=int, default=100, help="Number of rollouts per task")
    parser.add_argument("--single_task_id", type=int, default=None, help="If set, only evaluate a single task id (debug).")

    #################################################################################################################
    # Utils
    #################################################################################################################
    parser.add_argument("--run_id_note", type=str, default=f"test_libero_object", help="Extra note to add in run ID for logging")
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs", help="Local directory for eval logs")
    parser.add_argument("--rollout_root_dir", type=str, default="./rollouts", help="Root directory for saving rollout videos.")
    parser.add_argument("--rollout_per_task_dir", type=str2bool, default=False, help="If True, save videos under exp_name/task{N}_suffix/DATE so each task has its own folder (e.g. clean_baseline/task0_no_attack_no_defense/2026_03_07/).")
    parser.add_argument("--exp_name_task_suffix", type=str, default="_no_attack_no_defense", help="Suffix for per-task folder when rollout_per_task_dir is True (e.g. task0_no_attack_no_defense).")
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="Whether to also log results in Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="LIBERO_simulation_test", help="Name of W&B project to log to (use default!)")
    parser.add_argument("--wandb_entity", type=str, default="taowen_wang-rit", help="Name of entity to log under")
    parser.add_argument("--seed", type=int, default=7, help="Random Seed (for reproducibility)")
    # Patch control
    parser.add_argument("--use_patch", type=str2bool, default=True, help="Whether to apply adversarial patch")
    parser.add_argument("--patchroot", type=str, default="/spl_data/tw9146/openvla-main/run/white_patch_attack/a5083c2b-1186-4464-ab9f-1056211a2221/4000/patch.pt", help="")
    parser.add_argument("--x", type=int, default=2, help="")
    parser.add_argument("--y", type=int, default=2, help="")
    parser.add_argument("--angle", type=float, default=0, help="")
    parser.add_argument("--shx", type=float, default=0, help="")
    parser.add_argument("--shy", type=float, default=0, help="")
    parser.add_argument("--cudaid", type=int, default=2, help="")

    # Defense control (online, debug stage)
    parser.add_argument("--defense_enabled", type=str2bool, default=False, help="Enable online attention-based defense.")
    parser.add_argument("--defense_mode", type=str, default="known", choices=["known", "auto"],
                        help="Defense mode: 'known' (oracle patch location) or 'auto' (localize from heatmap).")
    parser.add_argument("--defense_attn_module", type=str, default=None, help="Optional attention module name to hook.")
    parser.add_argument("--defense_aggregate_mode", type=str, default="mean", help="Attention head aggregation mode.")
    # Known mode parameters
    parser.add_argument("--defense_patch_mass_threshold", type=float, default=0.25, help="Patch attention mass threshold (known mode).")
    parser.add_argument("--defense_use_entropy_gate", type=str2bool, default=False, help="Gate detection by entropy threshold (known mode).")
    parser.add_argument("--defense_entropy_threshold", type=float, default=None, help="Entropy threshold when gating is enabled (known mode).")
    parser.add_argument("--defense_patch_w", type=int, default=50, help="Patch width in pixels (known mode, axis-aligned box).")
    parser.add_argument("--defense_patch_h", type=int, default=50, help="Patch height in pixels (known mode, axis-aligned box).")
    # Auto mode parameters
    parser.add_argument("--defense_localizer_top_p", type=float, default=0.07, help="Top-p fraction for saliency thresholding (auto mode).")
    parser.add_argument("--defense_localizer_min_area", type=float, default=0.003, help="Minimum area ratio for candidate ROI (auto mode).")
    parser.add_argument("--defense_localizer_max_area", type=float, default=0.12, help="Maximum area ratio for candidate ROI (auto mode).")
    parser.add_argument("--defense_gate_theta_on", type=float, default=0.07, help="Threshold to trigger masking (auto mode, hysteresis on).")
    parser.add_argument("--defense_gate_theta_off", type=float, default=0.05, help="Threshold to stop masking (auto mode, hysteresis off).")
    parser.add_argument("--defense_gate_ema_alpha", type=float, default=0.3, help="EMA smoothing coefficient for mass (auto mode).")
    parser.add_argument("--defense_gate_check_every_k", type=int, default=3, help="Only check trigger every k frames when SAFE (auto mode).")
    # Purifier parameters
    parser.add_argument("--defense_purifier_strategy", type=str, default="mask_mean",
                        choices=["mask_mean", "mask_gray", "blend_mean", "blend_gray", "blur"],
                        help="Purifier strategy: mask_mean|mask_gray|blend_mean|blend_gray|blur.")
    parser.add_argument("--defense_purifier_pad", type=int, default=0, help="Pad patch box before purification (pixels).")
    parser.add_argument("--defense_purifier_alpha", type=float, default=1.0, help="Blend strength for blend_* strategies (0-1).")
    parser.add_argument("--defense_strength_min", type=float, default=0.35,
                        help="Minimum purifier strength for auto mode (0-1). Maps to [strength_min, strength_max] range.")
    parser.add_argument("--defense_strength_max", type=float, default=0.85,
                        help="Maximum purifier strength for auto mode (0-1). Set both to 1.0 for full masking in auto mode.")
    parser.add_argument("--defense_gray_value", type=int, default=127, help="Gray value when using mask_gray/blend_gray purifier.")
    parser.add_argument("--defense_recompute_action", type=str2bool, default=True, help="Recompute action using purified image when defense triggers.")
    parser.add_argument("--defense_debug", type=str2bool, default=False, help="Print defense debug logs to terminal and log file.")
    parser.add_argument("--defense_debug_every_step", type=str2bool, default=False, help="When enabled, print defense scores for every step (debug only).")
    parser.add_argument("--defense_debug_geometry", type=str2bool, default=False, help="When enabled, print projected gripper / arm geometry coordinates to the terminal (debug only).")
    parser.add_argument("--defense_debug_selector", type=str2bool, default=False, help="When enabled, print selector debug lines to the terminal (debug only).")
    parser.add_argument("--defense_viz", type=str2bool, default=False, help="If enabled, save side-by-side frames (policy input | heatmap overlay).")
    parser.add_argument("--defense_viz_alpha", type=float, default=0.45, help="Overlay alpha for heatmap visualization (0-1).")
    parser.add_argument("--defense_geometry_rotate_180", type=str2bool, default=True, help="Whether the policy image applies a 180-degree rotation to the raw camera frame. Projected geometry derives its own row / col alignment from this setting.")
    parser.add_argument("--defense_viz_joint_radius_px", type=int, default=4, help="Joint / gripper point radius in the defense visualization.")
    parser.add_argument("--defense_viz_link_line_thickness", type=int, default=2, help="Center-line thickness for projected arm links in the defense visualization.")
    parser.add_argument("--defense_viz_link_quad_thickness", type=int, default=2, help="Outline thickness for projected per-link quads in the defense visualization.")
    # Verifier parameters (counterfactual verification)
    parser.add_argument("--defense_verifier_enabled", type=str2bool, default=True, help="Enable counterfactual verifier (auto mode only).")
    parser.add_argument("--defense_verifier_min_mass_drop", type=float, default=0.15, help="Minimum relative ROI mass drop for verification (Tier 1).")
    parser.add_argument("--defense_verifier_min_entropy_gain", type=float, default=0.02, help="Minimum absolute entropy gain for verification (Tier 1).")
    parser.add_argument("--defense_verifier_entropy_hard", type=str2bool, default=False, help="If True, entropy_gain is a hard requirement for Tier 1.")
    parser.add_argument("--defense_verifier_max_main_drop", type=float, default=0.10, help="Maximum relative mainland mass drop (Tier 2: task preservation).")
    parser.add_argument("--defense_verifier_min_action_diff_l2", type=float, default=0.01, help="Minimum L2 action difference for verification (Tier 3).")
    parser.add_argument("--defense_verifier_min_action_diff_rel", type=float, default=0.05, help="Minimum relative action difference for verification (Tier 3).")
    parser.add_argument("--defense_verifier_min_gripper_diff", type=float, default=0.1, help="Minimum gripper action difference for verification (Tier 3).")
    parser.add_argument("--defense_verifier_use_action", type=str2bool, default=True, help="Enable action-level verification (Tier 3).")
    parser.add_argument("--defense_verifier_every_k", type=int, default=3, help="Verify every k frames during TRACK (auto mode).")
    parser.add_argument("--defense_require_verify", type=str2bool, default=False, help="Require verification to pass before purifying (strict mode).")
    # Controller (quality aligned with heatmap)
    parser.add_argument("--defense_min_trigger_mass_heatmap", type=float, default=0.02, help="Min ROI mass on heatmap to allow trigger (auto mode).")
    parser.add_argument("--defense_quality_mass_source", type=str, default="heatmap", choices=["heatmap", "grid"], help="Mass source for quality gate (auto mode).")

    # Safety Region Layer
    parser.add_argument("--defense_safety_region_enabled", type=str2bool, default=False, help="Enable safety region fusion layer.")
    parser.add_argument("--defense_safety_w_arm_core", type=float, default=1.0, help="Penalty weight for arm core region.")
    parser.add_argument("--defense_safety_w_arm_guard", type=float, default=0.6, help="Penalty weight for arm guard region.")
    parser.add_argument("--defense_safety_w_gripper_core", type=float, default=1.0, help="Penalty weight for gripper core region.")
    parser.add_argument("--defense_safety_w_gripper_guard", type=float, default=0.7, help="Penalty weight for gripper guard region.")
    parser.add_argument("--defense_safety_smooth_kernel", type=int, default=0, help="Optional smoothing kernel size for penalty map (0 disables).")

    # Pixel Mask Optimizer Layer
    parser.add_argument("--defense_pixel_mask_refine_enabled", type=str2bool, default=False, help="Enable pixel-level mask refinement.")
    parser.add_argument("--defense_pixel_lambda_safety", type=float, default=0.75, help="Safety penalty coefficient in pixel score.")
    parser.add_argument("--defense_pixel_score_quantile", type=float, default=0.65, help="Score quantile threshold for mask binarization.")
    parser.add_argument("--defense_pixel_min_area_ratio", type=float, default=0.08, help="Minimum selected pixel ratio in ROI.")
    parser.add_argument("--defense_pixel_min_cover_ratio", type=float, default=0.40, help="Minimum heatmap coverage ratio in ROI.")
    parser.add_argument("--defense_pixel_keep_largest_component", type=str2bool, default=True, help="Keep largest connected component in refined mask.")
    parser.add_argument("--defense_pixel_hard_forbid_core", type=str2bool, default=True, help="Disallow selecting core safety pixels during mask refinement.")

    # Temporal Conflict Layer
    parser.add_argument("--defense_temporal_conflict_enabled", type=str2bool, default=False, help="Enable temporal conflict resolver.")
    parser.add_argument("--defense_conflict_core_hard_on", type=float, default=0.10, help="Core-overlap threshold to enter HARD mode.")
    parser.add_argument("--defense_conflict_core_hard_off", type=float, default=0.04, help="Core-overlap threshold to exit HARD mode.")
    parser.add_argument("--defense_conflict_guard_soft_on", type=float, default=0.25, help="Guard-overlap threshold to enter SOFT mode.")
    parser.add_argument("--defense_conflict_guard_soft_off", type=float, default=0.12, help="Guard-overlap threshold to exit SOFT mode.")
    parser.add_argument("--defense_conflict_hard_on_frames", type=int, default=2, help="Consecutive frames required to trigger HARD conflict.")
    parser.add_argument("--defense_conflict_soft_on_frames", type=int, default=2, help="Consecutive frames required to trigger SOFT conflict.")
    parser.add_argument("--defense_conflict_soft_alpha", type=float, default=0.7, help="Strength scale applied under SOFT conflict mode.")
    
    # Multimodal Gripper Prior & Patch Selector parameters
    parser.add_argument("--defense_gripper_prior_enabled", type=str2bool, default=True, help="Enable multimodal geometric gripper prior.")
    parser.add_argument("--defense_gripper_site_names", type=str, default="grip_site,ft_frame", help="Comma-separated gripper site names used first for true camera projection.")
    parser.add_argument("--defense_gripper_body_names", type=str, default="right_hand,right_gripper,eef,leftfinger,rightfinger,finger_joint1_tip,finger_joint2_tip", help="Comma-separated gripper body names used as fallback when sites are unavailable.")
    parser.add_argument("--defense_gripper_segment_pairs", type=str, default="right_hand:ft_frame,ft_frame:grip_site,grip_site:finger_joint1_tip,grip_site:finger_joint2_tip", help="Comma-separated gripper segment pairs formatted as start:end.")
    parser.add_argument("--defense_gripper_name_prefixes", type=str, default="robot0_,Panda0_,Panda_", help="Comma-separated name prefixes tried when resolving gripper sites / bodies.")
    parser.add_argument("--defense_gripper_camera_name", type=str, default="agentview", help="Camera name used for true gripper projection.")
    parser.add_argument("--defense_gripper_point_radius_px", type=int, default=5, help="Point radius in pixels around each projected gripper keypoint.")
    parser.add_argument("--defense_gripper_core_thickness_px", type=int, default=10, help="Default thickness in pixels for projected gripper segments.")
    parser.add_argument("--defense_gripper_guard_scale", type=float, default=2.0, help="Guard thickness scale relative to gripper core thickness.")
    parser.add_argument("--defense_gripper_segment_core_thicknesses", type=str, default="", help="Optional comma-separated per-segment gripper core thickness overrides.")
    parser.add_argument("--defense_gripper_segment_guard_scales", type=str, default="", help="Optional comma-separated per-segment gripper guard-scale overrides.")
    parser.add_argument("--defense_gripper_min_valid_points", type=int, default=1, help="Minimum valid projected gripper points required to enable the per-frame gripper prior.")
    parser.add_argument("--defense_tau_g", type=float, default=0.3, help="Overlap threshold with GripperPrior for PatchSelector.")
    parser.add_argument("--defense_tau_arm", type=float, default=0.3, help="Overlap threshold with projected arm masks for PatchSelector.")
    parser.add_argument("--defense_tau_patch_strength", type=float, default=0.05, help="Minimum anomaly mass for PatchSelector.")
    parser.add_argument("--defense_near_task_tau", type=float, default=0.08, help="Minimum anomaly score for allowing near-task patch when enabled.")
    parser.add_argument("--defense_allow_near_task_patch", type=str2bool, default=False, help="Whether to allow near-task patch candidates to enter LOCKED.")
    parser.add_argument("--defense_selector_debug_enabled", type=str2bool, default=False, help="Enable selector debug metadata in defense results.")
    parser.add_argument("--defense_selector_debug_topk", type=int, default=3, help="Max number of selector candidates to log.")
    parser.add_argument("--defense_expected_patch_area_ratio", type=float, default=0.04, help="Expected patch area ratio for diagnostic scoring.")
    parser.add_argument("--defense_patch_area_sigma", type=float, default=0.03, help="Area sigma for diagnostic size prior.")
    parser.add_argument("--defense_patch_aspect_sigma", type=float, default=0.4, help="Aspect sigma for diagnostic square prior.")
    parser.add_argument("--defense_corner_prior_enabled", type=str2bool, default=False, help="Enable diagnostic corner prior for selector logging.")
    parser.add_argument(
        "--defense_corner_prior_type",
        type=str,
        default="top_right",
        choices=["top_right", "top_left", "bottom_right", "bottom_left", "none"],
        help="Corner prior type (top_right|top_left|bottom_right|bottom_left|none).",
    )
    parser.add_argument("--defense_corner_prior_sigma", type=float, default=0.35, help="Corner prior sigma for diagnostic scoring.")
    parser.add_argument("--defense_tau_protect", type=float, default=0.1, help="Max allowed overlap ratio of mask with GripperPrior.")
    parser.add_argument("--defense_tau_cover", type=float, default=0.5, help="Min required coverage ratio of the initial mask.")
    parser.add_argument("--defense_use_residual_candidate_grid", type=str2bool, default=False, help="Use geometry-residualized stable grid for candidate proposal and final evidence scoring.")
    parser.add_argument("--defense_geometry_residual_gamma", type=float, default=1.0, help="Exponent for geometry residual candidate-grid suppression.")
    parser.add_argument("--defense_geometry_guard_weight_arm", type=float, default=0.6, help="Soft occupancy weight for arm guard cells in residual candidate-grid suppression.")
    parser.add_argument("--defense_geometry_guard_weight_gripper", type=float, default=0.7, help="Soft occupancy weight for gripper guard cells in residual candidate-grid suppression.")
    parser.add_argument("--defense_arm_skeleton_enabled", type=str2bool, default=False, help="Enable arm skeleton prior built from simulator link poses.")
    parser.add_argument("--defense_arm_skeleton_source", type=str, default="body", choices=["body", "site"], help="Use body or site poses as keypoints for arm skeleton construction.")
    parser.add_argument("--defense_arm_body_names", type=str, default="base,link1,link2,link3,link4,link5,link6,link7,right_hand", help="Comma-separated body names for the arm skeleton keypoints.")
    parser.add_argument("--defense_arm_site_names", type=str, default="", help="Comma-separated site names for the arm skeleton keypoints when source=site.")
    parser.add_argument("--defense_arm_name_prefixes", type=str, default="robot0_,Panda0_,Panda_", help="Comma-separated name prefixes tried when resolving arm sites / bodies.")
    parser.add_argument("--defense_arm_camera_name", type=str, default="agentview", help="Camera name used for true arm skeleton projection.")
    parser.add_argument("--defense_arm_joint_radius_px", type=int, default=4, help="Joint marker radius in pixels added around every projected arm joint.")
    parser.add_argument("--defense_arm_core_thickness_px", type=int, default=14, help="Pixel thickness of the hard ArmCore corridor.")
    parser.add_argument("--defense_arm_guard_scale", type=float, default=1.8, help="Guard thickness scale relative to ArmCore.")
    parser.add_argument("--defense_arm_link_core_thicknesses", type=str, default="", help="Optional comma-separated per-link ArmCore thickness overrides.")
    parser.add_argument("--defense_arm_link_guard_scales", type=str, default="", help="Optional comma-separated per-link guard-scale overrides.")
    parser.add_argument("--defense_arm_min_valid_points", type=int, default=3, help="Minimum valid projected keypoints required for the arm skeleton prior.")

    # Recovery metrics / mechanism evidence logging. These flags only add
    # instrumentation; they do not alter the executed defense action.
    parser.add_argument("--metrics_enabled", type=str2bool, default=False, help="Enable recovery metrics JSONL instrumentation.")
    parser.add_argument("--metrics_action_recovery", type=str2bool, default=False, help="Run an extra clean-image forward and compute action recovery metrics.")
    parser.add_argument("--metrics_attention_recovery", type=str2bool, default=False, help="Record attention hijacking/suppression metrics when heatmaps are available.")
    parser.add_argument("--metrics_mask_recovery", type=str2bool, default=True, help="Record mask-vs-patch localization metrics.")
    parser.add_argument("--metrics_every_step", type=str2bool, default=True, help="Write frame metrics on every rollout step.")
    parser.add_argument("--metrics_sample_every_n", type=int, default=1, help="When metrics_every_step=False, write one frame row every N steps.")
    parser.add_argument("--metrics_save_jsonl", type=str2bool, default=True, help="Write recovery metrics to JSONL under local_log_dir.")
    parser.add_argument("--metrics_strict", type=str2bool, default=False, help="Raise JSONL logging errors instead of dropping bad metric rows.")
    parser.add_argument("--metrics_patch_w", type=int, default=50, help="Fallback patch width for ground-truth patch mask.")
    parser.add_argument("--metrics_patch_h", type=int, default=50, help="Fallback patch height for ground-truth patch mask.")
    parser.add_argument("--metrics_top_quantile", type=float, default=0.90, help="Attention quantile for Top-K attention IoU with patch.")

    # PRAC checker parameters
    parser.add_argument("--defense_prac_enabled", type=str2bool, default=False, help="Enable PRAC (Patch-wise Randomized Attention Consistency) checker (auto mode only).")
    parser.add_argument("--defense_prac_n_views", type=int, default=6, help="Number of random views for consensus attention (PRAC).")
    parser.add_argument("--defense_prac_patch_size", type=int, default=16, help="Patch size for random patch-wise perturbation (PRAC, pixels).")
    parser.add_argument("--defense_prac_mask_ratio", type=float, default=0.25, help="Fraction of patches to perturb (PRAC, 0-1).")
    parser.add_argument("--defense_prac_transform_mode", type=str, default="blur", choices=["blur", "gray", "noise"], help="Perturbation mode for patch masking (PRAC).")
    parser.add_argument("--defense_prac_tau_odr", type=float, default=1.8, help="ODR (Outlier Dependency Ratio) threshold for PRAC verdict (higher = more consistent).")
    parser.add_argument("--defense_prac_tau_mer", type=float, default=0.20, help="MER (Mainland Erosion Risk) threshold for PRAC verdict (lower = less overlap risk).")
    parser.add_argument("--defense_prac_max_attempts", type=int, default=2, help="Maximum re-localization attempts within one ACQUIRE frame (PRAC).")
    parser.add_argument("--defense_prac_seed", type=int, default=None, help="Random seed for PRAC perturbation (None = random).")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # CUDA_VISIBLE_DEVICES is already set at the top of the file (before torch import)
    # This ensures each process can independently use different GPUs when running multiple terminals
    # Double-check that it matches the parsed argument
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.cudaid):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cudaid)
        print(f"[*] Updated CUDA_VISIBLE_DEVICES={args.cudaid}")
    else:
        print(f"[*] CUDA_VISIBLE_DEVICES={args.cudaid} (already set correctly)")
    eval_libero(args)
