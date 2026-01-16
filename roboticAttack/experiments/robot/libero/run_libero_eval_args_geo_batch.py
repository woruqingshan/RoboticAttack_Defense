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
from typing import Optional, Union

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
    PatchAttentionAnomalyDetector,
    PatchBox,
    # Unified interface and auto-mode components
    UnifiedDefenseInterface,
    PatchAttentionLocalizer,
    TemporalGate,
    OnlinePatchDefenseController,
    CounterfactualVerifier,
)

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

def _maybe_pack_replay_frame(cfg, image_rgb, heatmap: Optional["np.ndarray"]):
    """Optionally concatenate the policy input and heatmap overlay side-by-side."""
    if not getattr(cfg, "defense_viz", False):
        return image_rgb
    if heatmap is None:
        return image_rgb
    overlay = _make_overlay_rgb(image_rgb, heatmap, alpha=getattr(cfg, "defense_viz_alpha", 0.45))
    try:
        return np.concatenate([image_rgb, overlay], axis=1)
    except Exception:
        return image_rgb


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
    if getattr(cfg, "defense_enabled", False):
        # Create hook (required for both modes)
        defense_hook = OnlineAttentionHook(
            model=model,
            attn_module_name=getattr(cfg, "defense_attn_module", None),
            aggregate_mode=getattr(cfg, "defense_aggregate_mode", "mean"),
            image_size=get_image_resize_size(cfg),
        )
        
        # Create purifier (required for both modes)
        defense_purifier = ImagePurifier(
            strategy=getattr(cfg, "defense_purifier_strategy", "mask_mean"),
            pad=getattr(cfg, "defense_purifier_pad", 0),
            gray_value=getattr(cfg, "defense_gray_value", 127),
            alpha=getattr(cfg, "defense_purifier_alpha", 0.8),
        )
        
        # Create unified interface based on mode
        defense_mode = getattr(cfg, "defense_mode", "known")  # Default: backward compatible
        
        if defense_mode == "known":
            # Known location mode (backward compatible)
            defense_detector = PatchAttentionAnomalyDetector(
                patch_mass_threshold=getattr(cfg, "defense_patch_mass_threshold", 0.25),
                entropy_threshold=getattr(cfg, "defense_entropy_threshold", None),
                use_entropy_gate=getattr(cfg, "defense_use_entropy_gate", False),
            )
            defense_interface = UnifiedDefenseInterface(
                hook=defense_hook,
                mode="known",
                detector=defense_detector,
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
                hold_frames=getattr(cfg, "defense_gate_hold_frames", 5),
                cooldown_frames=getattr(cfg, "defense_gate_cooldown_frames", 3),
            )
            
            # Create verifier if enabled
            verifier = None
            if getattr(cfg, "defense_verifier_enabled", True):  # Default: enabled
                verifier = CounterfactualVerifier(
                    min_mass_drop_rel=getattr(cfg, "defense_verifier_min_mass_drop", 0.15),
                    min_entropy_gain_abs=getattr(cfg, "defense_verifier_min_entropy_gain", 0.02),
                    max_main_mass_drop_rel=getattr(cfg, "defense_verifier_max_main_drop", 0.10),
                    min_action_diff_l2=getattr(cfg, "defense_verifier_min_action_diff_l2", 0.01),
                    min_action_diff_rel=getattr(cfg, "defense_verifier_min_action_diff_rel", 0.05),
                    min_gripper_diff=getattr(cfg, "defense_verifier_min_gripper_diff", 0.1),
                    use_action_verification=getattr(cfg, "defense_verifier_use_action", True),
                )
            
            controller = OnlinePatchDefenseController(
                hook=defense_hook,
                localizer=localizer,
                gate=gate,
                motion_threshold_cells=getattr(cfg, "defense_controller_motion_threshold", 1.2),
                motion_penalty_weight=getattr(cfg, "defense_controller_motion_penalty_weight", 0.3),
                tracker_iou_keep=getattr(cfg, "defense_controller_tracker_iou_keep", 0.30),
                tracker_ema=getattr(cfg, "defense_controller_tracker_ema", 0.50),
                top_k_candidates=getattr(cfg, "defense_top_k_candidates", 3),
                verifier=verifier,
                verify_every_k=getattr(cfg, "defense_verifier_every_k", 1),  # Verify on every trigger edge
                verify_block_frames=getattr(cfg, "defense_verifier_block_frames", 6),
                require_verify=getattr(cfg, "defense_require_verify", False),  # Default: False (backward compatible)
                # Backward compatibility: deprecated parameters are ignored by new controller
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

        task_defense_triggers = 0
        task_defense_patch_mass_sum = 0.0
        task_defense_steps_checked = 0

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
            episode_defense_triggers = 0
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size) # Preprocess image for model
                    if cfg.use_patch:
                        img = randomPatchTransform.simulation_random_patch(
                            img, patch, geometry=True, colorjitter=False,
                            angle=cfg.angle, shx=cfg.shx, shy=cfg.shy, position=(cfg.x, cfg.y)
                        )
                    img_for_policy = img

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img_for_policy,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action (this forward pass is also used to populate attention hooks)
                    if defense_interface is not None:
                        try:
                            defense_interface.clear()
                        except Exception as clear_error:
                            error_msg = f"FATAL DEFENSE CLEAR ERROR: {clear_error}"
                            print(error_msg)
                            print("Exiting immediately to prevent empty episode analysis...")
                            log_file.write(error_msg + "\n")
                            log_file.close()
                            sys.exit(1)
                    action = get_action(cfg, model, observation, task_description, processor=processor)

                    # Unified defense step (works for both known and auto modes)
                    heatmap_for_viz = None
                    if defense_interface is not None:
                        try:
                            # For auto mode, pass additional parameters
                            if defense_mode == "auto":
                                defense_result = defense_interface.step(
                                    image=img,
                                    purify_fn=lambda img, box: defense_purifier.purify(img, box),
                                    forward_fn=lambda img: get_action(cfg, model, {
                                        "full_image": img,
                                        "state": np.concatenate(
                                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                                        ),
                                    }, task_description, processor=processor),
                                    heatmap_fn=lambda: defense_hook.get_heatmap(),
                                )
                            else:
                                defense_result = defense_interface.step()
                        except Exception as defense_error:
                            # Exit immediately on defense errors to avoid empty episode analysis
                            error_msg = f"FATAL DEFENSE ERROR: {defense_error}"
                            print(error_msg)
                            print("Exiting immediately to prevent empty episode analysis...")
                            log_file.write(error_msg + "\n")
                            log_file.close()
                            sys.exit(1)
                        
                        # Get heatmap for visualization
                        if getattr(cfg, "defense_viz", False):
                            heatmap_for_viz = defense_result.heatmap
                        
                        # Statistics (backward compatible)
                        task_defense_steps_checked += 1
                        task_defense_patch_mass_sum += float(defense_result.patch_mass)
                        
                        # Debug output (backward compatible + verifier details)
                        if _defense_debug_every_step(cfg):
                            verifier_info = ""
                            if defense_result.verify_stats is not None:
                                vs = defense_result.verify_stats
                                verifier_info = (
                                    f" | VERIFY: verified={defense_result.verified} "
                                    f"roi_mass_drop={vs.get('roi_mass_rel_drop', 0):.3f} "
                                    f"entropy_gain={vs.get('entropy_gain', 0):.3f} "
                                    f"main_mass_drop={vs.get('main_mass_drop', 0):.3f} "
                                    f"action_diff_l2={vs.get('action_diff_l2', 0):.4f} "
                                    f"action_diff_rel={vs.get('action_diff_rel', 0):.3f} "
                                    f"gripper_diff={vs.get('gripper_diff', 0):.3f}"
                                )
                            _defense_debug_print(
                                cfg,
                                f"[DEFENSE][CHECK] step={t} patch_mass={defense_result.patch_mass:.4f} "
                                f"entropy={defense_result.entropy if defense_result.entropy is not None else 'NA'} "
                                f"state={defense_result.state} reason={defense_result.reason}{verifier_info}",
                                log_file=log_file,
                            )
                        
                        # Check if purification is needed (unified field)
                        if defense_result.should_purify and defense_result.roi_box is not None:
                            task_defense_triggers += 1
                            episode_defense_triggers += 1
                            
                            # Purify image (unified interface)
                            # Use strength from defense_result if available (auto mode provides dynamic strength)
                            img_for_policy = defense_purifier.purify(
                                img_for_policy,
                                defense_result.roi_box,
                                strength=getattr(defense_result, "strength", None)  # Use dynamic strength if available
                            )
                            
                            # Recompute action on purified image
                            observation["full_image"] = img_for_policy
                            if getattr(cfg, "defense_recompute_action", True):
                                defense_interface.clear()
                                action = get_action(cfg, model, observation, task_description, processor=processor)
                            
                            # Enhanced trigger log with verifier details
                            verifier_info = ""
                            if defense_result.verify_stats is not None:
                                vs = defense_result.verify_stats
                                verifier_info = (
                                    f" | VERIFY: verified={defense_result.verified} "
                                    f"roi_mass_drop={vs.get('roi_mass_rel_drop', 0):.3f} "
                                    f"entropy_gain={vs.get('entropy_gain', 0):.3f} "
                                    f"main_mass_drop={vs.get('main_mass_drop', 0):.3f} "
                                    f"action_diff_l2={vs.get('action_diff_l2', 0):.4f} "
                                    f"action_diff_rel={vs.get('action_diff_rel', 0):.3f} "
                                    f"gripper_diff={vs.get('gripper_diff', 0):.3f}"
                                )
                            _defense_debug_print(
                                cfg,
                                f"[DEFENSE][TRIGGER] step={t} patch_mass={defense_result.patch_mass:.4f} "
                                f"state={defense_result.state} strategy={defense_purifier.strategy} "
                                f"recompute={getattr(cfg, 'defense_recompute_action', True)}{verifier_info}",
                                log_file=log_file,
                            )

                    # Save replay frame:
                    # - default: policy input image
                    # - optional: side-by-side with real-time heatmap overlay
                    replay_images.append(_maybe_pack_replay_frame(cfg, img_for_policy, heatmap_for_viz))

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
            print(f"Saving replay video...")
            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
                exp_name=cfg.exp_name,
                rollout_root_dir=getattr(cfg, "rollout_root_dir", "./rollouts"),
            )
            _defense_debug_print(
                cfg,
                f"[DEFENSE][EPISODE_END] episode={task_episodes+1} success={done} triggers={episode_defense_triggers}",
                log_file=log_file,
            )

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
        avg_patch_mass = (task_defense_patch_mass_sum / task_defense_steps_checked) if task_defense_steps_checked > 0 else 0.0
        _defense_debug_print(
            cfg,
            f"[DEFENSE][TASK_END] task_id={task_id} episodes={task_episodes} success_rate={float(task_successes) / float(task_episodes):.3f} "
            f"steps_checked={task_defense_steps_checked} triggers={task_defense_triggers} avg_patch_mass={avg_patch_mass:.4f}",
            log_file=log_file,
        )
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
    # 追加模式打开文件并添加新内容
    with open(os.path.join(cfg.local_log_dir,cfg.task_suite_name+".txt"), "a") as file:
        file.write(f"success_rate/total:{float(total_successes) / float(total_episodes)}, num_episodes/total:{total_episodes} position_info:{cfg.angle}_{cfg.shx}_{cfg.shy}_{cfg.x}_{cfg.y} \n")  # 在新行添加内容

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
    parser.add_argument("--defense_gate_hold_frames", type=int, default=5, help="Minimum frames to hold masking state (auto mode).")
    parser.add_argument("--defense_gate_cooldown_frames", type=int, default=3, help="Cooldown frames after masking (auto mode).")
    parser.add_argument("--defense_controller_motion_threshold", type=float, default=1.2, help="Max centroid movement (grid cells) to be considered static (auto mode).")
    parser.add_argument("--defense_controller_motion_penalty_weight", type=float, default=0.3, help="Motion penalty weight for candidate scoring (auto mode).")
    parser.add_argument("--defense_controller_tracker_iou_keep", type=float, default=0.30, help="IoU threshold for ROI tracking stickiness (auto mode).")
    parser.add_argument("--defense_controller_tracker_ema", type=float, default=0.50, help="EMA alpha for ROI position smoothing (auto mode).")
    parser.add_argument("--defense_top_k_candidates", type=int, default=3, help="Number of top candidates to evaluate (auto mode).")
    # Deprecated parameters (kept for backward compatibility)
    parser.add_argument("--defense_controller_min_stable_frames", type=int, default=3, help="[DEPRECATED] Require this many consecutive static frames before trigger (auto mode).")
    parser.add_argument("--defense_controller_track_iou", type=float, default=0.2, help="[DEPRECATED] IoU threshold for ROI tracking stickiness (auto mode).")
    parser.add_argument("--defense_controller_max_jump", type=float, default=3.0, help="[DEPRECATED] Max jump distance (grid cells) to accept new ROI (auto mode).")
    # Purifier parameters
    parser.add_argument("--defense_purifier_strategy", type=str, default="mask_mean",
                        choices=["mask_mean", "mask_gray", "blend_mean", "blend_gray", "blur"],
                        help="Purifier strategy: mask_mean|mask_gray|blend_mean|blend_gray|blur.")
    parser.add_argument("--defense_purifier_pad", type=int, default=0, help="Pad patch box before purification (pixels).")
    parser.add_argument("--defense_purifier_alpha", type=float, default=0.8, help="Blend strength for blend_* strategies (0-1).")
    parser.add_argument("--defense_gray_value", type=int, default=127, help="Gray value when using mask_gray/blend_gray purifier.")
    parser.add_argument("--defense_recompute_action", type=str2bool, default=True, help="Recompute action using purified image when defense triggers.")
    parser.add_argument("--defense_debug", type=str2bool, default=False, help="Print defense debug logs to terminal and log file.")
    parser.add_argument("--defense_debug_every_step", type=str2bool, default=False, help="When enabled, print defense scores for every step (debug only).")
    parser.add_argument("--defense_viz", type=str2bool, default=False, help="If enabled, save side-by-side frames (policy input | heatmap overlay).")
    parser.add_argument("--defense_viz_alpha", type=float, default=0.45, help="Overlay alpha for heatmap visualization (0-1).")
    # Verifier parameters (counterfactual verification)
    parser.add_argument("--defense_verifier_enabled", type=str2bool, default=True, help="Enable counterfactual verifier (auto mode only).")
    parser.add_argument("--defense_verifier_min_mass_drop", type=float, default=0.15, help="Minimum relative ROI mass drop for verification (Tier 1).")
    parser.add_argument("--defense_verifier_min_entropy_gain", type=float, default=0.02, help="Minimum absolute entropy gain for verification (Tier 1).")
    parser.add_argument("--defense_verifier_max_main_drop", type=float, default=0.10, help="Maximum relative mainland mass drop (Tier 2: task preservation).")
    parser.add_argument("--defense_verifier_min_action_diff_l2", type=float, default=0.01, help="Minimum L2 action difference for verification (Tier 3).")
    parser.add_argument("--defense_verifier_min_action_diff_rel", type=float, default=0.05, help="Minimum relative action difference for verification (Tier 3).")
    parser.add_argument("--defense_verifier_min_gripper_diff", type=float, default=0.1, help="Minimum gripper action difference for verification (Tier 3).")
    parser.add_argument("--defense_verifier_use_action", type=str2bool, default=True, help="Enable action-level verification (Tier 3).")
    parser.add_argument("--defense_verifier_every_k", type=int, default=1, help="Verify every k trigger edges (1 = verify on every trigger).")
    parser.add_argument("--defense_verifier_block_frames", type=int, default=6, help="Block frames after verification failure.")
    parser.add_argument("--defense_require_verify", type=str2bool, default=False, help="Require verification to pass before purifying (strict mode).")

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
