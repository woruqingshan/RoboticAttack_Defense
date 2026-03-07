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
    PatchBox,
    # Unified interface and auto-mode components
    UnifiedDefenseInterface,
    PatchAttentionLocalizer,
    TemporalGate,
    OnlinePatchDefenseController,
    CounterfactualVerifier,
    NoOpVerifier,
    format_defense_log_line,
    defense_result_to_log_dict,
    # Multimodal geometry prior components
    GripperPrior,
    GripperPriorConfig,
    PatchSelector,
    PatchSelectorConfig,
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

def _maybe_pack_replay_frame(cfg, image_rgb, heatmap: Optional["np.ndarray"], gripper_box=None):
    """Optionally concatenate the policy input and heatmap overlay side-by-side."""
    # If a gripper box is provided, draw it on the image_rgb (and overlay if created)
    img_to_pack = image_rgb.copy()
    if gripper_box is not None:
        try:
            import cv2
            # Draw a green bounding box for the gripper prior
            cv2.rectangle(
                img_to_pack, 
                (int(gripper_box.x0), int(gripper_box.y0)), 
                (int(gripper_box.x1), int(gripper_box.y1)), 
                (0, 255, 0), 2
            )
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
            patch_selector = None
            if getattr(cfg, "defense_gripper_prior_enabled", True):
                gp_cfg = GripperPriorConfig(
                    radius_px=getattr(cfg, "defense_gripper_radius_px", 40)
                )
                gripper_prior = GripperPrior(gp_cfg)
                
                ps_cfg = PatchSelectorConfig(
                    tau_g=getattr(cfg, "defense_tau_g", 0.3),
                    tau_patch_strength=getattr(cfg, "defense_tau_patch_strength", 0.05)
                )
                patch_selector = PatchSelector(ps_cfg)

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
                patch_selector=patch_selector,
                tau_protect=getattr(cfg, "defense_tau_protect", 0.1),
                tau_cover=getattr(cfg, "defense_tau_cover", 0.5),
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
                    gripper_box_for_viz = None
                    if defense_interface is not None:
                        try:
                            # Extract EEF pos for Multimodal Geometric Prior
                            # We concatenate pos, quat (as axis-angle), and gripper qpos
                            eef_pos = np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
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
                                )
                            else:
                                defense_result = defense_interface.step(eef_pos=eef_pos)
                        except Exception as defense_error:
                            # Exit immediately on defense errors to avoid empty episode analysis
                            error_msg = f"FATAL DEFENSE ERROR: {defense_error}"
                            print(error_msg)
                            print("Exiting immediately to prevent empty episode analysis...")
                            log_file.write(error_msg + "\n")
                            log_file.close()
                            sys.exit(1)
                        
                        # Get heatmap and gripper box for visualization
                        if getattr(cfg, "defense_viz", False):
                            heatmap_for_viz = defense_result.heatmap
                            gripper_box_for_viz = getattr(defense_result, "gripper_box", None)
                        
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
                        
                        # Check if purification is needed (unified field)
                        if defense_result.should_purify and defense_result.roi_box is not None:
                            
                            # Purify image (unified interface)
                            # Use strength from defense_result if available (auto mode provides dynamic strength)
                            img_for_policy = defense_purifier.purify(
                                img_for_policy,
                                defense_result.roi_box,
                                strength=getattr(defense_result, "strength", None),  # Use dynamic strength if available
                            )
                            
                            # Recompute action on purified image
                            observation["full_image"] = img_for_policy
                            if getattr(cfg, "defense_recompute_action", True):
                                defense_interface.clear()
                                action = get_action(cfg, model, observation, task_description, processor=processor)
                            
                            _defense_debug_print(
                                cfg,
                                f"[DEFENSE][MASK] {format_defense_log_line(step=int(t), result=defense_result)}",
                                log_file=log_file,
                            )

                    # Save replay frame:
                    # - default: policy input image
                    # - optional: side-by-side with real-time heatmap overlay
                    replay_images.append(_maybe_pack_replay_frame(cfg, img_for_policy, heatmap_for_viz, gripper_box_for_viz))

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
                f"[DEFENSE][EPISODE_END] episode={task_episodes+1} success={done} "
                f"masked_frames={episode_masked_frames} acquire_count={episode_acquire_count} reacquire_count={episode_reacquire_count}",
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
        _defense_debug_print(
            cfg,
            f"[DEFENSE][TASK_END] task_id={task_id} episodes={task_episodes} "
            f"success_rate={float(task_successes) / float(task_episodes):.3f} "
            f"masked_frames={task_masked_frames} acquire_count={task_acquire_count} reacquire_count={task_reacquire_count}",
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
    parser.add_argument("--defense_viz", type=str2bool, default=False, help="If enabled, save side-by-side frames (policy input | heatmap overlay).")
    parser.add_argument("--defense_viz_alpha", type=float, default=0.45, help="Overlay alpha for heatmap visualization (0-1).")
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
    
    # Multimodal Gripper Prior & Patch Selector parameters
    parser.add_argument("--defense_gripper_prior_enabled", type=str2bool, default=True, help="Enable multimodal geometric gripper prior.")
    parser.add_argument("--defense_gripper_radius_px", type=int, default=40, help="Radius in pixels for the gripper protection zone.")
    parser.add_argument("--defense_tau_g", type=float, default=0.3, help="Overlap threshold with GripperPrior for PatchSelector.")
    parser.add_argument("--defense_tau_patch_strength", type=float, default=0.05, help="Minimum anomaly mass for PatchSelector.")
    parser.add_argument("--defense_tau_protect", type=float, default=0.1, help="Max allowed overlap ratio of mask with GripperPrior.")
    parser.add_argument("--defense_tau_cover", type=float, default=0.5, help="Min required coverage ratio of the initial mask.")

    # PRAC checker parameters
    parser.add_argument("--defense_prac_enabled", type=str2bool, default=True, help="Enable PRAC (Patch-wise Randomized Attention Consistency) checker (auto mode only).")
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
