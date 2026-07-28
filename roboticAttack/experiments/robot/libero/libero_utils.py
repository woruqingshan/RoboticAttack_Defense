"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import time

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from evaluation_tool.defense.geometry_alignment import (
    DEFAULT_POLICY_IMAGE_ROTATE_180,
    apply_policy_image_alignment,
    normalize_resize_size,
)

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


def _libero_env_debug_enabled() -> bool:
    """Return True when LIBERO environment initialization debug logs are enabled."""
    return os.environ.get("LIBERO_ENV_DEBUG", "0").lower() in ("1", "true", "yes", "y")


def _libero_env_debug_log(msg: str) -> None:
    """Print flushed LIBERO environment initialization logs."""
    if not _libero_env_debug_enabled():
        return
    line = f"[LIBERO_ENV] {time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language

    _libero_env_debug_log("get_libero_env entered")
    _libero_env_debug_log(f"model_family={model_family}")
    _libero_env_debug_log(f"resolution={resolution}")
    _libero_env_debug_log(f"task_language={task_description}")

    _libero_env_debug_log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    _libero_env_debug_log(f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}")
    _libero_env_debug_log(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")
    _libero_env_debug_log(f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')}")
    _libero_env_debug_log(f"LIBERO_DATASET_PATH={os.environ.get('LIBERO_DATASET_PATH')}")
    _libero_env_debug_log(f"ROBOTIC_ATTACK_MODEL_ROOT={os.environ.get('ROBOTIC_ATTACK_MODEL_ROOT')}")

    bddl_root = get_libero_path("bddl_files")
    _libero_env_debug_log(f"bddl_root={bddl_root}")
    _libero_env_debug_log(f"bddl_root_exists={os.path.exists(bddl_root)}")

    task_bddl_file = os.path.join(bddl_root, task.problem_folder, task.bddl_file)
    _libero_env_debug_log(f"task_problem_folder={task.problem_folder}")
    _libero_env_debug_log(f"task_bddl_file={task_bddl_file}")
    _libero_env_debug_log(f"task_bddl_exists={os.path.exists(task_bddl_file)}")

    default_dataset_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../LIBERO/libero/datasets")
    )
    libero_default_dataset_path = os.path.abspath(
        os.path.join("/home/zifeng/siyuan/code/LIBERO/libero/libero/../datasets")
    )
    _libero_env_debug_log(f"default_dataset_path_guess={default_dataset_path}")
    _libero_env_debug_log(f"default_dataset_path_guess_exists={os.path.exists(default_dataset_path)}")
    _libero_env_debug_log(f"libero_default_dataset_path={libero_default_dataset_path}")
    _libero_env_debug_log(f"libero_default_dataset_path_exists={os.path.exists(libero_default_dataset_path)}")

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    _libero_env_debug_log(f"OffScreenRenderEnv init start env_args={env_args}")

    env = OffScreenRenderEnv(**env_args)

    _libero_env_debug_log("OffScreenRenderEnv init done")
    _libero_env_debug_log("env.seed start")

    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state

    _libero_env_debug_log("env.seed done")
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size, rotate_180: bool = DEFAULT_POLICY_IMAGE_ROTATE_180):
    """Extracts image from observations and preprocesses it."""
    resize_size = normalize_resize_size(resize_size)
    img = apply_policy_image_alignment(obs["agentview_image"], rotate_180=bool(rotate_180))
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    exp_name="test",
    rollout_root_dir="./rollouts",
):
    """Saves an MP4 replay of an episode."""
    # NOTE: rollout_root_dir can be an absolute path (e.g., /data/.../test) to keep large artifacts off system disks.
    rollout_dir = os.path.join(str(rollout_root_dir), str(exp_name), DATE)
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den