"""Utils for evaluating the OpenVLA policy."""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Default model root directory (same as UADA_wrapper.py)
DEFAULT_MODEL_ROOT = Path(os.environ.get("ROBOTIC_ATTACK_MODEL_ROOT", "/data/zifeng/siyuan/data/models"))

# Path to custom code files in the codebase
CODEBASE_ROOT = Path(__file__).parent.parent.parent  # Go up to roboticAttack root (experiments/robot -> experiments -> roboticAttack)
CUSTOM_CODE_SOURCE = CODEBASE_ROOT / "prismatic" / "extern" / "hf"
REQUIRED_CUSTOM_FILES = [
    "configuration_prismatic.py",
    "modeling_prismatic.py",
    "processing_prismatic.py",
]

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def ensure_custom_code_files(model_dir: Path):
    """
    Ensure custom code files exist in the model directory.
    Copy them from the codebase if they don't exist.
    
    Args:
        model_dir: Path to the model directory
    """
    model_dir = Path(model_dir)
    for filename in REQUIRED_CUSTOM_FILES:
        source_file = CUSTOM_CODE_SOURCE / filename
        target_file = model_dir / filename
        
        if not target_file.exists() and source_file.exists():
            print(f"[*] Copying {filename} to model directory...")
            shutil.copy2(source_file, target_file)
        elif not source_file.exists():
            print(f"[WARNING] Source file {source_file} not found, skipping {filename}")


def fix_config_for_local_use(model_dir: Path):
    """
    Fix config.json and preprocessor_config.json to use local paths instead of HuggingFace paths.
    This prevents transformers from trying to download files from HuggingFace.
    
    Args:
        model_dir: Path to the model directory
    """
    model_dir = Path(model_dir)
    
    # Fix config.json
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Remove auto_map to prevent transformers from trying to download from HuggingFace
        # We manually register the classes in get_vla(), so auto_map is not needed
        if "auto_map" in config:
            del config["auto_map"]
            print(f"[*] Removed auto_map from config.json (using manual class registration)")
        
        # Fix _name_or_path to use local path
        if "_name_or_path" in config:
            config["_name_or_path"] = str(model_dir)
        
        # Write back
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Fix preprocessor_config.json (also contains auto_map)
    preprocessor_config_path = model_dir / "preprocessor_config.json"
    if preprocessor_config_path.exists():
        with open(preprocessor_config_path, 'r') as f:
            preprocessor_config = json.load(f)
        
        # Remove auto_map from preprocessor_config.json
        if "auto_map" in preprocessor_config:
            del preprocessor_config["auto_map"]
            print(f"[*] Removed auto_map from preprocessor_config.json (using manual class registration)")
        
        # Write back
        with open(preprocessor_config_path, 'w') as f:
            json.dump(preprocessor_config, f, indent=2)
    
    print(f"[*] Fixed config files for local use")


def resolve_model_source(repo_id: str, override_root: Optional[str] = None):
    """
    Resolve HuggingFace repo ID to local directory.
    FORCE local-only mode - will raise error if local model not found.
    Similar to UADA_wrapper.py implementation.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "openvla/openvla-7b-finetuned-libero-spatial")
        override_root: Optional override for model root directory
    
    Returns:
        tuple: (resolved_path, local_only) where local_only is always True
    """
    target_root = Path(override_root) if override_root else DEFAULT_MODEL_ROOT
    
    # Extract model name from repo_id (e.g., "openvla/openvla-7b-finetuned-libero-spatial" -> "openvla-7b-finetuned-libero-spatial")
    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    
    # Try multiple candidate paths
    candidate_paths = [
        target_root / model_name,  # Direct match: models/openvla-7b-finetuned-libero-spatial
        target_root.joinpath(*repo_id.split("/")),  # Nested: models/openvla/openvla-7b-finetuned-libero-spatial
        target_root / repo_id.replace("/", "-"),  # Dashed: models/openvla-openvla-7b-finetuned-libero-spatial
    ]
    
    for local_dir in candidate_paths:
        if local_dir.exists() and (local_dir / "config.json").exists():
            print(f"[*] Found local model at: {local_dir}")
            # Ensure custom code files exist in model directory
            ensure_custom_code_files(local_dir)
            # Fix config.json to use local paths
            fix_config_for_local_use(local_dir)
            return str(local_dir), True
    
    # If no local model found, raise error instead of attempting download
    raise FileNotFoundError(
        f"Local model not found for {repo_id}. "
        f"Searched in: {[str(p) for p in candidate_paths]}. "
        f"Please ensure the model is available locally at one of these paths. "
        f"Network download is disabled."
    )


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # FORCE offline mode - disable all network access
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16")
    print("[*] FORCED OFFLINE MODE - No network access allowed")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Resolve model source (FORCE local-only, no network access)
    model_root_override = getattr(cfg, 'model_root', None)
    resolved_checkpoint, local_only = resolve_model_source(cfg.pretrained_checkpoint, model_root_override)
    
    # Force local_files_only=True to prevent any network access
    assert local_only, "resolve_model_source must return local_only=True"

    # Try to use flash_attention_2 if available, otherwise use default attention
    attn_implementation = None
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
        print("[*] Using Flash Attention 2")
    except ImportError:
        print("[*] Flash Attention 2 not available, using default attention implementation")
        attn_implementation = None  # Use default

    vla = AutoModelForVision2Seq.from_pretrained(
        resolved_checkpoint,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,  # FORCE local-only, disable network access
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    # Use resolved checkpoint path for dataset statistics
    dataset_statistics_path = os.path.join(resolved_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    # FORCE offline mode - disable all network access
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Resolve model source (FORCE local-only, no network access)
    model_root_override = getattr(cfg, 'model_root', None)
    resolved_checkpoint, local_only = resolve_model_source(cfg.pretrained_checkpoint, model_root_override)
    
    # Force local_files_only=True to prevent any network access
    assert local_only, "resolve_model_source must return local_only=True"
    
    # Directly instantiate PrismaticProcessor from local files
    # This avoids transformers trying to download from HuggingFace
    from transformers import AutoImageProcessor, AutoTokenizer
    
    # Register image processor class (already done in get_vla, but ensure it's done here too)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    
    resolved_path = Path(resolved_checkpoint)
    
    # Load image processor and tokenizer directly from local path
    image_processor = AutoImageProcessor.from_pretrained(
        resolved_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    
    # Create PrismaticProcessor instance directly
    processor = PrismaticProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
