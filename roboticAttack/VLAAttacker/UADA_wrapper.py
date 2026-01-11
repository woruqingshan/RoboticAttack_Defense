import torch
from transformers import AutoConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from transformers import AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
import os
from pathlib import Path
from typing import Optional
import numpy as np
import wandb
import argparse
import random
import uuid
from white_patch.UADA import OpenVLAAttacker
from white_patch.openvla_dataloader import DATASET_INFO, get_dataloader

DEFAULT_MODEL_ROOT = Path(os.environ.get("ROBOTIC_ATTACK_MODEL_ROOT", "/data/zifeng/siyuan/data/models"))
DEFAULT_DATASET_ROOT = Path(os.environ.get("ROBOTIC_ATTACK_DATA_ROOT", "/data/zifeng/siyuan/data/datasets"))


def resolve_model_source(repo_id: str, override_root: Optional[str] = None):
    """Resolve HuggingFace repo ID to local directory when available."""
    target_root = Path(override_root) if override_root else DEFAULT_MODEL_ROOT
    candidate_paths = [
        target_root.joinpath(*repo_id.split("/")),      # nested structure
        target_root / repo_id.replace("/", "-"),        # dashed full repo id
        target_root / repo_id.split("/")[-1],           # last segment only
    ]
    for local_dir in candidate_paths:
        if local_dir.exists():
            return str(local_dir), True
    return repo_id, False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    exp_id = str(uuid.uuid4())
    if "bridge_orig" in args.dataset:
        vla_path = "openvla/openvla-7b"
    elif "libero_spatial" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-spatial"
    elif "libero_object" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-object"
    elif "libero_goal" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-goal"
    elif "libero_10" in args.dataset:
        vla_path = "openvla/openvla-7b-finetuned-libero-10"
    else:
        assert False, "Invalid dataset"
    set_seed(42)
    target = ''
    for i in args.maskidx:
        target += str(i)
    name = f"{args.dataset}_modifyLabel_MSEDistance_lr{format(args.lr, '.0e')}_iter{args.iter}_warmup{args.warmup}_target{target}_inner_loop{args.innerLoop}_patch_size{args.patch_size}_seed42-{exp_id}"
    if args.wandb_project != "false":
        wandb_run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,name=name, tags=args.tags)
        wandb.config = {"iteration":args.iter, "learning_rate": args.lr, "attack_target": args.maskidx,"accumulate_steps":args.accumulate}
    print(f"exp_id:{exp_id}")
    # NOTE: Make output path controllable from CLI so users can write to /data and
    # avoid filling the system disk. If --out_dir is omitted, fall back to the
    # historical default: <cwd>/run/UADA/<uuid>.
    if args.out_dir is not None:
        path = str(Path(args.out_dir).expanduser())
    else:
        pwd = os.getcwd()
    path = f"{pwd}/run/UADA/{exp_id}"

    AutoConfig.register("openvla", OpenVLAConfig)
    # NOTE: Required for loading OpenVLA processors from a local directory.
    # Without this registration, transformers may fail with:
    # "Unrecognized image processor ... Should have a `image_processor_type` ..."
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    quantization_config = None
    vla_path, local_only = resolve_model_source(vla_path, args.model_root)
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True, local_files_only=local_only)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    # NOTE: Reduce GPU memory usage during patch optimization.
    # - Disable KV cache (past_key_values) to avoid storing large attention caches.
    # - Enable gradient checkpointing to trade compute for memory.
    if hasattr(vla, "config") and hasattr(vla.config, "use_cache"):
        vla.config.use_cache = False
    if hasattr(vla, "gradient_checkpointing_enable"):
        vla.gradient_checkpointing_enable()
    # NOTE: We only optimize the adversarial patch. Freezing model parameters
    # avoids allocating gradients/optimizer state for the full OpenVLA model and
    # significantly reduces GPU memory usage.
    for param in vla.parameters():
        param.requires_grad_(False)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    vla = vla.to(device)
    os.makedirs(path, exist_ok=True)

    # NOTE: For LIBERO datasets, some setups store TFDS builders under a nested
    # subdirectory like: <dataset_root>/libero_rlds/libero_object_no_noops/...
    # If the expected dataset is not found directly under --dataset_root, fall
    # back to <dataset_root>/libero_rlds automatically.
    dataset_info = DATASET_INFO.get(args.dataset)
    if dataset_info is not None:
        expected_dataset_name = dataset_info[0]
        root = Path(args.dataset_root) if args.dataset_root else DEFAULT_DATASET_ROOT
        direct = root / expected_dataset_name
        nested = root / "libero_rlds" / expected_dataset_name
        if (not direct.exists()) and nested.exists():
            args.dataset_root = str(root / "libero_rlds")

    train_dataloader, val_dataloader = get_dataloader(
        batch_size=args.bs,
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        model_root=args.model_root,
    )
    openVLA_Attacker = OpenVLAAttacker(vla, processor, path,optimizer="adamW", resize_patch=args.resize_patch)

    # patch 224x224
    # patch_size=[3,22,22] - 1%
    # patch_size=[3,50,50] - 5%
    # patch_size=[3,70,70] - 10%
    # patch_size=[3,87,87] - 15%
    # patch_size=[3,100,100] - 20%
    openVLA_Attacker.patchattack_unconstrained(train_dataloader, val_dataloader, num_iter=args.iter,
                                               target_action=np.zeros(7), patch_size=args.patch_size, lr=args.lr,
                                               accumulate_steps=args.accumulate,
                                               maskidx=args.maskidx,
                                               warmup=args.warmup,
                                               filterGripTrainTo1=args.filterGripTrainTo1,
                                               geometry=args.geometry,
                                               innerLoop=args.innerLoop,
                                               args=args)

    print("Attack done!")
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maskidx',default='0', type=list_of_ints)
    parser.add_argument('--lr',default=1e-3, type=float)
    parser.add_argument('--device',default=1, type=int)
    parser.add_argument('--iter',default=2000, type=int) # 266933
    parser.add_argument('--accumulate',default=1, type=int)
    parser.add_argument('--bs',default=8, type=int)
    parser.add_argument('--warmup',default=20, type=int)
    parser.add_argument('--tags',nargs='+', default=[""])
    parser.add_argument('--filterGripTrainTo1', type=str2bool, nargs='?',default=False,
                        help='Remove the gripper 0 traning samples during the attack of target at grip to 0')
    parser.add_argument('--geometry', type=str2bool, nargs='?',default=True,
                        help='add geometry trans to path')
    parser.add_argument('--patch_size', default='3,50,50', type=list_of_ints)
    parser.add_argument('--wandb_project', default="xxx", type=str)
    parser.add_argument('--wandb_entity', default="xxx", type=str)
    parser.add_argument('--innerLoop', default=50, type=int)
    parser.add_argument('--dataset', default="bridge_orig", type=str)
    parser.add_argument('--resize_patch', type=str2bool, default=False)
    parser.add_argument('--reverse_direction', type=str2bool, default=True)
    parser.add_argument('--dataset_root', default=str(DEFAULT_DATASET_ROOT), type=str,
                        help="Optional override for dataset root directory.")
    parser.add_argument('--model_root', default=str(DEFAULT_MODEL_ROOT), type=str,
                        help="Optional override for model directory.")
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help=(
            "Optional output directory for this run. "
            "If set, all run artifacts (patch checkpoints, loss curves, pkl files) "
            "will be written here. If omitted, defaults to <cwd>/run/UADA/<uuid>."
        ),
    )
    return parser.parse_args()

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    args = arg_parser()
    print(f"Paramters:\n maskidx:{args.maskidx}\n lr:{args.lr} \n device:{args.device} \ntags:{args.tags}")
    main(args)
