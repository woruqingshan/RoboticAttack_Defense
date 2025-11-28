import os
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import AutoConfig, AutoImageProcessor
from torch.utils.data import Dataset, DataLoader, Subset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
import random
from typing import Optional
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_DATA_ROOT = Path(os.environ.get("ROBOTIC_ATTACK_DATA_ROOT", "/data/zifeng/siyuan/data/datasets"))
DEFAULT_MODEL_ROOT = Path(os.environ.get("ROBOTIC_ATTACK_MODEL_ROOT", "/data/zifeng/siyuan/data/models"))


def resolve_model_path(repo_id: str, model_root: Optional[str] = None):
    """Resolve HuggingFace repo ID to local directory when available."""
    target_root = Path(model_root) if model_root else DEFAULT_MODEL_ROOT
    candidate_paths = [
        target_root.joinpath(*repo_id.split("/")),
        target_root / repo_id.replace("/", "-"),
        target_root / repo_id.split("/")[-1],
    ]
    for local_dir in candidate_paths:
        if local_dir.exists():
            return str(local_dir), True
    return repo_id, False


def resolve_dataset_root(server: Optional[str], dataset_root: Optional[str]):
    """Resolve dataset root directory while keeping backward compatibility."""
    if dataset_root:
        return Path(dataset_root)
    if server:
        return Path(f"{server}/openvla-main/dataset")
    return DEFAULT_DATA_ROOT


DATASET_INFO = {
    "bridge_orig": ("bridge_orig", "openvla/openvla-7b"),
    "libero_spatial": ("libero_spatial_no_noops", "openvla/openvla-7b-finetuned-libero-spatial"),
    "libero_spatial_no_noops": ("libero_spatial_no_noops", "openvla/openvla-7b-finetuned-libero-spatial"),
    "libero_object": ("libero_object_no_noops", "openvla/openvla-7b-finetuned-libero-object"),
    "libero_object_no_noops": ("libero_object_no_noops", "openvla/openvla-7b-finetuned-libero-object"),
    "libero_goal": ("libero_goal_no_noops", "openvla/openvla-7b-finetuned-libero-goal"),
    "libero_goal_no_noops": ("libero_goal_no_noops", "openvla/openvla-7b-finetuned-libero-goal"),
    "libero_10": ("libero_10_no_noops", "openvla/openvla-7b-finetuned-libero-10"),
    "libero_10_no_noops": ("libero_10_no_noops", "openvla/openvla-7b-finetuned-libero-10"),
}


def _resolve_dataset_info(dataset: str, repo_override: Optional[str] = None) -> tuple[str, str]:
    """Return dataset directory name and default repo for given dataset."""
    info = DATASET_INFO.get(dataset)
    if info is None:
        if repo_override is None:
            raise AssertionError("Invalid dataset")
        dataset_name = dataset
        repo_id = repo_override
    else:
        dataset_name, repo_id = info
    if repo_override is not None:
        repo_id = repo_override
    return dataset_name, repo_id


def _build_components(
    dataset: str,
    server: Optional[str],
    dataset_root: Optional[str],
    model_root: Optional[str],
    repo_override: Optional[str] = None,
    processor: Optional[AutoProcessor] = None,
):
    dataset_name, repo_id = _resolve_dataset_info(dataset, repo_override)
    data_root_dir = resolve_dataset_root(server, dataset_root)

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    resolved_repo, local_only = resolve_model_path(repo_id, model_root)
    if processor is None:
        processor = AutoProcessor.from_pretrained(resolved_repo, trust_remote_code=True, local_files_only=local_only)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    prompt_builder = PurePromptBuilder if "v01" not in repo_id else VicunaV15ChatPromptBuilder
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        prompt_builder_fn=prompt_builder,
    )

    shuffle_buffer_size = 100_000
    image_aug = False

    train_dataset = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=shuffle_buffer_size,
        train=True,
        image_aug=image_aug,
    )
    val_dataset = RLDSDataset(
        data_root_dir,
        dataset_name,
        batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=shuffle_buffer_size,
        train=False,
        image_aug=image_aug,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    return processor, train_dataset, val_dataset, collator
def get_bridge_dataloader(batch_size, server=None, dataset_root=None, model_root=None):
    return get_dataloader(
        batch_size=batch_size,
        dataset="bridge_orig",
        server=server,
        dataset_root=dataset_root,
        model_root=model_root,
    )


def get_dataloader(batch_size, dataset, server=None, vla_path=None, dataset_root=None, model_root=None):
    processor, train_dataset, val_dataset, collator = _build_components(
        dataset=dataset,
        server=server,
        dataset_root=dataset_root,
        model_root=model_root,
        repo_override=vla_path,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,  # 32
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    return train_dataloader, val_dataloader


def get_dataset(
    dataset,
    server: Optional[str] = None,
    vla_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
    model_root: Optional[str] = None,
    processor: Optional[AutoProcessor] = None,
):
    processor, train_dataset, val_dataset, _ = _build_components(
        dataset=dataset,
        server=server,
        dataset_root=dataset_root,
        model_root=model_root,
        repo_override=vla_path,
        processor=processor,
    )
    return train_dataset, val_dataset, processor
