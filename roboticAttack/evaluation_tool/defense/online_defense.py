"""Online attention hook + heatmap extraction for patch defense."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from evaluation_tool.saliency.attention_hooks import (
    AttentionHookManager,
    aggregate_attention_heads,
    attention_vector_to_grid,
    cls_to_patch_attention,
)


@dataclass
class VisionTokenMeta:
    """Best-effort metadata for mapping attention tokens to an image grid."""

    num_patches: Optional[int]
    patch_grid_hw: Optional[Tuple[int, int]]
    special_token_count: int


def _infer_register_token_count(featurizer: torch.nn.Module) -> int:
    """Best-effort detection of register/special tokens following CLS."""

    candidate_counts = []
    num_prefix = getattr(featurizer, "num_prefix_tokens", None)
    if num_prefix is not None:
        candidate_counts.append(max(0, int(num_prefix) - 1))

    register_tokens = getattr(featurizer, "register_tokens", None)
    if register_tokens is not None and hasattr(register_tokens, "shape"):
        candidate_counts.append(int(register_tokens.shape[1]))

    global_tokens = getattr(featurizer, "global_tokens", None)
    if global_tokens is not None and hasattr(global_tokens, "shape"):
        candidate_counts.append(int(global_tokens.shape[1]))

    return max(candidate_counts) if candidate_counts else 0


def infer_vision_token_meta(model: torch.nn.Module) -> VisionTokenMeta:
    """Infer patch token count and grid size from OpenVLA vision backbone when available."""

    num_patches = None
    patch_grid_hw = None
    special_token_count = 0

    vision_backbone = getattr(model, "vision_backbone", None)
    if vision_backbone is None:
        vision_backbone = getattr(model, "vision_tower", None)

    if vision_backbone is not None:
        featurizer = getattr(vision_backbone, "featurizer", None)
        if featurizer is not None:
            patch_embed = getattr(featurizer, "patch_embed", None)
            if patch_embed is not None:
                if hasattr(patch_embed, "num_patches"):
                    num_patches = int(patch_embed.num_patches)
                if hasattr(patch_embed, "grid_size"):
                    grid_size = patch_embed.grid_size
                    if isinstance(grid_size, (tuple, list)):
                        patch_grid_hw = (int(grid_size[0]), int(grid_size[1]))
                    else:
                        dim = int(grid_size)
                        patch_grid_hw = (dim, dim)
            special_token_count = _infer_register_token_count(featurizer)

    return VisionTokenMeta(num_patches=num_patches, patch_grid_hw=patch_grid_hw, special_token_count=special_token_count)


def infer_default_attn_module_name(model: torch.nn.Module) -> str:
    """Infer a vision self-attention module name for hooking (OpenVLA TIMM ViT style)."""

    for backbone_name in ["vision_backbone", "vision_tower"]:
        if hasattr(model, backbone_name):
            backbone = getattr(model, backbone_name)
            featurizer = getattr(backbone, "featurizer", None)
            if featurizer is not None and hasattr(featurizer, "blocks"):
                blocks = featurizer.blocks
                if len(blocks) > 0:
                    last_idx = len(blocks) - 1
                    if hasattr(blocks[last_idx], "attn"):
                        return f"{backbone_name}.featurizer.blocks.{last_idx}.attn"
                    if hasattr(blocks[last_idx], "self_attn"):
                        return f"{backbone_name}.featurizer.blocks.{last_idx}.self_attn"

    block_attn = []
    for name, _module in model.named_modules():
        m = re.match(r"vision_(backbone|tower)\.featurizer\.blocks\.(\d+)\.attn$", name)
        if m:
            block_attn.append((int(m.group(2)), name))
    if block_attn:
        block_attn.sort(key=lambda x: x[0])
        return block_attn[-1][1]

    candidates = []
    for name, _module in model.named_modules():
        if "attn" in name.lower() and "cross" not in name.lower():
            if any(sub in name for sub in [".qkv", ".proj", ".proj_drop"]):
                continue
            candidates.append(name)
    if candidates:
        candidates.sort(key=lambda x: x.count("."))
        return candidates[-1]

    raise RuntimeError("Could not infer a vision self-attention module name to hook.")


class OnlineAttentionHook:
    """
    Register an attention hook on an existing OpenVLA model instance and expose heatmap extraction.

    Important:
    - This class does NOT load a model. It must be attached to the model used for online inference.
    - Attention tensors are populated when the model runs forward (e.g., during vla.predict_action()).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        attn_module_name: Optional[str] = None,
        aggregate_mode: str = "mean",
        image_size: int = 224,
        cache_key: str = "vision_attn",
    ) -> None:
        self.model = model
        self.aggregate_mode = aggregate_mode
        self.image_size = int(image_size)
        self.cache_key = cache_key
        self.meta = infer_vision_token_meta(model)

        module_name = attn_module_name or infer_default_attn_module_name(model)
        module_dict = dict(model.named_modules())
        if module_name not in module_dict:
            matches = [m for n, m in module_dict.items() if module_name in n]
            if not matches:
                raise KeyError(f"Attention module '{module_name}' not found in model.named_modules().")
            target_module = matches[-1]
        else:
            target_module = module_dict[module_name]

        self.hooks = AttentionHookManager()
        self.hooks.register_attention(target_module, self.cache_key)

    def clear(self) -> None:
        self.hooks.clear()

    def get_attention_tensor(self) -> torch.Tensor:
        attn = self.hooks.get(self.cache_key)
        if attn is None:
            raise RuntimeError("Attention tensor was not captured. Check hook module selection.")
        return attn

    def get_heatmap(self) -> np.ndarray:
        """Convert the latest cached attention tensor to a 2D heatmap resized to image_size x image_size."""

        attn_tensor = self.get_attention_tensor().to(torch.float32)
        aggregated = aggregate_attention_heads(attn_tensor, mode=self.aggregate_mode)

        cls_vec = cls_to_patch_attention(
            aggregated,
            exclude_cls=True,
            patch_token_count=self.meta.num_patches,
            special_token_count=self.meta.special_token_count,
        )

        grid, _grid_shape = attention_vector_to_grid(
            cls_vec,
            patch_token_count=self.meta.num_patches,
            grid_shape=self.meta.patch_grid_hw,
        )

        grid = grid[0].unsqueeze(0).unsqueeze(0)
        up = F.interpolate(grid, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        heatmap = up.squeeze().detach().cpu().numpy()
        return heatmap


