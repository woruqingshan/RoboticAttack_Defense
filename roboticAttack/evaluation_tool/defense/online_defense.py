"""Online attention hook + heatmap extraction for patch defense."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any

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

    return VisionTokenMeta(
        num_patches=num_patches,
        patch_grid_hw=patch_grid_hw,
        special_token_count=special_token_count,
    )


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
    Register an attention hook on an existing OpenVLA model instance and expose heatmap/grid extraction.

    Important:
    - This class does NOT load a model. It must be attached to the model used for online inference.
    - Attention tensors are populated when the model runs forward (e.g., during vla.predict_action()).
    - The script is expected to call `clear()` before each forward step; this also resets internal caches.
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

        # Caches valid only for the latest forward step (cleared by clear()).
        self._cached_grid_2d: Optional[torch.Tensor] = None  # [Hgrid,Wgrid] on CPU
        self._cached_grid_hw: Optional[Tuple[int, int]] = None
        self._cached_heatmap: Optional[np.ndarray] = None  # [image_size,image_size]

    def _invalidate_cache(self) -> None:
        """Clear only derived caches (grid, heatmap); keep captured attention tensor."""
        self._cached_grid_2d = None
        self._cached_grid_hw = None
        self._cached_heatmap = None

    def clear(self) -> None:
        """Clear hook buffers and derived caches."""
        self.hooks.clear()
        self._invalidate_cache()

    def invalidate_cache(self) -> None:
        """Clear only derived caches; keep captured attention tensor."""
        self._invalidate_cache()

    def run_and_capture(self, forward_fn: Callable[[], Any]) -> Any:
        """
        Helper for counterfactual / verification:
        - clear hook buffers and derived caches
        - run a forward pass (caller provides forward_fn)
        - return forward_fn output

        Usage:
            hook.run_and_capture(lambda: get_action(...))
            grid = hook.get_saliency_grid()
            hm = hook.get_heatmap()
        """
        self.clear()
        return forward_fn()

    def get_attention_tensor(self) -> torch.Tensor:
        attn = self.hooks.get(self.cache_key)
        if attn is None:
            raise RuntimeError("Attention tensor was not captured. Check hook module selection.")
        return attn

    def _compute_grid_2d(self) -> torch.Tensor:
        """Compute and cache CLS->patch attention grid for the latest forward step."""
        if self._cached_grid_2d is not None:
            return self._cached_grid_2d

        attn_tensor = self.get_attention_tensor().to(torch.float32)
        aggregated = aggregate_attention_heads(attn_tensor, mode=self.aggregate_mode)

        cls_vec = cls_to_patch_attention(
            aggregated,
            exclude_cls=True,
            patch_token_count=self.meta.num_patches,
            special_token_count=self.meta.special_token_count,
        )

        grid_bhw, grid_shape = attention_vector_to_grid(
            cls_vec,
            patch_token_count=self.meta.num_patches,
            grid_shape=self.meta.patch_grid_hw,
        )

        # grid_bhw: [B,Hgrid,Wgrid], online inference expects B==1.
        grid_2d = grid_bhw[0].detach().to("cpu")
        self._cached_grid_2d = grid_2d

        if grid_shape is not None:
            self._cached_grid_hw = (int(grid_shape[0]), int(grid_shape[1]))
            if self.meta.patch_grid_hw is None:
                self.meta.patch_grid_hw = self._cached_grid_hw
        else:
            self._cached_grid_hw = (int(grid_2d.shape[0]), int(grid_2d.shape[1]))
            if self.meta.patch_grid_hw is None:
                self.meta.patch_grid_hw = self._cached_grid_hw

        return self._cached_grid_2d

    def get_saliency_grid(self, force: bool = False) -> np.ndarray:
        """Return low-res CLS->patch attention grid as numpy (e.g., 16x16).
        
        Args:
            force: If True, invalidate cache and recompute from attention tensor.
        """
        if force:
            self._invalidate_cache()
        grid_2d = self._compute_grid_2d()
        g = grid_2d.detach().cpu().numpy().astype(np.float32, copy=False)
        # defensive: ensure non-negative (should already hold for attention, but keep robust)
        g = g - float(g.min())
        return g

    def get_grid_shape(self) -> Tuple[int, int]:
        """Return (Hgrid, Wgrid) for the saliency grid."""
        if self._cached_grid_hw is not None:
            return self._cached_grid_hw
        if self.meta.patch_grid_hw is not None:
            return self.meta.patch_grid_hw
        g = self._compute_grid_2d()
        return int(g.shape[0]), int(g.shape[1])

    def grid_bbox_to_patch_box(self, gx0: int, gy0: int, gx1: int, gy1: int):
        """Map grid bbox to pixel PatchBox in resized image coordinates [0, image_size]."""
        # Local import prevents hard runtime dependency loops.
        from .anomaly_detector import PatchBox  # type: ignore

        hgrid, wgrid = self.get_grid_shape()

        gx0 = int(np.clip(gx0, 0, wgrid))
        gx1 = int(np.clip(gx1, 0, wgrid))
        gy0 = int(np.clip(gy0, 0, hgrid))
        gy1 = int(np.clip(gy1, 0, hgrid))
        if gx1 < gx0:
            gx0, gx1 = gx1, gx0
        if gy1 < gy0:
            gy0, gy1 = gy1, gy0

        # Convert grid coords -> pixel coords (round reduces small-ROI bias).
        x0 = int(round(gx0 * self.image_size / float(wgrid)))
        x1 = int(round(gx1 * self.image_size / float(wgrid)))
        y0 = int(round(gy0 * self.image_size / float(hgrid)))
        y1 = int(round(gy1 * self.image_size / float(hgrid)))

        x0 = int(np.clip(x0, 0, self.image_size))
        x1 = int(np.clip(x1, 0, self.image_size))
        y0 = int(np.clip(y0, 0, self.image_size))
        y1 = int(np.clip(y1, 0, self.image_size))

        return PatchBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def get_heatmap(self, force: bool = False) -> np.ndarray:
        """Return upsampled heatmap (image_size x image_size) for visualization/legacy detector.
        
        Args:
            force: If True, invalidate cache and recompute from attention tensor.
        """
        if force:
            self._invalidate_cache()
        if self._cached_heatmap is not None:
            return self._cached_heatmap

        grid_2d = self._compute_grid_2d()
        grid = grid_2d.unsqueeze(0).unsqueeze(0)  # [1,1,Hgrid,Wgrid]
        up = F.interpolate(
            grid,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        heatmap = up.squeeze().detach().cpu().numpy()
        self._cached_heatmap = heatmap
        return heatmap
