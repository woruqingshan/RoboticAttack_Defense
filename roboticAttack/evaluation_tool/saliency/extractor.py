"""Extractor that hooks OpenVLA attention maps and produces heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from .attention_hooks import (
    AttentionHookManager,
    aggregate_attention_heads,
    attention_vector_to_grid,
    cls_to_patch_attention,
)


DEFAULT_MODEL_ROOT = Path("/data/zifeng/siyuan/data/models")


def resolve_model_source(repo_id: str, override_root: Optional[Union[str, Path]] = None) -> Tuple[str, bool]:
    """Resolve HuggingFace repo ID to a local directory when the weights exist offline."""

    target_root = Path(override_root) if override_root else DEFAULT_MODEL_ROOT
    candidate_paths = [
        target_root.joinpath(*repo_id.split("/")),
        target_root / repo_id.replace("/", "-"),
        target_root / repo_id.split("/")[-1],
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate), True
    return repo_id, False


def dataset_to_repo_id(dataset: str) -> str:
    dataset = dataset.lower()
    if "bridge_orig" in dataset:
        return "openvla/openvla-7b"
    if "libero_spatial" in dataset:
        return "openvla/openvla-7b-finetuned-libero-spatial"
    if "libero_object" in dataset:
        return "openvla/openvla-7b-finetuned-libero-object"
    if "libero_goal" in dataset:
        return "openvla/openvla-7b-finetuned-libero-goal"
    if "libero_10" in dataset:
        return "openvla/openvla-7b-finetuned-libero-10"
    raise ValueError(f"Unsupported dataset specifier: {dataset}")


def build_openvla_model(
    dataset: str,
    device: Union[str, torch.device] = "cuda:0",
    model_root: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.nn.Module, AutoProcessor]:
    """Load OpenVLA weights + processor with the prismatic registry helpers."""

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    repo_id = dataset_to_repo_id(dataset)
    resolved_path, local_only = resolve_model_source(repo_id, model_root)
    processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True, local_files_only=local_only)
    model = AutoModelForVision2Seq.from_pretrained(
        resolved_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    dev = torch.device(device if isinstance(device, str) else device)
    model = model.to(dev)
    model.eval()
    return model, processor


class VLAAttentionExtractor:
    """Wrapper that exposes get_saliency() using OpenVLA attention weights."""

    def __init__(
        self,
        dataset: str,
        instruction_template: Optional[str] = None,
        device: Union[str, torch.device] = "cuda:0",
        model_root: Optional[str] = None,
        attn_module_name: Optional[str] = None,
        aggregate_mode: str = "mean",
        image_size: int = 224,
    ) -> None:
        self.model, self.processor = build_openvla_model(dataset, device=device, model_root=model_root)
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.aggregate_mode = aggregate_mode
        self.image_size = image_size
        self.instruction_template = instruction_template

        # Get the number of vision patches from the model
        self.num_patches = None
        vision_backbone = getattr(self.model, "vision_backbone", None)
        if vision_backbone is not None:
            featurizer = getattr(vision_backbone, "featurizer", None)
            if featurizer is not None:
                patch_embed = getattr(featurizer, "patch_embed", None)
                if patch_embed is not None and hasattr(patch_embed, "num_patches"):
                    self.num_patches = int(patch_embed.num_patches)
        
        if self.num_patches is not None:
            print(f"[SAL] Detected {self.num_patches} vision patches per image.")
        else:
            print("[SAL] WARNING: Could not infer patch token count; "
                  "falling back to heuristic grid inference.")

        self.hooks = AttentionHookManager()
        self.attn_cache_key = "vision_attn"

        module_name = attn_module_name or self._infer_default_attn_module()
        target_module = self._resolve_module_by_name(module_name)
        self.hooks.register_attention(target_module, self.attn_cache_key)

    def _infer_default_attn_module(self) -> str:
        """Infer the default vision self-attention module name for hooking."""
        candidates = []
        
        # First, try to find vision_backbone (OpenVLA uses PrismaticVisionBackbone)
        # Check both vision_backbone and vision_tower for compatibility
        for backbone_name in ["vision_backbone", "vision_tower"]:
            if hasattr(self.model, backbone_name):
                backbone = getattr(self.model, backbone_name)
                if hasattr(backbone, "featurizer"):
                    featurizer = backbone.featurizer
                    if hasattr(featurizer, "blocks"):
                        # TIMM ViT structure: blocks[-1].attn
                        last_block_idx = len(featurizer.blocks) - 1
                        if hasattr(featurizer.blocks[last_block_idx], "attn"):
                            return f"{backbone_name}.featurizer.blocks.{last_block_idx}.attn"
                        elif hasattr(featurizer.blocks[last_block_idx], "self_attn"):
                            return f"{backbone_name}.featurizer.blocks.{last_block_idx}.self_attn"
        
        # Alternative: Find all blocks.attn modules and select the last one by block index
        block_attn_modules = []
        for name, module in self.model.named_modules():
            # Match pattern: vision_backbone.featurizer.blocks.{idx}.attn
            match = re.match(r"vision_(backbone|tower)\.featurizer\.blocks\.(\d+)\.attn$", name)
            if match:
                block_idx = int(match.group(2))
                block_attn_modules.append((block_idx, name))
        
        if block_attn_modules:
            # Sort by block index and return the last one
            block_attn_modules.sort(key=lambda x: x[0])
            return block_attn_modules[-1][1]
        
        # Search for vision attention modules with various naming patterns
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Module):
                continue
            
            # Pattern 1: Standard transformers style (vision_tower.vision_model.encoder.layers[-1].self_attn)
            if "vision" in name.lower() and name.endswith("self_attn"):
                candidates.append(name)
            
            # Pattern 2: TIMM ViT style (featurizer.blocks[-1].attn or .self_attn)
            if ("featurizer" in name or "vision" in name.lower()) and (
                name.endswith(".attn") or name.endswith(".self_attn")
            ):
                # Filter out sub-modules like .qkv, .proj, .proj_drop
                if not any(sub in name for sub in [".qkv", ".proj", ".proj_drop", ".q_norm", ".k_norm", ".attn_drop"]):
                    candidates.append(name)
            
            # Pattern 3: Look for attention modules in vision-related paths
            if any(keyword in name.lower() for keyword in ["vision", "featurizer", "backbone"]) and any(
                keyword in name.lower() for keyword in ["attn", "attention"]
            ):
                # Filter out cross-attention and other non-self-attention modules
                if "cross" not in name.lower() and not any(sub in name for sub in [".qkv", ".proj", ".proj_drop"]):
                    candidates.append(name)
        
        # Prefer the last layer's attention (usually most informative)
        if candidates:
            # Filter to only blocks.attn modules and sort by block index
            block_candidates = []
            for cand in candidates:
                match = re.search(r"blocks\.(\d+)\.attn", cand)
                if match:
                    block_idx = int(match.group(1))
                    block_candidates.append((block_idx, cand))
            
            if block_candidates:
                block_candidates.sort(key=lambda x: x[0])
                return block_candidates[-1][1]
            
            # Fallback: sort by depth (more dots = deeper)
            candidates.sort(key=lambda x: x.count("."))
            return candidates[-1]
        
        # Fallback: try to find any attention module in the model
        for name, module in self.model.named_modules():
            if "attn" in name.lower() and "cross" not in name.lower():
                # Prefer blocks.attn over sub-modules
                if "blocks." in name and ".attn" in name and not any(sub in name for sub in [".qkv", ".proj"]):
                    return name
        
        # If still not found, use debug helper to get all candidates
        debug_info = self.list_attention_modules(max_results=20)
        
        error_msg = (
            f"Could not find a vision self-attention module to hook.\n"
            f"Found {len(debug_info['vision_modules'])} vision-related modules, "
            f"{len(debug_info['attention_modules'])} attention modules, "
            f"and {len(debug_info['candidates'])} candidate modules.\n\n"
        )
        
        if debug_info['candidates']:
            error_msg += "Top candidate modules (most likely to work):\n"
            for i, candidate in enumerate(debug_info['candidates'][:10], 1):
                error_msg += f"  {i}. {candidate}\n"
            error_msg += "\n"
        
        if debug_info['vision_modules']:
            error_msg += "Vision-related modules (first 10):\n"
            for i, module in enumerate(debug_info['vision_modules'][:10], 1):
                error_msg += f"  {i}. {module}\n"
            error_msg += "\n"
        
        if debug_info['attention_modules']:
            error_msg += "Attention modules (first 10):\n"
            for i, module in enumerate(debug_info['attention_modules'][:10], 1):
                error_msg += f"  {i}. {module}\n"
            error_msg += "\n"
        
        error_msg += (
            "To see all available modules, you can call:\n"
            "  extractor = VLAAttentionExtractor(...)\n"
            "  debug_info = extractor.list_attention_modules()\n"
            "  print(debug_info)\n\n"
            "Or specify --attn-module manually using one of the module names above."
        )
        
        raise RuntimeError(error_msg)

    def _resolve_module_by_name(self, module_name: str) -> torch.nn.Module:
        module_dict = dict(self.model.named_modules())
        if module_name in module_dict:
            return module_dict[module_name]
        matches = [module for name, module in module_dict.items() if module_name in name]
        if matches:
            return matches[-1]
        
        # If module not found, list all available attention modules to help user
        print(f"\n{'='*80}")
        print(f"ERROR: Module name '{module_name}' not found in OpenVLA model.")
        print(f"{'='*80}\n")
        print("Available attention modules:")
        print("-" * 80)
        
        debug_info = self.list_attention_modules(max_results=50)
        
        if debug_info['candidates']:
            print("\nRecommended candidates (most likely to work):")
            for i, candidate in enumerate(debug_info['candidates'][:10], 1):
                print(f"  {i}. {candidate}")
        
        if debug_info['vision_modules']:
            print("\nAll vision-related modules:")
            for i, module in enumerate(debug_info['vision_modules'][:20], 1):
                print(f"  {i}. {module}")
            if len(debug_info['vision_modules']) > 20:
                print(f"  ... and {len(debug_info['vision_modules']) - 20} more")
        
        if debug_info['attention_modules']:
            print("\nAll attention modules:")
            for i, module in enumerate(debug_info['attention_modules'][:20], 1):
                print(f"  {i}. {module}")
            if len(debug_info['attention_modules']) > 20:
                print(f"  ... and {len(debug_info['attention_modules']) - 20} more")
        
        print(f"\n{'='*80}")
        print("Please specify one of the modules above using --attn-module parameter.")
        print(f"{'='*80}\n")
        
        raise KeyError(f"Module name '{module_name}' not present in OpenVLA model.")

    def list_attention_modules(self, max_results: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Debug helper method to list all potential attention modules in the model.
        
        Args:
            max_results: Maximum number of results per category. If None, returns all.
            
        Returns:
            Dictionary with keys:
                - 'vision_modules': All modules containing 'vision' or 'featurizer'
                - 'attention_modules': All modules containing 'attn' (excluding cross-attention)
                - 'candidates': Modules matching common attention patterns
        """
        all_modules = [name for name, _ in self.model.named_modules()]
        
        vision_modules = [
            name for name in all_modules 
            if "vision" in name.lower() or "featurizer" in name.lower() or "backbone" in name.lower()
        ]
        
        attention_modules = [
            name for name in all_modules 
            if "attn" in name.lower() and "cross" not in name.lower()
        ]
        
        # Find candidates matching common patterns
        candidates = []
        for name in all_modules:
            # Pattern 1: Standard transformers style
            if "vision" in name.lower() and name.endswith("self_attn"):
                candidates.append(name)
            # Pattern 2: TIMM ViT style
            elif ("featurizer" in name or "vision" in name.lower()) and (
                name.endswith(".attn") or name.endswith(".self_attn") or ".attn." in name
            ):
                candidates.append(name)
            # Pattern 3: Vision-related attention
            elif any(kw in name.lower() for kw in ["vision", "featurizer", "backbone"]) and any(
                kw in name.lower() for kw in ["attn", "attention"]
            ):
                if "cross" not in name.lower():
                    candidates.append(name)
        
        result = {
            "vision_modules": vision_modules[:max_results] if max_results else vision_modules,
            "attention_modules": attention_modules[:max_results] if max_results else attention_modules,
            "candidates": candidates[:max_results] if max_results else candidates,
        }
        
        return result

    def _prepare_instruction(self, instruction: Optional[str]) -> str:
        if instruction:
            return instruction
        if self.instruction_template:
            return self.instruction_template
        raise ValueError("Instruction text is required for OpenVLA inference.")

    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            return pil_image
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("NumPy images must be HxWx3.")
            if image.dtype != np.uint8:
                clipped = np.clip(image, 0, 255).astype(np.uint8)
            else:
                clipped = image
            return Image.fromarray(clipped)
        raise TypeError(f"Unsupported image type: {type(image)}")

    @torch.inference_mode()
    def get_saliency(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        instruction: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Run OpenVLA forward pass and convert the latest vision attention into a heatmap."""

        pil_image = self._prepare_image(image)
        instruction_text = self._prepare_instruction(instruction)
        inputs = self.processor(
            images=pil_image,
            text=instruction_text,
            return_tensors="pt",
        )
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Only convert pixel_values to bfloat16, keep other tensors (input_ids, attention_mask) as integers
                if key == "pixel_values":
                    inputs[key] = value.to(device=self.device, dtype=self.dtype)
                else:
                    inputs[key] = value.to(device=self.device)

        self.hooks.clear()
        _ = self.model(**inputs)
        attn_tensor = self.hooks.get(self.attn_cache_key)
        if attn_tensor is None:
            raise RuntimeError("Attention tensor was not captured. Check module selection.")

        heatmap = self._attention_to_image(attn_tensor)
        return {
            "heatmap": heatmap,
            "attention_raw": attn_tensor.detach().to(torch.float32).cpu().numpy(),
        }

    def _attention_to_image(self, attn_tensor: torch.Tensor) -> np.ndarray:

        attn_tensor = attn_tensor.to(torch.float32)
        aggregated = aggregate_attention_heads(attn_tensor, mode=self.aggregate_mode)
        
        # Print aggregated attention shape for debugging
        print(f"[SAL] Aggregated attention shape: {aggregated.shape}")
        
        # Extract CLS-to-patch attention vector
        cls_vec = cls_to_patch_attention(
            aggregated,
            exclude_cls=True,
            patch_token_count=self.num_patches,
        )
        
        # Print CLS vector truncation info
        if self.num_patches is not None:
            print(f"[SAL] CLS attention vector: {aggregated.shape[-1]} -> {cls_vec.shape[-1]} tokens "
                  f"(kept first {self.num_patches} patch tokens)")
        
        # Map 1D attention vector to 2D grid
        grid, grid_size = attention_vector_to_grid(
            cls_vec,
            patch_token_count=self.num_patches,
        )
        
        print(f"[SAL] Reshaped to grid: {grid_size}x{grid_size}")
        
        grid = grid[0].unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(grid, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        saliency = upsampled.squeeze().cpu().numpy()
        return saliency

