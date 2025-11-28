"""Utilities for capturing and aggregating attention tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch


@dataclass
class HookRecord:
    """Stores cached tensors produced by hook callbacks."""

    key: str
    tensor: torch.Tensor


class AttentionHookManager:
    """Minimal manager that tracks hook handles and cached tensors."""

    def __init__(self) -> None:
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._cache: Dict[str, HookRecord] = {}

    def register_attention(self, module: torch.nn.Module, key: str) -> None:
        """Register a forward hook that extracts attention weights from TIMM ViT attention modules."""
        
        # For TIMM ViT, try to hook into the attn submodule if it exists
        if hasattr(module, 'attn'):
            attn_module = module.attn
        else:
            attn_module = module
        
        # Store QKV output for manual attention computation
        qkv_output = {}
        
        def qkv_hook(_module, _inputs, output):
            """Hook to capture QKV output before attention computation."""
            qkv_output['tensor'] = output.detach()
        
        def attn_forward_hook(_module, _inputs, output):
            """Hook to extract attention weights from TIMM attention module."""
            attn_weights = None
            
            # Method 1: Try to compute attention weights from QKV
            if 'tensor' in qkv_output and _inputs and len(_inputs) > 0:
                try:
                    qkv = qkv_output['tensor']
                    B, N, C3 = qkv.shape
                    C = C3 // 3
                    
                    # Split QKV
                    q, k, v = qkv.chunk(3, dim=-1)  # Each: [B, N, C]
                    
                    # Try to infer num_heads from common configurations
                    # Common values: 8, 12, 16, 24, 32, 48, 64
                    # Try to find a divisor that makes sense
                    possible_heads = [8, 12, 16, 24, 32, 48, 64]
                    num_heads = None
                    for nh in possible_heads:
                        if C % nh == 0:
                            num_heads = nh
                            break
                    
                    # If no common divisor found, try to infer from C
                    if num_heads is None:
                        # Try to find a reasonable head_dim (typically 64 or 128)
                        for head_dim in [64, 128, 256]:
                            if C % head_dim == 0:
                                num_heads = C // head_dim
                                break
                    
                    # Default fallback: assume head_dim = 64
                    if num_heads is None:
                        head_dim = 64
                        num_heads = C // head_dim if C % head_dim == 0 else 8
                    else:
                        head_dim = C // num_heads
                    
                    # Reshape for multi-head attention
                    q = q.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, H, N, D]
                    k = k.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, H, N, D]
                    
                    # Compute attention scores: Q @ K^T / sqrt(head_dim)
                    scale = (head_dim ** -0.5)
                    attn_scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, N, N]
                    attn_weights = attn_scores.softmax(dim=-1)  # [B, H, N, N]
                    
                except Exception:
                    # If manual computation fails, continue to other methods
                    pass
            
            # Method 2: Check if output is already 4D attention weights
            if attn_weights is None:
                if isinstance(output, torch.Tensor) and output.ndim == 4:
                    # Check if shape looks like attention weights [B, H, N, N]
                    if output.shape[-1] == output.shape[-2]:
                        attn_weights = output
            
            # Method 3: Check module attributes
            if attn_weights is None:
                if hasattr(_module, '_attn_cache'):
                    attn_weights = _module._attn_cache
                elif hasattr(_module, 'attn_weights'):
                    attn_weights = _module.attn_weights
            
            # If we successfully extracted attention weights, cache them
            if attn_weights is not None:
                self._cache[key] = HookRecord(key=key, tensor=attn_weights.detach())
            else:
                # Raise error with helpful message
                output_shape = output.shape if isinstance(output, torch.Tensor) else type(output)
                raise RuntimeError(
                    f"Cannot extract attention weights from module {_module.__class__.__name__}. "
                    f"Output shape: {output_shape}. "
                    f"QKV available: {'tensor' in qkv_output}. "
                    f"Please check the module structure or specify a different --attn-module."
                )
        
        # Register hook on qkv layer to capture Q, K, V
        if hasattr(attn_module, 'qkv'):
            qkv_handle = attn_module.qkv.register_forward_hook(qkv_hook)
            self._handles.append(qkv_handle)
        
        # Register main hook on attention module forward
        handle = attn_module.register_forward_hook(attn_forward_hook)
        self._handles.append(handle)

    def register_activations_with_grads(self, module: torch.nn.Module, key: str) -> None:
        """Capture activations and their gradients for Grad-CAM style saliency."""

        def forward_hook(_module, _inputs, output):
            self._cache[f"{key}_activations"] = HookRecord(key=f"{key}_activations", tensor=output)

            def grad_hook(grad):
                self._cache[f"{key}_grads"] = HookRecord(key=f"{key}_grads", tensor=grad)
                return grad

            output.register_hook(grad_hook)

        handle = module.register_forward_hook(forward_hook)
        self._handles.append(handle)

    def get(self, key: str) -> Optional[torch.Tensor]:
        record = self._cache.get(key)
        return record.tensor if record else None

    def clear(self) -> None:
        self._cache.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._cache.clear()


def aggregate_attention_heads(attn: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Aggregate attention heads into a single matrix."""

    if attn.ndim != 4:
        raise ValueError(f"Expected attention tensor with 4 dims, got {attn.shape}")
    if mode == "mean":
        return attn.mean(dim=1)
    if mode == "max":
        return attn.max(dim=1).values
    if mode == "sum":
        return attn.sum(dim=1)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def cls_to_patch_attention(
    attn_matrix: torch.Tensor,
    cls_index: int = 0,
    exclude_cls: bool = True,
    patch_token_count: Optional[int] = None,
) -> torch.Tensor:
    """
    Extract the attention from CLS token to vision patch tokens.

    Args:
        attn_matrix: Attention matrix of shape [B, N, N].
        cls_index: Index of CLS token in the sequence (usually 0).
        exclude_cls: Whether to drop the CLS-to-CLS entry.
        patch_token_count: Number of *true* vision patch tokens.
            If provided, we keep at most this many entries after `exclude_cls`.

    Returns:
        CLS attention vector of shape [B, num_tokens_kept].
    """
    if attn_matrix.ndim != 3:
        raise ValueError(f"Expected aggregated attention with 3 dims, got {attn_matrix.shape}")
    
    # Take the attention row for the CLS token: [B, N]
    cls_vector = attn_matrix[:, cls_index, :]
    
    # Optionally drop CLS->CLS entry so that we only see other tokens
    if exclude_cls:
        cls_vector = cls_vector[:, 1:]  # [B, N-1]
    
    # If we know how many patch tokens exist, drop any extra special tokens
    if (patch_token_count is not None) and (cls_vector.shape[-1] > patch_token_count):
        before_truncate = cls_vector.shape[-1]
        cls_vector = cls_vector[:, :patch_token_count]  # [B, num_patches]
        print(f"[SAL] Token truncation: {before_truncate} -> {patch_token_count} "
              f"(removed {before_truncate - patch_token_count} non-patch tokens)")
    
    return cls_vector  # [B, num_tokens_kept]


def attention_vector_to_grid(
    attn_vector: torch.Tensor,
    patch_token_count: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Reshape a 1D attention vector into a 2D patch grid.

    Args:
        attn_vector: Attention over tokens, shape [B, T].
        patch_token_count: If provided, only the first `patch_token_count`
            entries are used, and the grid size is inferred from it.

    Returns:
        grid: [B, H, W] attention map.
        grid_size: H == W if possible.
    """
    # If caller knows the true number of patches, respect it first
    if (patch_token_count is not None) and (attn_vector.shape[-1] >= patch_token_count):
        attn_vector = attn_vector[..., :patch_token_count]
        token_count = patch_token_count
    else:
        token_count = attn_vector.shape[-1]
    
    grid_size = int(token_count ** 0.5)
    
    if grid_size * grid_size != token_count:
        # Fallback: keep previous heuristic, but it now only triggers when we don't know
        # patch_token_count or it's really not a perfect square
        # Common ViT patch grid sizes: 14×14=196, 16×16=256
        common_sizes = [196, 256, 400, 576]  # 14×14, 16×16, 20×20, 24×24
        
        # Find the largest square that fits
        valid_size = None
        for size in sorted(common_sizes, reverse=True):
            if size <= token_count:
                # Check if remaining tokens are reasonable (usually 1-10 special tokens)
                remaining = token_count - size
                if 0 <= remaining <= 10:
                    valid_size = size
                    break
        
        if valid_size is not None:
            # Extract only the patch tokens
            original_count = attn_vector.shape[-1]
            attn_vector = attn_vector[..., :valid_size]
            token_count = valid_size
            grid_size = int(token_count ** 0.5)
            print(f"[SAL] Info: Extracted {valid_size} patch tokens from {original_count} total tokens.")
        else:
            # Last resort: pad to nearest square
            original_count = token_count
            next_square = (grid_size + 1) ** 2
            padding_size = next_square - token_count
            padding = torch.zeros(*attn_vector.shape[:-1], padding_size, device=attn_vector.device, dtype=attn_vector.dtype)
            attn_vector = torch.cat([attn_vector, padding], dim=-1)
            token_count = next_square
            grid_size = int(token_count ** 0.5)
            print(f"[SAL] Warning: Token count {original_count} was padded to {token_count} for grid reshaping.")
    
    # Now safely reshape into a square grid
    bsz = attn_vector.shape[0]
    grid = attn_vector.view(bsz, grid_size, grid_size)
    return grid, grid_size


def compute_gradcam_weights(activations: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """Grad-CAM style channel weights."""

    if activations.shape != grads.shape:
        raise ValueError("Activations and gradients must share the same shape.")
    weights = grads.mean(dim=(2, 3), keepdim=True)
    saliency = torch.relu((weights * activations).sum(dim=1))
    return saliency

