"""Shared image / geometry alignment helpers for LIBERO policy space.

Raw camera frames and projected camera pixels do not start from the same convention:

- raw `sim.render()` images follow the OpenGL framebuffer orientation
- `project_points_from_world_to_camera()` returns clipped `(row, col)` image indices

Therefore image preprocessing and projected-point alignment must be handled by
separate transforms rather than reusing one global "rotate 180" rule.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_POLICY_IMAGE_ROTATE_180 = True


def normalize_resize_size(resize_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Normalize an image resize argument to (height, width)."""
    if isinstance(resize_size, int):
        return int(resize_size), int(resize_size)
    if isinstance(resize_size, tuple) and len(resize_size) == 2:
        return int(resize_size[0]), int(resize_size[1])
    raise ValueError(f"Unsupported resize_size={resize_size!r}")


def apply_policy_image_alignment(image: np.ndarray, rotate_180: bool = DEFAULT_POLICY_IMAGE_ROTATE_180) -> np.ndarray:
    """Apply the same geometric transform used by policy-image preprocessing."""
    if not rotate_180:
        return image
    return image[::-1, ::-1]


def get_projected_point_alignment_flips(policy_image_rotate_180: bool) -> Tuple[bool, bool]:
    """Return (flip_row, flip_col) for projected camera pixels.

    `project_points_from_world_to_camera()` already produces upright `(row, col)`
    image indices, while raw RGB observations are still in OpenGL framebuffer
    orientation. As a result:

    - if the policy image keeps the raw image unchanged, projected points need a
      vertical flip to match it
    - if the policy image rotates the raw frame by 180 degrees, the vertical
      discrepancy is cancelled and only the horizontal mirror remains
    """
    flip_row = not bool(policy_image_rotate_180)
    flip_col = bool(policy_image_rotate_180)
    return flip_row, flip_col


def map_projected_pixel_rc_to_policy_rc(
    point_rc: Tuple[int, int],
    render_hw: Tuple[int, int],
    policy_hw: Tuple[int, int],
    policy_image_rotate_180: bool = DEFAULT_POLICY_IMAGE_ROTATE_180,
) -> Tuple[int, int]:
    """Map one projected `(row, col)` pixel into policy-image coordinates."""
    render_h, render_w = int(render_hw[0]), int(render_hw[1])
    policy_h, policy_w = int(policy_hw[0]), int(policy_hw[1])
    rr = int(point_rc[0])
    cc = int(point_rc[1])
    flip_row, flip_col = get_projected_point_alignment_flips(policy_image_rotate_180)
    if flip_row:
        rr = render_h - 1 - rr
    if flip_col:
        cc = render_w - 1 - cc
    rr_float = ((rr + 0.5) * policy_h / max(render_h, 1)) - 0.5
    cc_float = ((cc + 0.5) * policy_w / max(render_w, 1)) - 0.5
    rr_out = int(np.clip(np.round(rr_float), 0, max(policy_h - 1, 0)))
    cc_out = int(np.clip(np.round(cc_float), 0, max(policy_w - 1, 0)))
    return rr_out, cc_out


def map_projected_points_rc_to_policy_rc(
    points_rc: Sequence[Tuple[int, int]],
    render_hw: Tuple[int, int],
    policy_hw: Tuple[int, int],
    policy_image_rotate_180: bool = DEFAULT_POLICY_IMAGE_ROTATE_180,
) -> List[Tuple[int, int]]:
    """Map projected `(row, col)` points into policy-image coordinates."""
    return [
        map_projected_pixel_rc_to_policy_rc(
            point_rc,
            render_hw,
            policy_hw,
            policy_image_rotate_180=policy_image_rotate_180,
        )
        for point_rc in points_rc
    ]


def points_rc_to_xyxy_box(points_rc: Sequence[Tuple[int, int]], hw: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Convert (row, col) points into an xyxy box in the same image space."""
    if not points_rc:
        return None
    rows = np.array([int(p[0]) for p in points_rc], dtype=np.int32)
    cols = np.array([int(p[1]) for p in points_rc], dtype=np.int32)
    h, w = int(hw[0]), int(hw[1])
    y0 = int(np.clip(rows.min(), 0, max(h - 1, 0)))
    y1 = int(np.clip(rows.max() + 1, 0, h))
    x0 = int(np.clip(cols.min(), 0, max(w - 1, 0)))
    x1 = int(np.clip(cols.max() + 1, 0, w))
    return (x0, y0, x1, y1)
