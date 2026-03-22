# -*- coding: utf-8 -*-
"""
GripperPrior (Step 0): Geometric prior for the gripper/task area from end-effector state.

Computes per-frame protection zones used to avoid treating the gripper as patch and to
constrain mask refinement. Input: eef_pos (e.g. from obs). Output: G_px (PatchBox in
pixel coords), G_grid (GridBox in attention grid). Used by PatchSelector for overlap
filtering and by verifier.refine_mask_with_constraints for tau_protect/tau_cover.

Image coordinate convention: (0,0) at top-left, x increases rightward, y increases
downward (row index). So "vertical" arm means arm base is above the gripper in the
image (smaller y); we extend G_px upward by decreasing y0 to get ArmRegion.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .temporal import GridBox

@dataclass
class GripperPriorConfig:
    radius_px: int = 40          # Radius (in pixels), generated protection box size is 2*radius x 2*radius
    
    # Projection parameters (Plan B: If only 3D pose is available, use camera parameters for projection)
    # MVP version provides simple linear mapping, allowing tuning and running without strict intrinsics/extrinsics
    cam_fx: float = 100.0        # Focal length / scaling factor X
    cam_fy: float = 100.0        # Focal length / scaling factor Y
    cam_cx: float = 128.0        # Image center X (based on 256x256 image)
    cam_cy: float = 128.0        # Image center Y
    
    # Offset compensation specific to the LIBERO camera (tune so green box centers on gripper)
    offset_x: float = 0.0
    offset_y: float = 0.0

    # Which world axis drives image u (horizontal) and v (vertical); sign in [-1, 1]
    # LIBERO camera may have u from x or y; try proj_u_axis="x" if green box is horizontally off
    proj_u_axis: str = "y"   # "x" | "y" -> use eef_pos[0] or eef_pos[1] for u
    proj_v_axis: str = "z"   # "z" | "y" -> use eef_pos[2] or eef_pos[1] for v
    sign_u: int = -1         # -1 or 1: u = cam_cx + sign_u * (val_u * cam_fx) + offset_x
    sign_v: int = -1         # -1 or 1: v = cam_cy + sign_v * (val_v * cam_fy) + offset_y

    # Arm region: extend G_px toward arm base to define "arm" zone (avoid misjudging arm as patch).
    # Image coords: (0,0)=top-left, x=right, y=down. "vertical" = arm above gripper (smaller y).
    arm_orientation: str = "vertical"   # "vertical" | "horizontal"
    arm_extend_px: int = 0               # Pixels to extend toward arm base; 0 = disabled
    arm_extend_ortho_px: int = 20        # Perpendicular extension (widen arm band)

class GripperPrior:
    def __init__(self, config: GripperPriorConfig):
        self.config = config

    def _project_3d_to_2d(self, eef_pos: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int]:
        """
        Heuristic projection: map eef_pos [x,y,z] to image (u,v). Axis and sign are
        configurable so the green box can be tuned to center on the gripper (try
        proj_u_axis="x" or sign_u/sign_v if the box is offset).
        """
        x, y, z = eef_pos[0], eef_pos[1], eef_pos[2]
        val_u = x if self.config.proj_u_axis == "x" else y
        val_v = z if self.config.proj_v_axis == "z" else y
        su = int(self.config.sign_u) if self.config.sign_u in (-1, 1) else -1
        sv = int(self.config.sign_v) if self.config.sign_v in (-1, 1) else -1
        u_float = self.config.cam_cx + su * (val_u * self.config.cam_fx) + self.config.offset_x
        v_float = self.config.cam_cy + sv * (val_v * self.config.cam_fy) + self.config.offset_y
        u = int(round(u_float))
        v = int(round(v_float))
        u = max(0, min(u, img_w - 1))
        v = max(0, min(v, img_h - 1))
        return u, v

    def _pixel_box_to_grid(
        self,
        x0: int, y0: int, x1: int, y1: int,
        img_w: int, img_h: int,
        gw: int, gh: int,
    ) -> GridBox:
        """Map pixel box (x0,y0,x1,y1) to grid-level GridBox (same convention as G_grid)."""
        gx0 = int(np.floor(x0 * gw / img_w))
        gy0 = int(np.floor(y0 * gh / img_h))
        gx1 = int(np.ceil(x1 * gw / img_w))
        gy1 = int(np.ceil(y1 * gh / img_h))
        gx0 = max(0, min(gx0, gw - 1))
        gy0 = max(0, min(gy0, gh - 1))
        gx1 = max(0, min(gx1, gw))
        gy1 = max(0, min(gy1, gh))
        if gx1 <= gx0:
            gx1 = gx0 + 1
        if gy1 <= gy0:
            gy1 = gy0 + 1
        return GridBox(gx0=gx0, gy0=gy0, gx1=gx1, gy1=gy1)

    def compute(
        self, 
        eef_pos: np.ndarray, 
        image_shape: Tuple[int, int], 
        grid_shape: Tuple[int, int]
    ) -> Tuple[Tuple[int, int, int, int], GridBox, Optional[Tuple[int, int, int, int]], Optional[GridBox]]:
        """
        Computes the protection box for the gripper/task area (G_px, G_grid) and
        optionally the arm region (ArmRegion) extended toward the arm base.

        Image coords: (0,0)=top-left, x right, y down. For "vertical" arm, arm base
        is above the gripper (smaller y), so we extend G_px upward (decrease y0).

        Args:
            eef_pos: End-effector pose, a numpy array like [x, y, z, ...]
            image_shape: Original image shape (H, W), e.g., (256, 256)
            grid_shape: Attention grid shape (gH, gW), e.g., (16, 16)

        Returns:
            G_px: Pixel-level gripper zone as (x0, y0, x1, y1)
            G_grid: Grid-level gripper zone
            arm_region_px: Pixel-level arm zone (x0,y0,x1,y1) or None if arm_extend_px==0
            arm_region_grid: Grid-level arm zone or None
        """
        img_h, img_w = image_shape
        gh, gw = grid_shape

        # 1. Compute pixel coordinates (u, v) of the gripper center in the image
        u, v = self._project_3d_to_2d(eef_pos, img_w, img_h)

        # 2. Construct pixel-level protection zone G_px as (x0, y0, x1, y1)
        r = self.config.radius_px
        x0 = max(0, u - r)
        y0 = max(0, v - r)
        x1 = min(img_w, u + r)
        y1 = min(img_h, v + r)
        g_px_tuple = (x0, y0, x1, y1)

        # 3. Map to grid-level protection zone G_grid (GridBox)
        G_grid = self._pixel_box_to_grid(x0, y0, x1, y1, img_w, img_h, gw, gh)

        # 4. Optionally compute arm region (extend toward arm base)
        arm_region_px: Optional[Tuple[int, int, int, int]] = None
        arm_region_grid: Optional[GridBox] = None
        if self.config.arm_extend_px > 0:
            ext = self.config.arm_extend_px
            ortho = self.config.arm_extend_ortho_px
            if self.config.arm_orientation == "vertical":
                # Arm above gripper: extend upward (smaller y)
                ax0 = max(0, x0 - ortho)
                ay0 = max(0, y0 - ext)
                ax1 = min(img_w, x1 + ortho)
                ay1 = y1  # bottom stays at G_px bottom
            else:
                # Horizontal: extend left and right (x)
                ax0 = max(0, x0 - ext)
                ay0 = max(0, y0 - ortho)
                ax1 = min(img_w, x1 + ext)
                ay1 = min(img_h, y1 + ortho)
            arm_region_px = (ax0, ay0, ax1, ay1)
            arm_region_grid = self._pixel_box_to_grid(ax0, ay0, ax1, ay1, img_w, img_h, gw, gh)

        return g_px_tuple, G_grid, arm_region_px, arm_region_grid
