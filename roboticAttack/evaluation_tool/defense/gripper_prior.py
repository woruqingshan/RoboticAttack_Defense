# -*- coding: utf-8 -*-
"""
GripperPrior (Step 0): Geometric prior for the gripper/task area from end-effector state.

Computes per-frame protection zones used to avoid treating the gripper as patch and to
constrain mask refinement. Input: eef_pos (e.g. from obs). Output: G_px (PatchBox in
pixel coords), G_grid (GridBox in attention grid). Used by PatchSelector for overlap
filtering and by verifier.refine_mask_with_constraints for tau_protect/tau_cover.
"""

from dataclasses import dataclass
from typing import Tuple
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
    
    # Offset compensation specific to the LIBERO camera
    offset_x: float = 0.0
    offset_y: float = 0.0

class GripperPrior:
    def __init__(self, config: GripperPriorConfig):
        self.config = config

    def _project_3d_to_2d(self, eef_pos: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int]:
        """
        MVP projection logic: projects 3D end-effector pose to 2D image plane.
        If the system can directly provide actual pixel coordinates, it can be computed externally
        and passed in, or this logic can be replaced.
        """
        # Assume eef_pos is [x, y, z]
        x, y, z = eef_pos[0], eef_pos[1], eef_pos[2]
        
        # Heuristic linear mapping (usually y is left-right, z is up-down, depending on LIBERO camera frame)
        # Parameters (fx, fy, cx, cy) can be fine-tuned based on the visualized bounding box
        u_float = self.config.cam_cx + (y * self.config.cam_fx) + self.config.offset_x
        v_float = self.config.cam_cy - (z * self.config.cam_fy) + self.config.offset_y
        
        u = int(round(u_float))
        v = int(round(v_float))
        
        # Clamp to image boundaries
        u = max(0, min(u, img_w - 1))
        v = max(0, min(v, img_h - 1))
        
        return u, v

    def compute(
        self, 
        eef_pos: np.ndarray, 
        image_shape: Tuple[int, int], 
        grid_shape: Tuple[int, int]
    ) -> Tuple[Tuple[int, int, int, int], GridBox]:
        """
        Computes the protection box for the gripper/task area in the image (G_px)
        and its mapping on the grid (G_grid). Returns pixel box as (x0, y0, x1, y1)
        to avoid circular import with anomaly_detector.PatchBox.

        Args:
            eef_pos: End-effector pose, a numpy array like [x, y, z, ...]
            image_shape: Original image shape (H, W), e.g., (256, 256)
            grid_shape: Attention grid shape (gH, gW), e.g., (16, 16)

        Returns:
            G_px: Pixel-level protection zone as (x0, y0, x1, y1)
            G_grid: Grid-level protection zone
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
        # Note: use floor/ceil to ensure the grid box completely covers the pixel box
        gx0 = int(np.floor(x0 * gw / img_w))
        gy0 = int(np.floor(y0 * gh / img_h))
        gx1 = int(np.ceil(x1 * gw / img_w))
        gy1 = int(np.ceil(y1 * gh / img_h))
        
        # Clamp to grid boundaries
        gx0 = max(0, min(gx0, gw - 1))
        gy0 = max(0, min(gy0, gh - 1))
        gx1 = max(0, min(gx1, gw))
        gy1 = max(0, min(gy1, gh))
        
        # Fix boundaries: if bounds are inverted or size is zero
        if gx1 <= gx0: gx1 = gx0 + 1
        if gy1 <= gy0: gy1 = gy0 + 1
        
        G_grid = GridBox(gx0=gx0, gy0=gy0, gx1=gx1, gy1=gy1)

        return g_px_tuple, G_grid
