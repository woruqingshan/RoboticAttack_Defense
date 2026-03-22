"""Arm skeleton prior built from simulator link poses and true camera projection.

This module provides an explicit joint-and-link representation:

    sim body / site world poses -> agentview projection -> joint points
    -> per-link oriented corridors -> ArmCore / ArmGuard masks

The exported result keeps backward-compatible aggregate masks / boxes while adding
per-link segment geometry that can be tuned independently in later steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)

from .geometry_alignment import (
    DEFAULT_POLICY_IMAGE_ROTATE_180,
    map_projected_points_rc_to_policy_rc,
)


DEFAULT_ARM_SKELETON_PRIOR_CONFIG = {
    "enabled": True,
    "geometry_source": "body",
    "body_names": ["base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "right_hand"],
    "site_names": [],
    "name_prefixes": ["", "robot0_", "Panda0_", "Panda_"],
    "camera_name": "agentview",
    "joint_radius_px": 4,
    "core_thickness_px": 14,
    "guard_scale": 1.8,
    "link_core_thicknesses": [],
    "link_guard_scales": [],
    "min_valid_points": 3,
}


@dataclass(frozen=True)
class GeometryRuntimeContext:
    """Runtime geometry context shared by rollout and defense."""

    sim: Any
    camera_name: str
    render_hw: Tuple[int, int]
    policy_hw: Tuple[int, int]
    policy_image_rotate_180: bool = DEFAULT_POLICY_IMAGE_ROTATE_180


@dataclass
class ArmSkeletonPriorConfig:
    """Configuration for arm skeleton extraction and rasterization."""

    enabled: bool = bool(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["enabled"])
    geometry_source: str = str(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["geometry_source"])  # "body" | "site"
    body_names: List[str] = field(
        default_factory=lambda: list(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["body_names"])
    )
    site_names: List[str] = field(default_factory=lambda: list(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["site_names"]))
    name_prefixes: List[str] = field(default_factory=lambda: list(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["name_prefixes"]))
    camera_name: str = str(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["camera_name"])
    joint_radius_px: int = int(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["joint_radius_px"])
    core_thickness_px: int = int(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["core_thickness_px"])
    guard_scale: float = float(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["guard_scale"])
    link_core_thicknesses: List[int] = field(
        default_factory=lambda: list(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["link_core_thicknesses"])
    )
    link_guard_scales: List[float] = field(
        default_factory=lambda: list(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["link_guard_scales"])
    )
    min_valid_points: int = int(DEFAULT_ARM_SKELETON_PRIOR_CONFIG["min_valid_points"])


@dataclass
class LinkSegment2D:
    """Explicit projected geometry for one arm link segment."""

    name: str
    start_joint_name: str
    end_joint_name: str
    start_point_policy_px: Tuple[int, int]
    end_point_policy_px: Tuple[int, int]
    core_thickness_px: int
    guard_thickness_px: int
    core_quad_xy: List[Tuple[int, int]]
    guard_quad_xy: List[Tuple[int, int]]
    core_box_px: Optional[Tuple[int, int, int, int]]
    guard_box_px: Optional[Tuple[int, int, int, int]]


@dataclass
class ArmSkeletonResult:
    """Projected arm skeleton and rasterized protection regions."""

    valid: bool
    joint_names_used: List[str]
    joint_points_world: List[Tuple[float, float, float]]
    joint_points_render_px: List[Tuple[int, int]]  # (row, col) from camera projection before policy alignment
    joint_points_policy_px: List[Tuple[int, int]]  # (row, col) in aligned policy-image space
    link_segments_2d: List[LinkSegment2D]
    per_link_core_masks_px: List[np.ndarray]
    per_link_guard_masks_px: List[np.ndarray]
    arm_core_mask_px: np.ndarray
    arm_guard_mask_px: np.ndarray
    arm_core_grid_mask: np.ndarray
    arm_guard_grid_mask: np.ndarray
    arm_core_box_px: Optional[Tuple[int, int, int, int]]
    arm_guard_box_px: Optional[Tuple[int, int, int, int]]


class ArmSkeletonPrior:
    """Generate arm protection masks from simulator geometry."""

    def __init__(self, config: ArmSkeletonPriorConfig):
        self.config = config

    def _lookup_body_or_site_point(self, sim: Any, name: str) -> Optional[Tuple[str, np.ndarray]]:
        """Resolve a body / site world point using candidate prefixes."""
        if self.config.geometry_source == "site":
            for prefix in self.config.name_prefixes:
                full_name = f"{prefix}{name}"
                try:
                    site_id = sim.model.site_name2id(full_name)
                    point = np.array(sim.data.site_xpos[site_id], dtype=np.float32)
                    return full_name, point
                except Exception:
                    continue
            return None

        for prefix in self.config.name_prefixes:
            full_name = f"{prefix}{name}"
            try:
                body_id = sim.model.body_name2id(full_name)
                point = np.array(sim.data.body_xpos[body_id], dtype=np.float32)
                return full_name, point
            except Exception:
                continue
        return None

    def _collect_world_points(self, sim: Any) -> Tuple[List[str], List[np.ndarray]]:
        """Collect ordered world-space keypoints along the arm chain."""
        names: Sequence[str] = self.config.site_names if self.config.geometry_source == "site" else self.config.body_names
        names_used: List[str] = []
        points: List[np.ndarray] = []
        for name in names:
            resolved = self._lookup_body_or_site_point(sim, name)
            if resolved is not None:
                resolved_name, point = resolved
                names_used.append(resolved_name)
                points.append(point)
        return names_used, points

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Convert a binary mask to an xyxy pixel box."""
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))

    @staticmethod
    def _segment_mask(
        height: int,
        width: int,
        p0_rc: Tuple[int, int],
        p1_rc: Tuple[int, int],
        thickness_px: float,
    ) -> np.ndarray:
        """Rasterize a thick line segment as a boolean mask."""
        radius = max(float(thickness_px) * 0.5, 1.0)
        y0, x0 = float(p0_rc[0]), float(p0_rc[1])
        y1, x1 = float(p1_rc[0]), float(p1_rc[1])
        yy, xx = np.mgrid[0:height, 0:width]
        vx = x1 - x0
        vy = y1 - y0
        seg_len_sq = (vx * vx) + (vy * vy)

        if seg_len_sq <= 1e-8:
            dist_sq = (xx - x0) ** 2 + (yy - y0) ** 2
            return dist_sq <= (radius * radius)

        t = ((xx - x0) * vx + (yy - y0) * vy) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        proj_x = x0 + t * vx
        proj_y = y0 + t * vy
        dist_sq = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
        return dist_sq <= (radius * radius)

    def _point_mask(self, hw: Tuple[int, int], point_rc: Tuple[int, int], radius_px: float) -> np.ndarray:
        """Rasterize a circular joint marker as a boolean mask."""
        height, width = int(hw[0]), int(hw[1])
        return self._segment_mask(height, width, point_rc, point_rc, radius_px * 2.0)

    @staticmethod
    def _segment_quad_xy(
        p0_rc: Tuple[int, int],
        p1_rc: Tuple[int, int],
        thickness_px: float,
        hw: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Build an oriented rectangle around one joint-to-joint link segment."""
        height, width = int(hw[0]), int(hw[1])
        y0, x0 = float(p0_rc[0]), float(p0_rc[1])
        y1, x1 = float(p1_rc[0]), float(p1_rc[1])
        vx = x1 - x0
        vy = y1 - y0
        seg_len = float(np.hypot(vx, vy))
        radius = max(float(thickness_px) * 0.5, 1.0)

        if seg_len <= 1e-8:
            corners = [
                (x0 - radius, y0 - radius),
                (x0 + radius, y0 - radius),
                (x0 + radius, y0 + radius),
                (x0 - radius, y0 + radius),
            ]
        else:
            px = -vy / seg_len
            py = vx / seg_len
            ox = px * radius
            oy = py * radius
            corners = [
                (x0 + ox, y0 + oy),
                (x0 - ox, y0 - oy),
                (x1 - ox, y1 - oy),
                (x1 + ox, y1 + oy),
            ]

        quad_xy: List[Tuple[int, int]] = []
        for x_float, y_float in corners:
            xx = int(np.clip(np.round(x_float), 0, max(width - 1, 0)))
            yy = int(np.clip(np.round(y_float), 0, max(height - 1, 0)))
            quad_xy.append((xx, yy))
        return quad_xy

    def _resolve_link_core_thickness(self, seg_idx: int) -> int:
        """Use per-link override when provided, otherwise fall back to the global default."""
        if 0 <= seg_idx < len(self.config.link_core_thicknesses):
            return int(max(1, self.config.link_core_thicknesses[seg_idx]))
        return int(max(1, self.config.core_thickness_px))

    def _resolve_link_guard_thickness(self, seg_idx: int, core_thickness: int) -> int:
        """Use per-link guard scale override when provided, otherwise fall back to the global default."""
        guard_scale = float(self.config.guard_scale)
        if 0 <= seg_idx < len(self.config.link_guard_scales):
            guard_scale = float(self.config.link_guard_scales[seg_idx])
        return int(max(core_thickness, round(core_thickness * max(guard_scale, 1.0))))

    def _build_segment_geometry(
        self,
        joint_names: Sequence[str],
        points_rc: Sequence[Tuple[int, int]],
        hw: Tuple[int, int],
    ) -> Tuple[List[LinkSegment2D], List[np.ndarray], List[np.ndarray]]:
        """Build explicit per-link segment objects and their masks."""
        segments: List[LinkSegment2D] = []
        per_link_core_masks: List[np.ndarray] = []
        per_link_guard_masks: List[np.ndarray] = []
        joint_radius = float(max(1, int(self.config.joint_radius_px)))

        if len(points_rc) < 2:
            return segments, per_link_core_masks, per_link_guard_masks

        for seg_idx, ((p0, p1), (name0, name1)) in enumerate(zip(zip(points_rc[:-1], points_rc[1:]), zip(joint_names[:-1], joint_names[1:]))):
            core_thickness = self._resolve_link_core_thickness(seg_idx)
            guard_thickness = self._resolve_link_guard_thickness(seg_idx, core_thickness)

            core_mask = self._segment_mask(int(hw[0]), int(hw[1]), p0, p1, float(core_thickness))
            guard_mask = self._segment_mask(int(hw[0]), int(hw[1]), p0, p1, float(guard_thickness))

            # Include circular joint regions so the segment-level mask remains connected at joints.
            core_mask |= self._point_mask(hw, p0, joint_radius)
            core_mask |= self._point_mask(hw, p1, joint_radius)
            guard_mask |= self._point_mask(hw, p0, joint_radius)
            guard_mask |= self._point_mask(hw, p1, joint_radius)

            per_link_core_masks.append(core_mask)
            per_link_guard_masks.append(guard_mask)
            segments.append(
                LinkSegment2D(
                    name=f"link_segment_{seg_idx}",
                    start_joint_name=name0,
                    end_joint_name=name1,
                    start_point_policy_px=p0,
                    end_point_policy_px=p1,
                    core_thickness_px=core_thickness,
                    guard_thickness_px=guard_thickness,
                    core_quad_xy=self._segment_quad_xy(p0, p1, float(core_thickness), hw),
                    guard_quad_xy=self._segment_quad_xy(p0, p1, float(guard_thickness), hw),
                    core_box_px=self._mask_to_box(core_mask),
                    guard_box_px=self._mask_to_box(guard_mask),
                )
            )

        return segments, per_link_core_masks, per_link_guard_masks

    @staticmethod
    def _mask_to_grid_mask(mask: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
        """Downsample a pixel mask to a boolean grid mask using per-cell max."""
        gh, gw = int(grid_shape[0]), int(grid_shape[1])
        h, w = int(mask.shape[0]), int(mask.shape[1])
        grid = np.zeros((gh, gw), dtype=bool)
        for gy in range(gh):
            y0 = int(np.floor(gy * h / gh))
            y1 = int(np.ceil((gy + 1) * h / gh))
            for gx in range(gw):
                x0 = int(np.floor(gx * w / gw))
                x1 = int(np.ceil((gx + 1) * w / gw))
                if y1 > y0 and x1 > x0 and bool(mask[y0:y1, x0:x1].any()):
                    grid[gy, gx] = True
        return grid

    def compute(
        self,
        ctx: GeometryRuntimeContext,
        grid_shape: Tuple[int, int],
    ) -> ArmSkeletonResult:
        """Build the arm skeleton prior for the current frame."""
        policy_hw = (int(ctx.policy_hw[0]), int(ctx.policy_hw[1]))
        empty_mask = np.zeros(policy_hw, dtype=bool)
        empty_grid = np.zeros((int(grid_shape[0]), int(grid_shape[1])), dtype=bool)
        invalid = ArmSkeletonResult(
            valid=False,
            joint_names_used=[],
            joint_points_world=[],
            joint_points_render_px=[],
            joint_points_policy_px=[],
            link_segments_2d=[],
            per_link_core_masks_px=[],
            per_link_guard_masks_px=[],
            arm_core_mask_px=empty_mask,
            arm_guard_mask_px=empty_mask.copy(),
            arm_core_grid_mask=empty_grid,
            arm_guard_grid_mask=empty_grid.copy(),
            arm_core_box_px=None,
            arm_guard_box_px=None,
        )

        if not self.config.enabled or ctx.sim is None:
            return invalid

        joint_names_used, points_world_np = self._collect_world_points(ctx.sim)
        if len(points_world_np) < int(self.config.min_valid_points):
            return invalid

        camera_name = str(ctx.camera_name or self.config.camera_name)
        try:
            world_to_camera = get_camera_transform_matrix(
                sim=ctx.sim,
                camera_name=camera_name,
                camera_height=int(ctx.render_hw[0]),
                camera_width=int(ctx.render_hw[1]),
            )
        except Exception:
            return invalid

        world_points = np.stack(points_world_np, axis=0).astype(np.float32)
        render_pixels = project_points_from_world_to_camera(
            points=world_points,
            world_to_camera_transform=world_to_camera,
            camera_height=int(ctx.render_hw[0]),
            camera_width=int(ctx.render_hw[1]),
        )

        render_points_rc = [(int(rc[0]), int(rc[1])) for rc in render_pixels]
        policy_points_rc = map_projected_points_rc_to_policy_rc(
            render_points_rc,
            render_hw=ctx.render_hw,
            policy_hw=ctx.policy_hw,
            policy_image_rotate_180=bool(ctx.policy_image_rotate_180),
        )

        link_segments_2d, per_link_core_masks, per_link_guard_masks = self._build_segment_geometry(
            joint_names_used,
            policy_points_rc,
            policy_hw,
        )
        arm_core_mask = np.zeros(policy_hw, dtype=bool)
        arm_guard_mask = np.zeros(policy_hw, dtype=bool)
        for link_mask in per_link_core_masks:
            arm_core_mask |= link_mask
        for link_mask in per_link_guard_masks:
            arm_guard_mask |= link_mask
        arm_core_grid = self._mask_to_grid_mask(arm_core_mask, grid_shape)
        arm_guard_grid = self._mask_to_grid_mask(arm_guard_mask, grid_shape)

        return ArmSkeletonResult(
            valid=True,
            joint_names_used=joint_names_used,
            joint_points_world=[tuple(map(float, p.tolist())) for p in world_points],
            joint_points_render_px=render_points_rc,
            joint_points_policy_px=policy_points_rc,
            link_segments_2d=link_segments_2d,
            per_link_core_masks_px=per_link_core_masks,
            per_link_guard_masks_px=per_link_guard_masks,
            arm_core_mask_px=arm_core_mask,
            arm_guard_mask_px=arm_guard_mask,
            arm_core_grid_mask=arm_core_grid,
            arm_guard_grid_mask=arm_guard_grid,
            arm_core_box_px=self._mask_to_box(arm_core_mask),
            arm_guard_box_px=self._mask_to_box(arm_guard_mask),
        )
