# -*- coding: utf-8 -*-
"""Gripper prior built from true simulator geometry and camera projection."""

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
from .temporal import GridBox


DEFAULT_GRIPPER_PRIOR_CONFIG = {
    "enabled": True,
    "site_names": ["grip_site", "ft_frame"],
    "body_names": ["right_hand", "right_gripper", "eef", "leftfinger", "rightfinger", "finger_joint1_tip", "finger_joint2_tip"],
    "segment_point_pairs": [
        ("right_hand", "ft_frame"),
        ("ft_frame", "grip_site"),
        ("grip_site", "finger_joint1_tip"),
        ("grip_site", "finger_joint2_tip"),
    ],
    "name_prefixes": ["", "robot0_", "Panda0_", "Panda_"],
    "camera_name": "agentview",
    "point_radius_px": 5,
    "core_thickness_px": 10,
    "guard_scale": 2.0,
    "segment_core_thicknesses": [],
    "segment_guard_scales": [],
    "min_valid_points": 1,
}


@dataclass
class GripperPriorConfig:
    """Configuration for gripper geometry extraction."""

    enabled: bool = bool(DEFAULT_GRIPPER_PRIOR_CONFIG["enabled"])
    site_names: List[str] = field(default_factory=lambda: list(DEFAULT_GRIPPER_PRIOR_CONFIG["site_names"]))
    body_names: List[str] = field(default_factory=lambda: list(DEFAULT_GRIPPER_PRIOR_CONFIG["body_names"]))
    segment_point_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: list(DEFAULT_GRIPPER_PRIOR_CONFIG["segment_point_pairs"])
    )
    name_prefixes: List[str] = field(default_factory=lambda: list(DEFAULT_GRIPPER_PRIOR_CONFIG["name_prefixes"]))
    camera_name: str = str(DEFAULT_GRIPPER_PRIOR_CONFIG["camera_name"])
    point_radius_px: int = int(DEFAULT_GRIPPER_PRIOR_CONFIG["point_radius_px"])
    core_thickness_px: int = int(DEFAULT_GRIPPER_PRIOR_CONFIG["core_thickness_px"])
    guard_scale: float = float(DEFAULT_GRIPPER_PRIOR_CONFIG["guard_scale"])
    segment_core_thicknesses: List[int] = field(
        default_factory=lambda: list(DEFAULT_GRIPPER_PRIOR_CONFIG["segment_core_thicknesses"])
    )
    segment_guard_scales: List[float] = field(
        default_factory=lambda: list(DEFAULT_GRIPPER_PRIOR_CONFIG["segment_guard_scales"])
    )
    min_valid_points: int = int(DEFAULT_GRIPPER_PRIOR_CONFIG["min_valid_points"])


@dataclass
class GripperSegment2D:
    """Explicit projected geometry for one gripper segment."""

    name: str
    start_point_name: str
    end_point_name: str
    start_point_policy_px: Tuple[int, int]
    end_point_policy_px: Tuple[int, int]
    core_thickness_px: int
    guard_thickness_px: int
    core_quad_xy: List[Tuple[int, int]]
    guard_quad_xy: List[Tuple[int, int]]
    core_box_px: Optional[Tuple[int, int, int, int]]
    guard_box_px: Optional[Tuple[int, int, int, int]]


@dataclass
class GripperPriorResult:
    """Projected gripper geometry in policy-image coordinates."""

    valid: bool
    point_keys_used: List[str]
    point_names_used: List[str]
    gripper_points_world: List[Tuple[float, float, float]]
    gripper_points_render_px: List[Tuple[int, int]]  # (row, col) from camera projection before policy alignment
    gripper_points_policy_px: List[Tuple[int, int]]  # (row, col) in aligned policy-image space
    gripper_segments_2d: List[GripperSegment2D]
    per_segment_core_masks_px: List[np.ndarray]
    per_segment_guard_masks_px: List[np.ndarray]
    gripper_core_mask_px: np.ndarray
    gripper_guard_mask_px: np.ndarray
    gripper_core_grid_mask: np.ndarray
    gripper_guard_grid_mask: np.ndarray
    gripper_core_box_px: Optional[Tuple[int, int, int, int]]
    gripper_guard_box_px: Optional[Tuple[int, int, int, int]]
    gripper_core_grid: Optional[GridBox]
    gripper_guard_grid: Optional[GridBox]


class GripperPrior:
    def __init__(self, config: GripperPriorConfig):
        self.config = config

    def _lookup_site_point(self, sim: Any, name: str) -> Optional[Tuple[str, np.ndarray]]:
        """Resolve a site world point using candidate prefixes."""
        for prefix in self.config.name_prefixes:
            full_name = f"{prefix}{name}"
            try:
                site_id = sim.model.site_name2id(full_name)
                point = np.array(sim.data.site_xpos[site_id], dtype=np.float32)
                return full_name, point
            except Exception:
                continue
        return None

    def _lookup_body_point(self, sim: Any, name: str) -> Optional[Tuple[str, np.ndarray]]:
        """Resolve a body world point using candidate prefixes."""
        for prefix in self.config.name_prefixes:
            full_name = f"{prefix}{name}"
            try:
                body_id = sim.model.body_name2id(full_name)
                point = np.array(sim.data.body_xpos[body_id], dtype=np.float32)
                return full_name, point
            except Exception:
                continue
        return None

    def _collect_world_points(self, sim: Any) -> Tuple[List[str], List[str], List[np.ndarray]]:
        """Collect gripper points with site-first priority and body fallback."""
        point_keys_used: List[str] = []
        names_used: List[str] = []
        points: List[np.ndarray] = []
        seen_names = set()

        for site_name in self.config.site_names:
            resolved = self._lookup_site_point(sim, site_name)
            if resolved is None:
                continue
            resolved_name, point = resolved
            if resolved_name in seen_names:
                continue
            seen_names.add(resolved_name)
            point_keys_used.append(site_name)
            names_used.append(resolved_name)
            points.append(point)

        for body_name in self.config.body_names:
            resolved = self._lookup_body_point(sim, body_name)
            if resolved is None:
                continue
            resolved_name, point = resolved
            if resolved_name in seen_names:
                continue
            seen_names.add(resolved_name)
            point_keys_used.append(body_name)
            names_used.append(resolved_name)
            points.append(point)

        return point_keys_used, names_used, points

    @staticmethod
    def _mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Convert a binary mask to an xyxy pixel box."""
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))

    @staticmethod
    def _point_mask(hw: Tuple[int, int], point_rc: Tuple[int, int], radius_px: float) -> np.ndarray:
        """Rasterize a circular marker around one projected gripper point."""
        height, width = int(hw[0]), int(hw[1])
        yy, xx = np.mgrid[0:height, 0:width]
        y0, x0 = float(point_rc[0]), float(point_rc[1])
        radius = max(float(radius_px), 1.0)
        dist_sq = (xx - x0) ** 2 + (yy - y0) ** 2
        return dist_sq <= (radius * radius)

    @staticmethod
    def _segment_mask(
        height: int,
        width: int,
        p0_rc: Tuple[int, int],
        p1_rc: Tuple[int, int],
        thickness_px: float,
    ) -> np.ndarray:
        """Rasterize a thick segment as a boolean mask."""
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

    @staticmethod
    def _segment_quad_xy(
        p0_rc: Tuple[int, int],
        p1_rc: Tuple[int, int],
        thickness_px: float,
        hw: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Build an oriented rectangle around one projected gripper segment."""
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

    def _resolve_segment_core_thickness(self, seg_idx: int) -> int:
        """Use per-segment override when provided, otherwise fall back to the global default."""
        if 0 <= seg_idx < len(self.config.segment_core_thicknesses):
            return int(max(1, self.config.segment_core_thicknesses[seg_idx]))
        return int(max(1, self.config.core_thickness_px))

    def _resolve_segment_guard_thickness(self, seg_idx: int, core_thickness: int) -> int:
        """Use per-segment guard scale override when provided, otherwise fall back to the global default."""
        guard_scale = float(self.config.guard_scale)
        if 0 <= seg_idx < len(self.config.segment_guard_scales):
            guard_scale = float(self.config.segment_guard_scales[seg_idx])
        return int(max(core_thickness, round(core_thickness * max(guard_scale, 1.0))))

    def _build_segment_geometry(
        self,
        point_keys_used: Sequence[str],
        points_rc: Sequence[Tuple[int, int]],
        hw: Tuple[int, int],
    ) -> Tuple[List[GripperSegment2D], List[np.ndarray], List[np.ndarray]]:
        """Build explicit gripper segment objects and their masks."""
        segments: List[GripperSegment2D] = []
        per_segment_core_masks: List[np.ndarray] = []
        per_segment_guard_masks: List[np.ndarray] = []
        point_index = {name: idx for idx, name in enumerate(point_keys_used)}
        point_radius = float(max(1, int(self.config.point_radius_px)))
        segment_pairs = list(self.config.segment_point_pairs)
        if not segment_pairs and len(points_rc) >= 2:
            segment_pairs = [(point_keys_used[idx], point_keys_used[idx + 1]) for idx in range(len(points_rc) - 1)]

        for seg_idx, (start_name, end_name) in enumerate(segment_pairs):
            if start_name not in point_index or end_name not in point_index:
                continue
            p0 = points_rc[point_index[start_name]]
            p1 = points_rc[point_index[end_name]]
            core_thickness = self._resolve_segment_core_thickness(seg_idx)
            guard_thickness = self._resolve_segment_guard_thickness(seg_idx, core_thickness)

            core_mask = self._segment_mask(int(hw[0]), int(hw[1]), p0, p1, float(core_thickness))
            guard_mask = self._segment_mask(int(hw[0]), int(hw[1]), p0, p1, float(guard_thickness))
            core_mask |= self._point_mask(hw, p0, point_radius)
            core_mask |= self._point_mask(hw, p1, point_radius)
            guard_mask |= self._point_mask(hw, p0, point_radius)
            guard_mask |= self._point_mask(hw, p1, point_radius)

            per_segment_core_masks.append(core_mask)
            per_segment_guard_masks.append(guard_mask)
            segments.append(
                GripperSegment2D(
                    name=f"gripper_segment_{seg_idx}",
                    start_point_name=start_name,
                    end_point_name=end_name,
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

        return segments, per_segment_core_masks, per_segment_guard_masks

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

    @staticmethod
    def _pixel_box_to_grid(
        box_xyxy: Tuple[int, int, int, int],
        img_hw: Tuple[int, int],
        grid_shape: Tuple[int, int],
    ) -> GridBox:
        """Map a pixel xyxy box to a grid-level GridBox."""
        x0, y0, x1, y1 = box_xyxy
        img_h, img_w = int(img_hw[0]), int(img_hw[1])
        gh, gw = int(grid_shape[0]), int(grid_shape[1])
        gx0 = int(np.floor(x0 * gw / max(img_w, 1)))
        gy0 = int(np.floor(y0 * gh / max(img_h, 1)))
        gx1 = int(np.ceil(x1 * gw / max(img_w, 1)))
        gy1 = int(np.ceil(y1 * gh / max(img_h, 1)))
        gx0 = max(0, min(gx0, gw - 1))
        gy0 = max(0, min(gy0, gh - 1))
        gx1 = max(0, min(gx1, gw))
        gy1 = max(0, min(gy1, gh))
        if gx1 <= gx0:
            gx1 = min(gw, gx0 + 1)
        if gy1 <= gy0:
            gy1 = min(gh, gy0 + 1)
        return GridBox(gx0=gx0, gy0=gy0, gx1=gx1, gy1=gy1)

    def compute(self, geometry_ctx: Any, grid_shape: Tuple[int, int]) -> GripperPriorResult:
        """Project gripper points into the current policy image."""
        policy_hw = (
            int(getattr(geometry_ctx, "policy_hw", (0, 0))[0]) if geometry_ctx is not None else 0,
            int(getattr(geometry_ctx, "policy_hw", (0, 0))[1]) if geometry_ctx is not None else 0,
        )
        empty_mask = np.zeros(policy_hw, dtype=bool)
        empty_grid = np.zeros((int(grid_shape[0]), int(grid_shape[1])), dtype=bool)
        invalid = GripperPriorResult(
            valid=False,
            point_keys_used=[],
            point_names_used=[],
            gripper_points_world=[],
            gripper_points_render_px=[],
            gripper_points_policy_px=[],
            gripper_segments_2d=[],
            per_segment_core_masks_px=[],
            per_segment_guard_masks_px=[],
            gripper_core_mask_px=empty_mask,
            gripper_guard_mask_px=empty_mask.copy(),
            gripper_core_grid_mask=empty_grid,
            gripper_guard_grid_mask=empty_grid.copy(),
            gripper_core_box_px=None,
            gripper_guard_box_px=None,
            gripper_core_grid=None,
            gripper_guard_grid=None,
        )

        if not self.config.enabled or geometry_ctx is None or getattr(geometry_ctx, "sim", None) is None:
            return invalid

        point_keys_used, names_used, points_world_np = self._collect_world_points(geometry_ctx.sim)
        if len(points_world_np) < int(self.config.min_valid_points):
            return invalid

        policy_hw = (int(geometry_ctx.policy_hw[0]), int(geometry_ctx.policy_hw[1]))
        render_hw = (int(geometry_ctx.render_hw[0]), int(geometry_ctx.render_hw[1]))
        camera_name = str(getattr(geometry_ctx, "camera_name", "") or self.config.camera_name)

        try:
            world_to_camera = get_camera_transform_matrix(
                sim=geometry_ctx.sim,
                camera_name=camera_name,
                camera_height=render_hw[0],
                camera_width=render_hw[1],
            )
        except Exception:
            return invalid

        world_points = np.stack(points_world_np, axis=0).astype(np.float32)
        render_pixels = project_points_from_world_to_camera(
            points=world_points,
            world_to_camera_transform=world_to_camera,
            camera_height=render_hw[0],
            camera_width=render_hw[1],
        )

        render_points_rc = [(int(rc[0]), int(rc[1])) for rc in render_pixels]
        policy_points_rc = map_projected_points_rc_to_policy_rc(
            render_points_rc,
            render_hw=render_hw,
            policy_hw=policy_hw,
            policy_image_rotate_180=bool(
                getattr(geometry_ctx, "policy_image_rotate_180", DEFAULT_POLICY_IMAGE_ROTATE_180)
            ),
        )

        segments_2d, per_segment_core_masks, per_segment_guard_masks = self._build_segment_geometry(
            point_keys_used,
            policy_points_rc,
            policy_hw,
        )

        core_mask = np.zeros(policy_hw, dtype=bool)
        guard_mask = np.zeros(policy_hw, dtype=bool)
        point_radius = float(max(1, int(self.config.point_radius_px)))
        for point_rc in policy_points_rc:
            core_mask |= self._point_mask(policy_hw, point_rc, point_radius)
            guard_mask |= self._point_mask(policy_hw, point_rc, point_radius)
        for seg_mask in per_segment_core_masks:
            core_mask |= seg_mask
        for seg_mask in per_segment_guard_masks:
            guard_mask |= seg_mask

        core_box = self._mask_to_box(core_mask)
        guard_box = self._mask_to_box(guard_mask)
        core_grid_mask = self._mask_to_grid_mask(core_mask, grid_shape)
        guard_grid_mask = self._mask_to_grid_mask(guard_mask, grid_shape)
        core_grid = None if core_box is None else self._pixel_box_to_grid(core_box, policy_hw, grid_shape)
        guard_grid = None if guard_box is None else self._pixel_box_to_grid(guard_box, policy_hw, grid_shape)

        return GripperPriorResult(
            valid=(guard_box is not None and guard_grid is not None),
            point_keys_used=point_keys_used,
            point_names_used=names_used,
            gripper_points_world=[tuple(map(float, p.tolist())) for p in world_points],
            gripper_points_render_px=render_points_rc,
            gripper_points_policy_px=policy_points_rc,
            gripper_segments_2d=segments_2d,
            per_segment_core_masks_px=per_segment_core_masks,
            per_segment_guard_masks_px=per_segment_guard_masks,
            gripper_core_mask_px=core_mask,
            gripper_guard_mask_px=guard_mask,
            gripper_core_grid_mask=core_grid_mask,
            gripper_guard_grid_mask=guard_grid_mask,
            gripper_core_box_px=core_box,
            gripper_guard_box_px=guard_box,
            gripper_core_grid=core_grid,
            gripper_guard_grid=guard_grid,
        )
