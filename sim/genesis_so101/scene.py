"""Minimal dual SO-101 handover scene for Genesis v1.3.1."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from radeon_oneloop.contracts import (
    CAMERA_KEYS,
    SO101_GRIPPER_MAX_RAD,
    SO101_GRIPPER_MIN_RAD,
    genesis_arm_to_lerobot,
    lerobot_arm_to_genesis,
    require_vector,
)

from .fetch_assets import fetch
from .gaussian_appearance import (
    SafeAppearanceBinding,
    composite_with_proxy_depth,
    composite_with_gaussian_depth,
    entity_segmentation_index,
)
from .handover_asset import (
    DEFAULT_COLLISION_MESH,
    DEFAULT_MESH,
    DEFAULT_TRELLIS2_FEM_PROXY,
    DEFAULT_TRELLIS2_FEM_PROXY_MANIFEST,
    DEFAULT_TRELLIS2_FEM_LIVE_SURFACE,
    DEFAULT_TRELLIS2_FEM_LIVE_VOLUME_MANIFEST,
    DEFAULT_TRELLIS2_MGPBD_DENSE_SURFACE,
    DEFAULT_TRELLIS2_MGPBD_DENSE_VOLUME,
    DEFAULT_TRELLIS2_MGPBD_DENSE_VOLUME_MANIFEST,
    DEFAULT_TRELLIS2_GRANULAR_SHELL,
    DEFAULT_TRELLIS2_GRANULAR_SHELL_MANIFEST,
    DEFAULT_TRELLIS2_PLUSH_COLLISION,
    DEFAULT_TRELLIS2_PLUSH_COLLISION_MANIFEST,
    add_rigid_visual_collision_urdf,
    add_rigid_proxy,
    load_spec,
)
from .fem_plush import FEMPlushConfig, FEMPlushObjectAdapter, FEMTetVisualBinding
from .granular_plush import StarTetVisualBinding
from .mgpbd_tet import (
    MGPBDTetConfig,
    MGPBDTetProvider,
    MGPBDTetSolver,
    make_mgpbd_plush_adapter_class,
    tetrahedralize_proxy,
)
from .plush_physics import (
    PlushObjectAdapter,
    PlushVisualBinding,
    XPBDPlushObjectAdapter,
    XPBDShellProvider,
    install_custom_vvert_rest_normal_transport,
)
from .xpbd_plush import XPBDPlushConfig, XPBDPlushSolver, build_plush_topology


HOME_ARM_ACTION = (0.0, -55.0, 70.0, 70.0, 0.0, 35.0)
HOME_ACTION = HOME_ARM_ACTION + HOME_ARM_ACTION
# The rigid benchmark home points the fingers steeply down and is unsuitable
# for initializing a finite-size held object above the table.  Keep that
# baseline unchanged; the plush demo starts from a higher, collision-free hold.
PLUSH_HOME_ARM_ACTION = (0.0, -40.0, 40.0, 40.0, 0.0, 35.0)
PLUSH_HOME_ACTION = PLUSH_HOME_ARM_ACTION + PLUSH_HOME_ARM_ACTION

ARM_BASE_SEPARATION_M = 0.40
RIGID_TABLE_TOP_Z = 0.410
PLUSH_TABLE_TOP_Z = 0.4545
# The unrotated SO-101 MJCF reaches forward along world -Y.  Separate the
# bases along world X so the two arms are side-by-side, not front-to-back.
MODEL_FORWARD_UNIT = (0.0, -1.0, 0.0)
MODEL_LATERAL_UNIT = (1.0, 0.0, 0.0)
# From the operator position at +Y looking forward toward -Y, screen-left is
# world +X. Keep the semantic left/right labels aligned with the real leaders.
LEFT_BASE_POS = (ARM_BASE_SEPARATION_M / 2.0, 0.0, 0.425)
RIGHT_BASE_POS = (-ARM_BASE_SEPARATION_M / 2.0, 0.0, 0.425)
# The Real2Sim demo keeps the calibrated targets at y=-0.26 m and moves the
# followers only eight centimetres toward the rear.  A larger shift would make
# those targets marginal for the SO-101 reach envelope.
EXPANDED_LEFT_BASE_POS = (ARM_BASE_SEPARATION_M / 2.0, 0.08, 0.425)
EXPANDED_RIGHT_BASE_POS = (-ARM_BASE_SEPARATION_M / 2.0, 0.08, 0.425)
EXPANDED_TABLE_CENTER_XY = (0.0, -0.10)
EXPANDED_TABLE_SIZE_XYZ = (1.40, 1.20, 0.05)
SAFETY_RAIL_HEIGHT_M = 0.030
SAFETY_RAIL_THICKNESS_M = 0.020
SHARED_BASE_EULER_DEG = (0.0, 0.0, 0.0)
OBJECT_START_POS = (0.10, -0.26, 0.47)
OBJECT_TARGET_POS = (-0.10, -0.26, 0.47)
PLUSH_TELEOP_START_XY = (0.0, -0.14)
SIM_GRIPPER_SOLVER_TOLERANCE_RAD = math.radians(5.0)
# A closed rigid hull remains the authoritative anti-penetration volume.  Its
# contact is deliberately less stiff than the robot/table contacts, while
# velocity retention and rolling resistance remove the hard-plastic bounce and
# endless rolling that looked wrong for a 40 g plush toy.
RIGID_PLUSH_CONTACT_SOL_PARAMS = (0.020, 1.15, 0.85, 0.95, 0.003, 0.50, 2.0)
RIGID_PLUSH_LINEAR_VELOCITY_RETENTION = 0.995
RIGID_PLUSH_ANGULAR_VELOCITY_RETENTION = 0.970
RIGID_PLUSH_SETTLE_CENTER_HEIGHT_M = 0.058
RIGID_PLUSH_SETTLE_LINEAR_SPEED_M_S = 0.020
RIGID_PLUSH_SETTLE_ANGULAR_SPEED_RAD_S = 0.45
MPM_PLUSH_YOUNGS_MODULUS_PA = 8.0e3
MPM_PLUSH_POISSONS_RATIO = 0.20
MPM_PLUSH_GRIPPER_FORCE_LIMIT_N = 0.8
MPM_PLUSH_VISUAL_UPDATE_INTERVAL = 2
XPBD_PLUSH_VISUAL_UPDATE_INTERVAL = 1
GRANULAR_PLUSH_VISUAL_UPDATE_INTERVAL = 1
# Physics remains at 120 Hz.  The full 294k-face TRELLIS surface only needs a
# 30 Hz custom-vertex refresh; updating it four times per control frame wastes
# the GPU budget without adding contact fidelity.
FEM_PLUSH_VISUAL_UPDATE_INTERVAL = 2
MGPBD_PLUSH_VISUAL_UPDATE_INTERVAL = 1


def contact_pair_force_total(
    geom_a: np.ndarray,
    geom_b: np.ndarray,
    forces: np.ndarray,
    first_geom_range: tuple[int, int],
    second_geom_range: tuple[int, int],
) -> float:
    """Sum contact-force magnitudes only between two entity geometry ranges."""

    geom_a = np.asarray(geom_a).reshape(-1)
    geom_b = np.asarray(geom_b).reshape(-1)
    forces = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
    if len(geom_a) != len(geom_b) or len(geom_a) != len(forces):
        raise ValueError("contact geometry and force arrays must have equal lengths")
    first_start, first_end = first_geom_range
    second_start, second_end = second_geom_range
    if first_start >= first_end or second_start >= second_end:
        raise ValueError("contact geometry ranges must be non-empty")
    a_first = (geom_a >= first_start) & (geom_a < first_end)
    b_first = (geom_b >= first_start) & (geom_b < first_end)
    a_second = (geom_a >= second_start) & (geom_a < second_end)
    b_second = (geom_b >= second_start) & (geom_b < second_end)
    selected = (a_first & b_second) | (b_first & a_second)
    if not np.any(selected):
        return 0.0
    return float(np.linalg.norm(forces[selected], axis=1).sum())


def relative_transform(parent_world: Any, child_world: Any) -> np.ndarray:
    """Return the child pose in parent coordinates for two world transforms."""
    parent = np.asarray(parent_world, dtype=np.float64)
    child = np.asarray(child_world, dtype=np.float64)
    if parent.shape != (4, 4) or child.shape != (4, 4):
        raise ValueError("parent and child transforms must both be 4x4")
    if not np.isfinite(parent).all() or not np.isfinite(child).all():
        raise ValueError("transforms must be finite")
    return np.linalg.solve(parent, child)


@dataclass
class SceneHandles:
    gs: Any
    scene: Any
    left: Any
    right: Any
    table: Any
    object: Any
    front_camera: Any
    hand_camera: Any
    object_visualization: bool
    object_mesh_path: Path
    object_physics_kind: str
    rigid_contact_object: Any | None
    plush_binding: Any | None
    table_top_z: float
    object_start_pos: tuple[float, float, float]
    object_target_pos: tuple[float, float, float]
    left_base_pos: tuple[float, float, float]
    right_base_pos: tuple[float, float, float]
    table_center_xy: tuple[float, float]
    table_size_xyz: tuple[float, float, float]
    safety_rail_height_m: float


class SO101HandoverTask:
    def __init__(self, handles: SceneHandles):
        self.handles = handles
        if handles.object_physics_kind in (
            "pbd-plush",
            "xpbd-plush",
            "granular-plush",
            "mgpbd-plush",
        ):
            handles.object.configure_grippers(
                handles.left.get_link("gripper"),
                handles.right.get_link("gripper"),
                handles.left.get_link("moving_jaw_so101_v1"),
                handles.right.get_link("moving_jaw_so101_v1"),
                handles.left.get_link("wrist"),
                handles.right.get_link("wrist"),
            )
        self._gripper_geom_ranges = tuple(
            tuple(
                (link.geom_start, link.geom_end)
                for link in (
                    arm.get_link("gripper"),
                    arm.get_link("moving_jaw_so101_v1"),
                )
            )
            for arm in (handles.left, handles.right)
        )
        self._gripper_saturation_count = [0, 0]
        self._gripper_max_excursion_rad = [0.0, 0.0]
        self._mgpbd_effective_gripper_percent = np.asarray(
            (35.0, 35.0), dtype=np.float64
        )
        self._mgpbd_contact_limiter_frames = [0, 0]
        self._mgpbd_gripper_contact_limited = [False, False]
        self._mgpbd_gripper_penetration_m = np.zeros(2, dtype=np.float64)
        self._mgpbd_gripper_peak_penetration_m = np.zeros(2, dtype=np.float64)
        self._mgpbd_gripper_preload_m = np.zeros(2, dtype=np.float64)
        self._mgpbd_gripper_peak_preload_m = np.zeros(2, dtype=np.float64)
        self._mgpbd_grasp_hold_percent = np.full(2, np.nan, dtype=np.float64)
        self._appearance_binding: SafeAppearanceBinding | None = None
        self._object_segmentation_index: int | None = None
        self._appearance_render_ms: list[float] = []
        self._appearance_clipped_fraction: list[float] = []
        self._appearance_composited_frames = 0
        self._appearance_fallback_frames = 0
        self._plush_grasp_update_step = 0
        self._mpm_visual_update_step = 0
        self._effective_object_start_pos = np.asarray(
            handles.object_start_pos, dtype=np.float32
        )

    def default_action(self) -> tuple[float, ...]:
        if self.handles.object_physics_kind in (
            "pbd-plush",
            "mpm-plush",
            "xpbd-plush",
            "granular-plush",
            "fem-plush",
            "mgpbd-plush",
        ):
            return PLUSH_HOME_ACTION
        return HOME_ACTION

    def _set_xpbd_gripper_closure(self, values: Sequence[float]) -> None:
        if self.handles.object_physics_kind not in (
            "xpbd-plush",
            "granular-plush",
            "mgpbd-plush",
        ):
            return
        self.handles.object.set_gripper_closure(
            1.0 - float(values[5]) / 100.0,
            1.0 - float(values[11]) / 100.0,
        )

    def _limit_mgpbd_gripper_contact(
        self, values: Sequence[float]
    ) -> np.ndarray:
        """Apply persistent two-finger-preload compliance to each jaw."""

        result = np.asarray(values, dtype=np.float64).copy()
        if self.handles.object_physics_kind != "mgpbd-plush":
            return result
        evidence = self.handles.object.gripper_contact_evidence()
        active = tuple(
            bool(value)
            for value in evidence.get(
                "persistent_active_by_arm", evidence["active_by_arm"]
            )
        )
        penetration = np.asarray(
            self.handles.object.gripper_penetration_by_arm(), dtype=np.float64
        ).reshape(2)
        preload = np.asarray(
            self.handles.object.gripper_contact_preload_by_arm(),
            dtype=np.float64,
        ).reshape(2)
        self._mgpbd_gripper_penetration_m[:] = penetration
        self._mgpbd_gripper_peak_penetration_m = np.maximum(
            self._mgpbd_gripper_peak_penetration_m, penetration
        )
        self._mgpbd_gripper_preload_m[:] = preload
        self._mgpbd_gripper_peak_preload_m = np.maximum(
            self._mgpbd_gripper_peak_preload_m, preload
        )
        target_preload_m = 0.00070
        free_closing_percent_per_step = 4.0
        contact_closing_percent_per_step = 0.75
        for arm_index, action_index in enumerate((5, 11)):
            requested = float(result[action_index])
            current = float(self._mgpbd_effective_gripper_percent[arm_index])
            if requested >= current:
                # Operator-requested opening always releases promptly.
                effective = min(requested, current + 10.0)
                compliant = False
                self._mgpbd_grasp_hold_percent[arm_index] = np.nan
            else:
                # Compliance must be bidirectional.  Merely stopping closure
                # cannot recover from a discrete jaw step that has already
                # driven the low-resolution cage into its volume barrier.
                # Back the jaw out until geometric penetration returns to the
                # contact boundary, even while the operator keeps requesting
                # closure.
                if penetration[arm_index] > 0.003:
                    recovery_change = float(
                        np.clip(
                            1000.0 * (penetration[arm_index] - 0.0015),
                            0.5,
                            4.0,
                        )
                    )
                    effective = min(100.0, current + recovery_change)
                    compliant = True
                    self._mgpbd_grasp_hold_percent[arm_index] = np.nan
                    self._mgpbd_contact_limiter_frames[arm_index] += 1
                    self._mgpbd_gripper_contact_limited[arm_index] = compliant
                    self._mgpbd_effective_gripper_percent[arm_index] = effective
                    result[action_index] = effective
                    continue
                closing_fraction = 1.0
                # Contact, rather than an arbitrary gripper percentage, is
                # the onset of compliance.  The previous ``current <= 50``
                # gate let a jaw that first touched this 10 cm plush near
                # 80% opening continue closing at full speed until it was
                # already deep inside the object.
                compliance_enabled = active[arm_index]
                if compliance_enabled:
                    closing_fraction = float(
                        np.clip(
                            1.0
                            - preload[arm_index] / target_preload_m,
                            0.0,
                            1.0,
                        )
                    )
                effective = max(
                    requested,
                    current
                    - (
                        contact_closing_percent_per_step
                        if compliance_enabled
                        else free_closing_percent_per_step
                    )
                    * closing_fraction,
                )
                compliant = compliance_enabled and closing_fraction < 0.999
                if compliant:
                    self._mgpbd_contact_limiter_frames[arm_index] += 1
                self._mgpbd_grasp_hold_percent[arm_index] = (
                    effective
                    if compliance_enabled and closing_fraction <= 0.05
                    else np.nan
                )
            self._mgpbd_gripper_contact_limited[arm_index] = compliant
            self._mgpbd_effective_gripper_percent[arm_index] = effective
            result[action_index] = effective
        return result

    def _enforce_mgpbd_gripper_pose(self, values: Sequence[float]) -> None:
        """Remove servo overshoot from the custom-contact gripper DOF."""

        if self.handles.object_physics_kind != "mgpbd-plush":
            return
        for arm, start in (
            (self.handles.left, 0),
            (self.handles.right, 6),
        ):
            gripper_position = np.asarray(
                [lerobot_arm_to_genesis(values[start : start + 6])[5]],
                dtype=np.float32,
            )
            gripper_index = np.asarray([5], dtype=np.int32)
            arm.set_dofs_position(
                gripper_position,
                dofs_idx_local=gripper_index,
                zero_velocity=False,
            )
            arm.set_dofs_velocity(
                np.zeros(1, dtype=np.float32),
                dofs_idx_local=gripper_index,
            )

    def reset(self, action: Sequence[float] | None = None) -> dict[str, Any]:
        if action is None:
            action = self.default_action()
        values = require_vector(action)
        if self.handles.object_physics_kind == "mgpbd-plush":
            self._mgpbd_effective_gripper_percent[:] = (
                float(values[5]),
                float(values[11]),
            )
            self._mgpbd_contact_limiter_frames[:] = (0, 0)
            self._mgpbd_gripper_contact_limited[:] = (False, False)
            self._mgpbd_gripper_penetration_m.fill(0.0)
            self._mgpbd_gripper_peak_penetration_m.fill(0.0)
            self._mgpbd_gripper_preload_m.fill(0.0)
            self._mgpbd_gripper_peak_preload_m.fill(0.0)
            self._mgpbd_grasp_hold_percent.fill(np.nan)
        left = np.asarray(lerobot_arm_to_genesis(values[:6]), dtype=np.float32)
        right = np.asarray(lerobot_arm_to_genesis(values[6:]), dtype=np.float32)
        self.handles.left.set_dofs_position(left)
        self.handles.right.set_dofs_position(right)
        self._plush_grasp_update_step = 0
        self._mpm_visual_update_step = 0
        self.handles.left.control_dofs_position(left)
        self.handles.right.control_dofs_position(right)
        self._set_xpbd_gripper_closure(values)
        if self.handles.object_physics_kind == "pbd-plush":
            self.handles.object.release_all_grasps()
            # The legacy rigid smoke starts the object in free space. A plush
            # handover instead begins with the left follower already holding
            # the team-owned toy, matching the real HIL episode contract.
            self._effective_object_start_pos = self._array(
                self.handles.object.collision_free_initial_center(0)
            ).reshape(3).astype(np.float32)
        else:
            self._effective_object_start_pos = np.asarray(
                self.handles.object_start_pos, dtype=np.float32
            )
        self.handles.object.set_pos(self._effective_object_start_pos)
        self.handles.object.set_quat(np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32))
        self.handles.object.set_dofs_velocity(np.zeros(6, dtype=np.float32))
        if self.handles.object_physics_kind == "pbd-plush":
            self.handles.object.attach_initial_left_grasp()
        if self.handles.plush_binding is not None:
            if self.handles.object_physics_kind == "mgpbd-plush":
                self.handles.object.update_visual()
            else:
                self.handles.plush_binding.update()
        self.handles.scene.step()
        self._enforce_mgpbd_gripper_pose(values)
        if self.handles.object_physics_kind in (
            "xpbd-plush",
            "granular-plush",
            "mgpbd-plush",
        ):
            self.handles.object.step_simulation()
        return self.observe(render=False)

    def object_initial_position(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._effective_object_start_pos)

    def object_target_position(self) -> tuple[float, float, float]:
        return self.handles.object_target_pos

    def step(self, action: Sequence[float], *, render: bool = False) -> dict[str, Any]:
        values = self._limit_mgpbd_gripper_contact(require_vector(action))
        self.handles.left.control_dofs_position(
            np.asarray(lerobot_arm_to_genesis(values[:6]), dtype=np.float32)
        )
        self.handles.right.control_dofs_position(
            np.asarray(lerobot_arm_to_genesis(values[6:]), dtype=np.float32)
        )
        self._set_xpbd_gripper_closure(values)
        if self.handles.object_physics_kind == "pbd-plush":
            self.handles.object.update_grasps(
                (values[5], values[11]),
                allow_new_attachment=(self._plush_grasp_update_step % 4 == 0),
            )
            self._plush_grasp_update_step += 1
        self.handles.scene.step()
        self._enforce_mgpbd_gripper_pose(values)
        if self.handles.object_physics_kind in (
            "xpbd-plush",
            "granular-plush",
            "mgpbd-plush",
        ):
            # Genesis advances the rigid arms; the custom Taichi XPBD object
            # is not part of Genesis' solver graph and must be stepped
            # explicitly before its visual binding is refreshed.  Omitting
            # this call left the plush frozen after reset and let every moving
            # jaw pass straight through its static shell.
            self.handles.object.step_simulation()
        if (
            self.handles.object_physics_kind == "mpm-plush"
            and self.handles.plush_binding is not None
            and self._mpm_visual_update_step % MPM_PLUSH_VISUAL_UPDATE_INTERVAL == 0
        ):
            # The viewer renders independently of camera capture. Keep its
            # appearance on the authoritative MPM particles every physics step
            # so visible geometry can never drift away from contact geometry.
            self.handles.plush_binding.update()
        if self.handles.object_physics_kind == "mpm-plush":
            self._mpm_visual_update_step += 1
        if (
            self.handles.object_physics_kind
            in ("xpbd-plush", "granular-plush", "mgpbd-plush")
            and self.handles.plush_binding is not None
            and self._mpm_visual_update_step
            % (
                GRANULAR_PLUSH_VISUAL_UPDATE_INTERVAL
                if self.handles.object_physics_kind == "granular-plush"
                else (
                    MGPBD_PLUSH_VISUAL_UPDATE_INTERVAL
                    if self.handles.object_physics_kind == "mgpbd-plush"
                    else XPBD_PLUSH_VISUAL_UPDATE_INTERVAL
                )
            )
            == 0
        ):
            if self.handles.object_physics_kind == "mgpbd-plush":
                self.handles.object.update_visual()
            else:
                self.handles.plush_binding.update()
        if self.handles.object_physics_kind in (
            "xpbd-plush",
            "granular-plush",
            "mgpbd-plush",
        ):
            self._mpm_visual_update_step += 1
        if (
            self.handles.object_physics_kind == "fem-plush"
            and self.handles.plush_binding is not None
            and self._mpm_visual_update_step % FEM_PLUSH_VISUAL_UPDATE_INTERVAL == 0
        ):
            self.handles.plush_binding.update()
        if self.handles.object_physics_kind == "fem-plush":
            self._mpm_visual_update_step += 1
        if self.handles.object_physics_kind == "rigid-plush":
            velocity = self._array(
                self.handles.object.get_dofs_velocity()
            ).reshape(6).astype(np.float32)
            velocity[:3] *= RIGID_PLUSH_LINEAR_VELOCITY_RETENTION
            velocity[3:] *= RIGID_PLUSH_ANGULAR_VELOCITY_RETENTION
            center_z = float(
                self._array(self.handles.object.get_pos()).reshape(3)[2]
            )
            if (
                center_z
                <= self.handles.table_top_z + RIGID_PLUSH_SETTLE_CENTER_HEIGHT_M
                and np.linalg.norm(velocity[:3])
                <= RIGID_PLUSH_SETTLE_LINEAR_SPEED_M_S
                and np.linalg.norm(velocity[3:])
                <= RIGID_PLUSH_SETTLE_ANGULAR_SPEED_RAD_S
            ):
                # Static fabric/table contact has a finite breakaway threshold.
                # Snapping only an already-near-rest toy prevents solver-scale
                # rocking without resisting deliberate gripper motion.
                velocity[:] = 0.0
            self.handles.object.set_dofs_velocity(velocity)
        return self.observe(render=render)

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def observe(
        self, *, render: bool = True, force_render: bool = False
    ) -> dict[str, Any]:
        left = self._observe_arm(self.handles.left, arm_index=0)
        right = self._observe_arm(self.handles.right, arm_index=1)
        observation: dict[str, Any] = {
            "observation.state": np.asarray(left + right, dtype=np.float32)
        }
        if render:
            if self.handles.plush_binding is not None:
                if self.handles.object_physics_kind == "mgpbd-plush":
                    self.handles.object.update_visual()
                else:
                    self.handles.plush_binding.update()
            self.handles.hand_camera.move_to_attach()
            front_rgb = self._render_camera(
                self.handles.front_camera, force_render=force_render
            )
            hand_rgb = self._render_camera(
                self.handles.hand_camera, force_render=force_render
            )
            observation[CAMERA_KEYS[0]] = self._array(front_rgb)
            observation[CAMERA_KEYS[1]] = self._array(hand_rgb)
        return observation

    def set_appearance_binding(
        self, binding: SafeAppearanceBinding | None
    ) -> None:
        """Attach an optional demo renderer without changing control or physics."""

        self._appearance_binding = binding
        self._object_segmentation_index = None

    def _render_camera(self, camera: Any, *, force_render: bool = False) -> np.ndarray:
        if self._appearance_binding is None:
            rgb, _, _, _ = camera.render(rgb=True, force_render=force_render)
            return self._array(rgb)

        rgb, depth, segmentation, _ = camera.render(
            rgb=True,
            depth=True,
            segmentation=True,
            force_render=force_render,
        )
        base_rgb = self._array(rgb)
        binding_result = self._appearance_binding.render_from_genesis(
            camera, self.handles.object
        )
        if binding_result.frame is None:
            self._appearance_fallback_frames += 1
            return base_rgb
        if self.handles.object_visualization:
            if self._object_segmentation_index is None:
                self._object_segmentation_index = entity_segmentation_index(
                    self.handles.scene, self.handles.object
                )
            object_mask = self._array(segmentation) == self._object_segmentation_index
            composite = composite_with_proxy_depth(
                base_rgb,
                self._array(depth),
                object_mask,
                binding_result.frame,
            )
        else:
            composite = composite_with_gaussian_depth(
                base_rgb,
                self._array(depth),
                binding_result.frame,
            )
        self._appearance_render_ms.append(binding_result.frame.render_ms)
        self._appearance_clipped_fraction.append(
            composite.gaussian_alpha_clipped_fraction
        )
        self._appearance_composited_frames += 1
        return composite.rgb_u8

    def appearance_diagnostics(self) -> dict[str, Any]:
        binding_metrics = (
            self._appearance_binding.metrics()
            if self._appearance_binding is not None
            else None
        )
        return {
            "enabled": self._appearance_binding is not None,
            "object_segmentation_index": self._object_segmentation_index,
            "object_visualization": self.handles.object_visualization,
            "object_mesh_path": str(self.handles.object_mesh_path),
            "compositor": (
                None
                if self._appearance_binding is None
                else (
                    "proxy_matte"
                    if self.handles.object_visualization
                    else "gaussian_self_depth"
                )
            ),
            "composited_frames": self._appearance_composited_frames,
            "fallback_frames": self._appearance_fallback_frames,
            "render_ms": {
                "mean": (
                    float(np.mean(self._appearance_render_ms))
                    if self._appearance_render_ms
                    else None
                ),
                "p95": (
                    float(np.percentile(self._appearance_render_ms, 95))
                    if self._appearance_render_ms
                    else None
                ),
                "max": (
                    float(np.max(self._appearance_render_ms))
                    if self._appearance_render_ms
                    else None
                ),
            },
            "gaussian_alpha_clipped_fraction": {
                "mean": (
                    float(np.mean(self._appearance_clipped_fraction))
                    if self._appearance_clipped_fraction
                    else None
                ),
                "max": (
                    float(np.max(self._appearance_clipped_fraction))
                    if self._appearance_clipped_fraction
                    else None
                ),
            },
            "binding": binding_metrics,
            "physics": self.object_physics_diagnostics(),
        }

    def object_physics_diagnostics(self) -> dict[str, Any]:
        if self.handles.object_physics_kind in (
            "pbd-plush",
            "mpm-plush",
            "xpbd-plush",
            "granular-plush",
            "fem-plush",
            "mgpbd-plush",
        ):
            return self.handles.object.diagnostics()
        if self.handles.object_physics_kind == "rigid-plush":
            return {
                "kind": "rigid-plush",
                "authoritative_contact": "closed_convex_rigid_core",
                "rigid_contact_identity_available": True,
                "contact_sol_params": list(RIGID_PLUSH_CONTACT_SOL_PARAMS),
                "linear_velocity_retention_per_step": (
                    RIGID_PLUSH_LINEAR_VELOCITY_RETENTION
                ),
                "angular_velocity_retention_per_step": (
                    RIGID_PLUSH_ANGULAR_VELOCITY_RETENTION
                ),
                "near_table_static_settle": {
                    "center_height_m": RIGID_PLUSH_SETTLE_CENTER_HEIGHT_M,
                    "linear_speed_m_s": RIGID_PLUSH_SETTLE_LINEAR_SPEED_M_S,
                    "angular_speed_rad_s": RIGID_PLUSH_SETTLE_ANGULAR_SPEED_RAD_S,
                },
                "rolling_friction_enabled": True,
                "torsional_friction_enabled": True,
                "deformable_geometry": False,
                "model_scope": "qualitative compliant-contact plush approximation",
            }
        return {
            "kind": "rigid",
            "rigid_contact_identity_available": True,
        }

    def _observe_arm(self, arm: Any, *, arm_index: int) -> tuple[float, ...]:
        values = self._array(arm.get_dofs_position()).reshape(-1).tolist()
        gripper = float(values[5])
        excursion = max(
            SO101_GRIPPER_MIN_RAD - gripper,
            gripper - SO101_GRIPPER_MAX_RAD,
            0.0,
        )
        if excursion > 0.0:
            self._gripper_saturation_count[arm_index] += 1
            self._gripper_max_excursion_rad[arm_index] = max(
                self._gripper_max_excursion_rad[arm_index], excursion
            )
        return genesis_arm_to_lerobot(
            values,
            gripper_tolerance_rad=SIM_GRIPPER_SOLVER_TOLERANCE_RAD,
        )

    def solver_limit_diagnostics(self) -> dict[str, Any]:
        return {
            "gripper_saturation_count": list(self._gripper_saturation_count),
            "gripper_max_excursion_deg": [
                math.degrees(value) for value in self._gripper_max_excursion_rad
            ],
            "gripper_hard_tolerance_deg": math.degrees(
                SIM_GRIPPER_SOLVER_TOLERANCE_RAD
            ),
            "mgpbd_contact_limiter": {
                "enabled": self.handles.object_physics_kind == "mgpbd-plush",
                "effective_gripper_percent": (
                    self._mgpbd_effective_gripper_percent.tolist()
                ),
                "limited_frames_by_arm": list(
                    self._mgpbd_contact_limiter_frames
                ),
                "controller": "continuous_two_finger_preload_impedance_servo",
                "current_penetration_m_by_arm": (
                    self._mgpbd_gripper_penetration_m.tolist()
                ),
                "peak_penetration_m_by_arm": (
                    self._mgpbd_gripper_peak_penetration_m.tolist()
                ),
                "current_two_finger_preload_m_by_arm": (
                    self._mgpbd_gripper_preload_m.tolist()
                ),
                "peak_two_finger_preload_m_by_arm": (
                    self._mgpbd_gripper_peak_preload_m.tolist()
                ),
                "controlled_contact": "minimum_preload_across_two_finger_roles",
                "target_two_finger_preload_m": 0.00070,
                "compliance_onset": "identity_specific_two_finger_contact",
                "closing_velocity_law": (
                    "contact_speed_times_1_minus_preload_over_target"
                ),
                "free_closing_change_percent_per_step": 4.0,
                "contact_closing_change_percent_per_step": 0.75,
                "penetration_recovery": {
                    "onset_m": 0.001,
                    "target_m": 0.00035,
                    "opening_change_percent_per_step": [0.5, 4.0],
                },
                "release_change_percent_per_step": 10.0,
                "contact_limited_by_arm": list(
                    self._mgpbd_gripper_contact_limited
                ),
                "held_contact_percent_by_arm": [
                    None if not np.isfinite(value) else float(value)
                    for value in self._mgpbd_grasp_hold_percent
                ],
                "driver": "measured_two_finger_custom_body_contact",
            },
        }

    def visual_state(self) -> dict[str, tuple[float, ...]]:
        """Return a read-only renderer snapshot in Genesis/world coordinates."""

        joints = np.concatenate(
            (
                self._array(self.handles.left.get_dofs_position()).reshape(-1),
                self._array(self.handles.right.get_dofs_position()).reshape(-1),
            )
        )
        position = self._array(self.handles.object.get_pos()).reshape(-1)
        quaternion = self._array(self.handles.object.get_quat()).reshape(-1)
        return {
            "joint_positions_rad": tuple(float(value) for value in joints),
            "object_position_m": tuple(float(value) for value in position),
            "object_quaternion_wxyz": tuple(float(value) for value in quaternion),
        }

    def gripper_object_contact_evidence(self) -> dict[str, object] | None:
        """Return current custom-solver contact identity, when available.

        Genesis' rigid contact manifold cannot identify contacts owned by the
        separate XPBD/MGPBD solver.  The adapter instead reports current
        boundary contacts for each concrete gripper collider and requires both
        fixed- and moving-finger roles before declaring an arm active.
        """

        if self.handles.object_physics_kind in (
            "xpbd-plush",
            "granular-plush",
            "mgpbd-plush",
        ):
            return self.handles.object.gripper_contact_evidence()
        return None

    def haptic_feedback_diagnostics(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, float], tuple[float, float]]:
        """Return efforts, all external forces, and gripper-object forces.

        All external contacts are reflected, including the target, table, and
        opposite arm; self-collisions are excluded. The fixed bases are placed
        clear of the tabletop, so ordinary support does not create a standing
        feedback signal. The third result is stricter and includes only contact
        between either gripper/finger pair and the handover object.
        """

        # Pull the scene contact manifold once. Repeating entity.get_contacts()
        # for every counterpart adds several GPU synchronizations and prevents
        # the 120 Hz simulation loop from meeting its deadline.
        contacts = self.handles.scene.rigid_solver.collider.get_contacts(
            as_tensor=True, to_torch=True
        )
        geom_a = self._array(contacts["geom_a"]).reshape(-1)
        geom_b = self._array(contacts["geom_b"]).reshape(-1)
        forces = self._array(contacts["force"]).reshape(-1, 3)

        def external_force_total(arm: Any) -> float:
            a_in_arm = (geom_a >= arm.geom_start) & (geom_a < arm.geom_end)
            b_in_arm = (geom_b >= arm.geom_start) & (geom_b < arm.geom_end)
            external = np.logical_xor(a_in_arm, b_in_arm)
            if not np.any(external):
                return 0.0
            return float(np.linalg.norm(forces[external], axis=1).sum())

        left_force = external_force_total(self.handles.left)
        right_force = external_force_total(self.handles.right)
        if self.handles.rigid_contact_object is None:
            # Genesis applies two-way PBD coupling forces to the robot links,
            # but does not expose those contacts in the rigid contact manifold.
            # Preserve the haptic transport shape and fail closed on identity-
            # specific object force until a PBD contact receipt is implemented.
            gripper_object_force = (0.0, 0.0)
        else:
            object_range = (
                self.handles.rigid_contact_object.geom_start,
                self.handles.rigid_contact_object.geom_end,
            )
            gripper_object_force = tuple(
                sum(
                    contact_pair_force_total(
                        geom_a,
                        geom_b,
                        forces,
                        gripper_range,
                        object_range,
                    )
                    for gripper_range in arm_ranges
                )
                for arm_ranges in self._gripper_geom_ranges
            )
        left_effort = np.zeros(6, dtype=np.float64)
        right_effort = np.zeros(6, dtype=np.float64)
        if left_force > 1e-6:
            left_effort = -self._array(
                self.handles.left.get_dofs_control_force()
            ).reshape(-1)
        if right_force > 1e-6:
            right_effort = -self._array(
                self.handles.right.get_dofs_control_force()
            ).reshape(-1)
        efforts = tuple(float(value) for value in np.concatenate((left_effort, right_effort)))
        return (
            efforts,
            (left_force, right_force),
            gripper_object_force,
        )

    def haptic_feedback(self) -> tuple[tuple[float, ...], tuple[float, float]]:
        """Return the existing all-external-contact haptic transport contract."""

        efforts, external_force, _ = self.haptic_feedback_diagnostics()
        return efforts, external_force

    def success(self) -> bool:
        position = self._array(self.handles.object.get_pos()).reshape(-1)
        target = np.asarray(self.handles.object_target_pos)
        return bool(np.linalg.norm(position - target) < 0.08)


def build(
    asset_root: Path,
    *,
    seed: int = 20260803,
    show_viewer: bool = False,
    workspace_texture: Path | None = None,
    front_camera_calibration: Path | None = None,
    front_camera_gui: bool = False,
    object_visualization: bool = True,
    object_urdf_path: Path | None = None,
    plush_visual_mesh_path: Path | None = None,
    plush_physics_mode: str = "granular",
    sim_hz: float = 120.0,
    fem_substeps: int = 2,
) -> tuple[SO101HandoverTask, SceneHandles]:
    import genesis as gs
    import genesis.utils.geom as gu

    fetch(asset_root)
    model = asset_root / "so101_new_calib.xml"
    if object_urdf_path is not None and plush_visual_mesh_path is not None:
        raise ValueError("rigid URDF and deformable plush modes are mutually exclusive")
    if plush_visual_mesh_path is not None and not object_visualization:
        raise ValueError("deformable plush mode requires object_visualization=True")
    if plush_physics_mode not in ("granular", "fem", "xpbd", "mgpbd"):
        raise ValueError(
            "plush_physics_mode must be 'granular', 'fem', 'xpbd', or 'mgpbd'"
        )
    allowed_sim_hz = (
        (30.0, 60.0, 120.0)
        if plush_visual_mesh_path is not None and plush_physics_mode == "mgpbd"
        else (
            (60.0, 120.0)
            if plush_visual_mesh_path is not None and plush_physics_mode == "fem"
            else ((120.0,) if plush_visual_mesh_path is not None else (60.0, 120.0))
        )
    )
    if sim_hz not in allowed_sim_hz:
        raise ValueError(f"Genesis handover sim_hz must be one of {allowed_sim_hz}")
    if not isinstance(fem_substeps, int) or fem_substeps not in (2, 4):
        raise ValueError("fem_substeps must be 2 or 4")
    spec = load_spec()
    fem_config = FEMPlushConfig()
    fem_config.validate()
    gs.init(backend=gs.amdgpu, seed=seed)
    if gs.backend != gs.amdgpu:
        raise RuntimeError(f"Genesis did not select the AMD GPU backend: {gs.backend}")
    use_fem_plush = plush_visual_mesh_path is not None and plush_physics_mode == "fem"
    use_granular_plush = (
        plush_visual_mesh_path is not None and plush_physics_mode == "granular"
    )
    use_mgpbd_plush = (
        plush_visual_mesh_path is not None and plush_physics_mode == "mgpbd"
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / sim_hz,
            substeps=fem_substeps if use_fem_plush else 2,
            gravity=(0.0, 0.0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            enable_joint_limit=True,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
        ),
        # SAP is robust but Genesis 1.3.1 restricts it to FP64; the measured
        # AMD APU baseline is seconds per frame.  The live path therefore uses
        # the native FP32 rigid/FEM coupler while retaining the same implicit
        # corotated volumetric solve.  The offline SAP gate remains separate.
        coupler_options=gs.options.LegacyCouplerOptions(
            rigid_pbd=not use_fem_plush,
            rigid_fem=use_fem_plush,
        ),
        fem_options=gs.options.FEMOptions(
            damping=fem_config.damping,
            floor_height=PLUSH_TABLE_TOP_Z if use_fem_plush else None,
            use_implicit_solver=True,
            n_newton_iterations=fem_config.newton_iterations,
            n_pcg_iterations=fem_config.pcg_iterations,
            pcg_threshold=fem_config.pcg_threshold,
            damping_alpha=fem_config.damping_alpha,
            damping_beta=fem_config.damping_beta,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=spec.pbd["particle_size_m"],
            max_stretch_solver_iterations=8,
            max_bending_solver_iterations=4,
            max_volume_solver_iterations=8,
            lower_bound=(-0.72, -0.72, 0.0),
            upper_bound=(0.72, 0.52, 1.0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.72, -0.72, 0.0),
            upper_bound=(0.72, 0.52, 1.0),
            grid_density=96,
        ),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.55, 0.55, 0.55),
            shadow=True,
            segmentation_level="link",
        ),
        viewer_options=gs.options.ViewerOptions(
            # Oblique from the semantic-left side so the initially held plush
            # is not hidden directly behind the left follower in the GUI.
            camera_pos=(0.72, 0.88, 0.82),
            camera_lookat=(0.0, -0.18, 0.52),
            camera_up=(0.0, 0.0, 1.0),
            camera_fov=55.0,
            res=(640, 480),
            # The full 294k-face TRELLIS visual and MGPBD share one Radeon
            # device. Rendering it at 30 Hz starves the MGPCG kernels and
            # reduced measured live control to ~4.4 Hz. Ten visual refreshes
            # per second are sufficient for this solver while preserving every
            # source face and prioritizing control/contact progress.
            max_FPS=10 if use_mgpbd_plush else (15 if use_fem_plush else 30),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    expanded_workspace = object_urdf_path is not None or (
        workspace_texture is not None
        and workspace_texture.name == "hil_fixed_front_v7_expanded_table_texture.png"
    )
    table_center_xy = (
        EXPANDED_TABLE_CENTER_XY if expanded_workspace else (0.0, 0.0)
    )
    table_size_xyz = (
        EXPANDED_TABLE_SIZE_XYZ if expanded_workspace else (1.2, 0.8, 0.05)
    )
    left_base_pos = EXPANDED_LEFT_BASE_POS if expanded_workspace else LEFT_BASE_POS
    right_base_pos = (
        EXPANDED_RIGHT_BASE_POS if expanded_workspace else RIGHT_BASE_POS
    )
    table_surface = None
    workspace_plane = Path(__file__).resolve().parent / "assets_generated" / (
        "workspace_plane_expanded.obj" if expanded_workspace else "workspace_plane.obj"
    )
    if workspace_texture is not None:
        workspace_texture = workspace_texture.resolve()
        if not workspace_texture.is_file():
            raise FileNotFoundError(f"workspace texture does not exist: {workspace_texture}")
        table_surface = gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(image_path=str(workspace_texture)),
            roughness=0.9,
        )
    table_top_z = (
        PLUSH_TABLE_TOP_Z
        if plush_visual_mesh_path is not None or object_urdf_path is not None
        else RIGID_TABLE_TOP_Z
    )
    if plush_visual_mesh_path is not None:
        plush_object_z = table_top_z + 0.09595 / 2.0 + 0.004
        object_start_pos = (
            PLUSH_TELEOP_START_XY[0], PLUSH_TELEOP_START_XY[1], plush_object_z
        )
        object_target_pos = (
            OBJECT_TARGET_POS[0], OBJECT_TARGET_POS[1], plush_object_z
        )
    elif object_urdf_path is not None:
        # The conservative TRELLIS contact hull is 95.95 mm high.  Start and
        # target three millimetres above the raised tabletop without overlap.
        rigid_object_z = table_top_z + 0.09595 / 2.0 + 0.003
        object_start_pos = (OBJECT_START_POS[0], OBJECT_START_POS[1], rigid_object_z)
        object_target_pos = (OBJECT_TARGET_POS[0], OBJECT_TARGET_POS[1], rigid_object_z)
    else:
        object_start_pos = OBJECT_START_POS
        object_target_pos = OBJECT_TARGET_POS
    table_box_surface = (
        gs.surfaces.Default(opacity=0.0)
        if use_fem_plush and workspace_texture is not None
        else (table_surface if workspace_texture is None else None)
    )
    table = scene.add_entity(
        gs.morphs.Box(
            pos=(table_center_xy[0], table_center_xy[1], table_top_z - 0.025),
            size=table_size_xyz,
            fixed=True,
            # The FEM solver's own floor handles plush support without the
            # legacy per-vertex rigid/FEM response that inverted boundary
            # tetrahedra. The rendered tabletop stays authoritative visually.
            collision=not use_fem_plush,
            visualization=workspace_texture is None or use_fem_plush,
        ),
        material=gs.materials.Rigid(friction=1.0, coup_friction=0.8),
        surface=table_box_surface,
    )
    if workspace_texture is not None:
        if not workspace_plane.is_file():
            raise FileNotFoundError(f"workspace UV plane is missing: {workspace_plane}")
        scene.add_entity(
            gs.morphs.Mesh(
                file=str(workspace_plane),
                pos=(table_center_xy[0], table_center_xy[1], table_top_z + 0.0005),
                fixed=True,
                collision=False,
                convexify=False,
                decimate=False,
            ),
            surface=table_surface,
        )
    if expanded_workspace:
        rail_surface = gs.surfaces.Default(
            color=(0.08, 0.08, 0.07, 1.0), roughness=0.95
        )
        rail_material = gs.materials.Rigid(friction=1.2)
        table_width, table_depth, _ = table_size_xyz
        rail_z = table_top_z + SAFETY_RAIL_HEIGHT_M / 2.0
        side_x = table_width / 2.0 - SAFETY_RAIL_THICKNESS_M / 2.0
        edge_y = table_depth / 2.0 - SAFETY_RAIL_THICKNESS_M / 2.0
        for position, size in (
            (
                (table_center_xy[0] - side_x, table_center_xy[1], rail_z),
                (SAFETY_RAIL_THICKNESS_M, table_depth, SAFETY_RAIL_HEIGHT_M),
            ),
            (
                (table_center_xy[0] + side_x, table_center_xy[1], rail_z),
                (SAFETY_RAIL_THICKNESS_M, table_depth, SAFETY_RAIL_HEIGHT_M),
            ),
            (
                (table_center_xy[0], table_center_xy[1] - edge_y, rail_z),
                (
                    table_width - 2.0 * SAFETY_RAIL_THICKNESS_M,
                    SAFETY_RAIL_THICKNESS_M,
                    SAFETY_RAIL_HEIGHT_M,
                ),
            ),
            (
                (table_center_xy[0], table_center_xy[1] + edge_y, rail_z),
                (
                    table_width - 2.0 * SAFETY_RAIL_THICKNESS_M,
                    SAFETY_RAIL_THICKNESS_M,
                    SAFETY_RAIL_HEIGHT_M,
                ),
            ),
        ):
            scene.add_entity(
                gs.morphs.Box(pos=position, size=size, fixed=True),
                material=rail_material,
                surface=rail_surface,
            )
    left = scene.add_entity(
        gs.morphs.MJCF(
            file=str(model), pos=left_base_pos, euler=SHARED_BASE_EULER_DEG
        ),
        material=gs.materials.Rigid(friction=1.0, coup_friction=1.0),
    )
    right = scene.add_entity(
        gs.morphs.MJCF(
            file=str(model), pos=right_base_pos, euler=SHARED_BASE_EULER_DEG
        ),
        material=gs.materials.Rigid(friction=1.0, coup_friction=1.0),
    )
    plush_soft = None
    plush_visual = None
    plush_fem = None
    plush_topology = None
    plush_xpbd_config = None
    mgpbd_config = None
    mgpbd_nodes = None
    mgpbd_elements = None
    mgpbd_tetrahedralization = None
    if plush_visual_mesh_path is not None:
        object_mesh_path = plush_visual_mesh_path.resolve()
        if not object_mesh_path.is_file():
            raise FileNotFoundError(f"plush visual mesh does not exist: {object_mesh_path}")
        proxy_manifest_path = (
            DEFAULT_TRELLIS2_FEM_LIVE_VOLUME_MANIFEST
            if use_fem_plush
            else (
                DEFAULT_TRELLIS2_FEM_PROXY_MANIFEST
                if use_mgpbd_plush
                else (
                DEFAULT_TRELLIS2_GRANULAR_SHELL_MANIFEST
                if use_granular_plush
                else DEFAULT_TRELLIS2_PLUSH_COLLISION_MANIFEST
                )
            )
        )
        proxy_manifest = json.loads(proxy_manifest_path.read_text(encoding="utf-8"))
        proxy_record = (
            proxy_manifest["surface"]
            if use_fem_plush
            else proxy_manifest["proxy"]
        )
        if not proxy_record.get("watertight", False):
            raise ValueError("deformable plush collision volume must be watertight")
        if use_fem_plush:
            if not proxy_manifest.get("quality_gate", {}).get("passed", False):
                raise ValueError("dense FEM proxy did not pass its quality gate")
        if use_mgpbd_plush:
            fem_proxy_manifest = json.loads(
                DEFAULT_TRELLIS2_FEM_PROXY_MANIFEST.read_text(encoding="utf-8")
            )
            if not fem_proxy_manifest["visual_enclosure_gate"].get("passed", False):
                raise ValueError("FEM proxy does not enclose the runtime visual mesh")
        if use_fem_plush:
            plush_fem = scene.add_entity(
                gs.morphs.Mesh(
                    file=str(DEFAULT_TRELLIS2_FEM_LIVE_SURFACE.resolve()),
                    pos=object_start_pos,
                    collision=True,
                    convexify=False,
                    decimate=False,
                    nobisect=True,
                    quality=True,
                    minratio=fem_config.tet_min_ratio,
                    mindihedral=fem_config.tet_min_dihedral_deg,
                    maxvolume=fem_config.tet_max_volume_m3,
                ),
                material=gs.materials.FEM.Elastic(
                    model="linear_corotated",
                    E=fem_config.youngs_modulus_pa,
                    nu=fem_config.poissons_ratio,
                    rho=fem_config.density_kg_m3,
                    friction_mu=fem_config.friction_mu,
                    hydroelastic_modulus=fem_config.hydroelastic_modulus_pa,
                ),
                surface=gs.surfaces.Default(opacity=0.0),
            )
        elif use_mgpbd_plush:
            # Leaders/rendering run at 30 Hz, but contact integration retains
            # the validated 120 Hz MGPBD step.  A 30 Hz gravity/contact step
            # moves the 10 cm toy by 10.9 mm before projection and is not a
            # stable contact discretization.
            mgpbd_config = MGPBDTetConfig(
                dt_s=1.0 / 120.0,
                shear_modulus_pa=float(
                    os.environ.get("ONELOOP_MGPBD_SHEAR_MODULUS_PA", "150000")
                ),
                relaxation=0.60,
                nonlinear_iterations=int(
                    os.environ.get("ONELOOP_MGPBD_NONLINEAR_ITERATIONS", "1")
                ),
                pcg_iterations=int(
                    os.environ.get("ONELOOP_MGPBD_PCG_ITERATIONS", "8")
                ),
                amg_strength_threshold=float(
                    os.environ.get(
                        "ONELOOP_MGPBD_AMG_STRENGTH_THRESHOLD", "0.10"
                    )
                ),
                smoother_iterations=1,
                damping_retention=float(
                    os.environ.get("ONELOOP_MGPBD_DAMPING_RETENTION", "0.965")
                ),
                gripper_friction=float(
                    os.environ.get("ONELOOP_MGPBD_GRIPPER_FRICTION", "1.4")
                ),
                particle_radius_m=float(
                    os.environ.get("ONELOOP_MGPBD_PARTICLE_RADIUS_M", "0.00035")
                ),
                contact_release_m=float(
                    os.environ.get("ONELOOP_MGPBD_CONTACT_RELEASE_M", "0.004")
                ),
                minimum_signed_volume_ratio=float(
                    os.environ.get(
                        "ONELOOP_MGPBD_MINIMUM_SIGNED_VOLUME_RATIO", "0.20"
                    )
                ),
                two_finger_transfer_closure_threshold=float(
                    os.environ.get(
                        "ONELOOP_MGPBD_TWO_FINGER_TRANSFER_THRESHOLD", "0.20"
                    )
                ),
                maximum_grasp_transport_m=float(
                    os.environ.get(
                        "ONELOOP_MGPBD_MAXIMUM_GRASP_TRANSPORT_M", "0.012"
                    )
                ),
                grasp_contact_persistence_frames=int(
                    os.environ.get(
                        "ONELOOP_MGPBD_GRASP_CONTACT_PERSISTENCE_FRAMES", "8"
                    )
                ),
                maximum_sweep_margin_m=float(
                    os.environ.get("ONELOOP_MGPBD_MAXIMUM_SWEEP_MARGIN_M", "0.003")
                ),
            )
            mgpbd_nodes, mgpbd_elements, mgpbd_tetrahedralization = (
                tetrahedralize_proxy(
                    DEFAULT_TRELLIS2_FEM_PROXY,
                    mgpbd_config,
                    enclosure_path=object_mesh_path,
                    volume_path=DEFAULT_TRELLIS2_MGPBD_DENSE_VOLUME,
                    volume_manifest_path=(
                        DEFAULT_TRELLIS2_MGPBD_DENSE_VOLUME_MANIFEST
                    ),
                )
            )
            mgpbd_object_z = (
                table_top_z
                - float(np.min(mgpbd_nodes[:, 2]))
                + mgpbd_config.particle_radius_m
                + 0.001
            )
            object_start_pos = (
                PLUSH_TELEOP_START_XY[0],
                PLUSH_TELEOP_START_XY[1],
                mgpbd_object_z,
            )
            object_target_pos = (
                OBJECT_TARGET_POS[0], OBJECT_TARGET_POS[1], mgpbd_object_z
            )
        else:
            plush_xpbd_config = XPBDPlushConfig(
                dt_s=1.0 / sim_hz,
                nominal_mass_kg=spec.nominal_mass_kg,
            )
            shell_path = (
                DEFAULT_TRELLIS2_GRANULAR_SHELL
                if use_granular_plush
                else DEFAULT_TRELLIS2_PLUSH_COLLISION
            )
            plush_topology = build_plush_topology(shell_path.resolve(), plush_xpbd_config)
        plush_visual = scene.add_entity(
            gs.morphs.Mesh(
                file=str(object_mesh_path),
                pos=object_start_pos,
                fixed=True,
                collision=False,
                convexify=False,
                decimate=False,
                enable_custom_vverts=True,
            )
        )
        # Custom-vvert rendering recomputes normals every frame.  Genesis uses
        # an unindexed, per-triangle normal path when ``surface.smooth`` is
        # false, which made the otherwise valid 60k-face TRELLIS mesh look
        # visibly shattered.  Keep the imported texture/material but force an
        # indexed mesh so dynamic updates use area-weighted vertex normals.
        for visual_geom in plush_visual.vgeoms:
            visual_geom.surface.smooth = True
        object_entity = plush_visual
        object_physics_kind = (
            "fem-plush"
            if use_fem_plush
            else (
                "mgpbd-plush"
                if use_mgpbd_plush
                else ("granular-plush" if use_granular_plush else "xpbd-plush")
            )
        )
        rigid_contact_object = None
    elif object_urdf_path is not None:
        object_mesh_path = object_urdf_path.resolve()
        if not object_visualization:
            raise ValueError("object_urdf_path requires object_visualization=True")
        object_entity = add_rigid_visual_collision_urdf(
            gs,
            scene,
            spec,
            urdf_path=object_mesh_path,
            pos=object_start_pos,
        )
        object_physics_kind = "rigid-plush"
        rigid_contact_object = object_entity
    else:
        object_mesh_path = (
            DEFAULT_MESH if object_visualization else DEFAULT_COLLISION_MESH
        ).resolve()
        object_entity = add_rigid_proxy(
            gs,
            scene,
            spec,
            mesh_path=object_mesh_path,
            pos=object_start_pos,
            visualization=object_visualization,
        )
        object_physics_kind = "rigid"
        rigid_contact_object = object_entity
    front_camera_parameters = {
        "position_m": (0.0, 0.95, 0.90),
        "lookat_m": (0.0, -0.18, 0.50),
        "up": (0.0, 0.0, 1.0),
        "vertical_fov_deg": 55.0,
    }
    if front_camera_calibration is not None:
        front_camera_calibration = front_camera_calibration.resolve()
        calibration = json.loads(front_camera_calibration.read_text(encoding="utf-8"))
        if not calibration.get("accepted", False):
            raise ValueError("front camera calibration did not pass its quality gate")
        if calibration.get("image_size_px") != [640, 480]:
            raise ValueError("front camera calibration must target 640x480 images")
        front_camera_parameters = calibration["genesis_camera"]
    front_camera = scene.add_camera(
        res=(640, 480),
        pos=front_camera_parameters["position_m"],
        lookat=front_camera_parameters["lookat_m"],
        up=front_camera_parameters["up"],
        fov=front_camera_parameters["vertical_fov_deg"],
        GUI=front_camera_gui,
    )
    hand_camera = scene.add_camera(
        res=(640, 480), pos=(0.20, -0.55, 0.72),
        lookat=(0.0, -0.26, 0.48), fov=65, GUI=False,
    )
    dynamic_normal_holder: dict[str, dict[str, object]] = {}
    build_stage_started = time.perf_counter()

    def report_build_stage(stage: str) -> None:
        print(
            "ONELOOP_BUILD_STAGE "
            f"stage={stage} elapsed_s={time.perf_counter() - build_stage_started:.3f}",
            flush=True,
        )

    if object_physics_kind in ("fem-plush", "mgpbd-plush"):
        # RasterizerContext creates its JIT during context.build(), while the
        # interactive viewer starts later in Visualizer.build(). Install the
        # full-resolution custom-vvert normal transport in that narrow window.
        # A post-scene.build installation is too late: the viewer's first
        # reset otherwise begins recomputing 294k face normals on CPU and can
        # prevent scene.build() from returning at all.
        rasterizer_context = scene.visualizer.context
        original_context_build = rasterizer_context.build

        def build_context_with_plush_normal_transport(built_scene: object) -> None:
            report_build_stage("rasterizer_context_build_begin")
            original_context_build(built_scene)
            report_build_stage("rasterizer_context_build_complete")
            dynamic_normal_holder["diagnostics"] = (
                install_custom_vvert_rest_normal_transport(
                    scene,
                    plush_visual,
                )
            )
            report_build_stage("normal_transport_installed")
            rasterizer_context.build = original_context_build

        rasterizer_context.build = build_context_with_plush_normal_transport
    report_build_stage("scene_build_begin")
    scene.build()
    report_build_stage("scene_build_complete")
    if object_physics_kind == "rigid-plush":
        for geom in rigid_contact_object.geoms:
            geom.set_sol_params(np.asarray(RIGID_PLUSH_CONTACT_SOL_PARAMS))
    plush_binding = None
    if object_physics_kind == "fem-plush":
        if plush_fem is None or plush_visual is None:
            raise RuntimeError("FEM plush build state is incomplete")
        plush_binding = FEMTetVisualBinding(plush_fem, plush_visual)
        if "diagnostics" not in dynamic_normal_holder:
            raise RuntimeError("FEM dynamic normal transport was not installed")
        plush_binding.embedding_diagnostics["dynamic_normal_smoothing"] = (
            dynamic_normal_holder["diagnostics"]
        )
        object_entity = FEMPlushObjectAdapter(plush_fem, plush_binding, fem_config)
        object_entity.update_visual()
    elif object_physics_kind == "mgpbd-plush":
        if (
            mgpbd_config is None
            or mgpbd_nodes is None
            or mgpbd_elements is None
            or mgpbd_tetrahedralization is None
            or plush_visual is None
        ):
            raise RuntimeError("MGPBD plush build state is incomplete")
        prototype = plush_visual.get_vverts()
        report_build_stage("mgpbd_solver_begin")
        mgpbd_solver = MGPBDTetSolver(
            mgpbd_nodes,
            mgpbd_elements,
            mgpbd_config,
            initial_center_m=object_start_pos,
            total_mass_kg=spec.nominal_mass_kg,
            table_height_m=table_top_z,
            device=prototype.device,
            tetrahedralization=mgpbd_tetrahedralization,
        )
        report_build_stage("mgpbd_solver_complete")
        plush_soft = MGPBDTetProvider(mgpbd_solver)
        # The dense volume encloses every runtime TRELLIS vertex, so bind each
        # visual point to one containing tet with exact barycentric weights.
        # This removes the 25 mm all-node Gaussian field that blurred local
        # contact and tore the appearance into independently moving patches.
        report_build_stage("visual_binding_begin")
        plush_binding = FEMTetVisualBinding(
            plush_soft, plush_visual, candidate_count=96
        )
        report_build_stage("visual_binding_complete")
        if "diagnostics" not in dynamic_normal_holder:
            raise RuntimeError("MGPBD dynamic normal transport was not installed")
        plush_binding.embedding_diagnostics["dynamic_normal_smoothing"] = (
            dynamic_normal_holder["diagnostics"]
        )
        adapter_class = make_mgpbd_plush_adapter_class()
        object_entity = adapter_class(
            mgpbd_solver,
            plush_soft,
            plush_binding,
            table_height_m=table_top_z,
        )
        object_entity.update_visual()
        report_build_stage("mgpbd_visual_update_complete")
    elif object_physics_kind in ("xpbd-plush", "granular-plush"):
        if plush_topology is None or plush_xpbd_config is None or plush_visual is None:
            raise RuntimeError("XPBD plush build state is incomplete")
        plush_solver = XPBDPlushSolver(
            plush_topology,
            plush_xpbd_config,
            initial_center_m=object_start_pos,
            table_height_m=table_top_z,
            maximum_boxes=6,
        )
        plush_soft = XPBDShellProvider(plush_solver, plush_visual)
        if object_physics_kind == "granular-plush":
            plush_binding = StarTetVisualBinding(
                plush_soft,
                plush_visual,
                plush_topology.shell_faces,
            )
        else:
            plush_binding = PlushVisualBinding(plush_soft, plush_visual, supports=4)
        object_entity = XPBDPlushObjectAdapter(
            plush_solver,
            plush_soft,
            plush_binding,
            table_height_m=table_top_z,
        )
        object_entity.update_visual()
    handles = SceneHandles(
        gs,
        scene,
        left,
        right,
        table,
        object_entity,
        front_camera,
        hand_camera,
        object_visualization,
        object_mesh_path,
        object_physics_kind,
        rigid_contact_object,
        plush_binding,
        table_top_z,
        object_start_pos,
        object_target_pos,
        left_base_pos,
        right_base_pos,
        table_center_xy,
        table_size_xyz,
        SAFETY_RAIL_HEIGHT_M if expanded_workspace else 0.0,
    )
    task = SO101HandoverTask(handles)
    task.reset()
    # Preserve the explicitly configured world-space camera view at the home
    # pose, then follow the gripper with that full translation *and rotation*.
    # A translation-only attachment silently replaces the look-at rotation
    # with identity and can leave the hand camera staring into the table.
    gripper = left.get_link("gripper")
    link_T = gu.trans_quat_to_T(gripper.get_pos(), gripper.get_quat())
    if hasattr(link_T, "detach"):
        link_T = link_T.detach().cpu().numpy()
    camera_T = hand_camera.get_transform()
    if hasattr(camera_T, "detach"):
        camera_T = camera_T.detach().cpu().numpy()
    hand_camera.attach(gripper, relative_transform(link_T, camera_T))
    return task, handles
