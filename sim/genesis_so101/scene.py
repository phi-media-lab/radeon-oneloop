"""Minimal dual SO-101 handover scene for Genesis v1.3.1."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
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
    entity_segmentation_index,
)
from .handover_asset import DEFAULT_MESH, add_rigid_proxy, load_spec


HOME_ARM_ACTION = (0.0, -55.0, 70.0, 70.0, 0.0, 35.0)
HOME_ACTION = HOME_ARM_ACTION + HOME_ARM_ACTION

ARM_BASE_SEPARATION_M = 0.40
# The unrotated SO-101 MJCF reaches forward along world -Y.  Separate the
# bases along world X so the two arms are side-by-side, not front-to-back.
MODEL_FORWARD_UNIT = (0.0, -1.0, 0.0)
MODEL_LATERAL_UNIT = (1.0, 0.0, 0.0)
# From the operator position at +Y looking forward toward -Y, screen-left is
# world +X. Keep the semantic left/right labels aligned with the real leaders.
LEFT_BASE_POS = (ARM_BASE_SEPARATION_M / 2.0, 0.0, 0.425)
RIGHT_BASE_POS = (-ARM_BASE_SEPARATION_M / 2.0, 0.0, 0.425)
SHARED_BASE_EULER_DEG = (0.0, 0.0, 0.0)
OBJECT_START_POS = (0.10, -0.26, 0.47)
OBJECT_TARGET_POS = (-0.10, -0.26, 0.47)
SIM_GRIPPER_SOLVER_TOLERANCE_RAD = math.radians(5.0)


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


class SO101HandoverTask:
    def __init__(self, handles: SceneHandles):
        self.handles = handles
        self._gripper_saturation_count = [0, 0]
        self._gripper_max_excursion_rad = [0.0, 0.0]
        self._appearance_binding: SafeAppearanceBinding | None = None
        self._object_segmentation_index: int | None = None
        self._appearance_render_ms: list[float] = []
        self._appearance_clipped_fraction: list[float] = []
        self._appearance_composited_frames = 0
        self._appearance_fallback_frames = 0

    def reset(self, action: Sequence[float] = HOME_ACTION) -> dict[str, Any]:
        values = require_vector(action)
        left = np.asarray(lerobot_arm_to_genesis(values[:6]), dtype=np.float32)
        right = np.asarray(lerobot_arm_to_genesis(values[6:]), dtype=np.float32)
        self.handles.left.set_dofs_position(left)
        self.handles.right.set_dofs_position(right)
        self.handles.left.control_dofs_position(left)
        self.handles.right.control_dofs_position(right)
        self.handles.object.set_pos(np.asarray(OBJECT_START_POS, dtype=np.float32))
        self.handles.object.set_quat(np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32))
        self.handles.object.set_dofs_velocity(np.zeros(6, dtype=np.float32))
        self.handles.scene.step()
        return self.observe(render=False)

    def step(self, action: Sequence[float], *, render: bool = False) -> dict[str, Any]:
        values = require_vector(action)
        self.handles.left.control_dofs_position(
            np.asarray(lerobot_arm_to_genesis(values[:6]), dtype=np.float32)
        )
        self.handles.right.control_dofs_position(
            np.asarray(lerobot_arm_to_genesis(values[6:]), dtype=np.float32)
        )
        self.handles.scene.step()
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

    def haptic_feedback(self) -> tuple[tuple[float, ...], tuple[float, float]]:
        """Return contact-gated simulated joint reaction efforts and force totals.

        All external contacts are reflected, including the target, table, and
        opposite arm; self-collisions are excluded. The fixed bases are placed
        clear of the tabletop, so ordinary support does not create a standing
        feedback signal.
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
        return efforts, (left_force, right_force)

    def success(self) -> bool:
        position = self._array(self.handles.object.get_pos()).reshape(-1)
        target = np.asarray(OBJECT_TARGET_POS)
        return bool(np.linalg.norm(position - target) < 0.08)


def build(
    asset_root: Path,
    *,
    seed: int = 20260803,
    show_viewer: bool = False,
    workspace_texture: Path | None = None,
    front_camera_calibration: Path | None = None,
    front_camera_gui: bool = False,
) -> tuple[SO101HandoverTask, SceneHandles]:
    import genesis as gs
    import genesis.utils.geom as gu

    fetch(asset_root)
    model = asset_root / "so101_new_calib.xml"
    gs.init(backend=gs.amdgpu, seed=seed)
    if gs.backend != gs.amdgpu:
        raise RuntimeError(f"Genesis did not select the AMD GPU backend: {gs.backend}")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 120.0, substeps=2, gravity=(0.0, 0.0, -9.81)
        ),
        rigid_options=gs.options.RigidOptions(enable_collision=True, enable_joint_limit=True),
        vis_options=gs.options.VisOptions(
            ambient_light=(0.55, 0.55, 0.55),
            shadow=True,
            segmentation_level="link",
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, 1.05, 0.90),
            camera_lookat=(0.0, -0.18, 0.50),
            camera_up=(0.0, 0.0, 1.0),
            camera_fov=55.0,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    table_surface = None
    workspace_plane = Path(__file__).resolve().parent / "assets_generated" / "workspace_plane.obj"
    if workspace_texture is not None:
        workspace_texture = workspace_texture.resolve()
        if not workspace_texture.is_file():
            raise FileNotFoundError(f"workspace texture does not exist: {workspace_texture}")
        table_surface = gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(image_path=str(workspace_texture)),
            roughness=0.9,
        )
    table = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.385),
            size=(1.2, 0.8, 0.05),
            fixed=True,
            visualization=workspace_texture is None,
        ),
        surface=table_surface if workspace_texture is None else None,
    )
    if workspace_texture is not None:
        if not workspace_plane.is_file():
            raise FileNotFoundError(f"workspace UV plane is missing: {workspace_plane}")
        scene.add_entity(
            gs.morphs.Mesh(
                file=str(workspace_plane),
                pos=(0.0, 0.0, 0.4105),
                fixed=True,
                collision=False,
                convexify=False,
                decimate=False,
            ),
            surface=table_surface,
        )
    left = scene.add_entity(
        gs.morphs.MJCF(
            file=str(model), pos=LEFT_BASE_POS, euler=SHARED_BASE_EULER_DEG
        )
    )
    right = scene.add_entity(
        gs.morphs.MJCF(
            file=str(model), pos=RIGHT_BASE_POS, euler=SHARED_BASE_EULER_DEG
        )
    )
    object_entity = add_rigid_proxy(
        gs,
        scene,
        load_spec(),
        mesh_path=DEFAULT_MESH,
        pos=OBJECT_START_POS,
    )
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
    scene.build()
    handles = SceneHandles(
        gs, scene, left, right, table, object_entity, front_camera, hand_camera
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
