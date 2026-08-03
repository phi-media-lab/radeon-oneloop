"""Minimal dual SO-101 handover scene for Genesis v1.3.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from radeon_oneloop.contracts import (
    CAMERA_KEYS,
    genesis_arm_to_lerobot,
    lerobot_arm_to_genesis,
    require_vector,
)

from .fetch_assets import fetch


HOME_ACTION = (
    15.0, -55.0, 70.0, 70.0, 0.0, 35.0,
    -15.0, 55.0, 70.0, -70.0, 0.0, 35.0,
)


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
    object: Any
    front_camera: Any
    hand_camera: Any


class SO101HandoverTask:
    def __init__(self, handles: SceneHandles):
        self.handles = handles

    def reset(self, action: Sequence[float] = HOME_ACTION) -> dict[str, Any]:
        values = require_vector(action)
        left = np.asarray(lerobot_arm_to_genesis(values[:6]), dtype=np.float32)
        right = np.asarray(lerobot_arm_to_genesis(values[6:]), dtype=np.float32)
        self.handles.left.set_dofs_position(left)
        self.handles.right.set_dofs_position(right)
        self.handles.left.control_dofs_position(left)
        self.handles.right.control_dofs_position(right)
        self.handles.object.set_pos(np.asarray((0.0, -0.12, 0.47), dtype=np.float32))
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

    def observe(self, *, render: bool = True) -> dict[str, Any]:
        left = genesis_arm_to_lerobot(
            self._array(self.handles.left.get_dofs_position()).reshape(-1).tolist()
        )
        right = genesis_arm_to_lerobot(
            self._array(self.handles.right.get_dofs_position()).reshape(-1).tolist()
        )
        observation: dict[str, Any] = {
            "observation.state": np.asarray(left + right, dtype=np.float32)
        }
        if render:
            self.handles.hand_camera.move_to_attach()
            front_rgb, _, _, _ = self.handles.front_camera.render(rgb=True)
            hand_rgb, _, _, _ = self.handles.hand_camera.render(rgb=True)
            observation[CAMERA_KEYS[0]] = self._array(front_rgb)
            observation[CAMERA_KEYS[1]] = self._array(hand_rgb)
        return observation

    def success(self) -> bool:
        position = self._array(self.handles.object.get_pos()).reshape(-1)
        target = np.asarray((0.0, 0.20, 0.47))
        return bool(np.linalg.norm(position - target) < 0.08)


def build(
    asset_root: Path, *, seed: int = 20260803, show_viewer: bool = False
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
            ambient_light=(0.55, 0.55, 0.55), shadow=True
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(pos=(0.0, 0.0, 0.385), size=(1.2, 0.8, 0.05), fixed=True)
    )
    left = scene.add_entity(
        gs.morphs.MJCF(
            file=str(model), pos=(0.0, -0.32, 0.425), euler=(0.0, 0.0, 90.0)
        )
    )
    right = scene.add_entity(
        gs.morphs.MJCF(
            file=str(model), pos=(0.0, 0.32, 0.425), euler=(0.0, 0.0, -90.0)
        )
    )
    object_entity = scene.add_entity(
        gs.morphs.Sphere(pos=(0.0, -0.12, 0.47), radius=0.035),
        material=gs.materials.Rigid(rho=450.0, friction=0.8),
        surface=gs.surfaces.Default(color=(0.85, 0.12, 0.12, 1.0)),
    )
    front_camera = scene.add_camera(
        res=(640, 480), pos=(0.85, -0.85, 0.95),
        lookat=(0.0, 0.0, 0.50), fov=55, GUI=False,
    )
    hand_camera = scene.add_camera(
        res=(640, 480), pos=(0.0, -0.60, 0.72),
        lookat=(0.0, 0.0, 0.48), fov=65, GUI=False,
    )
    scene.build()
    handles = SceneHandles(
        gs, scene, left, right, object_entity, front_camera, hand_camera
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
