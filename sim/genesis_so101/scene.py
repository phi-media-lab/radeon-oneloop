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
        res=(480, 640), pos=(0.85, -0.85, 0.95),
        lookat=(0.0, 0.0, 0.50), fov=55, GUI=False,
    )
    hand_camera = scene.add_camera(
        res=(480, 640), pos=(0.0, -0.60, 0.72),
        lookat=(0.0, 0.0, 0.48), fov=65, GUI=False,
    )
    scene.build()
    hand_camera.attach(
        left.get_link("gripper"), gu.trans_to_T(np.asarray((0.06, 0.0, 0.04)))
    )
    handles = SceneHandles(
        gs, scene, left, right, object_entity, front_camera, hand_camera
    )
    task = SO101HandoverTask(handles)
    task.reset()
    return task, handles

