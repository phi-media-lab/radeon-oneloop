"""Granular-plush visual cage and native Genesis comparison model.

The physical object has three deliberately separate layers:

* the accepted path uses a custom closed XPBD shell and discrete hard grains;
* the textured TRELLIS.2 mesh owns pixels only and is embedded in a star-tet
  deformation cage formed by the closed convex shell surface.

MGPBD and homogeneous FEM remain useful solver/material baselines, but neither
represents a granular fill.  The native Genesis PBD-cloth + MPM-sand classes
below are retained as a measured negative comparison: the closed PBD shell was
numerically unstable while unconstrained MPM sand dispersed.  The exact
``StarTetVisualBinding`` is shared by the accepted custom-XPBD mainline.  Heavy
Genesis, Torch, and SciPy imports remain lazy so pure topology tests run in the
lightweight development environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .fem_plush import locate_points_in_tets


@dataclass(frozen=True)
class GranularPlushConfig:
    """Settings for the native Genesis PBD+MPM comparison experiment."""

    sim_hz: float = 120.0
    substeps: int = 40
    pbd_particle_size_m: float = 0.006
    pbd_hash_cell_size_m: float = 0.012
    pbd_stretch_iterations: int = 6
    pbd_bending_iterations: int = 4
    shell_areal_density_kg_m2: float = 0.35
    shell_static_friction: float = 1.20
    shell_kinetic_friction: float = 1.00
    shell_stretch_compliance_m_n: float = 1.0e-8
    shell_bending_compliance_rad_n: float = 1.0e-7
    shell_stretch_relaxation: float = 0.80
    shell_bending_relaxation: float = 0.55
    shell_air_resistance: float = 0.003
    mpm_particle_size_m: float = 0.006
    mpm_grid_density: float = 96.0
    shell_scale: float = 1.0
    core_scale: float = 0.94
    core_youngs_modulus_pa: float = 5.0e4
    core_poissons_ratio: float = 0.20
    core_effective_density_kg_m3: float = 105.0
    core_friction_angle_deg: float = 48.0
    visual_embedding_candidate_count: int = 96

    def validate(self) -> None:
        if self.sim_hz <= 0.0 or self.substeps < 1:
            raise ValueError("granular plush clock must be positive")
        if self.pbd_particle_size_m <= 0.0 or self.mpm_particle_size_m <= 0.0:
            raise ValueError("granular plush particle sizes must be positive")
        if self.pbd_hash_cell_size_m < self.pbd_particle_size_m:
            raise ValueError("PBD hash cell cannot be smaller than a shell particle")
        if self.pbd_stretch_iterations < 1 or self.pbd_bending_iterations < 1:
            raise ValueError("PBD shell iteration counts must be positive")
        if self.shell_areal_density_kg_m2 <= 0.0:
            raise ValueError("shell areal density must be positive")
        if self.shell_static_friction < 0.0 or self.shell_kinetic_friction < 0.0:
            raise ValueError("shell friction cannot be negative")
        if not 0.0 <= self.shell_stretch_relaxation <= 1.0:
            raise ValueError("shell stretch relaxation must be in [0, 1]")
        if not 0.0 <= self.shell_bending_relaxation <= 1.0:
            raise ValueError("shell bending relaxation must be in [0, 1]")
        if self.shell_scale < 1.0:
            raise ValueError("shell scale must preserve the conservative enclosure")
        if not 0.0 < self.core_scale < self.shell_scale:
            raise ValueError("granular core scale must be in (0, shell_scale)")
        if self.mpm_grid_density <= 0.0 or self.core_youngs_modulus_pa <= 0.0:
            raise ValueError("MPM grid density and Young's modulus must be positive")
        if not -1.0 < self.core_poissons_ratio < 0.5:
            raise ValueError("core Poisson ratio must be in (-1, 0.5)")
        if self.core_effective_density_kg_m3 <= 0.0:
            raise ValueError("core effective density must be positive")
        if not 0.0 < self.core_friction_angle_deg < 90.0:
            raise ValueError("core friction angle must be in (0, 90) degrees")
        if self.visual_embedding_candidate_count < 1:
            raise ValueError("visual embedding candidate count must be positive")

    @property
    def dt_s(self) -> float:
        return 1.0 / self.sim_hz

    def to_dict(self) -> dict[str, float | int]:
        self.validate()
        return asdict(self)


def star_tet_cage(
    shell_vertices: np.ndarray, shell_faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Form a center-to-surface tetrahedral fan for a closed convex cage.

    The conservative TRELLIS contact proxy is an appearance-derived convex
    hull.  Joining every oriented surface triangle to one interior center
    partitions that hull into tetrahedra and gives every enclosed visual point
    exact affine coordinates.  Face winding is irrelevant for point location.
    """

    vertices = np.asarray(shell_vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(shell_faces, dtype=np.int64).reshape(-1, 3)
    if len(vertices) < 4 or len(faces) < 4:
        raise ValueError("star-tet cage needs a non-empty closed triangle mesh")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("shell face index lies outside the vertex array")
    center = vertices.mean(axis=0, keepdims=True)
    cage_vertices = np.concatenate((vertices, center), axis=0)
    center_index = len(vertices)
    cage_tets = np.column_stack(
        (
            np.full(len(faces), center_index, dtype=np.int64),
            faces,
        )
    )
    volumes6 = np.abs(
        np.einsum(
            "ij,ij->i",
            np.cross(
                cage_vertices[cage_tets[:, 1]] - cage_vertices[cage_tets[:, 0]],
                cage_vertices[cage_tets[:, 2]] - cage_vertices[cage_tets[:, 0]],
            ),
            cage_vertices[cage_tets[:, 3]] - cage_vertices[cage_tets[:, 0]],
        )
    )
    if np.any(volumes6 <= 1.0e-15):
        raise ValueError("star-tet cage contains a degenerate surface triangle")
    return cage_vertices, cage_tets


class ClosedShellCageVisualBinding:
    """Embed a textured visual exactly inside the deforming PBD shell cage."""

    def __init__(self, shell: object, visual: object, *, candidate_count: int = 96):
        import torch

        self.shell = shell
        self.visual = visual
        self.shell_vvert_start = int(shell._vvert_start)
        self.shell_vvert_count = int(shell.n_vverts)
        rest_shell = np.asarray(shell._vverts, dtype=np.float64).reshape(-1, 3)
        shell_faces = np.asarray(shell._vfaces, dtype=np.int64).reshape(-1, 3)
        rest_visual_tensor = visual.get_vverts().detach().reshape(-1, 3).clone()
        rest_visual = rest_visual_tensor.cpu().numpy().astype(np.float64, copy=False)
        cage_vertices, cage_tets = star_tet_cage(rest_shell, shell_faces)
        element_indices, weights, diagnostics = locate_points_in_tets(
            rest_visual,
            cage_vertices,
            cage_tets,
            candidate_count=candidate_count,
            tolerance=3.0e-5,
        )
        vertex_indices = cage_tets[element_indices]
        self.vertex_indices = torch.as_tensor(
            vertex_indices, dtype=torch.long, device=rest_visual_tensor.device
        )
        self.weights = torch.as_tensor(
            weights, dtype=rest_visual_tensor.dtype, device=rest_visual_tensor.device
        )
        self.rest_visual = rest_visual_tensor
        self.embedding_diagnostics = {
            **diagnostics,
            "binding": "closed_convex_shell_star_tet_barycentric",
            "shell_visual_vertices": self.shell_vvert_count,
            "shell_faces": int(len(shell_faces)),
            "virtual_center_vertices": 1,
        }

    def current_shell_vertices(self):
        self.shell.solver.update_render_fields()
        all_positions = self.shell.solver.vverts_render.pos.to_torch()
        start = self.shell_vvert_start
        stop = start + self.shell_vvert_count
        return all_positions[start:stop, 0, :]

    def current_visual_vertices(self):
        import torch

        shell_positions = self.current_shell_vertices()
        center = shell_positions.mean(dim=0, keepdim=True)
        cage_positions = torch.cat((shell_positions, center), dim=0)
        return (
            cage_positions[self.vertex_indices] * self.weights.unsqueeze(-1)
        ).sum(dim=1)

    def update(self) -> None:
        import torch

        with torch.no_grad():
            self.visual.set_vverts(self.current_visual_vertices())


class StarTetVisualBinding:
    """Exact star-tet visual embedding for any Torch shell-position provider.

    The custom Taichi XPBD solver exposes its closed shell through a lightweight
    provider.  Unlike the retired nearest-particle residual skinning, this
    binding reconstructs every TRELLIS vertex from one containing tetrahedron,
    so adjacent visual triangles cannot be pulled apart by unrelated supports.
    """

    def __init__(
        self,
        shell_provider: object,
        visual: object,
        shell_faces: np.ndarray,
        *,
        candidate_count: int = 96,
    ):
        import torch

        self.shell_provider = shell_provider
        self.visual = visual
        rest_shell_tensor = (
            shell_provider.get_particles_pos().detach().reshape(-1, 3).clone()
        )
        rest_visual_tensor = visual.get_vverts().detach().reshape(-1, 3).clone()
        rest_shell = rest_shell_tensor.cpu().numpy().astype(np.float64, copy=False)
        rest_visual = rest_visual_tensor.cpu().numpy().astype(np.float64, copy=False)
        cage_vertices, cage_tets = star_tet_cage(rest_shell, shell_faces)
        element_indices, weights, diagnostics = locate_points_in_tets(
            rest_visual,
            cage_vertices,
            cage_tets,
            candidate_count=candidate_count,
            tolerance=3.0e-5,
        )
        self.vertex_indices = torch.as_tensor(
            cage_tets[element_indices],
            dtype=torch.long,
            device=rest_shell_tensor.device,
        )
        self.weights = torch.as_tensor(
            weights,
            dtype=rest_shell_tensor.dtype,
            device=rest_shell_tensor.device,
        )
        self.rest_shell = rest_shell_tensor
        # ``XPBDPlushObjectAdapter`` deliberately accepts either the retired
        # nearest-particle binder or this exact cage binder.  Keep the common
        # pose/diagnostic attributes so resets cannot depend on which visual
        # transport is selected.
        self.rest_particles = rest_shell_tensor
        self.rest_visual = rest_visual_tensor
        self.rest_center = self.rest_particles.mean(dim=0)
        self.rest_particles_centered = self.rest_particles - self.rest_center
        self.rest_visual_centered = self.rest_visual - self.rest_center
        self.embedding_diagnostics = {
            **diagnostics,
            "binding": "custom_xpbd_closed_shell_star_tet_barycentric",
            "shell_vertices": int(len(rest_shell)),
            "shell_faces": int(len(shell_faces)),
            "virtual_center_vertices": 1,
        }

    def current_shell_vertices(self):
        return self.shell_provider.get_particles_pos().reshape(-1, 3)

    def rigid_transform(self):
        """Return the shell, centroid and best-fit row-vector rotation."""

        import torch

        shell_positions = self.current_shell_vertices()
        center = shell_positions.mean(dim=0)
        centered = shell_positions - center
        covariance = self.rest_particles_centered.T @ centered
        left, _singular, right_t = torch.linalg.svd(covariance)
        rotation = left @ right_t
        if torch.linalg.det(rotation) < 0:
            left = left.clone()
            left[:, -1] *= -1.0
            rotation = left @ right_t
        return shell_positions, center, rotation

    def current_visual_vertices(self):
        import torch

        shell_positions = self.current_shell_vertices()
        center = shell_positions.mean(dim=0, keepdim=True)
        cage_positions = torch.cat((shell_positions, center), dim=0)
        return (
            cage_positions[self.vertex_indices] * self.weights.unsqueeze(-1)
        ).sum(dim=1)

    def update(self) -> None:
        import torch

        with torch.no_grad():
            self.visual.set_vverts(self.current_visual_vertices())


class GranularPlushObjectAdapter:
    """Expose the composite PBD/MPM body through the task object interface."""

    def __init__(
        self,
        shell: object,
        core: object,
        binding: ClosedShellCageVisualBinding,
        config: GranularPlushConfig,
    ):
        self.shell = shell
        self.core = core
        self.binding = binding
        self.config = config
        self.rest_shell_particles = shell.get_particles_pos().detach().reshape(-1, 3).clone()
        self.rest_core_particles = core.get_particles_pos().detach().reshape(-1, 3).clone()
        self.rest_center = self.rest_shell_particles.mean(dim=0)
        self.rest_shell_centered = self.rest_shell_particles - self.rest_center

    def _rigid_transform(self) -> tuple[Any, Any, Any]:
        import torch

        shell = self.shell.get_particles_pos().reshape(-1, 3)
        center = shell.mean(dim=0)
        centered = shell - center
        covariance = self.rest_shell_centered.T @ centered
        left, _singular, right_t = torch.linalg.svd(covariance)
        rotation = left @ right_t
        if torch.linalg.det(rotation) < 0:
            left = left.clone()
            left[:, -1] *= -1.0
            rotation = left @ right_t
        return shell, center, rotation

    def get_pos(self) -> np.ndarray:
        _shell, center, _rotation = self._rigid_transform()
        return center.detach().cpu().numpy().astype(np.float32)

    def get_quat(self) -> np.ndarray:
        from .plush_physics import matrix_to_quaternion_wxyz

        _shell, _center, rotation = self._rigid_transform()
        return matrix_to_quaternion_wxyz(
            rotation.detach().cpu().numpy()
        ).astype(np.float32)

    def set_pos(self, position: object) -> None:
        import torch

        target = torch.as_tensor(
            np.asarray(position, dtype=np.float32).reshape(3),
            dtype=self.rest_shell_particles.dtype,
            device=self.rest_shell_particles.device,
        )
        shell = self.shell.get_particles_pos().reshape(-1, 3)
        core = self.core.get_particles_pos().reshape(-1, 3)
        delta = target - shell.mean(dim=0)
        self.shell.set_particles_pos(shell + delta)
        self.core.set_particles_pos(core + delta)

    def set_quat(self, quaternion: object) -> None:
        import torch

        from .plush_physics import quaternion_wxyz_to_matrix

        desired = torch.as_tensor(
            quaternion_wxyz_to_matrix(quaternion),
            dtype=self.rest_shell_particles.dtype,
            device=self.rest_shell_particles.device,
        )
        shell, center, current = self._rigid_transform()
        delta_rotation = current.T @ desired
        core = self.core.get_particles_pos().reshape(-1, 3)
        self.shell.set_particles_pos((shell - center) @ delta_rotation + center)
        self.core.set_particles_pos((core - center) @ delta_rotation + center)

    def set_dofs_velocity(self, velocity: object) -> None:
        values = np.asarray(velocity, dtype=np.float32).reshape(-1)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("granular plush reset velocity must contain six finite values")
        if np.any(np.abs(values) > 1.0e-8):
            raise ValueError("granular plush adapter currently supports only zero reset velocity")
        self.shell.set_particles_vel(0.0)
        self.core.set_particles_vel(0.0)

    def update_visual(self) -> None:
        self.binding.update()

    def diagnostics(self) -> dict[str, object]:
        shell = self.shell.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
        core = self.core.get_particles_pos().detach().cpu().numpy().reshape(-1, 3)
        return {
            "kind": "native_genesis_closed_cloth_shell_plus_mpm_granular_core",
            "authoritative_contact": (
                "Genesis_rigid_PBD_and_rigid_MPM_with_two_way_MPM_PBD_coupling"
            ),
            "shell_particles": int(len(shell)),
            "core_material_points": int(len(core)),
            "shell_center_m": shell.mean(axis=0).tolist(),
            "shell_extents_m": np.ptp(shell, axis=0).tolist(),
            "core_center_m": core.mean(axis=0).tolist(),
            "core_extents_m": np.ptp(core, axis=0).tolist(),
            "configuration": self.config.to_dict(),
            "visual_embedding": dict(self.binding.embedding_diagnostics),
            "material_interpretation": (
                "tight fabric boundary plus pressure-dependent frictional granular continuum"
            ),
            "not_homogeneous_solid": True,
        }
