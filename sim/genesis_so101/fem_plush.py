"""Volumetric FEM plush support for the TRELLIS.2 handover object.

The physical object is a low-resolution, watertight tetrahedral body.  The
textured TRELLIS.2 surface is embedded into the rest tetrahedra with
barycentric coordinates, so the renderer follows the continuous volumetric
deformation instead of interpolating unrelated nearest surface vertices.

Genesis provides the Radeon FEM and SAP contact kernels.  Imports that require
Genesis, Torch, or SciPy stay inside methods so topology tests remain runnable
on the lightweight development host.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FEMPlushConfig:
    """Material and solver settings for a tightly stuffed effective solid."""

    # A tightly filled fabric shell is much less compressible than the old
    # homogeneous-foam baseline.  Keep nu safely below the singular 0.5 limit
    # while giving the native corotated FEM a separate bulk response instead
    # of relying on a post-contact volume barrier.
    youngs_modulus_pa: float = 2.5e5
    poissons_ratio: float = 0.45
    # 40 g nominal mass / 3.9881e-4 m^3 dense watertight proxy volume.
    density_kg_m3: float = 100.3
    friction_mu: float = 1.80
    hydroelastic_modulus_pa: float = 5.0e5
    tet_max_volume_m3: float = -1.0
    tet_min_ratio: float = 1.2
    tet_min_dihedral_deg: int = 10
    damping: float = 0.12
    damping_alpha: float = 0.45
    damping_beta: float = 0.001
    newton_iterations: int = 1
    pcg_iterations: int = 12
    pcg_threshold: float = 5.0e-4
    sap_iterations: int = 2
    sap_pcg_iterations: int = 30
    sap_pcg_threshold: float = 2.0e-4

    def validate(self) -> None:
        if self.youngs_modulus_pa <= 0.0:
            raise ValueError("FEM Young's modulus must be positive")
        if not -1.0 < self.poissons_ratio < 0.5:
            raise ValueError("FEM Poisson ratio must be in (-1, 0.5)")
        if self.density_kg_m3 <= 0.0 or self.friction_mu < 0.0:
            raise ValueError("FEM density must be positive and friction nonnegative")
        if self.hydroelastic_modulus_pa <= 0.0:
            raise ValueError("FEM hydroelastic modulus must be positive")
        if self.tet_max_volume_m3 == 0.0:
            raise ValueError("tet max volume must be negative (automatic) or positive")
        if self.tet_min_ratio <= 0.0 or self.tet_min_dihedral_deg < 0:
            raise ValueError("invalid TetGen quality settings")
        for name, value in (
            ("newton_iterations", self.newton_iterations),
            ("pcg_iterations", self.pcg_iterations),
            ("sap_iterations", self.sap_iterations),
            ("sap_pcg_iterations", self.sap_pcg_iterations),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.pcg_threshold <= 0.0 or self.sap_pcg_threshold <= 0.0:
            raise ValueError("PCG thresholds must be positive")

    def to_dict(self) -> dict[str, float | int]:
        self.validate()
        return asdict(self)


def tet_barycentric_coordinates(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Return barycentric coordinates for paired points and tetrahedra.

    ``points`` has shape ``(..., 3)`` and ``tetrahedra`` has shape
    ``(..., 4, 3)``.  Singular candidates are returned as NaNs, allowing the
    caller to reject them while considering neighbouring tetrahedra.
    """

    points = np.asarray(points, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.float64)
    if tetrahedra.shape[:-2] != points.shape[:-1] or tetrahedra.shape[-2:] != (4, 3):
        raise ValueError("paired tetrahedra must have shape points.shape[:-1] + (4, 3)")
    base = tetrahedra[..., 3, :]
    matrix = np.stack(
        (
            tetrahedra[..., 0, :] - base,
            tetrahedra[..., 1, :] - base,
            tetrahedra[..., 2, :] - base,
        ),
        axis=-1,
    )
    rhs = points - base
    flat_matrix = matrix.reshape(-1, 3, 3)
    flat_rhs = rhs.reshape(-1, 3)
    determinants = np.linalg.det(flat_matrix)
    valid = np.abs(determinants) > 1.0e-14
    first_three = np.full((len(flat_matrix), 3), np.nan, dtype=np.float64)
    if np.any(valid):
        first_three[valid] = np.linalg.solve(flat_matrix[valid], flat_rhs[valid])
    barycentric = np.concatenate(
        (first_three, 1.0 - first_three.sum(axis=1, keepdims=True)), axis=1
    )
    return barycentric.reshape(points.shape[:-1] + (4,))


def locate_points_in_tets(
    points: np.ndarray,
    vertices: np.ndarray,
    elements: np.ndarray,
    *,
    candidate_count: int = 48,
    batch_size: int = 2048,
    tolerance: float = 2.0e-5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Embed points in a tetrahedral volume using nearest-centroid candidates.

    The TRELLIS visual is enclosed by the conservative convex FEM proxy, so a
    containing tetrahedron should exist for every vertex.  A small tolerance
    absorbs surface roundoff; a failure is fatal rather than silently binding
    a visual vertex outside the physical body.
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    elements = np.asarray(elements, dtype=np.int64).reshape(-1, 4)
    if not len(points) or not len(vertices) or not len(elements):
        raise ValueError("tet embedding inputs must be non-empty")
    if int(elements.min()) < 0 or int(elements.max()) >= len(vertices):
        raise ValueError("tet element index lies outside the vertex array")
    if candidate_count < 1 or batch_size < 1 or tolerance < 0.0:
        raise ValueError("invalid tet locator settings")

    candidate_count = min(int(candidate_count), len(elements))
    centers = vertices[elements].mean(axis=1)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centers)
    except ModuleNotFoundError:
        tree = None
    chosen_elements = np.empty(len(points), dtype=np.int32)
    chosen_weights = np.empty((len(points), 4), dtype=np.float32)
    minimum_weights = np.empty(len(points), dtype=np.float64)

    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        batch = points[start:stop]
        if tree is not None:
            _distances, candidates = tree.query(
                batch, k=candidate_count, workers=-1
            )
        else:
            squared_distances = np.sum(
                (batch[:, None, :] - centers[None, :, :]) ** 2, axis=-1
            )
            candidates = np.argpartition(
                squared_distances, candidate_count - 1, axis=1
            )[:, :candidate_count]
        if candidates.ndim == 1:
            candidates = candidates[:, None]
        candidate_tets = vertices[elements[candidates]]
        repeated_points = np.broadcast_to(batch[:, None, :], candidates.shape + (3,))
        weights = tet_barycentric_coordinates(repeated_points, candidate_tets)
        scores = np.nanmin(weights, axis=-1)
        scores[~np.isfinite(scores)] = -np.inf
        best = np.argmax(scores, axis=1)
        rows = np.arange(len(batch))
        best_scores = scores[rows, best]
        selected_elements = candidates[rows, best].astype(np.int32)
        selected = weights[rows, best]
        # A centroid-nearest query can miss a thin surface tet even when the
        # point is enclosed.  Escalate only those rare misses to an exhaustive
        # element check; this keeps normal startup fast without weakening the
        # enclosure invariant.
        for row in np.flatnonzero(best_scores < -tolerance):
            all_weights = tet_barycentric_coordinates(
                np.broadcast_to(batch[row], (len(elements), 3)),
                vertices[elements],
            )
            all_scores = np.nanmin(all_weights, axis=-1)
            all_scores[~np.isfinite(all_scores)] = -np.inf
            all_best = int(np.argmax(all_scores))
            selected_elements[row] = all_best
            selected[row] = all_weights[all_best]
            best_scores[row] = all_scores[all_best]
        if np.any(best_scores < -tolerance):
            failures = int(np.count_nonzero(best_scores < -tolerance))
            worst = float(np.min(best_scores))
            raise ValueError(
                f"{failures} visual vertices are outside the FEM volume; "
                f"worst barycentric weight={worst:.6g}"
            )
        # Clamp only tolerance-scale negative weights and renormalize so the
        # rest-pose reconstruction remains a convex affine combination.
        selected = np.maximum(selected, 0.0)
        selected /= selected.sum(axis=1, keepdims=True)
        chosen_elements[start:stop] = selected_elements
        chosen_weights[start:stop] = selected.astype(np.float32)
        minimum_weights[start:stop] = best_scores

    reconstruction = np.einsum(
        "ni,nij->nj", chosen_weights, vertices[elements[chosen_elements]]
    )
    errors = np.linalg.norm(reconstruction - points, axis=1)
    diagnostics: dict[str, float | int] = {
        "visual_vertices": int(len(points)),
        "physics_vertices": int(len(vertices)),
        "physics_tetrahedra": int(len(elements)),
        "candidate_count": int(candidate_count),
        "minimum_barycentric_weight": float(np.min(minimum_weights)),
        "reconstruction_error_m_max": float(np.max(errors)),
        "reconstruction_error_m_p95": float(np.percentile(errors, 95)),
    }
    return chosen_elements, chosen_weights, diagnostics


class FEMTetVisualBinding:
    """Drive a textured visual by barycentric embedding in FEM tetrahedra."""

    def __init__(self, fem: object, visual: object, *, candidate_count: int = 96):
        import torch

        self.fem = fem
        self.visual = visual
        rest_physics = fem.get_state().pos.detach().reshape(-1, 3).clone()
        rest_visual = visual.get_vverts().detach().reshape(-1, 3).clone()
        self.rest_physics = rest_physics
        self.rest_visual = rest_visual
        self.rest_center = rest_physics.mean(dim=0)
        self.rest_physics_centered = rest_physics - self.rest_center
        element_indices, weights, diagnostics = locate_points_in_tets(
            rest_visual.cpu().numpy(),
            rest_physics.cpu().numpy(),
            np.asarray(fem.elems, dtype=np.int64),
            candidate_count=candidate_count,
        )
        self.element_indices_np = element_indices
        self.elements_np = np.asarray(fem.elems, dtype=np.int64)
        vertex_indices = self.elements_np[element_indices]
        self.vertex_indices = torch.as_tensor(
            vertex_indices, dtype=torch.long, device=rest_physics.device
        )
        self.weights = torch.as_tensor(
            weights, dtype=rest_physics.dtype, device=rest_physics.device
        )
        faces: list[np.ndarray] = []
        vertex_offset = 0
        for vgeom in visual.vgeoms:
            local_faces = np.asarray(vgeom.vmesh.faces, dtype=np.int64).reshape(-1, 3)
            faces.append(local_faces + vertex_offset)
            vertex_offset += int(vgeom.n_vverts)
        if vertex_offset != len(rest_visual) or not faces:
            raise RuntimeError("FEM visual face topology does not match custom vertices")
        self.faces = torch.as_tensor(
            np.concatenate(faces, axis=0),
            dtype=torch.long,
            device=rest_physics.device,
        )
        rest_triangles = rest_visual[self.faces]
        self.rest_face_cross = torch.linalg.cross(
            rest_triangles[:, 1] - rest_triangles[:, 0],
            rest_triangles[:, 2] - rest_triangles[:, 0],
        )
        self.rest_face_double_area = torch.linalg.norm(
            self.rest_face_cross, dim=1
        )
        self.rest_face_valid = self.rest_face_double_area > 1.0e-12
        self.rest_face_double_area = self.rest_face_double_area.clamp_min(1.0e-12)
        self._last_visual_positions = rest_visual.detach().clone()
        self.embedding_diagnostics = {
            **diagnostics,
            "visual_faces": int(len(self.faces)),
        }

    def current_positions(self):
        return self.fem.get_state().pos.reshape(-1, 3)

    def rigid_transform(self) -> tuple[Any, Any, Any]:
        import torch

        positions = self.current_positions()
        center = positions.mean(dim=0)
        centered = positions - center
        covariance = self.rest_physics_centered.T @ centered
        left, _singular, right_t = torch.linalg.svd(covariance)
        rotation = left @ right_t
        if torch.linalg.det(rotation) < 0:
            left = left.clone()
            left[:, -1] *= -1.0
            rotation = left @ right_t
        return positions, center, rotation

    def update(self) -> None:
        import torch

        with torch.no_grad():
            positions = self.current_positions()
            visual_positions = (
                positions[self.vertex_indices] * self.weights.unsqueeze(-1)
            ).sum(dim=1)
            self._last_visual_positions = visual_positions.detach().clone()
            self.visual.set_vverts(visual_positions)

    def surface_quality_diagnostics(self) -> dict[str, object]:
        """Measure visible triangle folding independently of FEM tet validity."""

        import torch

        with torch.no_grad():
            _positions, _center, rotation = self.rigid_transform()
            triangles = self._last_visual_positions[self.faces]
            face_cross = torch.linalg.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            double_area = torch.linalg.norm(face_cross, dim=1)
            rigid_rest_cross = self.rest_face_cross @ rotation
            orientation_cosine = torch.sum(
                face_cross * rigid_rest_cross, dim=1
            ) / (
                double_area.clamp_min(1.0e-12) * self.rest_face_double_area
            )
            area_ratio = double_area / self.rest_face_double_area
            finite_values = torch.isfinite(area_ratio) & torch.isfinite(
                orientation_cosine
            )
            evaluable = finite_values & self.rest_face_valid
            flipped = evaluable & (orientation_cosine <= 0.0)
            current_degenerate = evaluable & (area_ratio <= 1.0e-3)
            finite_ratio = area_ratio[evaluable]
            source_degenerate = int(torch.count_nonzero(~self.rest_face_valid))
            if not len(finite_ratio):
                return {
                    "finite": False,
                    "faces": int(len(self.faces)),
                    "flipped_faces": int(len(self.faces)),
                    "degenerate_faces": int(len(self.faces)),
                }
            return {
                "finite": bool(torch.all(finite_values)),
                "faces": int(len(self.faces)),
                "evaluated_faces": int(torch.count_nonzero(evaluable)),
                "source_degenerate_faces": source_degenerate,
                "flipped_faces": int(torch.count_nonzero(flipped)),
                "flipped_fraction": float(torch.mean(flipped.float())),
                "degenerate_faces": source_degenerate
                + int(torch.count_nonzero(current_degenerate)),
                "degenerate_fraction": float(
                    (
                        source_degenerate
                        + int(torch.count_nonzero(current_degenerate))
                    )
                    / len(self.faces)
                ),
                "area_ratio": {
                    "minimum": float(torch.min(finite_ratio)),
                    "p01": float(torch.quantile(finite_ratio, 0.01)),
                    "median": float(torch.median(finite_ratio)),
                    "p99": float(torch.quantile(finite_ratio, 0.99)),
                    "maximum": float(torch.max(finite_ratio)),
                },
            }


class FEMPlushObjectAdapter:
    """Expose FEM state through the handover task's object interface."""

    def __init__(self, fem: object, binding: FEMTetVisualBinding, config: FEMPlushConfig):
        self.fem = fem
        self.binding = binding
        self.config = config
        self.rest_positions = binding.rest_physics.detach().clone()
        self.rest_center = self.rest_positions.mean(dim=0)
        elements = np.asarray(fem.elems, dtype=np.int64)
        rest_np = self.rest_positions.cpu().numpy()
        rest_tets = rest_np[elements]
        self._elements = elements
        self._rest_signed_six_volume = np.einsum(
            "ij,ij->i",
            np.cross(rest_tets[:, 1] - rest_tets[:, 0], rest_tets[:, 2] - rest_tets[:, 0]),
            rest_tets[:, 3] - rest_tets[:, 0],
        )

    def get_pos(self) -> np.ndarray:
        _positions, center, _rotation = self.binding.rigid_transform()
        return center.detach().cpu().numpy().astype(np.float32)

    def get_quat(self) -> np.ndarray:
        from .plush_physics import matrix_to_quaternion_wxyz

        _positions, _center, rotation = self.binding.rigid_transform()
        return matrix_to_quaternion_wxyz(rotation.detach().cpu().numpy()).astype(np.float32)

    def set_pos(self, position: object) -> None:
        self.fem.set_position(np.asarray(position, dtype=np.float32).reshape(3))

    def set_quat(self, quaternion: object) -> None:
        from .plush_physics import quaternion_wxyz_to_matrix

        center = (
            self.binding.current_positions().mean(dim=0).detach().cpu().numpy()
        )
        rotation = quaternion_wxyz_to_matrix(quaternion)
        rest = self.rest_positions.detach().cpu().numpy()
        rest_center = self.rest_center.detach().cpu().numpy()
        positions = (rest - rest_center) @ rotation + center
        self.fem.set_position(positions.astype(np.float32))

    def set_dofs_velocity(self, velocity: object) -> None:
        values = np.asarray(velocity, dtype=np.float32).reshape(-1)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("FEM reset velocity must contain six finite values")
        if np.any(np.abs(values) > 1.0e-8):
            raise ValueError("FEM adapter currently supports only zero reset velocity")
        self.fem.set_velocity(np.zeros(3, dtype=np.float32))

    def update_visual(self) -> None:
        self.binding.update()

    def diagnostics(self) -> dict[str, object]:
        positions = self.binding.current_positions().detach().cpu().numpy()
        tets = positions[self._elements]
        current_signed_six_volume = np.einsum(
            "ij,ij->i",
            np.cross(tets[:, 1] - tets[:, 0], tets[:, 2] - tets[:, 0]),
            tets[:, 3] - tets[:, 0],
        )
        valid_rest = np.abs(self._rest_signed_six_volume) > 1.0e-14
        volume_ratio = np.full(len(tets), np.nan, dtype=np.float64)
        volume_ratio[valid_rest] = (
            current_signed_six_volume[valid_rest]
            / self._rest_signed_six_volume[valid_rest]
        )
        finite = volume_ratio[np.isfinite(volume_ratio)]
        finite_mask = np.isfinite(volume_ratio)
        inverted_mask = finite_mask & (volume_ratio <= 0.0)
        rest_volume = np.abs(self._rest_signed_six_volume) / 6.0
        total_rest_volume = float(np.sum(rest_volume))
        inverted_rest_volume = rest_volume[inverted_mask]
        median_rest_volume = float(np.median(rest_volume[valid_rest]))
        significant_inversion = inverted_mask & (
            rest_volume >= median_rest_volume * 0.1
        )
        current_extents = np.ptp(positions, axis=0)
        rest_extents = np.ptp(self.rest_positions.detach().cpu().numpy(), axis=0)
        return {
            "kind": "fem-plush",
            "solver": "Genesis_FEM_implicit_corotated_FP32",
            "effective_material": "tightly_stuffed_volumetric_continuum",
            "physics_vertices": int(len(positions)),
            "physics_tetrahedra": int(len(self._elements)),
            "visual_vertices": int(len(self.binding.rest_visual)),
            "current_center_m": positions.mean(axis=0).tolist(),
            "current_extents_m": current_extents.tolist(),
            "extent_ratio_to_rest": (current_extents / rest_extents).tolist(),
            "material": self.config.to_dict(),
            "embedding": dict(self.binding.embedding_diagnostics),
            "visual_surface_quality": self.binding.surface_quality_diagnostics(),
            "volume_ratio": {
                "p05": float(np.percentile(finite, 5)),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95)),
                "minimum": float(np.min(finite)),
                "inverted_tetrahedra": int(np.count_nonzero(inverted_mask)),
                "significantly_inverted_tetrahedra": int(
                    np.count_nonzero(significant_inversion)
                ),
                "inverted_rest_volume_fraction": (
                    float(np.sum(inverted_rest_volume) / total_rest_volume)
                    if total_rest_volume > 0.0
                    else None
                ),
                "rest_tet_volume_m3": {
                    "minimum": float(np.min(rest_volume[valid_rest])),
                    "median": median_rest_volume,
                    "p95": float(np.percentile(rest_volume[valid_rest], 95)),
                    "maximum": float(np.max(rest_volume[valid_rest])),
                },
                "maximum_inverted_rest_tet_volume_m3": (
                    float(np.max(inverted_rest_volume))
                    if len(inverted_rest_volume)
                    else 0.0
                ),
            },
            "rigid_contact_identity_available": False,
            "contact": "native_legacy_rigid_FEM_surface_contact",
            "synthetic_attachment": False,
        }
